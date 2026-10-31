"""End-to-end check of the confidence_gate middleware on a live server with the issue-91 clips.

Per clip: an AmphionEval-shaped request (system block, non-streaming, max_tokens 512) and an
audiollm-go-shaped one (user-only template, streaming with usage, decodeBudget). Then prefix
continuations: a control's own transcript rolled back by 5 tokens as audiollm-go does (must pass) and the device's background text
(must come back empty with ``rejected``). Optionally compares the runtime-shaped decisions with
the offline probe (confprobe.py items) at the server's threshold. Prints counts, no transcripts.

    python confgate_smoke.py --server URL --model NAME --template T.jinja --clips ROOT [--probe items.jsonl --threshold T]
"""
import argparse, base64, json, re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests, soundfile as sf

NO_SPEECH = "language None<asr_text>"


def budget(samples):
    return min(512, max(128, 96 + (samples * 24 + 15999) // 16000))


def post(args, body):
    r = requests.post(f"{args.server}/v1/chat/completions", json={"model": args.model, "temperature": 0, **body},
                      timeout=600, stream=bool(body.get("stream")))
    r.raise_for_status()
    if not body.get("stream"):
        j = r.json()
        return j["choices"][0]["message"]["content"], j["choices"][0]["finish_reason"], j["amphion_confidence"], j.get("usage"), j["choices"][0]["logprobs"]
    content, finish, report, usage, done = "", None, None, None, False
    for line in r.iter_lines(decode_unicode=True):
        if not line:
            continue
        data = line.removeprefix("data: ")
        if data == "[DONE]":
            done = True
            break
        chunk = json.loads(data)
        usage = chunk.get("usage") or usage
        report = chunk.get("amphion_confidence") or report
        for c in chunk["choices"]:
            content += c["delta"].get("content") or ""
            finish = c["finish_reason"] or finish
    assert done, "stream ended without [DONE]"
    return content, finish, report, usage, None


def audio_part(path):
    x, sr = sf.read(path, dtype="float32")
    assert sr == 16000
    import io
    buf = io.BytesIO(); sf.write(buf, x, sr, format="WAV", subtype="PCM_16")
    return {"type": "input_audio", "input_audio": {"data": base64.b64encode(buf.getvalue()).decode(), "format": "wav"}}, len(x)


def main():
    ap = argparse.ArgumentParser()
    for a in ("--server", "--model", "--template", "--clips"):
        ap.add_argument(a, required=True)
    ap.add_argument("--probe")
    ap.add_argument("--threshold", type=float)
    args = ap.parse_args()
    template = open(args.template).read()
    rows = [json.loads(l) for l in open(Path(args.clips) / "manifest.jsonl")]

    def one(row):
        part, samples = audio_part(Path(args.clips) / row["file"])
        user = [{"role": "user", "content": [part]}]
        ae = post(args, {"messages": [{"role": "system", "content": ""}] + user, "max_tokens": 512})
        rt = post(args, {"messages": user, "chat_template": template, "max_tokens": budget(samples), "stream": True,
                         "stream_options": {"include_usage": True}})
        rt_json = post(args, {"messages": user, "chat_template": template, "max_tokens": budget(samples)})
        out = {"file": row["file"], "bg": row["file"].startswith("hallucination/"), "ae": ae, "rt": rt, "rt_json": rt_json}
        if not out["bg"] and not rt[2]["rejected"]:
            # audiollm-go rolls a prefix back by tokens (partialPrefix), so it stays token-aligned
            head, _, text = rt[0].partition("<asr_text>")
            tokens = requests.post(f"{args.server}/tokenize", json={"model": args.model, "prompt": text, "add_special_tokens": False}).json()["tokens"]
            kept = requests.post(f"{args.server}/detokenize", json={"model": args.model, "tokens": tokens[:max(1, len(tokens) - 5)]}).json()["prompt"]
            prefix = head + "<asr_text>" + kept
            out["prefix_keep"] = post(args, {"messages": user + [{"role": "assistant", "content": prefix}], "chat_template": template,
                                             "continue_final_message": True, "add_generation_prompt": False, "max_tokens": budget(samples)})
            out["logprobs"] = post(args, {"messages": user, "chat_template": template, "max_tokens": budget(samples), "logprobs": True})
        if out["bg"] and row["device_048"]:
            prefix = "language Chinese<asr_text>" + row["device_048"][:8]
            out["prefix_bg"] = post(args, {"messages": user + [{"role": "assistant", "content": prefix}], "chat_template": template,
                                           "continue_final_message": True, "add_generation_prompt": False, "max_tokens": budget(samples),
                                           "stream": True, "stream_options": {"include_usage": True}})
        return out

    with ThreadPoolExecutor(8) as pool:
        results = list(pool.map(one, rows))
    bg = [r for r in results if r["bg"]]
    ctrl = [r for r in results if not r["bg"]]
    rej = lambda r, k: r[k][2]["rejected"]
    print(f"background clips {len(bg)}: AE-shaped rejected {sum(rej(r, 'ae') for r in bg)}, runtime-shaped rejected {sum(rej(r, 'rt') for r in bg)}")
    print(f"controls {len(ctrl)}: AE-shaped kept {sum(not rej(r, 'ae') for r in ctrl)}, runtime-shaped kept {sum(not rej(r, 'rt') for r in ctrl)}")
    already_empty = sum(1 for r in bg if r["rt"][2]["score"] is None)
    print(f"  (runtime-shaped outputs already empty before the gate: {already_empty})")
    bad = []
    for r in results:
        for k in ("ae", "rt", "rt_json"):
            content, finish, report = r[k][:3]
            if report["rejected"] and (content != NO_SPEECH or finish != "stop"):
                bad.append((r["file"], k, "rejected but content/finish wrong"))
            if finish != "stop" and not report["rejected"]:
                bad.append((r["file"], k, f"finish {finish}"))
        # two separate decodes: logprobs jitter with vLLM batching, so compare the outcome only
        if r["rt"][2]["rejected"] != r["rt_json"][2]["rejected"] or (not r["rt"][2]["rejected"] and r["rt"][0] != r["rt_json"][0]):
            bad.append((r["file"], "stream vs json outcome differ", r["rt"][2], r["rt_json"][2]))
        if r["rt"][3] is None or r["rt"][3]["completion_tokens"] <= 0:
            bad.append((r["file"], "stream usage missing"))
        if "prefix_keep" in r and (r["prefix_keep"][2]["rejected"] or not r["prefix_keep"][2]["prefix"] or r["prefix_keep"][2]["head_tokens"] == 0):
            bad.append((r["file"], "control prefix continuation", r["prefix_keep"][2]))
        if "logprobs" in r and not r["logprobs"][4]:
            bad.append((r["file"], "client logprobs dropped"))
        if "prefix_bg" in r and not (r["prefix_bg"][2]["rejected"] and r["prefix_bg"][0] == "" and r["prefix_bg"][1] == "stop"):
            bad.append((r["file"], "background prefix continuation not rejected", r["prefix_bg"][2]))
    print(f"prefix continuations: controls {sum('prefix_keep' in r for r in ctrl)} (must pass), "
          f"background {sum('prefix_bg' in r for r in bg)} rejected {sum(r['prefix_bg'][2]['rejected'] for r in bg if 'prefix_bg' in r)}")
    if args.probe:
        probe = {json.loads(l)["id"].removeprefix("clip:"): json.loads(l) for l in open(args.probe) if '"issue91"' in l}
        differ = []
        for r in results:
            p = probe[r["file"]]
            lp = p["lp"][:16]
            loop = p["finish"] == "length" and bool(re.search(r"(.{1,4}?)\1{7,}", re.sub(r"\W", "", p["text"])))
            expected = bool(lp) and bool(re.sub(r"\W", "", p["text"])) and (sum(lp) / len(lp) < args.threshold or loop)
            if expected != r["rt"][2]["rejected"]:
                differ.append((r["file"].split("/")[1][:8], expected, r["rt"][2]))
        print(f"runtime-shaped decisions vs offline probe: {len(results) - len(differ)}/{len(results)} agree", differ or "")
    print("problems:", bad or "none")


if __name__ == "__main__":
    main()
