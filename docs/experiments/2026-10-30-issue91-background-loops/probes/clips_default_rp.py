"""Send the issue-91 clips to a server whose DEFAULT sampling params carry repetition_penalty.

The request omits repetition_penalty, so a loop-free result proves the server-side
default reaches the sampler the way AmphionEval's requests will see it.
"""
import argparse, base64, io, json, re, concurrent.futures as cf
import requests, soundfile as sf

ap = argparse.ArgumentParser()
ap.add_argument("--root", required=True)
ap.add_argument("--port", type=int, required=True)
ap.add_argument("--model", required=True)
ap.add_argument("--out", required=True)
args = ap.parse_args()

rows = [json.loads(l) for l in open(f"{args.root}/manifest.jsonl")]

def call(x):
    b = io.BytesIO(); sf.write(b, x, 16000, format="WAV", subtype="PCM_16")
    body = {"model": args.model, "temperature": 0, "max_tokens": 512, "messages": [
        {"role": "system", "content": ""},
        {"role": "user", "content": [{"type": "input_audio", "input_audio": {
            "data": base64.b64encode(b.getvalue()).decode(), "format": "wav"}}]}]}
    r = requests.post(f"http://127.0.0.1:{args.port}/v1/chat/completions", json=body, timeout=300)
    r.raise_for_status()
    c = r.json()["choices"][0]["message"]["content"] or ""
    return c.partition("<asr_text>")[2] if "<asr_text>" in c else c

jobs = []
for r in rows:
    x, sr = sf.read(f"{args.root}/{r['file']}", dtype="float32")
    for s0 in (0, 0.3):
        jobs.append((r["file"], s0, x[int(s0 * sr):]))
with cf.ThreadPoolExecutor(8) as ex:
    futs = {ex.submit(call, y): (f, s0) for f, s0, y in jobs}
    out = [dict(zip(("file", "shift"), futs[fu]), text=fu.result()) for fu in cf.as_completed(futs)]
with open(args.out, "w") as fh:
    for o in sorted(out, key=lambda o: (o["file"], o["shift"])):
        fh.write(json.dumps(o, ensure_ascii=False) + "\n")

def loop(s):
    s = re.sub(r"[\s，。？！、,.?!]", "", s)
    return bool(re.search(r"(.{1,4}?)\1{7,}", s))

h = [o for o in out if o["file"].startswith("hall")]
c = [o for o in out if o["file"].startswith("control")]
print(f"hall: loops {sum(loop(o['text']) for o in h)}/{len(h)}, empty {sum(not o['text'].strip() for o in h)}")
print(f"control: loops {sum(loop(o['text']) for o in c)}/{len(c)}, empty {sum(not o['text'].strip() for o in c)}")
for o in c:
    print("control", o["shift"], o["text"][:80])
