"""Causal test of "false alarms under-attend to the enrollment" by attention-score intervention at the decision token.

Observation (04): at LLM layers 18-21 the decision position (``language``) puts 0.087-0.104 of its attention on the
enrollment audio tokens for correct rejections but 0.034-0.068 for false alarms (FA). If FAs happen *because* the
decision under-consults the enrollment, adding log(k) to the pre-softmax scores from the decision query to the
enrollment keys (softmax renormalises) should flip FAs to `` None``; a matched boost of the mixture (all of it, or a
random block of mixture tokens as large as the enrollment) should not do so as efficiently.

Subcommands:

- ``prepare``: from ts-asr-benchmark v12 states (``states-full.npz``), the sample ids (all FAs, seeded draws of
  hits and correct rejections; HF/vLLM first-token disagreements excluded) and the heads of layers 18-21 with the
  largest enrollment-attention gap rejection - FA (plus a seeded random head set as control).
- ``run``: per sample, the exact input of extract_states.py truncated after ``language``; a full forward gives the
  baseline decision and the KV cache, the cache is cropped before ``language`` and the decision token alone is
  rerun for every setting (batched), with an additive bias on the attention mask of the chosen layers / heads at
  the chosen key positions. The bias only touches the decision query row (the only query in the rerun), so the
  KV of all earlier positions are unchanged: this is an intervention on what the decision reads, nothing else.
- ``report``: decision changes per setting relative to the unpatched baseline.
"""
import argparse
import copy
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from extract_states import Encoded, segments  # noqa: E402  validated input building

NONE_ID = 2240
LATE = (18, 19, 20, 21)  # layers where the rejection/FA enrollment-attention gap was observed (04)
EARLY = (12, 13, 14, 15)  # comparison block, where the presence probe already reaches AUC 0.997 (04)
KS = (1.5, 2.0, 3.0)
DOSE_KS = (5.0, 10.0)  # dose-response beyond the rejection-level enrollment mass that k=3 already reaches
BIAS_LAYERS = tuple(range(12, 22))
SUPPRESS_KS = (0.5, 1 / 3)  # necessity check: does *lowering* enrollment attention turn rejections into FAs?
TOP_HEADS = 5
N_HITS, N_REJ = 1000, 500
MAX_ROWS = 20  # settings per batched rerun (bounds the repeated KV cache)
MIN_AGREEMENT = 0.99
MASS_LAYERS = tuple(range(12, 22))  # attention mass recorded after the intervention


def prepare(args):
    z = np.load(args.states, allow_pickle=True)
    keep = z["final_argmax"] == z["emitted_token"]
    absent, empty = z["reference_empty"], z["prediction_empty"]
    groups = {"fa": keep & absent & ~empty, "hit": keep & ~absent & ~empty, "rej": keep & absent & empty}
    rng = np.random.default_rng(args.seed)
    ids = z["id"]
    chosen = {"fa": np.flatnonzero(groups["fa"]),
              "hit": np.sort(rng.choice(np.flatnonzero(groups["hit"]), N_HITS, replace=False)),
              "rej": np.sort(rng.choice(np.flatnonzero(groups["rej"]), N_REJ, replace=False))}
    heads = z["attn_mass_heads"][:, :, :, 0].astype(np.float32)  # (S, 28, 16) enrollment mass
    gap = heads[groups["rej"]].mean(0) - heads[groups["fa"]].mean(0)  # (28, 16)
    cand = [(l, h) for l in LATE for h in range(gap.shape[1])]
    ranked = sorted(cand, key=lambda lh: -gap[lh])
    top = ranked[:TOP_HEADS]
    rest = [lh for lh in cand if lh not in top]
    rand = [rest[i] for i in np.random.default_rng(args.seed + 1).choice(len(rest), TOP_HEADS, replace=False)]
    out = Path(args.out_dir)
    with open(out / "ids.txt", "w") as f:
        for g in ("fa", "hit", "rej"):
            for i in chosen[g]:
                f.write(f"{ids[i]}\n")
    info = {"states": args.states, "seed": args.seed, "counts": {g: int(len(v)) for g, v in chosen.items()},
            "group_of": {str(ids[i]): g for g, v in chosen.items() for i in v},
            "top_heads": [{"layer": l, "head": h, "rej": float(heads[groups["rej"], l, h].mean()),
                           "fa": float(heads[groups["fa"], l, h].mean()),
                           "hit": float(heads[groups["hit"], l, h].mean()), "gap": float(gap[l, h])} for l, h in top],
            "random_heads": [{"layer": l, "head": h, "gap": float(gap[l, h])} for l, h in rand],
            "gap_share_top_heads": float(sum(gap[lh] for lh in top) / sum(gap[lh] for lh in cand)),
            "layer_mass": {str(l): {g: float(z["attn_mass"][groups[g], l, 0].mean()) for g in groups}
                           for l in range(12, 24)}}
    (out / "heads.json").write_text(json.dumps(info, indent=1) + "\n")
    print(json.dumps({k: v for k, v in info.items() if k != "group_of"}, indent=1))


def settings(info, which="main"):
    if which == "l12-21":  # follow-up: is the L12-21 enrollment boost specific? same rows for all three targets
        rows = [{"name": "baseline", "target": None, "layers": [], "heads": None, "k": 1.0},
                {"name": "control", "target": None, "layers": [], "heads": None, "k": 1.0}]
        for target in ("enroll", "mix", "rand"):
            for k in (3.0, 10.0):
                rows.append({"name": f"{target} L12-21 k={k:g}", "target": target, "layers": list(BIAS_LAYERS),
                             "heads": None, "k": k})
        return rows
    top = [(d["layer"], d["head"]) for d in info["top_heads"]]
    rand = [(d["layer"], d["head"]) for d in info["random_heads"]]
    rows = [{"name": "baseline", "target": None, "layers": [], "heads": None, "k": 1.0},
            {"name": "control", "target": None, "layers": [], "heads": None, "k": 1.0}]
    for target in ("enroll", "mix", "rand"):
        for tag, layers in (("L18-21", LATE), ("L12-15", EARLY)):
            for k in KS + DOSE_KS:
                rows.append({"name": f"{target} {tag} k={k:g}", "target": target, "layers": list(layers),
                             "heads": None, "k": k})
    for layer in LATE:
        for k in KS:
            rows.append({"name": f"enroll L{layer} k={k:g}", "target": "enroll", "layers": [layer], "heads": None, "k": k})
    for tag, hs in (("top5", top), ("rand5", rand)):
        for k in KS + DOSE_KS:
            rows.append({"name": f"enroll {tag}-heads k={k:g}", "target": "enroll", "layers": sorted({l for l, _ in hs}),
                         "heads": [list(x) for x in hs], "k": k})
    for k in (3.0, 10.0):
        rows.append({"name": f"enroll L12-21 k={k:g}", "target": "enroll", "layers": list(BIAS_LAYERS), "heads": None, "k": k})
    for k in SUPPRESS_KS:
        rows.append({"name": f"enroll L18-21 k={k:.2g}", "target": "enroll", "layers": list(LATE), "heads": None, "k": k})
    return rows


def run(args):
    from swift.model import get_model_processor
    from swift.template import get_template

    import open_audio_llm.integrations.ms_swift.register_qwen3_asr  # noqa: F401

    out_dir = Path(args.out_dir)
    info = json.loads((out_dir / "heads.json").read_text())
    grid = settings(info, args.settings)
    run_dir = Path(args.run_dir)
    predictions = {p["id"]: p for p in map(json.loads, open(run_dir / "predictions.jsonl")) if p["status"] == "success"}
    by_id = {s["id"]: s for s in map(json.loads, open(run_dir / "samples.jsonl"))}
    wanted = [line.strip() for line in open(out_dir / "ids.txt") if line.strip()]
    if args.limit:
        wanted = wanted[:: max(1, len(wanted) // args.limit)][: args.limit]
    samples = [by_id[i] for i in wanted]
    print(f"{len(samples)} samples, {len(grid)} settings", flush=True)

    model, processor = get_model_processor(args.checkpoint, model_type="amphion_asr_1.7b", torch_dtype=torch.bfloat16,
                                           attn_impl="eager", device_map="cuda:0")
    model.eval()
    thinker = model.thinker
    tower, text = thinker.audio_tower, thinker.model
    tower.config._attn_implementation = "sdpa"  # global audio attention, as extract_states.py
    assert text.config._attn_implementation == "eager"
    tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor
    template = get_template(processor, max_length=None, template_type="amphion_asr_1.7b")
    template.set_mode("train")
    language_id = tokenizer.convert_tokens_to_ids("language")
    assert tokenizer.encode(" None", add_special_tokens=False) == [NONE_ID]
    n_heads = text.config.num_attention_heads
    dev = "cuda:0"

    state = {"h0": None, "bias": {}, "seg": None, "mass": {}}

    def embed_hook(module, inputs):
        state["h0"] = inputs[0]

    def bias_hook(layer):
        def hook(module, a, kw):
            bias = state["bias"].get(layer)
            if bias is None:
                return None
            mask = kw.get("attention_mask")
            kw["attention_mask"] = bias if mask is None else mask[:, :, -1:, : bias.shape[-1]] + bias
            return a, kw
        return hook

    def mass_hook(layer):
        def hook(module, inputs, output):
            seg = state["seg"]
            w = output[1][:, :, -1].float()  # (b, heads, K) decision row
            state["mass"][layer] = torch.stack([w[..., seg["enroll"]].sum(-1).mean(-1),
                                                w[..., seg["mix"]].sum(-1).mean(-1)], -1)  # (b, 2)
        return hook

    text.layers[0].register_forward_pre_hook(embed_hook)
    for layer in BIAS_LAYERS:
        text.layers[layer].self_attn.register_forward_pre_hook(bias_hook(layer), with_kwargs=True)
    for layer in MASS_LAYERS:
        text.layers[layer].self_attn.register_forward_hook(mass_hook(layer))

    S, G = len(samples), len(grid)
    decision = np.zeros((S, G), np.int64)
    none_margin = np.zeros((S, G), np.float32)
    mass = np.zeros((S, G, len(MASS_LAYERS), 2), np.float32)  # head-mean enroll / mixture mass after intervention
    full_argmax = np.zeros(S, np.int64)
    emitted = np.zeros(S, np.int64)
    n_mix = np.zeros(S, np.int64)
    rand_size = np.zeros(S, np.int64)
    logit_maxdiff = np.zeros(S, np.float32)

    def rerun(tail, cache, pos, rows, key_len, cols):
        b = len(rows)
        state["bias"] = {}
        for layer in BIAS_LAYERS:
            bias = torch.zeros(b, n_heads, 1, key_len, device=dev)
            used = False
            for i, row in enumerate(rows):
                if layer not in row["layers"]:
                    continue
                hs = [h for l, h in row["heads"] if l == layer] if row["heads"] else list(range(n_heads))
                c = cols[row["target"]]
                bias[i, torch.tensor(hs)[:, None], 0, c[None, :]] = math.log(row["k"])
                used = True
            if used:
                state["bias"][layer] = bias.to(torch.bfloat16)
        cache_b = copy.deepcopy(cache)
        if b > 1:
            cache_b.batch_repeat_interleave(b)
        positions = torch.tensor([pos], device=dev)
        out = text(inputs_embeds=tail.expand(b, -1, -1), position_ids=positions.view(1, -1).expand(b, -1),
                   past_key_values=cache_b, cache_position=positions, use_cache=True)
        state["bias"] = {}
        m = torch.stack([state["mass"][l] for l in MASS_LAYERS], 1)  # (b, layers, 2)
        return thinker.lm_head(out.last_hidden_state[:, -1]).float(), m

    loader = torch.utils.data.DataLoader(Encoded(samples, predictions, template), batch_size=None,
                                         num_workers=args.workers, prefetch_factor=4 if args.workers else None)
    start = time.time()
    for done, (index, encoded) in enumerate(loader):
        batch = template.data_collator([encoded])
        labels = batch.pop("labels")[0]
        dec = int((labels != -100).nonzero()[0])
        input_ids = batch["input_ids"][0]
        assert int(input_ids[dec]) == language_id
        seg = segments(input_ids, int(encoded["enroll_n_frames"]), tokenizer)
        state["seg"] = seg
        T = dec + 1
        for key in ("input_ids", "attention_mask"):
            batch[key] = batch[key][:, :T]
        batch = {k: v.to(dev) if torch.is_tensor(v) else v for k, v in batch.items()}
        batch["input_features"] = batch["input_features"].to(torch.bfloat16)
        mix = torch.arange(seg["mix"].start, seg["mix"].stop)
        size = min(seg["n_enroll"], len(mix))  # mixtures shorter than the enrollment: the whole mixture
        offset = int(np.random.default_rng(args.seed * 1_000_003 + index).integers(0, len(mix) - size + 1))
        cols = {"enroll": torch.arange(seg["enroll"].start, seg["enroll"].stop), "mix": mix,
                "rand": mix[offset: offset + size], None: torch.arange(0)}
        with torch.inference_mode():
            outputs = model(**batch, use_cache=True)
            full_logits = outputs.logits[0, -1].float()
            cache = outputs.past_key_values
            cache.crop(dec)
            tail = state["h0"][:, dec:T]
            base, base_mass = rerun(tail, cache, dec, [grid[0]], T, cols)
            logits, masses = [base], [base_mass]
            for s in range(1, G, MAX_ROWS):
                lg, ms = rerun(tail, cache, dec, grid[s:s + MAX_ROWS], T, cols)
                logits.append(lg)
                masses.append(ms)
            logits, masses = torch.cat(logits), torch.cat(masses)
        others = logits.clone()
        others[:, NONE_ID] = -float("inf")
        decision[index] = logits.argmax(-1).cpu().numpy()
        none_margin[index] = (logits[:, NONE_ID] - others.max(-1).values).cpu().numpy()
        mass[index] = masses.cpu().numpy()
        full_argmax[index] = int(full_logits.argmax())
        emitted[index] = int(input_ids[dec + 1])
        n_mix[index] = len(mix)
        rand_size[index] = size
        logit_maxdiff[index] = float((base[0] - full_logits).abs().max())
        if (done + 1) % 200 == 0:
            print(f"{done + 1}/{S} {(done + 1) / (time.time() - start):.2f} samples/s", flush=True)
    elapsed = time.time() - start

    z = np.load(info["states"], allow_pickle=True)
    pos_of = {str(i): n for n, i in enumerate(z["id"])}
    rows_states = np.array([pos_of[s["id"]] for s in samples])
    stored = z["final_argmax"][rows_states]
    stored_mass = z["attn_mass"][rows_states][:, list(MASS_LAYERS), 0]
    summary = {
        "samples": S, "settings": G, "seconds": elapsed, "samples_per_s": S / elapsed,
        "agreement_full_vs_stored_hf": float((full_argmax == stored).mean()),
        "agreement_full_vs_vllm": float((full_argmax == emitted).mean()),
        "agreement_rerun_baseline_vs_full": float((decision[:, 0] == full_argmax).mean()),
        "agreement_batched_control_vs_baseline": float((decision[:, 1] == decision[:, 0]).mean()),
        "rerun_logit_maxdiff_max": float(logit_maxdiff.max()),
        "baseline_enroll_mass_absdiff_vs_stored_max": float(np.abs(mass[:, 0, :, 0] - stored_mass).max()),
        "rand_block_smaller_than_enroll": int((rand_size < 65).sum()),
        "checkpoint": args.checkpoint, "run_dir": args.run_dir,
    }
    summary["pass"] = summary["agreement_full_vs_stored_hf"] >= MIN_AGREEMENT
    np.savez(out_dir / "patch.npz", id=np.array([s["id"] for s in samples]),
             group=np.array([info["group_of"][s["id"]] for s in samples]), decision=decision,
             none_margin=none_margin, mass=mass, mass_layers=np.array(MASS_LAYERS), full_argmax=full_argmax,
             stored_argmax=stored, emitted_token=emitted, n_mix=n_mix, rand_size=rand_size, grid=np.array(json.dumps(grid)))
    (out_dir / "patch.summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


def report(args):
    out_dir = Path(args.out_dir)
    z = np.load(out_dir / "patch.npz")
    grid = json.loads(str(z["grid"]))
    group = z["group"]
    base_none = z["decision"][:, 0] == NONE_ID
    fa = (group == "fa") & ~base_none
    hit = (group == "hit") & ~base_none
    rej = (group == "rej") & base_none
    late = [list(z["mass_layers"]).index(l) for l in LATE]
    rows = []
    for g, s in enumerate(grid):
        none = z["decision"][:, g] == NONE_ID
        r = {"name": s["name"], "fa_flipped": int((fa & none).sum()), "hits_harmed": int((hit & none).sum()),
             "rej_changed": int((rej & ~none).sum()),
             "fa_rate": float((fa & none).sum() / fa.sum()), "hit_harm_rate": float((hit & none).sum() / hit.sum()),
             "rej_change_rate": float((rej & ~none).sum() / rej.sum()),
             # mean change of the FA None-margin (nat) and of the head-mean enrollment / mixture mass at L18-21
             "fa_margin_shift": float((z["none_margin"][fa, g] - z["none_margin"][fa, 0]).mean()),
             "fa_enroll_mass_l18_21": float(z["mass"][fa][:, g, late, 0].mean()),
             "fa_mix_mass_l18_21": float(z["mass"][fa][:, g, late, 1].mean())}
        r["rate_ratio"] = r["fa_rate"] / r["hit_harm_rate"] if r["hit_harm_rate"] else None
        rows.append(r)
    result = {"n": {"fa": int(fa.sum()), "hit": int(hit.sum()), "rej": int(rej.sum())},
              "baseline_rej_enroll_mass_l18_21": float(z["mass"][rej][:, 0, late, 0].mean()),
              "baseline_hit_enroll_mass_l18_21": float(z["mass"][hit][:, 0, late, 0].mean()), "rows": rows}
    (out_dir / "patching.json").write_text(json.dumps(result, indent=1) + "\n")
    print({k: v for k, v in result.items() if k != "rows"})
    print(f"{'setting':28s} {'FA->None':>9s} {'hit->None':>9s} {'rej->text':>9s} {'dMarginFA':>9s} {'enrFA':>6s} {'mixFA':>6s}")
    for r in rows:
        print(f"{r['name']:28s} {r['fa_flipped']:4d} {r['fa_rate']:5.1%} {r['hits_harmed']:4d} {r['hit_harm_rate']:5.1%}"
              f" {r['rej_changed']:4d} {r['fa_margin_shift']:+9.2f} {r['fa_enroll_mass_l18_21']:6.3f} {r['fa_mix_mass_l18_21']:6.3f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--states", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--seed", type=int, default=0)
    r = sub.add_parser("run")
    r.add_argument("--checkpoint", required=True)
    r.add_argument("--run-dir", required=True)
    r.add_argument("--out-dir", required=True)
    r.add_argument("--limit", type=int, default=0, help="smoke test on an evenly spaced subset")
    r.add_argument("--settings", choices=("main", "l12-21"), default="main")
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--workers", type=int, default=8)
    q = sub.add_parser("report")
    q.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    {"prepare": prepare, "run": run, "report": report}[args.cmd](args)


if __name__ == "__main__":
    main()
