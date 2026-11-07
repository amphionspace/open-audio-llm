"""Training-free activation steering of the AmphionASR first-token decision (`` None`` = target absent vs a
language token).

Three subcommands:

- ``fit``: from ts-asr-benchmark v12 states only (``states-full.npz`` of extract_states.py), per layer l the
  steering direction v_l = mean residual[l] over target-absent & correctly-empty samples minus the mean over
  target-present & non-empty samples, plus the layer-12 presence probe (same fit as probe_transfer.py) for the
  adaptive variant. The unit step along v_l is ``UNIT_FRACTION`` x the median residual norm at that layer.
- ``run``: per sample of an AmphionEval run, the exact input of extract_states.py (same template, audio,
  SDPA audio tower, eager LLM) truncated after the decision token ``language``; one full forward gives the
  baseline decision and the KV cache. The cache is cropped to the prompt part before the "tail" (all tokens
  after the last audio pad, up to and including the decision token: ``<|audio_end|>``, ``<|im_end|>``, the
  assistant header and ``language``), and every steering setting reruns only that tail, batched. Steering adds
  alpha * unit_l * v_l / |v_l| to the output of decoder layer l (= residual[l] of the states npz) either at
  the decision position only (``dec``) or at every tail position (``tail``). ``adaptive`` scales alpha by
  (1 - P_present) of the layer-12 probe read at the decision position in the unsteered tail.
- ``report``: aggregates the per-sample decisions into FA removed / hits lost per setting.

Only the logits at the decision position are read; they depend on nothing after it (causal LM), so
teacher forcing beyond ``language`` is unnecessary.
"""
import argparse
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from extract_states import Encoded, select  # noqa: E402  validated input building

LAYERS = (12, 14, 16, 18, 20)
ALPHAS = (0.5, 1.0, 2.0, 4.0, 8.0)
UNIT_FRACTION = 0.1  # unit step = 10% of the median |residual[l]| at the decision position (v12)
PROBE_LAYER = 12
NONE_ID = 2240
MAX_ROWS = 34  # steering rows per tail forward (bounds the repeated KV cache memory)
MIN_AGREEMENT = 0.99


def fit(args):
    z = np.load(args.states, allow_pickle=True)
    keep = z["final_argmax"] == z["emitted_token"]
    absent, empty = z["reference_empty"], z["prediction_empty"]
    neg, pos = keep & absent & empty, keep & ~absent & ~empty
    residual = z["residual"]
    out, stats = {}, {}
    for layer in LAYERS:
        x = residual[:, layer].astype(np.float32)
        v = x[neg].mean(0) - x[pos].mean(0)
        median_norm = float(np.median(np.linalg.norm(x[keep], axis=1)))
        out[f"dir_{layer}"] = v / np.linalg.norm(v)
        out[f"unit_{layer}"] = np.float32(UNIT_FRACTION * median_norm)
        proj = x @ out[f"dir_{layer}"]
        stats[layer] = {"v_norm": float(np.linalg.norm(v)), "median_residual_norm": median_norm,
                        "unit": float(out[f"unit_{layer}"]),
                        "proj_mean": {k: float(proj[m].mean()) for k, m in
                                      (("absent_empty", neg), ("present_text", pos), ("false_alarm", keep & absent & ~empty))}}
    # Presence probe, identical to probe_transfer.py (standardised, class-balanced BCE, L2 1e-3, LBFGS).
    dev = "cuda"
    xt = torch.as_tensor(residual[keep][:, PROBE_LAYER].astype(np.float32), device=dev)
    yt = torch.as_tensor(~absent[keep], dtype=torch.float32, device=dev)
    mu, sd = xt.mean(0), xt.std(0) + 1e-4
    w = torch.zeros(xt.shape[1], device=dev, requires_grad=True)
    b = torch.zeros((), device=dev, requires_grad=True)
    weight = torch.where(yt > 0, 0.5 / yt.mean(), 0.5 / (1 - yt.mean()))
    opt = torch.optim.LBFGS([w, b], max_iter=300, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = (weight * torch.nn.functional.binary_cross_entropy_with_logits(
            ((xt - mu) / sd) @ w + b, yt, reduction="none")).mean() + 1e-3 * (w ** 2).sum()
        loss.backward()
        return loss

    opt.step(closure)
    with torch.no_grad():
        p = torch.sigmoid(((xt - mu) / sd) @ w + b)
        stats["probe_train_acc"] = float(((p > 0.5).float() == yt).float().mean())
    out.update(probe_mu=mu.cpu().numpy(), probe_sd=sd.cpu().numpy(), probe_w=w.detach().cpu().numpy(),
               probe_b=np.float32(b.item()))
    np.savez(args.out, layers=np.array(LAYERS), **out)
    stats.update(states=args.states, n_absent_empty=int(neg.sum()), n_present_text=int(pos.sum()),
                 unit_fraction=UNIT_FRACTION, probe_layer=PROBE_LAYER)
    Path(args.out).with_suffix(".json").write_text(json.dumps(stats, indent=1) + "\n")
    print(json.dumps(stats, indent=1))


def settings():
    # "control" (no steering) runs inside the first batched tail forward: checks batched vs single-row numerics.
    rows = [{"layer": None, "alpha": 0.0, "where": None, "mode": "baseline"},
            {"layer": None, "alpha": 0.0, "where": None, "mode": "control"}]
    for layer in LAYERS:
        for alpha in ALPHAS:
            for where in ("dec", "tail"):
                for mode in ("fixed", "adaptive"):
                    rows.append({"layer": layer, "alpha": alpha, "where": where, "mode": mode})
    return rows


def run(args):
    from swift.model import get_model_processor
    from swift.template import get_template

    import open_audio_llm.integrations.ms_swift.register_qwen3_asr  # noqa: F401

    d = np.load(args.directions)
    run_dir = Path(args.run_dir)
    predictions = {p["id"]: p for p in map(json.loads, open(run_dir / "predictions.jsonl")) if p["status"] == "success"}
    samples = select([json.loads(line) for line in open(run_dir / "samples.jsonl")], predictions, args)
    print(f"{len(samples)} samples ({sum(not s['reference'].strip() for s in samples)} target-absent)", flush=True)

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
    pad_id = tokenizer.convert_tokens_to_ids("<|audio_pad|>")
    assert tokenizer.encode(" None", add_special_tokens=False) == [NONE_ID]

    dev = "cuda:0"
    dirs = {l: torch.as_tensor(d[f"dir_{l}"], device=dev) * float(d[f"unit_{l}"]) for l in LAYERS}
    probe = {k: torch.as_tensor(d[f"probe_{k}"], device=dev) for k in ("mu", "sd", "w", "b")}
    grid = settings()

    # Hooks. residual[l] (states npz) = output of text.layers[l - 1].
    state = {"h0": None, "delta": {}, "probe_state": None}

    def embed_hook(module, inputs):
        state["h0"] = inputs[0]

    def layer_hook(layer):
        def hook(module, inputs, output):
            hidden = output[0] if isinstance(output, tuple) else output
            if layer == PROBE_LAYER:
                state["probe_state"] = hidden[:, -1].float()
            delta = state["delta"].get(layer)
            if delta is None:
                return None
            hidden = hidden + delta.to(hidden.dtype)
            return (hidden,) + tuple(output[1:]) if isinstance(output, tuple) else hidden
        return hook

    text.layers[0].register_forward_pre_hook(embed_hook)
    for layer in sorted(set(LAYERS) | {PROBE_LAYER}):
        text.layers[layer - 1].register_forward_hook(layer_hook(layer))

    S, G = len(samples), len(grid)
    decision = np.zeros((S, G), np.int64)
    none_margin = np.zeros((S, G), np.float32)  # logit(None) - max logit over the other tokens
    full_argmax = np.zeros(S, np.int64)
    emitted = np.zeros(S, np.int64)
    p_present = np.zeros(S, np.float32)
    tail_len = np.zeros(S, np.int64)
    base_logit_maxdiff = np.zeros(S, np.float32)

    def tail_forward(tail, cache, p, rows):
        """Rerun the tail with a per-row steering delta for each row in ``rows`` (list of grid entries)."""
        b, k = len(rows), tail.shape[1]
        state["delta"] = {}
        for layer in LAYERS:
            delta = torch.zeros(b, k, tail.shape[2], device=dev)
            used = False
            for i, row in enumerate(rows):
                if row["layer"] != layer:
                    continue
                scale = row["alpha"] * (1 - p_cur if row["mode"] == "adaptive" else 1.0)
                if row["where"] == "dec":
                    delta[i, -1] = scale * dirs[layer]
                else:
                    delta[i, :] = scale * dirs[layer]
                used = True
            if used:
                state["delta"][layer] = delta
        cache_b = copy.deepcopy(cache)
        if b > 1:
            cache_b.batch_repeat_interleave(b)
        positions = torch.arange(p, p + k, device=dev)
        out = text(inputs_embeds=tail.expand(b, -1, -1), position_ids=positions.view(1, -1).expand(b, -1),
                   past_key_values=cache_b, cache_position=positions, use_cache=True)
        state["delta"] = {}
        return thinker.lm_head(out.last_hidden_state[:, -1]).float()

    loader = torch.utils.data.DataLoader(Encoded(samples, predictions, template), batch_size=None,
                                         num_workers=args.workers, prefetch_factor=4 if args.workers else None)
    start = time.time()
    for done, (index, encoded) in enumerate(loader):
        batch = template.data_collator([encoded])
        labels = batch.pop("labels")[0]
        dec = int((labels != -100).nonzero()[0])
        input_ids = batch["input_ids"][0]
        assert int(input_ids[dec]) == language_id
        T = dec + 1
        for key in ("input_ids", "attention_mask"):
            batch[key] = batch[key][:, :T]
        batch = {k: v.to(dev) if torch.is_tensor(v) else v for k, v in batch.items()}
        batch["input_features"] = batch["input_features"].to(torch.bfloat16)
        p = int((input_ids[:T] == pad_id).nonzero()[-1]) + 1  # first tail position
        with torch.inference_mode():
            outputs = model(**batch, use_cache=True)
            full_logits = outputs.logits[0, -1].float()
            cache = outputs.past_key_values
            cache.crop(p)
            tail = state["h0"][:, p:T]
            base = tail_forward(tail, cache, p, [grid[0]])[0]
            ps = state["probe_state"][0]
            p_cur = float(torch.sigmoid(((ps - probe["mu"]) / probe["sd"]) @ probe["w"] + probe["b"]))
            logits = [base[None]]
            for s in range(1, G, MAX_ROWS):
                logits.append(tail_forward(tail, cache, p, grid[s:s + MAX_ROWS]))
            logits = torch.cat(logits)
        top = logits.argmax(-1)
        others = logits.clone()
        others[:, NONE_ID] = -float("inf")
        decision[index] = top.cpu().numpy()
        none_margin[index] = (logits[:, NONE_ID] - others.max(-1).values).cpu().numpy()
        full_argmax[index] = int(full_logits.argmax())
        emitted[index] = int(input_ids[dec + 1])
        p_present[index] = p_cur
        tail_len[index] = T - p
        base_logit_maxdiff[index] = float((base - full_logits).abs().max())
        if (done + 1) % 200 == 0:
            print(f"{done + 1}/{S} {(done + 1) / (time.time() - start):.2f} samples/s", flush=True)
    elapsed = time.time() - start

    ref_empty = np.array([not s["reference"].strip() for s in samples])
    pred_empty = np.array([not predictions[s["id"]]["prediction"].strip() for s in samples])
    summary = {
        "samples": S, "seconds": elapsed, "samples_per_s": S / elapsed,
        "agreement_full_vs_vllm": float((full_argmax == emitted).mean()),
        "none_agreement_full_vs_vllm": float(((full_argmax == NONE_ID) == (emitted == NONE_ID)).mean()),
        "agreement_tail_baseline_vs_full": float((decision[:, 0] == full_argmax).mean()),
        "agreement_batched_control_vs_baseline": float((decision[:, 1] == decision[:, 0]).mean()),
        "tail_baseline_logit_maxdiff_max": float(base_logit_maxdiff.max()),
        "tail_len": [int(tail_len.min()), int(tail_len.max())],
        "checkpoint": args.checkpoint, "run_dir": args.run_dir, "limit": args.limit, "seed": args.seed,
        "directions": args.directions,
    }
    summary["pass"] = summary["agreement_full_vs_vllm"] >= MIN_AGREEMENT
    np.savez(args.out, id=np.array([s["id"] for s in samples]), reference_empty=ref_empty, prediction_empty=pred_empty,
             decision=decision, none_margin=none_margin, full_argmax=full_argmax, emitted_token=emitted,
             p_present=p_present, tail_len=tail_len, grid=np.array(json.dumps(grid)))
    Path(args.out).with_suffix(".summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


def report(args):
    result = {"sets": {}}
    for spec in args.sets:
        name, path = spec.split("=", 1)
        z = np.load(path)
        grid = json.loads(str(z["grid"]))
        absent = z["reference_empty"]
        base_none = z["decision"][:, 0] == NONE_ID
        fa0, hit0, miss0 = absent & ~base_none, ~absent & ~base_none, ~absent & base_none
        rows = []
        for g, setting in enumerate(grid):
            none = z["decision"][:, g] == NONE_ID
            rows.append({**setting, "fa_after": int((absent & ~none).sum()), "fa_removed": int((fa0 & none).sum()),
                         "fa_new": int((absent & base_none & ~none).sum()), "hits_lost": int((hit0 & none).sum()),
                         "misses_fixed": int((miss0 & ~none).sum())})
            rows[-1]["net"] = rows[-1]["fa_removed"] - rows[-1]["fa_new"] - rows[-1]["hits_lost"] + rows[-1]["misses_fixed"]
        result["sets"][name] = {
            "npz": path, "n": len(absent), "absent": int(absent.sum()), "fa_baseline": int(fa0.sum()),
            "hits_baseline": int(hit0.sum()), "misses_baseline": int(miss0.sum()),
            "baseline_vs_vllm_agreement": float((z["full_argmax"] == z["emitted_token"]).mean()),
            "baseline_none_vs_vllm_agreement": float(((z["decision"][:, 0] == NONE_ID) == (z["emitted_token"] == NONE_ID)).mean()),
            "grid": rows}
        best = sorted(rows[1:], key=lambda r: -r["net"])[:5]
        print(name, {k: v for k, v in result["sets"][name].items() if k != "grid"})
        for r in best:
            print("  ", r)
    Path(args.out).write_text(json.dumps(result, indent=1) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fit")
    f.add_argument("--states", required=True)
    f.add_argument("--out", required=True)
    r = sub.add_parser("run")
    r.add_argument("--checkpoint", required=True)
    r.add_argument("--run-dir", required=True)
    r.add_argument("--directions", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--limit", type=int, default=0)
    r.add_argument("--ids-file", default=None)
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--workers", type=int, default=8)
    p = sub.add_parser("report")
    p.add_argument("--out", required=True)
    p.add_argument("sets", nargs="+", help="NAME=steer npz")
    args = parser.parse_args()
    {"fit": fit, "run": run, "report": report}[args.cmd](args)


if __name__ == "__main__":
    main()
