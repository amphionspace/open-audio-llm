"""Task 12 follow-up: would an explicit pair objective separate look-alikes where GRPO only shifts the threshold?

On the pairs of ``paired_attribution`` (one rendered target-present mixture; the same mixture with the
enrollment swapped for a same-corpus look-alike), two ascent directions are accumulated at S:

- ``pair-margin``: grad of m(twin) - m(present), m the presence margin at the decision
  (log p(None) - max language). Raising it says "None for the look-alike, speak for the real speaker" on the
  very same audio, so a shift shared by both cancels by construction.
- ``encoder-cos``: grad of cos(enroll, mix | present) - cos(enroll, mix | twin), mean-pooled audio-tower
  states of the last encoder layer (task 04: speaker signal is strongest there); only audio-tower parameters
  get gradient. This is a speaker-contrastive loss on the encoder alone.

As in ``grad_attribution``, each direction (unit-normalised) is dotted with the held-out margin gradients per
module group, to see how one step moves the decision on hard negatives versus target-present samples.
Usage: pair_objectives.py --out DIR [--pairs 128] [--per-family 40] [--seed 0]
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import grad_attribution as ga  # noqa: E402
import paired_attribution as pa  # noqa: E402

DIRECTIONS = ("pair-margin", "encoder-cos")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", required=True)
    parser.add_argument("--pairs", type=int, default=128)
    parser.add_argument("--per-family", type=int, default=40)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--start", default=ga.START)
    parser.add_argument("--v12-run", default=ga.V12_RUN)
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    from open_audio_llm.data.catalog_resolver import LhotseCatalogAudioResolver
    from open_audio_llm.tsasr.sep_token import conv_token_count

    model, template, tokenizer = ga.load_model(args.start)
    template.set_mode("train")
    frozen = ("embed_tokens", "conv2d1", "conv2d2", "conv2d3", "conv_out")
    for n, p in model.named_parameters():
        if any(f in n for f in frozen):
            p.requires_grad_(False)
    params = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    groups_of = [ga.module_group(n) for n, _ in params]
    group_names = sorted(set(groups_of))
    tower = model.thinker.audio_tower
    captured = {}
    tower.layers[-1].register_forward_hook(lambda m, i, o: captured.__setitem__("enc", o[0] if isinstance(o, tuple) else o))
    language_id = tokenizer.convert_tokens_to_ids("language")
    ids = [tokenizer.encode(w, add_special_tokens=False)[0] for w in (" None", " Chinese", " English")]

    def forward(prompt):
        row = {"messages": [*prompt["messages"], {"role": "assistant", "content": "language None<asr_text>"}],
               "audios": prompt["audios"], "mix_wav": prompt["mix_wav"]}
        encoded = template.encode(row)
        batch = template.data_collator([encoded])
        labels = batch.pop("labels")
        batch = ga.to_device(batch)
        m = ga.margin(model, batch, labels, ids, language_id)
        states = captured["enc"].float()
        n_enroll = conv_token_count(int(encoded["enroll_n_frames"]))
        cos = torch.nn.functional.cosine_similarity(states[:n_enroll].mean(0), states[n_enroll + 1:].mean(0), dim=0)
        return m, cos

    pa_args = argparse.Namespace(prompts_per_type=args.pairs, seed=args.seed)
    dataset, _, chosen = pa.build_pairs(pa_args)
    buffers = {d: [torch.zeros(p.shape, dtype=torch.float32, device="cuda:1") for _, p in params] for d in DIRECTIONS}
    pair_stats = []
    start = time.time()
    for k in range(args.pairs):
        present, twin = pa.RENDERED[2 * k], pa.RENDERED[2 * k + 1]
        row = {}
        for sign, prompt, name in ((-1.0, present, "present"), (1.0, twin, "twin")):
            m, cos = forward(prompt)
            row[f"margin_{name}"], row[f"cos_{name}"] = float(m.detach()), float(cos.detach())
            # pair-margin: +grad m(twin) - grad m(present); encoder-cos: +grad cos(present) - grad cos(twin)
            for direction, objective, weight in (("pair-margin", m, sign), ("encoder-cos", cos, -sign)):
                model.zero_grad(set_to_none=True)
                objective.backward(retain_graph=direction == "pair-margin")
                for buf, (n, p) in zip(buffers[direction], params):
                    if p.grad is not None:
                        buf.add_(p.grad.to("cuda:1", torch.float32), alpha=weight)
        pair_stats.append(row)
        if (k + 1) % 16 == 0:
            print(f"pairs {k + 1}/{args.pairs} {time.time() - start:.0f}s", flush=True)
    model.zero_grad(set_to_none=True)
    norms = {d: float(sum(b.double().pow(2).sum() for b in bufs).sqrt()) for d, bufs in buffers.items()}
    print(json.dumps(norms), flush=True)

    samples = ga.held_out(args)
    resolver = LhotseCatalogAudioResolver("/workspace/data/registry/catalog", "/workspace/data/registry/roots.json")
    group_index = [group_names.index(g) for g in groups_of]
    dots = np.zeros((len(samples), len(DIRECTIONS), len(group_names)))
    margins = np.zeros(len(samples))
    for s_i, sample in enumerate(samples):
        batch = ga.margin_batch(sample, template, resolver)
        labels = batch.pop("labels")
        batch = ga.to_device(batch)
        model.zero_grad(set_to_none=True)
        m = ga.margin(model, batch, labels, ids, language_id)
        m.backward()
        margins[s_i] = float(m.detach())
        acc = torch.zeros(len(DIRECTIONS), len(group_names), dtype=torch.float64, device="cuda:1")
        for j, ((n, p), gi) in enumerate(zip(params, group_index)):
            g = p.grad.to("cuda:1", torch.float32)
            for c, d in enumerate(DIRECTIONS):
                acc[c, gi] += (g * buffers[d][j]).sum().double() / max(norms[d], 1e-30)
        dots[s_i] = acc.cpu().numpy()
        if (s_i + 1) % 100 == 0:
            print(f"held-out {s_i + 1}/{len(samples)} {time.time() - start:.0f}s", flush=True)

    family = [s["source"].split(":", 1)[1].split("/", 1)[0] for s in samples]
    np.savez(out / "pair_objectives.npz", dots=dots, components=np.array(DIRECTIONS), module_groups=np.array(group_names),
             family=np.array(family), role=np.array([ga.FAMILIES[f] for f in family]), margin_S=margins)
    with open(out / "pairs.jsonl", "w") as handle:
        for k, row in enumerate(pair_stats):
            handle.write(json.dumps({"record_id": pa.RENDERED[2 * k]["record_id"],
                                     "twin_enrollment": pa.RENDERED[2 * k + 1]["mix_info"].get("twin_enrollment"), **row}) + "\n")
    role = np.array([ga.FAMILIES[f] for f in family])
    total = dots.sum(-1)
    table = {d: {r: float(total[role == r, c].mean()) for r in ("hard-neg", "easy-neg", "pos")} for c, d in enumerate(DIRECTIONS)}
    by_module = {d: {g: {r: float(dots[role == r, c, gi].mean()) for r in ("hard-neg", "pos")} for gi, g in enumerate(group_names)}
                 for c, d in enumerate(DIRECTIONS)}
    stats = {"twin_speaks_at_S": float(np.mean([r["margin_twin"] < 0 for r in pair_stats])),
             "present_none_at_S": float(np.mean([r["margin_present"] > 0 for r in pair_stats])),
             "cos_present_mean": float(np.mean([r["cos_present"] for r in pair_stats])),
             "cos_twin_mean": float(np.mean([r["cos_twin"] for r in pair_stats])),
             "cos_auc_present_vs_twin": float(np.mean([[a["cos_present"] > b["cos_twin"] for b in pair_stats] for a in pair_stats]))}
    summary = {"norms": norms, "pairs": stats, "effects": table, "by_module": by_module, "args": vars(args)}
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps({"pairs": stats, "effects": table}, indent=1), flush=True)


if __name__ == "__main__":
    main()
