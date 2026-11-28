"""Aggregate teacher_signal.py shards: per model and input, how well each input predicts the training label.

Token-weighted over samples: text -log p(label) per token, text argmax accuracy, text entropy per token,
KL(input || mix) per answer token (all parts, as the task-03 loss averaged it) and per text token, mean P(" None") at
the language word, and the share of text tokens where the input beats the mixture (lower -log p).
Split by language, overlap bin, share of digital silence in the clean target, and mixture kind.
Usage: teacher_signal_report.py --out REPORT.json NAME=SHARD_DIR [NAME=SHARD_DIR ...] [--wandb RUN_NAME]
"""
import argparse
import collections
import glob
import json
import math

VIEWS = ("mix", "clean_zero", "clean_floor", "clean_crop")
OVERLAP_BINS = ("0", "0-0.6", "0.6-0.9", ">=0.9")  # the 2026-11-19 retrospective's bins
ZERO_BINS = ("<0.25", "0.25-0.5", ">=0.5")  # share of the clean target that is digital silence


def overlap_bin(r):
    return "0" if r == 0 else "0-0.6" if r < 0.6 else "0.6-0.9" if r < 0.9 else ">=0.9"


def zero_bin(z):
    return "<0.25" if z < 0.25 else "0.25-0.5" if z < 0.5 else ">=0.5"


def load(shard_dir):
    rows = []
    for path in sorted(glob.glob(f"{shard_dir}/shard-*.jsonl")):
        rows += [json.loads(line) for line in open(path) if line.strip()]
    return rows


def aggregate(rows):
    c = {v: collections.Counter() for v in VIEWS}
    for r in rows:
        n_prefix, n_text, n_end = r["n_tokens"]
        text = slice(n_prefix, n_prefix + n_text)
        mix_text = r["mix"]["nll"][text]
        for v in VIEWS:
            x = r[v]
            t = c[v]
            t["samples"] += 1
            t["text_tokens"] += n_text
            t["all_tokens"] += n_prefix + n_text + n_end
            t["text_nll"] += sum(x["nll"][text])
            t["prefix_nll"] += sum(x["nll"][:n_prefix])
            t["text_correct"] += x["correct"][1]
            t["text_entropy"] += x["entropy"][1]
            t["kl_all"] += sum(x["kl_to_mix"])
            t["kl_text"] += x["kl_to_mix"][1]
            t["p_none"] += math.exp(x["none_logp"])
            t["beats_mix"] += sum(a < b - 1e-3 for a, b in zip(x["nll"][text], mix_text))
    out = {}
    for v, t in c.items():
        if not t["samples"]:
            continue
        out[v] = {
            "samples": t["samples"],
            "text_nll": round(t["text_nll"] / t["text_tokens"], 4),
            "prefix_nll": round(t["prefix_nll"] / t["samples"], 4),
            "text_acc": round(100 * t["text_correct"] / t["text_tokens"], 2),
            "text_entropy": round(t["text_entropy"] / t["text_tokens"], 4),
            "kl_all": round(t["kl_all"] / t["all_tokens"], 4),
            "kl_text": round(t["kl_text"] / t["text_tokens"], 4),
            "p_none": round(t["p_none"] / t["samples"], 4),
            "beats_mix": round(100 * t["beats_mix"] / t["text_tokens"], 1),
        }
    return out


def groups(rows):
    by = collections.defaultdict(list)
    for r in rows:
        by[("all",)].append(r)
        by[("language", r["language"])].append(r)
        by[("overlap", overlap_bin(r["overlap"]))].append(r)
        by[("zero_frac", zero_bin(r["zero_frac"]))].append(r)
        by[("kind", r["kind"])].append(r)
        if r["language"] == "zh":
            by[("zh overlap", overlap_bin(r["overlap"]))].append(r)
            by[("zh zero_frac", zero_bin(r["zero_frac"]))].append(r)
    return by


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", required=True)
    parser.add_argument("--wandb", help="W&B run name; logs the report as summary keys")
    parser.add_argument("models", nargs="+", help="NAME=SHARD_DIR")
    args = parser.parse_args()
    report, lines = {}, []
    for spec in args.models:
        name, shard_dir = spec.split("=", 1)
        rows = load(shard_dir)
        report[name] = {"/".join(key): aggregate(sub) for key, sub in sorted(groups(rows).items())}
        lines.append(f"== {name}: {len(rows)} samples")
        lines.append(f"{'group':26s} {'input':12s} {'n':>5s} {'textNLL':>8s} {'acc%':>6s} {'ent':>6s} "
                     f"{'KLall':>6s} {'KLtext':>6s} {'pNone':>6s} {'beat%':>6s}")
        for group, views in report[name].items():
            for v, m in views.items():
                lines.append(f"{group:26s} {v:12s} {m['samples']:5d} {m['text_nll']:8.4f} {m['text_acc']:6.2f} "
                             f"{m['text_entropy']:6.3f} {m['kl_all']:6.3f} {m['kl_text']:6.3f} {m['p_none']:6.3f} "
                             f"{m['beats_mix']:6.1f}")
    text = "\n".join(lines)
    print(text)
    json.dump(report, open(args.out, "w"), ensure_ascii=False, indent=1)
    open(args.out.rsplit(".", 1)[0] + ".txt", "w").write(text + "\n")
    if args.wandb:
        import wandb

        summary = {}
        for name, by in report.items():
            for group, views in by.items():
                for v, m in views.items():
                    for k, value in m.items():
                        summary[f"teacher_signal/{name}/{group}/{v}/{k}"] = value
        run = wandb.init(entity="1016097967-amphion", project="open-audio-llm", name=args.wandb, job_type="eval",
                         config={"script": "runs/ts-overlap-20261119/scripts/teacher_signal.py", "models": args.models})
        run.summary.update(summary)
        run.save(args.out, policy="now")
        url = run.url
        run.finish()
        print(f"W&B {url}: {len(summary)} keys", flush=True)


if __name__ == "__main__":
    main()
