"""tasks/09: per-layer enrollment selectivity from existing ref_score.py shards (no GPU).

Question: does the audio tower's target selectivity still grow at its last layers (more depth, e.g. a looped
encoder, may help), or does it level off before layer 24 (more depth of the same kind is unlikely to help)?

Selectivity of a layer = 1 - cosine between the states under the true and a swapped enrollment, averaged over the
mixture tokens of one region (overlap / interferer_only / target_only), as in the retrospective. ref_score.py stores
it for the audio-tower input + 24 layers and the LLM embedding + 28 layers.

Per target-overlap bin and layer: mean selectivity per region, and the Spearman correlation across samples between
the layer's overlap-region selectivity and the sample's teacher-forced reference -log p per token (negative = more
selective samples are fitted better).
Usage: layer_selectivity.py OUT.json NAME=SHARD_DIR [NAME=SHARD_DIR ...]
"""
import glob
import json
import sys

import numpy as np

BINS = ("0-0.6", "0.6-0.9", ">=0.9")
REGIONS = ("overlap", "interferer_only", "target_only")


def overlap_bin(ratio):
    if not ratio:
        return None
    return "0-0.6" if ratio < 0.6 else "0.6-0.9" if ratio < 0.9 else ">=0.9"


def spearman(x, y):
    rx, ry = np.argsort(np.argsort(x)), np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1])


def summarize(shard_dir):
    rows = {b: [] for b in BINS}
    for path in sorted(glob.glob(f"{shard_dir}/shard-*.jsonl")):
        for r in map(json.loads, open(path)):
            b = overlap_bin(r["labels"].get("target_overlap_ratio"))
            regions = r["swap"].get("regions") or {}
            if b is None or r["ref"]["n_tokens"] == 0 or "overlap" not in regions:
                continue
            rows[b].append(r)
    out = {}
    for b, rs in rows.items():
        nll = np.array([-r["ref"]["logp_sum"] / r["ref"]["n_tokens"] for r in rs])
        entry = {"n": len(rs)}
        for part in ("audio", "llm"):
            for region in REGIONS:
                have = [r for r in rs if r["swap"]["regions"].get(region, {}).get("n", 0) > 0]
                sel = np.array([[1 - c for c in r["swap"]["regions"][region][part]] for r in have])
                entry[f"{part}/{region}"] = [round(float(v), 4) for v in sel.mean(0)]
            sel_ov = np.array([[1 - c for c in r["swap"]["regions"]["overlap"][part]] for r in rs])
            entry[f"{part}/overlap_vs_nll_spearman"] = [
                round(spearman(sel_ov[:, k], nll), 4) if sel_ov[:, k].std() > 0 else None for k in range(sel_ov.shape[1])
            ]
        out[b] = entry
    return out


def main():
    out_path, specs = sys.argv[1], [a.split("=", 1) for a in sys.argv[2:]]
    result = {name: summarize(d) for name, d in specs}
    json.dump(result, open(out_path, "w"), indent=1)
    for name, bins in result.items():
        for b, e in bins.items():
            print(f"== {name} overlap {b} (n={e['n']})")
            print("layer " + " ".join(f"{k:>6d}" for k in range(len(e["audio/overlap"]))))
            for key in ("audio/overlap", "audio/interferer_only", "audio/target_only", "audio/overlap_vs_nll_spearman"):
                print(f"{key.split('/', 1)[1][:16]:16s}"[:6] + " " + " ".join(
                    f"{v:6.3f}" if v is not None else "     -" for v in e[key]) + f"   <- audio {key.split('/', 1)[1]}")
            for key in ("llm/overlap", "llm/interferer_only", "llm/overlap_vs_nll_spearman"):
                print("      " + " ".join(f"{v:6.3f}" if v is not None else "     -" for v in e[key])
                      + f"   <- llm {key.split('/', 1)[1]}")


if __name__ == "__main__":
    main()
