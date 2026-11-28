"""tasks/10: summarise probe_ts_sampling.py shards and log the summary to its own W&B run.

Errors are character-weighted (sum of edits / sum of reference length), the retrospective's in-utterance error with
empty outputs counted as fully deleted. "better" means a sample's error below the greedy output's.
usage: grpo_headroom_report.py OUT.json RUN_NAME SHARD.jsonl [SHARD.jsonl ...]   (oal-train python, W&B key in env)
"""
import json
import sys

import wandb


def main():
    out, name, paths = sys.argv[1], sys.argv[2], sys.argv[3:]
    rows = [json.loads(line) for path in paths for line in open(path)]
    n = len(rows)
    chars = sum(r["ref_len"] for r in rows)

    def weighted(values):
        return sum(v * r["ref_len"] for v, r in zip(values, rows)) / chars

    greedy = [r["greedy"] for r in rows]
    best = [min(r["samples"]) for r in rows]
    mean = [sum(r["samples"]) / len(r["samples"]) for r in rows]
    wrong = [r for r in rows if r["greedy"] > 0]
    better = [r for r in wrong if min(r["samples"]) < r["greedy"]]
    perfect = [r for r in wrong if min(r["samples"]) == 0]
    split = [r for r in rows if len(set(r["samples"])) > 1]
    wrong_chars = sum(r["greedy"] * r["ref_len"] for r in wrong)
    summary = {
        "samples": n,
        "draws_per_sample": len(rows[0]["samples"]),
        "error/greedy": weighted(greedy),
        "error/sample_mean": weighted(mean),
        "error/best_of_n": weighted(best),
        "greedy_wrong/samples": len(wrong),
        "greedy_wrong/share_with_a_better_draw": len(better) / len(wrong),
        "greedy_wrong/share_with_a_perfect_draw": len(perfect) / len(wrong),
        "greedy_wrong/share_never_better": 1 - len(better) / len(wrong),
        "greedy_wrong/error_chars_on_samples_with_a_better_draw":
            sum(r["greedy"] * r["ref_len"] for r in better) / wrong_chars,
        "greedy_wrong/mean_better_draws": sum(sum(s < r["greedy"] for s in r["samples"]) for r in wrong) / len(wrong),
        "groups/share_with_reward_spread": len(split) / n,
        "groups/share_all_draws_equal": 1 - len(split) / n,
        "draws/empty_share": sum(r["empty"] for r in rows) / (n * len(rows[0]["samples"])),
    }
    json.dump(summary, open(out, "w"), indent=1)
    for key, value in summary.items():
        print(f"{key:58s} {value:.4f}" if isinstance(value, float) else f"{key:58s} {value}")
    run = wandb.init(project="open-audio-llm", entity="1016097967-amphion", name=name, job_type="grpo-headroom-probe",
                     config={"shards": paths, "metric": "in-utterance error, char-weighted (decompose_errors.py)"})
    run.summary.update(summary)
    run.finish()


if __name__ == "__main__":
    main()
