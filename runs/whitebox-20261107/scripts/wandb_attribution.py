"""Log task 12 attempts (grad_attribution.py + analyze_attribution.py outputs) to W&B and print each run URL.
Run where the W&B proxy works (an ACP node). Usage: wandb_attribution.py ATTEMPT_DIR...
"""
import json
import sys
from pathlib import Path

import wandb


def main():
    for attempt in map(Path, sys.argv[1:]):
        summary = json.loads((attempt / "summary.json").read_text())
        analysis = json.loads((attempt / "analysis.json").read_text())
        start = Path(summary["start"]).parent.name if summary["start"].endswith("checkpoint") else "ts-robust-073-avg3"
        run = wandb.init(entity="1016097967-amphion", project="open-audio-llm", job_type="analysis",
                         name=f"AmphionASR-whitebox-12-grad-attribution-{attempt.name}-from-{start}",
                         config={**summary["args"], "start": summary["start"], "deltas": summary["deltas"],
                                 "data_config": summary["data_config"], "v12_run": summary["v12_run"],
                                 "backend": "HF transformers bf16, SDPA (audio tower global window)"})
        table = wandb.Table(columns=["component", "module", "hard_neg", "easy_neg", "pos"])
        metrics = {}
        for name, row in analysis["components"].items():
            metrics.update({f"{name}/{k}": row[k] for k in ("hard-neg", "easy-neg", "pos", "discrimination")})
            for module, values in row["by_module"].items():
                table.add_data(name, module, values["hard-neg"], values["easy-neg"], values["pos"])
        for name, row in analysis["actual_margin_change"].items():
            metrics.update({f"actual/{name}/{k}": row[k] for k in ("hard-neg", "easy-neg", "pos", "first_order_corr")})
        for kind, row in analysis["rollouts"].items():
            metrics.update({f"rollouts/{kind}/{k}": v for k, v in row.items()})
        metrics.update({f"norm/{k}": v for k, v in summary["norms"].items()})
        run.log({"by_module": table, **metrics})
        artifact = wandb.Artifact(f"whitebox-12-grad-attribution-{attempt.name}", type="analysis-results")
        for f in ("summary.json", "analysis.json", "rollouts.jsonl"):
            artifact.add_file(str(attempt / f))
        run.log_artifact(artifact)
        print("URL", run.id, run.url, flush=True)
        run.finish()


if __name__ == "__main__":
    main()
