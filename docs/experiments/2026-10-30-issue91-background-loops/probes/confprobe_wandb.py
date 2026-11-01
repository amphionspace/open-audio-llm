"""Log the confidence-probe summary (confprobe_report.py --json) to W&B and verify the remote run.

Aggregates only: no transcripts or audio leave the machine. With WANDB_MODE=offline the run is
written locally for a later wandb sync and the remote check is skipped.

    WANDB_HTTP_PROXY=... WANDB_HTTPS_PROXY=... python confprobe_wandb.py summary.json report.txt RUN_NAME
"""
import json, os, sys

import wandb

summary_path, report_path, name = sys.argv[1:4]
summary = json.load(open(summary_path))
flat = {f"counts/{k}": v for k, v in summary["counts"].items()}
flat.update({f"auc/{k}": v for k, v in summary["auc"].items() if v is not None})
for k, v in summary["baseline"].items():
    flat.update({f"baseline/{k}/{m}": x for m, x in v.items() if x is not None})
for k, v in summary["points"].items():
    flat.update({f"point/{k}/{m}": x for m, x in v.items() if x is not None})
run = wandb.init(entity="1016097967-amphion", project="open-audio-llm", name=name, id=name, resume="allow",
                 job_type="eval", tags=["issue91", "confidence-rejection"],
                 config={"model": "/workspace/data/ae-runs/ts-robust-054-avg3/checkpoint-avg3", "backend": "vllm 0.18.0 served",
                         "prompt": "user-only (audiollm-go runtime), no system block", "temperature": 0,
                         "max_tokens": "audiollm-go decodeBudget: min(512, max(128, 96 + 24/s))", "top_logprobs": 20,
                         "sets": ["issue91 clips 33 background + 4 control", "tdtech/scene-asr@v1-20261028 asr-only (4600)",
                                  "open-audio-llm/asr-babble-bench@v3-20261023 (3200)", "wenetspeech test_meeting (8370)"]})
run.summary.update(flat)
run.log({"report": wandb.Html("<pre>" + open(report_path).read().replace("&", "&amp;").replace("<", "&lt;") + "</pre>")})
url = run.url
run.finish()
if os.environ.get("WANDB_MODE") == "offline":
    raise SystemExit(f"offline run written under {run.dir}; sync it, then check the remote run")
api = wandb.Api()
remote = api.run(f"1016097967-amphion/open-audio-llm/{name}")
print(url, remote.state, len([k for k in remote.summary.keys() if k.startswith("point/")]), "point metrics on remote")
