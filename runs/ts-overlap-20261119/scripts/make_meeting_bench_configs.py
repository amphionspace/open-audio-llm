"""tasks/27: AmphionEval configs of the whole-meeting benchmark for eval_lanes.sh (TEMPLATES=configs/meeting-bench).

The data config comes from audio-data-contract's scripts/ts_real/meeting_eval_set.py (the eval set's groups and
descriptions); its entries are split into four lanes of similar size, one AmphionEval run each:
a4ramc (AISHELL-4 meeting-v3, RAMC), alibase (AliMeeting base, -xrec), alinear (-enear, -near), alisnr (-snr).

    python make_meeting_bench_configs.py ADC_REPO VERSION
"""
import subprocess
import sys
from pathlib import Path

import yaml

E = Path(__file__).resolve().parents[1]
OUT = E / "configs" / "meeting-bench" / "meeting"
LANES = {
    "a4ramc": lambda s: "-aishell4-" in s or "-ramc-" in s,
    "alibase": lambda s: "-alimeeting-" in s and (s.endswith(("-pos", "-neg")) or s.endswith("-xrec")),
    "alinear": lambda s: "-alimeeting-" in s and s.endswith(("-enear", "-near")),
    "alisnr": lambda s: "-alimeeting-" in s and s.endswith("-snr"),
}
TEMPLATE = """# tasks/27 整场会议基准（{lane} 通道）：由 scripts/make_meeting_bench_configs.py 生成，eval_lanes.sh 只改模型名、服务地址与输出目录。
suite_id: asr-regression-native@1
model_name: meeting-bench
evaluator: ts-asr
data_config: {data}
tsasr_server_url: http://127.0.0.1:0
tsasr_served_model: meeting-bench
worker_timeout: 600
output_root: /workspace/data/ae-runs/meeting-bench
parameters:
  split_group: evaluation
  samples_per_source: 0
  batch_size: 32
  max_new_tokens: 512
  seed: 42
  enroll_seconds: 5
wandb:
  entity: 1016097967-amphion
  project: open-audio-llm
publish:
  enabled: false
  model_info:
    origin: ours
    mode: offline
    params: 1.7B
    license: internal
"""


def main():
    adc, version = Path(sys.argv[1]), sys.argv[2]
    data = yaml.safe_load(subprocess.run(
        ["/workspace/workspace/mingdong/miniconda3/envs/adc-dev/bin/python", str(adc / "scripts/ts_real/meeting_eval_set.py"),
         "--version", version], check=True, capture_output=True, text=True).stdout)
    OUT.mkdir(parents=True, exist_ok=True)
    seen = set()
    for lane, keep in LANES.items():
        entries = [e for e in data["evaluation"] if keep(e["split"])]
        labels = {e["label"] for e in entries}
        seen |= labels
        groups = [{**g, "labels": [x for x in g["labels"] if x in labels]} for g in data["groups"]]
        lane_data = {**data, "groups": [g for g in groups if g["labels"]], "evaluation": entries}
        path = OUT / f"data-{lane}.yaml"
        path.write_text(yaml.safe_dump(lane_data, allow_unicode=True, sort_keys=False, width=200), encoding="utf-8")
        (OUT / f"after-training-{lane}.yaml").write_text(TEMPLATE.format(lane=lane, data=path), encoding="utf-8")
        print(lane, sorted(labels))
    missing = {e["label"] for e in data["evaluation"]} - seen
    if missing:
        raise SystemExit(f"splits in no lane: {sorted(missing)}")
    (OUT.parent / "VERSION").write_text(version + "\n")


if __name__ == "__main__":
    main()
