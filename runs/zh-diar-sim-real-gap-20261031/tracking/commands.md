# 本目录实际执行的命令（2026-10-31）

环境：`/workspace/workspace/mingdong/miniconda3/envs/oal-train/bin/python`，均在本目录下、`env -u PYTHONPATH`。

```bash
# 解码噪声（tasks/01-decode-noise）
open-audio-llm prepare --config configs/probe-noise.yaml
python scripts/noise_report.py --run round1-step0=<第一轮 A 组 step-0 选模 evaluation 目录> --run default-r1=... --run default-r2=... --run default-r3=... --output tracking/noise-default.json

# 回放前两轮选模
PYTHONPATH=shared/code/src python scripts/replay_decisions.py --select-config ../zh-diar-aug-acp-20261030/configs/select-h.yaml \
  --baseline-eval ../zh-diar-ts-aux-acp-20261028/tasks/03-select-a/attempts/001/artifacts/checkpoints/step-0/selection/attempts/001/artifacts/evaluation \
  --arm A=../zh-diar-ts-aux-acp-20261028/tasks/03-select-a ...（C D E F G H I 同理） --output tracking/replay-decisions.json

# 对话结构统计
python scripts/label_stats.py --catalog ../zh-diar-aug-acp-20261030/shared/catalog --roots ../zh-diar-aug-acp-20261030/shared/roots.json --sample 400 --source <name>=<dataset>@<version>:<split> ... --output tracking/label-stats-train.json

# 变体构建（PYTHONPATH=amphiondata/src）
python scripts/build_dev_variants.py \
  --panel-records /workspace/data/datasets/meeting_selection_panel/versions/v1-20261008/records/alimeeting_dev.jsonl.gz \
  --panel-audio-dir /workspace/data/datasets/meeting_selection_panel/source/v1-20261008/audio \
  --far-dir /workspace/data/datasets/alimeeting/source/openslr-119-local-20260826/Eval_Ali/Eval_Ali_far/audio_dir \
  --near-dir /workspace/data/datasets/alimeeting/source/openslr-119-local-20260826/Eval_Ali/Eval_Ali_near/audio_dir \
  --rir-manifest /workspace/data/datasets/real_rir_slr28/versions/prepared-20260908/recordings/real_rir_slr28_recordings_all.jsonl.gz \
  --noise-manifest /workspace/data/datasets/musan/versions/legacy-20260804/noise_recordings/musan_recordings_noise.jsonl.gz \
  --output /workspace/workspace/mingdong/data/simreal-gap-20261031/dev-variants --catalog-dir shared/catalog-variants --workers 8
python scripts/prefix_variant_ids.py shared/catalog-variants .   # 记录 id 改成 "<变体>/<面板 id>"（AmphionEval 要求全局唯一）
open-audio-llm prepare --config configs/probe-variants.yaml        # attempt 001 因 id 重复失败，002 完成

# 去重叠
python scripts/build_dev_sequential.py --panel-records <同上> --near-dir <同上> --output <同上> --catalog-dir shared/catalog-variants
open-audio-llm prepare --config configs/probe-sequential.yaml

# 分档
PYTHONPATH=shared/code/src python scripts/overlap_breakdown.py --item <name>=<label>:<predictions.jsonl>[,...] --output tracking/overlap-breakdown-*.json
PYTHONPATH=shared/code/src python scripts/seq_compare.py --predictions <tasks/03-sequential-eval/.../predictions.jsonl> --output tracking/seq-compare.json
```
