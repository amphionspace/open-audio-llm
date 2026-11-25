#!/usr/bin/env bash
# tasks/27 ACP 1-node job: the whole-meeting TS benchmark (configs/meeting-bench, from make_meeting_bench_configs.py) for one
# cached model on the deployment stack (pack-mtp serve.sh without MTP, eval_lanes.sh: four lanes of one ASR + one TS server on
# the 8 GPUs), then with QC=1 plain-ASR transcripts of the version's base segments and enrollments (meeting_asr_qc.py,
# one ASR server per GPU) for the annotation text check.
# W&B goes through the dev-machine forwarder on 17922, which admits this node only after its IP is added (the wandb-ready flag).
# usage: job_meeting_bench.sh NAME TAG [QC] [CATALOG]   (outputs under /workspace/data/ts-overlap-20261119/task27-bench-TAG)
set -u; umask 000
NAME=$1; TAG=$2; QC=${3:-0}; CATALOG=${4:-/workspace/data/registry/catalog}
O=/workspace/data/ts-overlap-20261119/task27-bench-$TAG; E=/workspace/workspace/mingdong/repo/oal-clean-teacher/runs/ts-overlap-20261119
VERSION=$(cat $E/configs/meeting-bench/VERSION)
echo "node ip $(hostname -I) $(date -u +%FT%TZ) version $VERSION catalog $CATALOG" > $O/logs/node.txt
for _ in $(seq 120); do [ -f $O/wandb-ready ] && break; sleep 20; done
[ -f $O/wandb-ready ] || { echo "no wandb-ready flag $(date -u +%FT%TZ)" >> $O/logs/node.txt; exit 3; }
cd $E; WB_PROXY=http://10.119.17.252:17922 TEMPLATES=$E/configs/meeting-bench CATALOG=$CATALOG bash scripts/eval_lanes.sh $NAME $TAG 0 19900 \
  "meeting/a4ramc" "meeting/alibase" "meeting/alinear" "meeting/alisnr" > $O/logs/eval.log 2>&1
echo "bench rc=$? $(date -u +%FT%TZ)" >> $O/logs/node.txt
[ "$QC" = 1 ] || exit 0
ENVS=/workspace/workspace/mingdong/miniconda3/envs; P=/workspace/workspace/mingdong/repo/oal-joint-mtp/runs/joint-mtp-spec-20261107/shared/pack-mtp
M=$(bash /workspace/workspace/mingdong/scratch/acp-dev/ae-models.sh fetch $NAME) || { echo "fetch $NAME failed" >> $O/logs/node.txt; exit 2; }
export no_proxy=localhost,127.0.0.1 NO_PROXY=localhost,127.0.0.1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
mkdir -p $O/qc; pids=(); urls=()
for g in 0 1 2 3 4 5 6 7; do p=$((19960 + g))
  CUDA_VISIBLE_DEVICES=$g MODEL=$M PYTHON=$ENVS/vllm-serving/bin/python PORT=$p NAME=qc-asr-$p GPU_UTIL=0.8 AMPHION_MTP_TOKENS=0 \
    setsid bash $P/serve.sh asr --max-num-seqs 64 > $O/logs/serve-qc-$p.log 2>&1 & pids+=($!); urls+=(http://127.0.0.1:$p)
done
for p in $(seq 19960 19967); do n=0
  until curl -s -m 5 http://127.0.0.1:$p/v1/models >/dev/null; do n=$((n + 1)); [ $n -gt 80 ] && { echo "qc server $p not ready" >> $O/logs/node.txt; exit 4; }; sleep 15; done
done
env -u PYTHONPATH $ENVS/amphion-eval/bin/python $E/scripts/meeting_asr_qc.py $VERSION $O/qc/asr.jsonl "${urls[@]}" > $O/logs/qc.log 2>&1
echo "qc rc=$? $(date -u +%FT%TZ)" >> $O/logs/node.txt
for x in "${pids[@]}"; do kill -TERM -$x 2>/dev/null; done; wait 2>/dev/null; exit 0
