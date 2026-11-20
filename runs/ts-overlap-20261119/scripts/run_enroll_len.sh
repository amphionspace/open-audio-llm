#!/usr/bin/env bash
# tasks/02-enroll-length: 11.17 on Chinese TS v12 with 3/5/8/10/12-second enrollment, one TS server (one GPU) per length.
# Usage on an ACP inbox runner: bash run_enroll_len.sh   (no arguments; lengths, ports and outputs are in configs/enroll-len/)
set -u; umask 000
E=/workspace/workspace/mingdong/repo/oal-clean-teacher/runs/ts-overlap-20261119
ENVS=/workspace/workspace/mingdong/miniconda3/envs; PY=$ENVS/vllm-serving/bin/python
P=/workspace/workspace/mingdong/repo/oal-joint-mtp/runs/joint-mtp-spec-20261107/shared/pack-mtp   # same serve.sh as the 11.17 gate run
NAME=whitebox-merge-v18a-es116inc
M=$(bash /workspace/workspace/mingdong/scratch/acp-dev/ae-models.sh fetch $NAME) || { echo "fetch $NAME failed"; exit 2; }
A=$E/tasks/02-enroll-length/attempts/001; L=$A/logs; mkdir -p $L
python3 - "$M" "$NAME" > $A/handoff.json <<'PY'
import json, sys
print(json.dumps({"framework": "open-audio-llm", "schema_version": 2, "checkpoint": sys.argv[1], "base_model": "", "tuner_type": "full",
                  "selection": "final", "global_step": 0, "training_run": sys.argv[2]}))
PY
source /workspace/workspace/mingdong/.config/open-audio-llm/wandb-cluster.sh >/dev/null 2>&1
export no_proxy=localhost,127.0.0.1 NO_PROXY=localhost,127.0.0.1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1 AMPHION_EVAL_HOME=/workspace/data/ae-state
export WANDB_HTTP_PROXY=http://10.119.17.252:17920 WANDB_HTTPS_PROXY=http://10.119.17.252:17920
export AUDIO_DATA_CATALOG=/workspace/data/registry/catalog AUDIO_DATA_ROOTS_FILE=/workspace/data/registry/roots.json
echo "node ip $(hostname -I)"
for k in $(seq 1 80); do curl -s -m 10 -o /dev/null -x $WANDB_HTTPS_PROXY https://api.wandb.ai && break; [ $k = 80 ] && { echo "W&B proxy unreachable"; exit 5; }; sleep 15; done
echo "W&B proxy ok $(date -u +%FT%TZ)"
lane() { local g=$1 sec=$2 port=$((19800+$2))
  CUDA_VISIBLE_DEVICES=$g MODEL=$M PYTHON=$PY PORT=$port NAME=1117-enroll${sec}s-tsasr GPU_UTIL=0.8 AMPHION_MTP_TOKENS=0 \
    AMPHION_TSASR_ENROLL_SEC=$sec setsid bash $P/serve.sh tsasr --max-num-seqs 64 > $L/serve-L$sec.log 2>&1 & local s=$!
  until curl -s -m 5 http://127.0.0.1:$port/v1/models >/dev/null; do kill -0 $s 2>/dev/null || { echo "server L$sec died"; return 1; }; sleep 15; done
  grep -h "AMPHION_TSASR_ENROLL_SEC\|enroll" $L/serve-L$sec.log | head -2; echo "server L$sec ready $(date -u +%FT%TZ)"
  env -u PYTHONPATH $ENVS/amphion-eval/bin/ae open-audio-llm after-training --handoff $A/handoff.json \
    --config $E/configs/enroll-len/after-training-v12-L$sec.yaml > $L/eval-L$sec.log 2>&1
  local e=$?; echo "L$sec exit=$e $(date -u +%FT%TZ)"; kill -TERM -$s; wait $s 2>/dev/null; return $e; }
pids=(); i=0
for sec in 3 5 8 10 12; do lane $i $sec & pids+=($!); i=$((i+1)); sleep 30; done
rc=0; for x in "${pids[@]}"; do wait $x || rc=1; done; echo "done rc=$rc $(date -u +%FT%TZ)"; exit $rc
