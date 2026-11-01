#!/usr/bin/env bash
# 准确率闸门：起一个 TS 服务变体（open-audio-llm serve），跑 v12 全量（AmphionEval），结束后按自己的 PID 停服务。
# 用法：run_gate.sh <suffix|fp8|prompt5s>
# prompt5s 用原样服务 serve-base.yaml，并在 18194 起 prompt 改写代理（"3-second" -> "5-second"）。
set -u
V=$1
SCRIPTS=$(cd "$(dirname "$0")" && pwd)
RUN=$(cd "$SCRIPTS/.." && pwd)
C=$RUN/configs/gate
L=$RUN/tasks/02-accuracy-gates/logs; mkdir -p $L
OAL=/workspace/workspace/mingdong/miniconda3/envs/oal-train/bin/open-audio-llm
PY=/workspace/workspace/mingdong/miniconda3/envs/oal-train/bin/python
AE=/workspace/workspace/mingdong/miniconda3/envs/amphion-eval/bin/ae
source /workspace/workspace/mingdong/.config/open-audio-llm/wandb-cluster.sh
# W&B 出口被封（10-30 起），离线记录，恢复后 wandb sync 补录。
export WANDB_MODE=offline
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1 AMPHION_EVAL_HOME=/workspace/data/ae-state AUDIO_DATA_CATALOG=/workspace/data/registry/catalog AUDIO_DATA_ROOTS_FILE=/workspace/data/registry/roots.json
unset PYTHONPATH
SERVE=$V; [ "$V" = prompt5s ] && SERVE=base
PORT=$(awk '/--port:/{print $2}' $C/serve-$SERVE.yaml)
GPU=$(awk -F'[][]' '/^  gpus:/{print $2}' $C/serve-$SERVE.yaml)
for p in $PORT 18194; do
  curl -s -m 3 http://127.0.0.1:$p/v1/models >/dev/null && { echo "$V: port $p busy" >> $L/status.txt; exit 1; }
  ps -eo cmd | grep -v grep | grep -q -- "--port $p" && { echo "$V: port $p in ps" >> $L/status.txt; exit 1; }
done
(cd $RUN/../.. && CUDA_VISIBLE_DEVICES=$GPU exec $OAL serve --config $C/serve-$SERVE.yaml) > $L/serve-$V.log 2>&1 &
PS=$!
echo "$V launcher=$PS gpu=$GPU port=$PORT $(date -u +%H:%MZ)" >> $L/status.txt
for i in $(seq 1 180); do
  curl -s -m 5 http://127.0.0.1:$PORT/v1/models >/dev/null && break
  kill -0 $PS 2>/dev/null || { echo "$V: server died $(date -u +%H:%MZ)" >> $L/status.txt; exit 1; }
  sleep 10
done
curl -s -m 5 http://127.0.0.1:$PORT/version > $L/server-version-$V.json
PP=
if [ "$V" = prompt5s ]; then
  $PY $SCRIPTS/prompt_rewrite_proxy.py --backend http://127.0.0.1:$PORT --port 18194 \
    --old "The clip starts with a 3-second enrollment" --new "The clip starts with a 5-second enrollment" > $L/proxy.log 2>&1 &
  PP=$!
  sleep 5
fi
echo "$V ready $(date -u +%H:%MZ)" >> $L/status.txt
$AE open-audio-llm after-training --handoff $C/handoff.json --config $C/after-training-v12-$V.yaml > $L/eval-$V.log 2>&1
echo "$V eval exit=$? $(date -u +%H:%MZ)" >> $L/status.txt
[ -n "$PP" ] && { curl -s -m 5 http://127.0.0.1:18194/proxy-stats > $L/proxy-stats.json; kill $PP; }
VP=$(ps -eo pid,ppid,args | awk -v p=$PS '$2==p' | grep -- "--port $PORT" | awk '{print $1}')
kill $VP $PS 2>/dev/null
echo "$V stopped vllm=$VP $(date -u +%H:%MZ)" >> $L/status.txt
