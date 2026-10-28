#!/usr/bin/env bash
# MOSS-Transcribe-Diarize 0.9B 在本轮全部面板上的同口径基线（用户 2026-10-28：“和moss-0.9B在相同切分方式下的性能对比，而不是看论文数字”）。
# 起一个 MOSS 服务（官方固定提交的 vLLM nightly，conda env moss-vllm，configs/serve-moss09b.yaml），依次跑 configs/eval-moss09b-*.yaml
# （由本轮面板模板生成：同一数据、切分、AmphionEval 0.10.0 打分），最后按记下的 PID 停服务。
# GPU：空闲显存最多、且不是 GPU5（另一会话）和本轮选模用的 GPU 2 / 3 的卡。端口 8762，被占则退出。
set -u
RUN=/workspace/workspace/mingdong/repo/oal-diar-restart/runs/zh-diar-ts-aux-acp-20261028
OAL=/workspace/workspace/mingdong/miniconda3/envs/oal-train/bin/open-audio-llm
PORT=8762 FREE_MB=40000
cd $RUN
curl -s -m 3 http://127.0.0.1:$PORT/v1/models >/dev/null && { echo "port $PORT busy"; exit 1; }
G=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F', ' '$1!=5 && $1!=2 && $1!=3 {print 81920-$2, $1}' | sort -rn | head -1)
[ "${G% *}" -ge $FREE_MB ] || { echo "no GPU with $FREE_MB MB free (best: $G)"; exit 1; }
G=${G#* }
sed -i "/^  gpus:/s/.*/  gpus: [$G]/" configs/serve-moss09b.yaml
mkdir -p tracking/moss-compare
env -u PYTHONPATH CUDA_VISIBLE_DEVICES=$G $OAL serve --config configs/serve-moss09b.yaml > tracking/moss-compare/serve.log 2>&1 &
LP=$!; echo "moss09b launcher=$LP GPU $G port $PORT $(date -u +%H:%MZ)" | tee -a tracking/moss-compare/pids.txt
until curl -s -m 3 http://127.0.0.1:$PORT/v1/models | grep -q moss-transcribe-diarize; do
  kill -0 $LP 2>/dev/null || { echo "server died"; tail -5 tracking/moss-compare/serve.log; exit 1; }; sleep 15
done
echo "ready $(date -u +%H:%MZ)"
for p in panel meeting180 chime6 fixed338 aishell5-dev cs-dialogue-test many70; do
  env -u PYTHONPATH $OAL eval --config configs/eval-moss09b-$p.yaml > tracking/moss-compare/eval-$p.log 2>&1
  echo "$p exit=$? $(date -u +%H:%MZ)"
done
VP=$(ps -eo pid,ppid,args | awk -v p=$LP '$2==p' | grep -- "--port $PORT" | awk '{print $1}')
kill $VP $LP 2>/dev/null; for i in $(seq 1 30); do curl -s -m 3 http://127.0.0.1:$PORT/v1/models >/dev/null || break; sleep 2; done
curl -s -m 3 http://127.0.0.1:$PORT/v1/models >/dev/null && echo "$PORT still up" || echo "stopped moss09b launcher=$LP vllm=$VP $(date -u +%H:%MZ)" | tee -a tracking/moss-compare/pids.txt
