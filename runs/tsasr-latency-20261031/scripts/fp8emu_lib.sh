# run_fp8emu.sh 里的服务启停与评测函数（第二段 run_fp8emu_rest.sh 复用；run_fp8emu.sh 运行中不能改，故抽出副本）。
set -u
SCRIPTS=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
RUN=$(cd "$SCRIPTS/.." && pwd)
C=$RUN/configs/fp8emu
L=$RUN/tasks/07-fp8-h20-emu/logs; mkdir -p $L
OAL=/workspace/workspace/mingdong/miniconda3/envs/oal-train/bin/open-audio-llm
AE=/workspace/workspace/mingdong/miniconda3/envs/amphion-eval/bin/ae
source /workspace/workspace/mingdong/.config/open-audio-llm/wandb-cluster.sh
# W&B 出口被封（10-30 起），离线记录，结束后经代理 wandb sync 补录。
export WANDB_MODE=offline
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1 AMPHION_EVAL_HOME=/workspace/data/ae-state AUDIO_DATA_CATALOG=/workspace/data/registry/catalog AUDIO_DATA_ROOTS_FILE=/workspace/data/registry/roots.json
unset PYTHONPATH
status() { echo "$* $(date -u +%m-%dT%H:%MZ)" >> $L/status.txt; }

start() {  # start <tsasr|asr>; sets LAUNCHER_<name>
  local port gpu
  port=$(awk '/--port:/{print $2}' $C/serve-$1.yaml)
  gpu=$(awk -F'[][]' '/^  gpus:/{print $2}' $C/serve-$1.yaml)
  curl -s -m 3 http://127.0.0.1:$port/v1/models >/dev/null && { status "$1: port $port busy"; exit 1; }
  ps -eo args | grep -v grep | grep -q -- "--port $port" && { status "$1: port $port in ps"; exit 1; }
  (cd $RUN/../.. && CUDA_VISIBLE_DEVICES=$gpu exec $OAL serve --config $C/serve-$1.yaml) > $L/serve-$1.log 2>&1 &
  eval "LAUNCHER_${1}=$!"
  status "$1 launcher=$! gpu=$gpu port=$port"
}

ready() {  # ready <tsasr|asr>: wait for the server, then require the emulation in its engine process
  local port launcher
  port=$(awk '/--port:/{print $2}' $C/serve-$1.yaml)
  eval "launcher=\$LAUNCHER_${1}"
  for i in $(seq 1 180); do
    curl -s -m 5 http://127.0.0.1:$port/v1/models >/dev/null && break
    kill -0 $launcher 2>/dev/null || { status "$1: server died"; return 1; }
    sleep 10
  done
  curl -s -m 5 http://127.0.0.1:$port/version > $L/server-version-$1.json || { status "$1: not ready"; return 1; }
  # open-audio-llm serve writes the server output to the attempt's process log, named in its first JSON line.
  local plog
  plog=$(grep -o '"log": "[^"]*"' $L/serve-$1.log | head -1 | cut -d'"' -f4)
  grep -q 'fp8-sm90-emu\] activation scales' $plog || { status "$1: emulation not active ($plog)"; return 1; }
  status "$1 ready, $(grep -c 'fp8-sm90-emu\] quantize' $plog) FP8 linear layers, $plog"
}

stop() {  # stop <tsasr|asr>
  local port launcher vllm
  port=$(awk '/--port:/{print $2}' $C/serve-$1.yaml)
  eval "launcher=\$LAUNCHER_${1}"
  vllm=$(ps -eo pid,ppid,args | awk -v p=$launcher '$2==p' | grep -- "--port $port" | awk '{print $1}')
  kill $vllm $launcher 2>/dev/null
  status "$1 stopped vllm=$vllm"
}

evaluate() {  # evaluate <name>
  $AE open-audio-llm after-training --handoff $C/handoff.json --config $C/after-training-$1.yaml > $L/eval-$1.log 2>&1
  status "$1 eval exit=$?"
}

# Shared cards: other users' jobs come and go, so each server takes the card with the most free
# memory at launch (recorded in its config and attempt) and retries on another card if startup fails.
launch() {  # launch <tsasr|asr>
  local gpu
  for try in 1 2 3 4 5 6; do
    gpu=$(nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader,nounits \
      | awk -F', ' -v used=",$USED," 'index(used, ","$1",")==0 {print $3-$2, $1}' | sort -n | tail -1 | awk '{print $2}')
    sed -i "s/^  gpus: \[[0-9]\]/  gpus: [$gpu]/" $C/serve-$1.yaml
    start $1
    if ready $1; then USED="$USED,$gpu"; return 0; fi
    stop $1
    sleep 60
  done
  return 1
}
