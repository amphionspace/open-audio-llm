#!/usr/bin/env bash
# 第二段。run_fp8emu.sh 跑完 v12、警务四集和警务单说后，单个 ASR 服务在共享卡上太慢：ASR 全量加热词全量
# 估计要十几个小时。改为三路并行：
# - 原 ASR 服务（18200）等它正在跑的评测结束，接着跑剩下的小集和热词全量；
# - 另起两个 ASR 服务（18201、18202），各跑 ASR 全量的一半（按数据源整份切分）。
# 用法：run_fp8emu_rest.sh <18200 上正在跑的 ae PID> <原 ASR 服务的 launcher PID> <18200 上接着跑的评测...>
source "$(dirname "$0")/fp8emu_lib.sh"
CURRENT=$1
LAUNCHER_asr=$2
shift 2
USED=,$(awk -F'[][]' '/^  gpus:/{print $2}' $C/serve-asr.yaml)
status "--- rest: running ae=$CURRENT, asr launcher=$LAUNCHER_asr, then $*, cards in use $USED"
(
  while kill -0 $CURRENT 2>/dev/null; do sleep 30; done
  status "ae $CURRENT finished (exit state in its run.json)"
  for name in "$@"; do evaluate $name; done
  stop asr
) &
A=$!
launch asrb || { wait $A; exit 1; }
if launch asrc; then
  (evaluate asrfull-shard1; stop asrb) &
  (evaluate asrfull-shard2; stop asrc) &
else
  (evaluate asrfull-shard1; evaluate asrfull-shard2; stop asrb) &
fi
wait
status "rest done"
