#!/usr/bin/env bash
# 噪声参照：不开 FP8、其余服务设置与 FP8 ASR 服务相同，重跑警务、警务单说、babble、CHiME-4。
# 这几个集的 bf16 基线只有一次运行，判断不了 FP8 的微小差异是否超出重跑抖动。
source "$(dirname "$0")/fp8emu_lib.sh"
# fp8emu_lib.sh 的 ready 要求日志里有 FP8 模拟标记（bf16 服务永远没有），这里改成反向核对：服务起来且没有模拟。
ready() {
  local port launcher plog
  port=$(awk '/--port:/{print $2}' $C/serve-$1.yaml)
  eval "launcher=\$LAUNCHER_${1}"
  for i in $(seq 1 180); do
    curl -s -m 5 http://127.0.0.1:$port/v1/models >/dev/null && break
    kill -0 $launcher 2>/dev/null || { status "$1: server died"; return 1; }
    sleep 10
  done
  curl -s -m 5 http://127.0.0.1:$port/version > $L/server-version-$1.json || { status "$1: not ready"; return 1; }
  plog=$(grep -o '"log": "[^"]*"' $L/serve-$1.log | head -1 | cut -d'"' -f4)
  grep -q 'fp8-sm90-emu' $plog && { status "$1: unexpected FP8 emulation ($plog)"; return 1; }
  status "$1 ready without FP8, $plog"
}
USED=,5  # 18201 还在跑 ASR 分片 1
status "--- bf16 rerun"
launch bf16rerun || exit 1
for name in police police-single babble chime4; do evaluate rerun-$name; done
stop bf16rerun
status "bf16 rerun done"
