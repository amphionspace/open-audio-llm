#!/usr/bin/env bash
# Evaluate one cached model on 11.17's suites, each lane = one ASR + one TS server on two GPUs (pack-mtp/serve.sh, no MTP,
# the stack of the 11.17 gate runs). Configs are written to configs/eval-TAG/ from the 11.17-era templates.
# Usage: [PACK=DIR] [TEMPLATES=DIR] [CATALOG=DIR] eval_lanes.sh CACHE_NAME TAG GPU_BASE PORT_BASE "LANE1 suites" "LANE2 suites" ...
# (suite = <template dir>/<name>; TEMPLATES holds the template dirs, CATALOG is the data catalog, default the registry)
set -u; umask 000
NAME=$1; T=$2; GB=$3; PB=$4; shift 4
ENVS=/workspace/workspace/mingdong/miniconda3/envs; PY=$ENVS/vllm-serving/bin/python
E=/workspace/workspace/mingdong/repo/oal-clean-teacher/runs/ts-overlap-20261119
M=$(bash /workspace/workspace/mingdong/scratch/acp-dev/ae-models.sh fetch $NAME) || { echo "fetch $NAME failed"; exit 2; }
# PACK picks the serving runtime; the looped encoder (tasks/11) needs shared/pack-loop, the old runtime skips its weights.
P=${PACK:-/workspace/workspace/mingdong/repo/oal-joint-mtp/runs/joint-mtp-spec-20261107/shared/pack-mtp}
G=${TEMPLATES:-/workspace/workspace/mingdong/repo/oal-grpo/runs/amphionasr2-grpo-20261101/configs}
O=$E/configs/eval-$T; L=$O/logs; mkdir -p $L
OUT=/workspace/data/ae-runs/$NAME
python3 - "$M" "$NAME" > $O/handoff.json <<'PY'
import json, sys
print(json.dumps({"framework": "open-audio-llm", "schema_version": 2, "checkpoint": sys.argv[1], "base_model": "", "tuner_type": "full",
                  "selection": "final", "global_step": 0, "training_run": sys.argv[2]}))
PY
mk() { local src=$1 s=$2 pa=$3 pt=$4
  sed -E -e "s#^model_name: .*#model_name: $NAME#" -e "s#^output_root: .*#output_root: $OUT#" \
      -e "s#^server_url: .*#server_url: http://127.0.0.1:$pa#" -e "s#^served_model: .*#served_model: $T-asr-$pa#" \
      -e "s#^tsasr_server_url: .*#tsasr_server_url: http://127.0.0.1:$pt#" -e "s#^tsasr_served_model: .*#tsasr_served_model: $T-tsasr-$pt#" \
      $G/$src/after-training-$s.yaml > $O/after-training-$s.yaml
  grep -v "^\s*#" $O/after-training-$s.yaml | grep -q "sumfull-es116\|v18avg3\|19565" && { echo "stale field in $s"; return 3; }; return 0; }
export no_proxy=localhost,127.0.0.1 NO_PROXY=localhost,127.0.0.1
lane() { local ga=$1 gt=$2 pa=$3 pt=$4; shift 4
  CUDA_VISIBLE_DEVICES=$ga MODEL=$M PYTHON=$PY PORT=$pa NAME=$T-asr-$pa GPU_UTIL=0.5 AMPHION_MTP_TOKENS=0 setsid bash $P/serve.sh asr --max-num-seqs 48 > $L/serve-asr-$pa.log 2>&1 & local a=$!
  CUDA_VISIBLE_DEVICES=$gt MODEL=$M PYTHON=$PY PORT=$pt NAME=$T-tsasr-$pt GPU_UTIL=0.5 AMPHION_MTP_TOKENS=0 setsid bash $P/serve.sh tsasr --max-num-seqs 48 > $L/serve-tsasr-$pt.log 2>&1 & local b=$!
  for p in $pa $pt; do until curl -s -m 5 http://127.0.0.1:$p/v1/models >/dev/null; do kill -0 $a $b 2>/dev/null || { echo "server died ($pa/$pt)"; kill -TERM -$a -$b 2>/dev/null; return 1; }; sleep 15; done; done
  echo "servers $pa/$pt ready $(date -u +%FT%TZ)"
  local r=0
  for x in "$@"; do local src=${x%%/*} s=${x##*/}; mk $src $s $pa $pt || { r=1; continue; }
    env -u PYTHONPATH $ENVS/amphion-eval/bin/ae open-audio-llm after-training --handoff $O/handoff.json --config $O/after-training-$s.yaml > $L/eval-$s.log 2>&1
    e=$?; echo "$T $s exit=$e $(date -u +%FT%TZ)"; [ $e = 0 ] || r=1; done
  kill -TERM -$a -$b; wait $a $b 2>/dev/null; return $r; }
source /workspace/workspace/mingdong/.config/open-audio-llm/wandb-cluster.sh >/dev/null 2>&1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1 AMPHION_EVAL_HOME=/workspace/data/ae-state
# W&B goes through a dev-machine forwarder that only admits listed node IPs; WB_PROXY picks which one.
WB=${WB_PROXY:-http://10.119.17.252:17920}
export WANDB_HTTP_PROXY=$WB WANDB_HTTPS_PROXY=$WB
export AUDIO_DATA_CATALOG=${CATALOG:-/workspace/data/registry/catalog} AUDIO_DATA_ROOTS_FILE=/workspace/data/registry/roots.json
pids=(); i=0
for spec in "$@"; do lane $((GB+2*i)) $((GB+2*i+1)) $((PB+10*i)) $((PB+10*i+1)) $spec & pids+=($!); i=$((i+1)); sleep 60; done
rc=0; for x in "${pids[@]}"; do wait $x || rc=1; done; echo "$T evals done rc=$rc $(date -u +%FT%TZ)"; exit $rc
