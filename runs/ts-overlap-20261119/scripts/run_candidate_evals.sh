#!/usr/bin/env bash
# tasks/16: a candidate on the 23 release suites of 11.17 not covered by eval_lanes.sh (configs/eval-cand-TAG from
# make_candidate_configs.py), four lanes of one ASR + one TS server (pack-mtp serve.sh without MTP, as 11.17's release
# evals), suites dealt round-robin. W&B through this experiment's forward.
# usage: [SUITES_FILE=F] [LANES=N] run_candidate_evals.sh MODEL_NAME TAG [GPU_BASE PORT_BASE]
#   SUITES_FILE: suites to run (first column; default configs/eval-cand-TAG/suites.txt); LANES: lanes, two GPUs each (default 4).
set -u; umask 000
NAME=$1; T=$2; GB=${3:-0}; PB=${4:-19700}
ENVS=/workspace/workspace/mingdong/miniconda3/envs; PY=$ENVS/vllm-serving/bin/python
E=/workspace/workspace/mingdong/repo/oal-clean-teacher/runs/ts-overlap-20261119
P=/workspace/workspace/mingdong/repo/oal-joint-mtp/runs/joint-mtp-spec-20261107/shared/pack-mtp
C=$E/configs/eval-cand-$T; L=$C/logs; mkdir -p $L
M=$(bash /workspace/workspace/mingdong/scratch/acp-dev/ae-models.sh fetch $NAME | tail -1) || { echo "fetch $NAME failed"; exit 2; }
python3 - "$M" "$NAME" > $C/handoff.json <<'PY'
import json, sys
print(json.dumps({"framework": "open-audio-llm", "schema_version": 2, "checkpoint": sys.argv[1], "base_model": "", "tuner_type": "full",
                  "selection": "final", "global_step": 0, "training_run": sys.argv[2]}))
PY
source /workspace/workspace/mingdong/.config/open-audio-llm/wandb-cluster.sh >/dev/null 2>&1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1 AMPHION_EVAL_HOME=/workspace/data/ae-state
export WANDB_HTTP_PROXY=http://10.119.17.252:17922 WANDB_HTTPS_PROXY=http://10.119.17.252:17922
export AUDIO_DATA_CATALOG=/workspace/data/registry/catalog AUDIO_DATA_ROOTS_FILE=/workspace/data/registry/roots.json
export no_proxy=localhost,127.0.0.1 NO_PROXY=localhost,127.0.0.1
mapfile -t suites < <(awk '{print $1}' ${SUITES_FILE:-$C/suites.txt})
NL=${LANES:-4}
lane() { local i=$1 ga=$((GB + 2 * $1)) gt=$((GB + 2 * $1 + 1)) pa=$((PB + 10 * $1)) pt=$((PB + 10 * $1 + 1)) r=0 k s e
  CUDA_VISIBLE_DEVICES=$ga MODEL=$M PYTHON=$PY PORT=$pa NAME=$T-asr-$pa GPU_UTIL=0.5 AMPHION_MTP_TOKENS=0 setsid bash $P/serve.sh asr --max-num-seqs 48 > $L/serve-asr-$pa.log 2>&1 & local a=$!
  CUDA_VISIBLE_DEVICES=$gt MODEL=$M PYTHON=$PY PORT=$pt NAME=$T-tsasr-$pt GPU_UTIL=0.5 AMPHION_MTP_TOKENS=0 setsid bash $P/serve.sh tsasr --max-num-seqs 48 > $L/serve-tsasr-$pt.log 2>&1 & local b=$!
  for k in $pa $pt; do until curl -s -m 5 http://127.0.0.1:$k/v1/models >/dev/null; do kill -0 $a $b 2>/dev/null || { echo "lane $i server died"; kill -TERM -$a -$b 2>/dev/null; return 1; }; sleep 15; done; done
  echo "lane $i servers $pa/$pt ready $(date -u +%FT%TZ)"
  for k in "${!suites[@]}"; do [ $((k % NL)) = $i ] || continue; s=${suites[$k]}
    sed -e "s/__ASR_PORT__/$pa/g" -e "s/__TS_PORT__/$pt/g" $C/after-training-$s.yaml > $C/lane$i-$s.yaml
    env -u PYTHONPATH $ENVS/amphion-eval/bin/ae open-audio-llm after-training --handoff $C/handoff.json --config $C/lane$i-$s.yaml > $L/eval-$s.log 2>&1
    e=$?; echo "$T $s exit=$e $(date -u +%FT%TZ)"; [ $e = 0 ] || r=1
  done
  kill -TERM -$a -$b; wait $a $b 2>/dev/null; return $r; }
pids=(); for i in $(seq 0 $((NL - 1))); do lane $i & pids+=($!); sleep 30; done
rc=0; for p in "${pids[@]}"; do wait $p || rc=1; done; echo "$T candidate evals done rc=$rc $(date -u +%FT%TZ)"; exit $rc
