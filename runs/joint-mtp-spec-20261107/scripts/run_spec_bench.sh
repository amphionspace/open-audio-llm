#!/usr/bin/env bash
# 01-spec-bench on the ACP GRPO dev node: prepares the vLLM draft directory from v16 avg3's mtp/, then runs the four
# spec_check lanes (base, MTP k1/k2/k3; one GPU each) at once. Usage: run_spec_bench.sh ATTEMPT_DIR
set -uo pipefail
umask 000
R=/workspace/workspace/mingdong/repo/oal-joint-mtp/runs/joint-mtp-spec-20261107; A=$1; cd $R; mkdir -p $A/logs
PY=/workspace/workspace/mingdong/miniconda3/envs/oal-train/bin/python
M=/workspace/data/ae-runs/amphionasr2-grpo-v16avg3-ckpt400/checkpoint; D=shared/draft-v16avg3
echo "start $(date -u +%FT%TZ)" > $A/status.txt
mkdir -p $D && cp $M/mtp/model.safetensors $M/mtp/config.json $D/ && chmod 644 $D/* || exit 3
PYTHONPATH=/workspace/workspace/mingdong/repo/oal-joint-mtp/src $PY - "$D" "$M/config.json" <<'PY' || exit 3
import json, sys
from open_audio_llm.tsasr.vllm_mtp import write_vllm_config
text = json.load(open(sys.argv[2]))["thinker_config"]["text_config"]
write_vllm_config(sys.argv[1], max_position_embeddings=text["max_position_embeddings"])
print(json.load(open(sys.argv[1] + "/config.json"))["architectures"])
PY
source /workspace/workspace/mingdong/.config/open-audio-llm/wandb-cluster.sh >/dev/null
export WANDB_HTTP_PROXY=http://10.119.17.252:17920 WANDB_HTTPS_PROXY=http://10.119.17.252:17920
unset no_proxy NO_PROXY
export PYTHONPATH=/workspace/workspace/mingdong/repo/oal-joint-mtp/src
cp ${CONFIG:-configs/spec-check.yaml} $A/effective.yaml
pids=(); for lane in ${LANES:-0 1 2 3}; do $PY scripts/spec_check.py --config ${CONFIG:-configs/spec-check.yaml} --lane $lane > $A/logs/lane$lane.log 2>&1 & pids+=($!); done
rc=0; for x in "${pids[@]}"; do wait $x || rc=1; done
echo "lanes rc=$rc $(date -u +%FT%TZ)" >> $A/status.txt
$PY scripts/show_spec.py $A > $A/summary.txt 2>&1
$PY scripts/compare_spec.py $A/artifacts > $A/compare.txt 2>&1
echo "done rc=$rc $(date -u +%FT%TZ)" >> $A/status.txt; exit $rc
