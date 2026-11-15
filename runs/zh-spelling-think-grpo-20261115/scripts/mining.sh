#!/usr/bin/env bash
# 01-mining: the release round's (oal-grpo round 10) mining on the round-2 think checkpoint-3000, think off.
# Same lanes, sources, candidates and epochs as oal-grpo b1-grpo-v16-mining.sh; only the model, code-v2, data config and
# output dir differ. 7 plain-ASR lanes (hotword/nonspeech + miss) on serve-asr-{0..6}, then 7 TS lanes on serve-tsasr-{0..6},
# 16 samples per prompt; scripts/mining_plans_n16.py splits the kept prompts into five plans -> tasks/01-mining/plan-*.jsonl.
set -u
umask 002
E=/workspace/workspace/mingdong/repo/oal-think-exp/runs/zh-spelling-think-grpo-20261115; cd $E
ENVS=/workspace/workspace/mingdong/miniconda3/envs; C=configs/mining; O=tasks/01-mining; mkdir -p $O
START=$(bash /workspace/workspace/mingdong/scratch/acp-dev/ae-models.sh fetch \
  whai:open-audio-llm/runs/zh-spelling-think-r2-20261115/tasks/01-train-think/attempts/001/artifacts/training/checkpoint-3000/) || exit 5
grep -q "path: $START\$" $C/serve-asr-0.yaml || { echo "start checkpoint moved to $START; rerun scripts/make_configs.py"; exit 5; }
echo "start $START $(date -u +%FT%TZ)"
for i in $(seq 360); do [ $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1>2000' | wc -l) -eq 0 ] && break; sleep 10; done
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
source /workspace/workspace/mingdong/.config/open-audio-llm/wandb-cluster.sh
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1 NUMPY_MADVISE_HUGEPAGE=0 \
  WANDB_ENTITY=1016097967-amphion WANDB_PROJECT=open-audio-llm WANDB_DIR=$E/$O \
  WANDB_HTTP_PROXY=http://10.119.17.252:17920 WANDB_HTTPS_PROXY=http://10.119.17.252:17920
MINE="env PYTHONPATH=shared/code-v2/src:shared/code-v2/audio-data-contract-src $ENVS/oal-train-vllm/bin/python -m open_audio_llm.scripts.grpo_mining --data_config configs/train-data-grpo.yaml --workers 6 --concurrency 48 --samples 16"
start_servers() {  # $1 = serve-asr | serve-tsasr, $2 = first port
  local kind=$1; pids=(); ports=()
  for n in 0 1 2 3 4 5 6; do ports+=($(( $2 + n )))
    $ENVS/oal-train/bin/open-audio-llm serve --config $C/$kind-$n.yaml > $O/$kind-$n.log 2>&1 & pids+=($!); done
  for n in 0 1 2 3 4 5 6; do
    until curl -s --noproxy '*' -m 5 http://127.0.0.1:${ports[$n]}/v1/models >/dev/null 2>&1; do
      kill -0 ${pids[$n]} 2>/dev/null || { echo "$kind $n exited $(date -u +%FT%TZ)"; kill -TERM "${pids[@]}" 2>/dev/null; exit 1; }
      sleep 15
    done
  done; echo "$kind servers ready $(date -u +%FT%TZ)"
}
stop_servers() { kill -TERM "${pids[@]}" 2>/dev/null; wait "${pids[@]}" 2>/dev/null; sleep 30; }
lanes=()
start_servers serve-asr 19400
mpids=(); names=()
for n in 0 1 2 3 4 5 6; do L=$O/plain-lane-$n; mkdir -p $L; lanes+=($L)
  $MINE --source "hw_grpo_candidates@v1-20261101:zh" --source "ts_nonspeech_pool@env-20261101:all" \
    --source "hard_hw_sft@hard-20261014:train" --source "ts_nonspeech_pool@v1-20261016:all" --source "ts_nonspeech_pool@babble-20261024:all" \
    --candidates 900 --epoch $((20 + n)) --server http://127.0.0.1:${ports[$n]} --model mine-r2think-asr --output $L/kept.jsonl > $L/run.log 2>&1 &
  mpids+=($!); names+=(plain-$n)
done
S=()
for k in alimeeting_sdm@asr-20261014-punc-20261102:train aishell4@asr-20261014-punc-20261102:train \
         threed_speaker_zh_far_asr@train-punc-v1-20261102:train asr_distant_talk_pool@v1-20261102-punc-20261102:text_zh \
         ami_sdm@asr-20261014-punc-20261102:train asr_distant_talk_pool@v1-20261102:text_en \
         wenetspeech_clean@train-punc-v1-20261102:train aishell2@train-punc-v1-20261102:train kespeech@train-punc-v1-20261102:train \
         gigaspeech@train-punc-v1-20261102:train librispeech@train-punc-v1-20261102:train magicdata_ramc@train-punc-v1-20261102:train; do
  S+=(--source "$k"); done
for n in 0 1 2 3 4 5 6; do L=$O/miss-lane-$n; mkdir -p $L; lanes+=($L)
  $MINE "${S[@]}" --candidates 400 --epoch $((40 + n)) --server http://127.0.0.1:${ports[$n]} --model mine-r2think-asr --output $L/kept.jsonl > $L/run.log 2>&1 &
  mpids+=($!); names+=(miss-$n)
done
rc=0; for i in "${!mpids[@]}"; do wait ${mpids[$i]}; x=$?; echo "${names[$i]} exit $x $(date -u +%FT%TZ)"; [ $x -ne 0 ] && rc=$x; done
stop_servers; [ $rc -eq 0 ] || exit $rc
start_servers serve-tsasr 19410
T=()
for k in alimeeting_sdm@turn-taking-20261014-utts-punc-20261102:{train_2spk,train_3spk,train_neg} \
         aishell4@turn-taking-20261014-utts-punc-20261102:{train_2spk,train_3spk,train_neg} \
         ami_sdm@turn-taking-20261014-utts-punc-20261102:{train_2spk,train_3spk,train_neg} \
         magicdata_ramc@turn-taking-20261014-utts-punc-20261102:{train_2spk,train_neg} \
         chime6@turn-taking-20261014:{train_3spk,train_neg} \
         emilia2_dialog@dialog-turn-taking-20261013-enroll:{train_2spk,train_neg} \
         ts_real_train@v2-20261018-utts-punc-20261102:{alimeeting,aishell4} ts_real_train@v1-20261017-utts-punc-20261102:{ramc,alimeeting,aishell4} \
         ts_real_train@v3-20261020-utts-punc-20261102:{alimeeting_en_mn,alimeeting_en_mf} \
         ts_real_train@v3-20261020-utts-punc-20261102:emilia2_1spk ts_real_train@en-v1-20261101-utts-punc-20261102:{ami,dipco} \
         ts_real_train@en-v1-20261101-utts:{icsi,notsofar} \
         ts_scene_pool@multi-20261017-punchw-20261102:zh ts_scene_pool@multi-20261017-punchw2-20261102:en \
         ts_scene_pool@alt-twins-20261024-punchw-20261102:zh ts_scene_pool@alt-twins-20261024-punchw2-20261102:en \
         ts_scene_pool@solo-20261020-punchw-20261102:zh ts_scene_pool@solo-20261020-punchw2-20261102:en \
         ts_scene_pool@handover-20261021-punchw-20261102:zh ts_scene_pool@handover-20261021-punchw2-20261102:en \
         ts_scene_pool@xsession-near-20261023-punchw-20261102:zh ts_scene_pool@magicdata-near-20261023-punchw-20261102:zh \
         ts_scene_pool@cvaccent-near-20261101-punchw2-20261102:en ts_scene_pool@hard-20261016-punchw-20261102:zh \
         ts_scene_pool@hard-20261016-punchw2-20261102:en \
         aishellmix@pack3-current-audio-record-20260910-punc-20261102:train_neg librimix@960-current-audio-record-20260910-punc-20261102:train_neg; do
  T+=(--source "$k"); done
mpids=()
for n in 0 1 2 3 4 5 6; do L=$O/ts-lane-$n; mkdir -p $L; lanes+=($L); url=http://127.0.0.1:${ports[$n]}
  $MINE "${T[@]}" --candidates 120 --epoch $((30 + n)) --server $url --model mine-r2think-tsasr \
    --ts_server $url --ts_model mine-r2think-tsasr --output $L/kept.jsonl > $L/run.log 2>&1 & mpids+=($!)
done
rc=0; for n in 0 1 2 3 4 5 6; do wait ${mpids[$n]}; x=$?; echo "ts lane $n exit $x $(date -u +%FT%TZ)"; [ $x -ne 0 ] && rc=$x; done
stop_servers; [ $rc -eq 0 ] || exit $rc
$ENVS/oal-train/bin/python -I scripts/mining_plans_n16.py $O AmphionASR-spelling-think-grpo-mining-r2think3000-n16-direction-split "${lanes[@]}" > $O/plans.log 2>&1; x=$?
echo "plans exit $x $(date -u +%FT%TZ)"; head -1 $O/plans.log; exit $x
