#!/usr/bin/env bash
# tasks/28 ACP 2-node job (both nodes run this): task 25's training (job_encreplay.sh: LLM frozen, audio tower trained; simulated
# scenes CE + mask aux; replay KL on configs/train-data-encreplay.yaml; 30k steps) plus the meeting stream (MEETING_PLAN, CE on real
# meetings sent the way speaker-enroll sends them), then task 25's evaluation of steps 10k/20k/30k on the 16 GPUs.
# The simulator modules (gen/, tse/) and the evaluation lists are the ones task 25 ran, under P (scratchpad copy).
# W&B goes through the dev-machine forwarder on 17923 (admits node 0 only after its IP is added, then the wandb-ready flag).
# usage: job_meeting_train.sh   (outputs under /workspace/data/ts-overlap-20261119/task28-meeting)
set -u
NR=${SENSECORE_PYTORCH_NODE_RANK:-0}; NN=${SENSECORE_PYTORCH_NNODES:-1}; O=/workspace/data/ts-overlap-20261119/task28-meeting; mkdir -p $O/logs
E=/workspace/workspace/mingdong/repo/oal-clean-teacher/runs/ts-overlap-20261119
P=/workspace/workspace/mingdong/.claude/tmp/claude-1001/-workspace-workspace-mingdong-repo-open-audio-llm/ca9aa6d0-4ecc-4cfc-b526-5322bf6c9de5/scratchpad/ovdiag/probe
echo "node $NR of $NN ip $(hostname -I) master ${MASTER_ADDR:-?}:${MASTER_PORT:-?} $(date -u +%FT%TZ)" > $O/logs/node-$NR.txt
export POOLS=train_final2 REPLAY_CONFIG=$E/configs/train-data-encreplay.yaml MEETING_PLAN=$O/plan HEAD_LAYERS=5,11 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 AUDIO_DATA_ROOTS_FILE=/workspace/data/registry/roots.json
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 NCCL_IB_TIMEOUT=22 NCCL_IB_RETRY_CNT=13 NCCL_IB_AR_THRESHOLD=0
export PYTHONPATH=$E/shared/code-v4/src:$E/shared/code-v4/audio-data-contract-src
source /workspace/workspace/mingdong/.config/open-audio-llm/wandb-cluster.sh >/dev/null 2>&1
export WANDB_HTTP_PROXY=http://10.119.17.252:17923 WANDB_HTTPS_PROXY=http://10.119.17.252:17923 WANDB_DIR=$O WANDB_READY_FILE=$O/wandb-ready
ENV=/workspace/workspace/mingdong/miniconda3/envs/oal-train/bin
$ENV/torchrun --nnodes $NN --node_rank $NR --nproc_per_node 8 --master_addr $MASTER_ADDR --master_port $MASTER_PORT \
  $P/ft_enc_replay_meeting.py $O/ckpt 30000 4 4 4.0 meeting-001 10000 1.0 6 > $O/logs/train-node$NR.log 2>&1
echo "train rc=$? node $NR $(date -u +%FT%TZ)" >> $O/logs/node-$NR.txt
until [ -f $O/ckpt/step30000.pt ] || grep -q "Traceback" $O/logs/train-node0.log 2>/dev/null; do sleep 30; done
chmod -R a+rX $O; export OMP_NUM_THREADS=8
for st in 10000 20000 30000; do mkdir -p $O/eval-$st/mask; done
pids=()
for g in 0 1 2 3 4 5 6 7; do s=$((NR * 8 + g))
  ( for st in 10000 20000 30000; do C=$O/ckpt/step$st.pt; [ -f $C ] || continue
      CUDA_VISIBLE_DEVICES=$g $ENV/python $P/decode_adapter.py ft $C $P/test-list.jsonl $P/../subset/manifest.jsonl $O/eval-$st/heavy-$s.jsonl orig,clean $s 16
      CUDA_VISIBLE_DEVICES=$g $ENV/python $P/decode_adapter.py ft $C $P/safety-list.jsonl /workspace/data/ts-overlap-20261119/task19-tse/safety/manifest.jsonl $O/eval-$st/safety-$s.jsonl mix $s 16
      CUDA_VISIBLE_DEVICES=$g $ENV/python $P/mask_probe.py $C $O/eval-$st/mask 0 $s 16
    done ) > $O/logs/eval-$s.log 2>&1 & pids+=($!)
done
rc=0; for p in "${pids[@]}"; do wait $p || rc=1; done; chmod -R a+rX $O
echo "eval rc=$rc node $NR $(date -u +%FT%TZ)" >> $O/logs/node-$NR.txt
# The job ends when node 0 exits; in task 25 that cut node 1's evaluation short (mask probe 520/615).
if [ $NR = 0 ]; then for r in $(seq 1 $((NN - 1))); do until grep -q "eval rc=" $O/logs/node-$r.txt 2>/dev/null; do sleep 30; done; done; fi
