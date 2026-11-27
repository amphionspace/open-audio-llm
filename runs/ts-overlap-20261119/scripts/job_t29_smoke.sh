#!/usr/bin/env bash
# tasks/29 local 1-GPU smoke of ft_enc_replay.py with every task-29 stream on (ARM simneg-asr): SMOKE=1 logs the simulated
# and meeting streams every 20 steps, W&B offline; checks the target-absent families and hard cosine actually sent,
# the second replay KL (~0 at step 1), memory and step time.  usage: job_t29_smoke.sh GPU [STEPS]
set -u
G=$1; N=${2:-100}
E=/workspace/workspace/mingdong/repo/oal-clean-teacher/runs/ts-overlap-20261119
P=/workspace/workspace/mingdong/.claude/tmp/claude-1001/-workspace-workspace-mingdong-repo-open-audio-llm/ca9aa6d0-4ecc-4cfc-b526-5322bf6c9de5/scratchpad/ovdiag/probe
D=/workspace/data/ts-overlap-20261119; O=$D/task29-simneg-asranchor/smoke; mkdir -p $O/logs
export POOLS=train_final2 REPLAY_CONFIG=$E/configs/train-data-encreplay-t29.yaml MEETING_PLAN=$D/task28-meeting-002/plan HEAD_LAYERS=5,11
export SIM_NEG=0.15 SPK_EMB=$D/task29-simneg-asranchor/speaker-embeddings.npz ASR_REPLAY_CONFIG=$E/configs/train-data-asrreplay-t29.yaml ASR_REPLAY_WORKERS=4
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 AUDIO_DATA_ROOTS_FILE=/workspace/data/registry/roots.json HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export SMOKE=1 WANDB_MODE=offline WANDB_DIR=$O PYTHONPATH=$E/shared/code-v4/src:$E/shared/code-v4/audio-data-contract-src
echo "start $(hostname -I) gpu $G $(date -u +%FT%TZ)" > $O/logs/node.txt
CUDA_VISIBLE_DEVICES=$G /workspace/workspace/mingdong/miniconda3/envs/oal-train/bin/python $P/ft_enc_replay_t29.py $O/ckpt $N 4 4 4.0 t29-smoke 100000 1.0 6 > $O/logs/train.log 2>&1
echo "train rc=$? $(date -u +%FT%TZ)" >> $O/logs/node.txt
