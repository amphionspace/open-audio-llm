#!/usr/bin/env bash
# zh-ctx-clue-target-20261117: the two implicit arms, rerun after context_correction_synth_zh@implicit-v1-20261117 was
# registered (audio-data-contract #271) and synced to the registry; 30-chain's attempt stopped at the data preflight.
# Configs were written by 30-chain's configs stage; train both one after the other on GPUs 0-7, then evaluate both.
set -u
C=/workspace/workspace/mingdong/repo/oal-clue-target/runs/zh-ctx-clue-target-20261117/scripts/node_chain.sh
ARMS="clue-target-implicit clue-target-implicit-noexplain" exec bash $C train eval
