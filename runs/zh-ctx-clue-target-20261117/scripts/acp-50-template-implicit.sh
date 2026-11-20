#!/usr/bin/env bash
# zh-ctx-clue-target-20261117, question 3 (2026-11-20): train template-implicit-noexplain on GPUs 0-7, then evaluate its
# checkpoint-3000 on context-correction v2, think off and on, on GPUs 0/1. Configs by make_configs.py on the dev machine.
set -u
C=/workspace/workspace/mingdong/repo/oal-clue-target/runs/zh-ctx-clue-target-20261117/scripts/node_chain.sh
ARMS="template-implicit-noexplain" exec bash $C train eval
