#!/usr/bin/env bash
# zh-ctx-clue-target-20261117, question 5: G (template-no-synth)'s context-correction eval v2, think off and on, on
# GPUs 0/1 of one ACP node (node_chain.sh eval fetches the latest completed checkpoint-3000 of the arm).
set -u; umask 002
ARMS="template-no-synth" bash /workspace/workspace/mingdong/repo/oal-clue-target/runs/zh-ctx-clue-target-20261117/scripts/node_chain.sh eval
