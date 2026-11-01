"""Write an acceptance-only spec_check config: each draft at one k, single stream, both modes.

Usage: make_accept_config.py ATTEMPT OUTPUT.yaml NAME=METHOD:K:DRAFT_DIR...  (variants fill GPUs 4-7, one lane each)
"""
import copy
import sys
from pathlib import Path

import yaml

EXP = Path(__file__).resolve().parents[1]
PY = "/workspace/workspace/mingdong/miniconda3/envs/vllm-serving/bin/python"
attempt, output, *specs = sys.argv[1:]
p = copy.deepcopy(yaml.safe_load((EXP / "configs/spec-check.yaml").read_text())["parameters"])
p["output"] = str(EXP / f"tasks/02-serve-bench/attempts/{attempt}/artifacts")
p["concurrency"] = [1]
p["variants"], p["lanes"] = [], []
for index, spec in enumerate(specs):
    name, value = spec.split("=", 1)
    method, k, draft = value.split(":", 2)
    p["variants"].append({"name": name, "python": PY, "arguments": ["--speculative-config",
                         '{"method": "%s", "model": "%s", "num_speculative_tokens": %s}' % (method, draft, k)]})
    p["lanes"].append({"gpu": 4 + index, "port": 18700 + 10 * index, "variants": [name]})
p["wandb"] = {"enabled": True, "mode": "online", "entity": "1016097967-amphion", "project": "open-audio-llm",
              "name": f"mtp-draft-head-20261031-a{attempt}-accept"}
head = f"# 02 的 {attempt}：只测接受率（单路、同一批 306 + 300 条请求，协议同 spec-check.yaml）。\n"
(EXP / output).write_text(head + yaml.safe_dump({"version": 1, "parameters": p}, allow_unicode=True, sort_keys=False, width=250))
