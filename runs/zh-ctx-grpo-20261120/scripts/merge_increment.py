"""Add a model's increment over its start to another model: out = base + (candidate - start), tensor by tensor.

The 11.17 package was built this way (GRPO v18 avg3 plus es116's Spanish increment, no further training); here the
context-correction increment (avg3 - 11.06, J's SFT and this GRPO) goes onto 11.17 (README "合并诊断"). Sums are taken
in float32 and stored in the base's dtype. The base's other files (config, tokenizer, mtp/) are copied unchanged:
its MTP draft was trained on the base backbone, which changes acceptance, not greedy output.
usage: merge_increment.py BASE_DIR START_DIR CANDIDATE_DIR OUTPUT_DIR
"""
import json
import shutil
import sys
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

WEIGHTS = "model.safetensors"


def main():
    base, start, cand, out = map(Path, sys.argv[1:5])
    out.mkdir(parents=True, exist_ok=False)
    tensors, moved = {}, 0
    with safe_open(base / WEIGHTS, "pt") as b, safe_open(start / WEIGHTS, "pt") as s, \
            safe_open(cand / WEIGHTS, "pt") as c:
        keys = set(b.keys())
        assert keys == set(s.keys()) == set(c.keys()), "the three models must have the same tensors"
        for key in sorted(keys):
            x, y, z = b.get_tensor(key), s.get_tensor(key), c.get_tensor(key)
            assert x.shape == y.shape == z.shape, key
            delta = z.float() - y.float()
            moved += bool(delta.abs().max() > 0)
            tensors[key] = (x.float() + delta).to(x.dtype).contiguous()
        metadata = b.metadata()
    save_file(tensors, out / WEIGHTS, metadata=metadata)
    for item in base.iterdir():
        if item.name == WEIGHTS:
            continue
        (shutil.copytree if item.is_dir() else shutil.copy2)(item, out / item.name)
    (out / "merge.json").write_text(json.dumps({
        "formula": "base + (candidate - start)", "base": str(base), "start": str(start), "candidate": str(cand),
        "tensors": len(keys), "tensors_changed_by_increment": moved}, indent=1) + "\n")
    print(f"{len(keys)} tensors, {moved} with a non-zero increment -> {out}")


if __name__ == "__main__":
    main()
