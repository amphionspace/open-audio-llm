"""One model with another's tensors under a key prefix: out = BASE, except every tensor whose key starts with PREFIX,
which comes from DONOR (README "拆分诊断"). The 11.17 package was put together the same way, by component (LLM
increment from one SFT, audio-tower increment from another). The base's other files (config, tokenizer, mtp/) are
copied unchanged.
usage: swap_component.py BASE_DIR DONOR_DIR PREFIX OUTPUT_DIR   (PREFIX e.g. thinker.audio_tower.)
"""
import json
import shutil
import sys
from pathlib import Path

from safetensors import safe_open
from safetensors.torch import save_file

WEIGHTS = "model.safetensors"


def main():
    base, donor, prefix, out = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3], Path(sys.argv[4])
    out.mkdir(parents=True, exist_ok=False)
    tensors, swapped = {}, 0
    with safe_open(base / WEIGHTS, "pt") as b, safe_open(donor / WEIGHTS, "pt") as d:
        keys = set(b.keys())
        assert keys == set(d.keys()), "the two models must have the same tensors"
        for key in sorted(keys):
            x = d.get_tensor(key) if key.startswith(prefix) else b.get_tensor(key)
            swapped += key.startswith(prefix)
            assert list(x.shape) == list(b.get_slice(key).get_shape()), key
            tensors[key] = x.contiguous()
        metadata = b.metadata()
    assert swapped, f"no tensor starts with {prefix}"
    save_file(tensors, out / WEIGHTS, metadata=metadata)
    for item in base.iterdir():
        if item.name != WEIGHTS:
            (shutil.copytree if item.is_dir() else shutil.copy2)(item, out / item.name)
    (out / "swap.json").write_text(json.dumps({"base": str(base), "donor": str(donor), "prefix": prefix,
                                               "tensors": len(keys), "from_donor": swapped}, indent=1) + "\n")
    print(f"{len(keys)} tensors, {swapped} from the donor -> {out}")


if __name__ == "__main__":
    main()
