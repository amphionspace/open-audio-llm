"""Weight interpolation of a round-2 checkpoint with the 11.06 release, the way the release itself was made.

OUT = (1 - alpha) * CHECKPOINT + alpha * RELEASE for every floating-point tensor; both must hold the same tensors.
Tokenizer, config and chat template files come from CHECKPOINT.

usage: interp_release.py <out dir> <checkpoint dir> [alpha=0.5]
"""
import shutil
import sys
from pathlib import Path

from safetensors import safe_open
from safetensors.torch import save_file

RELEASE = Path("/workspace/data/ae-runs/amphionasr2-grpo-v10interp050/checkpoint")


def main():
    out, checkpoint = Path(sys.argv[1]), Path(sys.argv[2])
    alpha = float(sys.argv[3]) if len(sys.argv) > 3 else 0.5
    ours = safe_open(str(checkpoint / "model.safetensors"), "pt")
    release = safe_open(str(RELEASE / "model.safetensors"), "pt")
    if set(ours.keys()) != set(release.keys()):
        raise SystemExit(f"tensor names differ: {sorted(set(ours.keys()) ^ set(release.keys()))[:10]}")
    merged = {}
    for name in ours.keys():
        tensor = ours.get_tensor(name)
        if tensor.is_floating_point():
            other = release.get_tensor(name)
            if other.shape != tensor.shape:
                raise SystemExit(f"{name}: {tuple(tensor.shape)} vs {tuple(other.shape)}")
            tensor = ((1 - alpha) * tensor.float() + alpha * other.float()).to(tensor.dtype)
        merged[name] = tensor
    out.mkdir(parents=True, exist_ok=True)
    save_file(merged, str(out / "model.safetensors"), metadata={"format": "pt"})
    for path in checkpoint.iterdir():
        if path.is_file() and path.suffix in (".json", ".txt", ".jinja") and path.name != "trainer_state.json":
            shutil.copy(path, out / path.name)
    print(f"{out}: alpha {alpha}, {len(merged)} tensors")


if __name__ == "__main__":
    main()
