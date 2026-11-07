"""Training-free merges of the new-prompt GRPO updates (task arithmetic per module; 2606.18521, 2507.17107).

S = SFT 073 avg3; D15 = v15 avg3 - S; D16 = v16 avg3 - v15 interp (= S + D15/2). Released-style models:
v16 interp = S + D15/2 + D16/2, v16 avg3 = S + D15/2 + D16. Each recipe gives (c15, c16) per tensor group.
Usage: layer_merge.py RECIPE OUT_DIR"""
import re
import shutil
import sys
from pathlib import Path

from safetensors import safe_open
from safetensors.torch import save_file

AE = Path("/workspace/data/ae-runs")
S = AE / "ts-robust-073-avg3/checkpoint-avg3/model.safetensors"
V15 = AE / "amphionasr2-grpo-v15avg3-ckpt400/checkpoint/model.safetensors"
V15I = AE / "amphionasr2-grpo-v15interp050/checkpoint/model.safetensors"
V16 = AE / "amphionasr2-grpo-v16avg3-ckpt400/checkpoint/model.safetensors"
MIDDLE = set(range(10, 20))  # LLM layers 10-19: where both GRPO updates are largest and most aligned


def layer(name):
    m = re.search(r"thinker\.model\.layers\.(\d+)\.", name)
    return int(m.group(1)) if m else None


RECIPES = {
    # full GRPO update in the audio tower, half elsewhere (as in the interpolation)
    "audio-full": lambda n: (0.5, 1.0) if "audio_tower" in n else (0.5, 0.5),
    # the reverse: half in the audio tower, full in the LLM
    "llm-full": lambda n: (0.5, 0.5) if "audio_tower" in n else (0.5, 1.0),
    # both rounds at full strength (near-orthogonal updates, so the sum keeps each)
    "sum-full": lambda n: (1.0, 1.0),
    # extrapolation past the sum: both rounds scaled up (task arithmetic with lambda > 1)
    "sum-1.25": lambda n: (1.25, 1.25),
    "sum-1.5": lambda n: (1.5, 1.5),
    # v15 at 3/4: the customer-TS sentence the sum drops first appears after v15
    "d15-0.75": lambda n: (0.75, 1.0),
    # full update only in LLM layers 10-19, half elsewhere
    "mid-full": lambda n: (0.5, 1.0) if layer(n) in MIDDLE else (0.5, 0.5),
}

recipe, out = RECIPES[sys.argv[1]], Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)
f = {k: safe_open(str(p), "pt") for k, p in (("s", S), ("v15", V15), ("v15i", V15I), ("v16", V16))}
merged = {}
for name in f["s"].keys():
    s = f["s"].get_tensor(name)
    if not s.is_floating_point():
        merged[name] = s
        continue
    c15, c16 = recipe(name)
    d15 = f["v15"].get_tensor(name).float() - s.float()
    d16 = f["v16"].get_tensor(name).float() - f["v15i"].get_tensor(name).float()
    merged[name] = (s.float() + c15 * d15 + c16 * d16).to(s.dtype)
save_file(merged, str(out / "model.safetensors"), metadata={"format": "pt"})
src = V16.parent
for p in src.iterdir():
    if p.is_file() and p.name not in ("model.safetensors", "trainer_state.json", "training_args.bin"):
        shutil.copy(p, out / p.name)
(out / ".complete").touch()
print(sys.argv[1], "->", out)
