"""Merge two GRPO branches trained from the same start B (task arithmetic): out = B + c * ((X - B) + (Y - B)).
c = 1 adds both updates in full; c = 0.5 is the plain average of X and Y. Usage: branch_merge.py B X Y C OUT_DIR"""
import shutil
import sys
from pathlib import Path

from safetensors import safe_open
from safetensors.torch import save_file

b, x, y, c, out = sys.argv[1], sys.argv[2], sys.argv[3], float(sys.argv[4]), Path(sys.argv[5])
out.mkdir(parents=True, exist_ok=True)
f = {k: safe_open(str(Path(p) / "model.safetensors"), "pt") for k, p in (("b", b), ("x", x), ("y", y))}
merged = {}
for name in f["b"].keys():
    t = f["b"].get_tensor(name)
    if not t.is_floating_point():
        merged[name] = t
        continue
    base = t.float()
    merged[name] = (base + c * ((f["x"].get_tensor(name).float() - base) + (f["y"].get_tensor(name).float() - base))).to(t.dtype)
save_file(merged, str(out / "model.safetensors"), metadata={"format": "pt"})
for p in Path(b).iterdir():
    if p.is_file() and p.suffix in (".json", ".txt", ".jinja") and p.name != "trainer_state.json":
        shutil.copy(p, out / p.name)
(out / ".complete").touch()
print("->", out)
