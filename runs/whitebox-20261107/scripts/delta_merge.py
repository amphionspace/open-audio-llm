"""Task arithmetic over GRPO updates: OUT = BASE + sum_i c_i * (X_i - B_i), accumulated in fp32.
Usage: delta_merge.py OUT_DIR BASE C1 X1 B1 [C2 X2 B2 ...]  (directories holding model.safetensors)"""
import shutil
import sys
from pathlib import Path

from safetensors import safe_open
from safetensors.torch import save_file

out, base, rest = Path(sys.argv[1]), sys.argv[2], sys.argv[3:]
if len(rest) % 3:
    raise SystemExit("terms come in triples: coefficient, trained, its start")
terms = [(float(rest[i]), rest[i + 1], rest[i + 2]) for i in range(0, len(rest), 3)]
open_ = lambda d: safe_open(str(Path(d) / "model.safetensors"), "pt")
b = open_(base)
handles = [(c, open_(x), open_(s)) for c, x, s in terms]
merged = {}
for name in b.keys():
    t = b.get_tensor(name)
    if not t.is_floating_point():
        merged[name] = t
        continue
    acc = t.float()
    for c, x, s in handles:
        acc = acc + c * (x.get_tensor(name).float() - s.get_tensor(name).float())
    merged[name] = acc.to(t.dtype)
out.mkdir(parents=True, exist_ok=True)
save_file(merged, str(out / "model.safetensors"), metadata={"format": "pt"})
for p in Path(base).iterdir():
    if p.is_file() and p.suffix in (".json", ".txt", ".jinja") and p.name != "trainer_state.json":
        shutil.copy(p, out / p.name)
(out / ".complete").touch()
print("->", out, terms)
