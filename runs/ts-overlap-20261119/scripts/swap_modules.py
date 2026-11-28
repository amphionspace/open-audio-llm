"""tasks/30 language-ID localization: a full model dir in BASE's format (11.17's files and tensor set) whose tensors of one
module group are taken from DONOR (the original Qwen3-ASR-1.7B, sharded). GROUP: enc = thinker.audio_tower.*,
llm = thinker.model.* and thinker.lm_head.weight, all = every tensor DONOR has. Tensors only BASE has (11.17's
audio_tower.sep_token) stay BASE's. usage: swap_modules.py BASE_DIR DONOR_DIR OUT_DIR GROUP   (oal-train python)"""
import json, pathlib, shutil, sys
from safetensors import safe_open
from safetensors.torch import save_file

base, donor, out, group = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2]), pathlib.Path(sys.argv[3]), sys.argv[4]
pick = {"enc": lambda k: k.startswith("thinker.audio_tower."),
        "llm": lambda k: k.startswith("thinker.model.") or k == "thinker.lm_head.weight",
        "all": lambda k: True}[group]
h = safe_open(str(base / "model.safetensors"), "pt")
index = json.loads((donor / "model.safetensors.index.json").read_text())["weight_map"]
shards = {f: safe_open(str(donor / f), "pt") for f in set(index.values())}
w, n = {}, 0
for k in h.keys():
    t = h.get_tensor(k)
    if k in index and pick(k):
        d = shards[index[k]].get_tensor(k)
        assert d.shape == t.shape, f"{k}: {tuple(d.shape)} vs {tuple(t.shape)}"
        t, n = d.to(t.dtype), n + 1
    w[k] = t.contiguous()
save_file(w, str(out / "model.safetensors"), metadata=h.metadata())
for f in base.iterdir():
    if f.name != "model.safetensors" and f.is_file():
        shutil.copy2(f, out / f.name)
print(f"{group}: {n} of {len(w)} tensors from {donor}")
