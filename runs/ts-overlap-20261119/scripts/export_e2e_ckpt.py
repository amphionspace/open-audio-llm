"""Write a full model dir from a ft_e2e2.py checkpoint: 11.17's files, its weights overwritten by the checkpoint's
fine-tuned tensors (mask heads dropped). The tied lm_head is not a separate parameter in training, so it is set to the
fine-tuned embed_tokens. usage: export_e2e_ckpt.py BASE_DIR CKPT_PT OUT_DIR   (oal-train python)"""
import shutil, sys, pathlib, torch
from safetensors import safe_open
from safetensors.torch import save_file
base, ckpt, out = pathlib.Path(sys.argv[1]), sys.argv[2], pathlib.Path(sys.argv[3])
st = torch.load(ckpt, map_location="cpu", mmap=True)["state"]
h = safe_open(str(base / "model.safetensors"), "pt"); keys = list(h.keys())
assert set(st) <= set(keys), "checkpoint names missing from the base model"
w = {k: (st[k] if k in st else h.get_tensor(k)).to(h.get_tensor(k).dtype).contiguous() for k in keys}
tied = [k for k in keys if k not in st]; assert tied == ["thinker.lm_head.weight"], tied
assert torch.equal(h.get_tensor("thinker.lm_head.weight"), h.get_tensor("thinker.model.embed_tokens.weight")), "base lm_head is not tied"
w["thinker.lm_head.weight"] = w["thinker.model.embed_tokens.weight"].clone()
save_file(w, str(out / "model.safetensors"), metadata=h.metadata())
for f in base.iterdir():
    if f.name != "model.safetensors" and f.is_file(): shutil.copy2(f, out / f.name)
print(len(st), "fine-tuned tensors written over", len(keys), "from", ckpt)
