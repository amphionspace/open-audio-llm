"""Where the follow-up 2 arms moved, vs 11.17 (CPU): relative change per module group (audio tower, each LLM layer
block, final norm, lm_head) and of the lm_head rows of the presence-decision tokens.
usage: decision_rows_delta.py BASE_DIR NAME=DIR ..."""
import sys, collections
from safetensors import safe_open
ROWS = {"None": 2240, "Chinese": 8453, "English": 6364, "Spanish": 15154, "<asr_text>": 151704}
def group(k):
    if "audio_tower" in k: return "audio_tower"
    if "lm_head" in k: return "lm_head"
    if "embed_tokens" in k: return "embed_tokens"
    if ".layers." in k:
        i = int(k.split(".layers.")[1].split(".")[0]); return f"llm_layers_{i // 7 * 7:02d}-{i // 7 * 7 + 6:02d}"
    return k.split("thinker.")[-1]
base = safe_open(sys.argv[1] + "/model.safetensors", "pt")
keys = [k for k in base.keys() if k.startswith("thinker.") and "mtp" not in k]
for arg in sys.argv[2:]:
    name, d = arg.split("=", 1); f = safe_open(d + "/model.safetensors", "pt")
    num, den = collections.Counter(), collections.Counter()
    for k in keys:
        a = base.get_tensor(k).float(); b = f.get_tensor(k).float(); g = group(k)
        num[g] += (b - a).pow(2).sum().item(); den[g] += a.pow(2).sum().item()
        if k == "thinker.lm_head.weight":
            rows = {t: f"{100 * ((b[i] - a[i]).norm() / a[i].norm()).item():.3f}%" for t, i in ROWS.items()}
            mean_row = 100 * ((b - a).norm(dim=1) / a.norm(dim=1).clamp_min(1e-12)).mean().item()
    print(name, " ".join(f"{g}={100 * (num[g] / den[g]) ** 0.5:.3f}%" for g in sorted(num)))
    print("   lm_head rows", rows, f"mean row {mean_row:.3f}%")
