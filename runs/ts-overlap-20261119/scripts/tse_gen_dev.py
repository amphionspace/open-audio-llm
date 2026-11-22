"""Fixed validation scenes for the extractor from dev-split speakers (disjoint from train and from v12).
usage: tse_gen_dev.py OUT_DIR N   (amphion-eval python)"""
import json, random, sys
from pathlib import Path
import soundfile as sf
sys.path.insert(0, str(Path(__file__).resolve().parent))
import tse_data
out, n = Path(sys.argv[1]), int(sys.argv[2]); (out / "wav").mkdir(parents=True, exist_ok=True)
recipe, pools, splits, n_spk = tse_data.setup("dev"); print("dev pools", n_spk, len(splits), "splits", flush=True)
rng = random.Random("tse-dev-20261122")
with open(out / "manifest.jsonl", "w") as f:
    for i in range(n):
        name, e, m, t = tse_data.scene(rng, pools, splits); key = f"dev-{i:04d}"
        for s, w in (("enroll", e), ("mix", m), ("target", t)):
            sf.write(out / "wav" / f"{key}.{s}.wav", w, tse_data.SR, subtype="FLOAT")
        f.write(json.dumps({"key": key, "split": name}) + "\n")
print("done", n)
