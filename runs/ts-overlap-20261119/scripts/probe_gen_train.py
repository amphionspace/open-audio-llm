"""Probe training scenes: the v12 generator (audio-data-contract 3f9c7ca, recipe v4-20261022) with every speech pool
switched from the corpora's test splits to their train splits (speakers disjoint from v12), same split parameters,
for the target-present splits of the v12 test subset. Per scene: 5 s enrollment, mixture, clean (= target + noise,
interferers removed exactly), reference text, labels. usage: gen_train.py GEN_DIR SPLITS_JSON N_PER_SPLIT OUT_DIR WORKERS"""
import json, random, sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np, soundfile as sf, yaml
gen, split_names, n_per, out, workers = Path(sys.argv[1]), json.load(open(sys.argv[2])), int(sys.argv[3]), Path(sys.argv[4]), int(sys.argv[5])
sys.path.insert(0, str(gen))
import build, mixer
from audio_data_contract import load_catalog
SRC = Path("/workspace/data/datasets/ts_asr_benchmark/source/v4-20261022")
SR = mixer.SAMPLE_RATE
recipe = yaml.safe_load((gen / "recipe.yaml").read_text())
for pool in recipe["pools"].values():
    for s in pool.get("sources", []):
        s["split"] = "train"
recipe["seed"] = "probe-train-20261122"
roots = json.loads(Path("/workspace/data/registry/roots.json").read_text())
catalog = load_catalog(Path("/workspace/data/registry/catalog"))
pools = build.build_pools(recipe, catalog, roots)
pools["__enroll_files__"] = build.enrollment_files(recipe, roots)
pools["__noise__"] = {k: build.noise_files(spec, catalog, roots) for k, spec in recipe["noise"].items()}
extra, rir_pools = build.prepared_noise(SRC / "noise")
pools["__noise__"].update(extra); pools["__rir_pools__"] = rir_pools
pools["__rirs__"] = sorted(str(p) for p in (SRC / "rir").glob("*.wav"))
pools["__rir_pools__"]["dns5_sim"] = list(pools["__rirs__"])
pools["__rirs__"] = sorted(p for v in pools["__rir_pools__"].values() for p in v)
build.G.update(recipe=recipe, pools=pools)
splits = {s["name"]: s for s in build.expand_splits(recipe)}
sys.path.insert(0, str(Path(__file__).parent))
from ovr_stems_lib import stems

def job(arg):
    name, index = arg; split = splits[name]
    for attempt in range(build.ATTEMPTS):
        rng = random.Random(f"{recipe['seed']}:{split['params'].get('pair', name)}:{index}:{attempt}")
        try:
            plan = mixer.plan_scene(rng, split["params"], pools, None); break
        except mixer.PlanError:
            continue
        except Exception as exc:
            return {"split": name, "index": index, "error": f"{type(exc).__name__}: {exc}"}
    else:
        return {"split": name, "index": index, "error": "no plan"}
    if plan["params"].get("mix_effect"):
        return {"split": name, "index": index, "error": "mix_effect"}
    target = next((t for t in plan["tracks"] if t.role == "target"), None)
    if target is None:
        return {"split": name, "index": index, "error": "no target"}
    t, o, n = stems(plan, build, mixer)
    load = lambda utt: build.read_audio(utt.path)
    enroll = mixer.render_enrollment(plan, load, lambda p: sf.read(p, dtype="float32")[0], lambda kind, path: build._cached_noise(path))
    c = 5 * SR; enroll = np.pad(enroll[:c], (0, max(0, c - len(enroll))))
    key = f"{name}-{index:04d}"
    for suffix, w in (("enroll", enroll), ("mix", t + o + n), ("clean", t + n)):
        sf.write(out / "wav" / f"{key}.{suffix}.wav", np.clip(w, -1, 1).astype(np.float32), SR, subtype="PCM_16")
    labels, _ = build.describe(plan, split, index)
    return {"key": key, "split": name, "reference": build._label_text(target.items), "labels": labels}

if __name__ == "__main__":
    (out / "wav").mkdir(parents=True, exist_ok=True)
    jobs = [(s, i) for s in split_names for i in range(n_per)]
    ok = bad = 0
    with ProcessPoolExecutor(workers) as pool, open(out / "manifest.jsonl", "w", encoding="utf-8") as f, open(out / "errors.jsonl", "w") as ferr:
        for r in pool.map(job, jobs, chunksize=4):
            if "error" in r: ferr.write(json.dumps(r) + "\n"); bad += 1
            else: f.write(json.dumps(r, ensure_ascii=False) + "\n"); ok += 1
    print(ok, "scenes,", bad, "skipped")
