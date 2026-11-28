"""Re-plan v12 (ts_asr_benchmark v4-20261022) scenes with the generator at audio-data-contract 3f9c7ca and render stems:
target image (its RIR and gain), interferer image, noise (scene noise + floor), all with the mixture's own scaling.
A sample is kept only if target + interferers + noise reproduces the stored mixture (PCM_16 tolerance) and has no
nonlinear mix effect. usage: stems.py GEN_DIR MANIFEST OUT_DIR   (amphion-eval python)"""
import json, random, sys
from pathlib import Path
import numpy as np, soundfile as sf, yaml
gen, manifest, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
sys.path.insert(0, str(gen))
import build, mixer
from audio_data_contract import load_catalog
SRC = Path("/workspace/data/datasets/ts_asr_benchmark/source/v4-20261022")
recipe = yaml.safe_load((gen / "recipe.yaml").read_text())
roots = json.loads(Path("/workspace/data/registry/roots.json").read_text())
catalog = load_catalog(Path("/workspace/data/registry/catalog"))
pools = build.build_pools(recipe, catalog, roots)
pools["__enroll_files__"] = build.enrollment_files(recipe, roots)
pools["__noise__"] = {k: build.noise_files(spec, catalog, roots) for k, spec in recipe["noise"].items()}
extra, rir_pools = build.prepared_noise(SRC / "noise")
pools["__noise__"].update(extra); pools["__rir_pools__"] = rir_pools
pools["__rirs__"] = sorted(str(p) for p in (SRC / "rir").glob("*.wav"))
data = np.load(SRC / "speaker-embeddings.npz", allow_pickle=False)
vectors = dict(zip(data["keys"].tolist(), data["vectors"]))
similarity = lambda a, b: float(vectors[a] @ vectors[b])
pools["__rir_pools__"]["dns5_sim"] = list(pools["__rirs__"])
pools["__rirs__"] = sorted(p for v in pools["__rir_pools__"].values() for p in v)
build.G.update(recipe=recipe, pools=pools)
splits = {s["name"]: s for s in build.expand_splits(recipe)}
SR = mixer.SAMPLE_RATE

def plan_for(split, index):
    for attempt in range(build.ATTEMPTS):
        rng = random.Random(f"{recipe['seed']}:{split['params'].get('pair', split['name'])}:{index}:{attempt}")
        try:
            return mixer.plan_scene(rng, split["params"], pools, similarity)
        except mixer.PlanError:
            continue
    raise RuntimeError("no plan")

def stems(plan):
    total = int(plan["total"] * SR)
    parts = {"target": np.zeros(total + int(0.4 * SR), np.float32), "other": np.zeros(total + int(0.4 * SR), np.float32)}
    for track in plan["tracks"]:
        rir = build.read_audio(track.rir) if track.rir else None
        dest = parts["target" if track.role == "target" else "other"]
        for utt, start in track.items:
            w = mixer.normalize_rms(build.read_audio(utt.path))
            if rir is not None:
                w = mixer.apply_rir(w, rir)
            w = w * 10.0 ** (track.gain_db / 20.0)
            begin = int(start * SR); end = min(begin + len(w), total)
            if end <= begin: continue
            truncated = end - begin < len(w); w = w[: end - begin].copy()
            if truncated:
                fade = min(len(w), int(mixer.FADE_SECONDS * SR)); w[-fade:] *= np.linspace(1.0, 0.0, fade, dtype=np.float32)
            dest[begin:end] += w
    t, o = parts["target"][:total], parts["other"][:total]
    speech = t + o; noise = np.zeros(total, np.float32)
    spec = plan.get("noise")
    if spec:
        seg = mixer._fit(build._cached_noise(spec["file"]) if spec["kind"] != "quiet" else
                         np.random.default_rng(0).standard_normal(SR * 30).astype(np.float32), total, spec["offset"])
        noise = seg * (mixer.rms(speech) / mixer.rms(seg) * 10.0 ** (-spec["snr_db"] / 20.0))
    mix = speech + noise; peak = np.abs(mix).max(); k = (0.5 / peak) * 10.0 ** (plan.get("mix_gain_db", 0.0) / 20.0)
    floor = (plan.get("floor") or {}).get("mix")
    fl = mixer.floor_noise(total, floor, lambda kind, path: build._cached_noise(path)) if floor else np.zeros(total, np.float32)
    return (t * k).astype(np.float32), (o * k).astype(np.float32), (noise * k + fl).astype(np.float32)

(out / "stems").mkdir(parents=True, exist_ok=True)
RUN = "/workspace/data/ae-runs/whitebox-merge-v18a-es116inc/runs/run-20261117T032925Z-961b82b3/samples.jsonl"
source = {json.loads(l)["id"]: json.loads(l)["source"] for l in open(RUN, encoding="utf-8")}
report = []
for line in open(manifest, encoding="utf-8"):
    r = json.loads(line)
    if not source[r["id"]].startswith("ts_asr_benchmark@v4-20261022:"):
        report.append({"id": r["id"], "ok": False, "why": source[r["id"]].split(":")[0]}); continue
    split_name, index = r["id"].rsplit(":", 1); split = splits[split_name]
    plan = plan_for(split, int(index))
    stored, _ = sf.read(SRC / "audio" / split_name / f"{split_name}-{int(index):04d}-mix.wav", dtype="float32")
    if plan["params"].get("mix_effect"):
        report.append({"id": r["id"], "ok": False, "why": "mix_effect"}); continue
    t, o, n = stems(plan)
    err = float(np.abs(np.clip(t + o + n, -1, 1) - stored).max()) if len(stored) == len(t) else None
    ok = err is not None and err < 2.0 / 32768
    report.append({"id": r["id"], "ok": ok, "max_abs_err": err})
    if ok:
        for name, w in (("target", t), ("other", o), ("noise", n)):
            sf.write(out / "stems" / f"{r['key']}.{name}.wav", w, SR, subtype="FLOAT")
with open(out / "stems-report.jsonl", "w") as f:
    for x in report: f.write(json.dumps(x) + "\n")
print(sum(x["ok"] for x in report), "/", len(report), "reproduced")
