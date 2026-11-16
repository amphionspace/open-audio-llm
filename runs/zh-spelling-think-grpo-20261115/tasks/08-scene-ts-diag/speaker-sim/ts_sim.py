#!/usr/bin/env python3
"""ECAPA cosine similarity between enrollment and mixture audio for several TS record groups.
Usage: ts_sim.py OUTDIR   (writes OUTDIR/sim_<group>.csv)"""
import csv, glob, gzip, json, os, random, sys
from multiprocessing import Pool
import numpy as np

ROOTS = json.load(open("/workspace/data/registry/roots.json"))
DS = "/workspace/data/datasets"
MODEL = "/workspace/model/wespeaker-ecapa-tdnn512-LM/voxceleb_ECAPA512_LM.onnx"
SEED = 20261116
_idx = {}


def jl(p):
    with (gzip.open(p, "rt") if p.endswith(".gz") else open(p)) as f:
        for l in f:
            if l.strip():
                yield json.loads(l)


def index(dataset_id, version):
    key = (dataset_id, version)
    if key not in _idx:
        d = {}
        for p in glob.glob(f"{DS}/{dataset_id}/versions/{version}/audio[-_]index*.jsonl.gz"):
            for r in jl(p):
                d[r["cut_id"]] = r
        assert d, f"no audio index for {key}"
        _idx[key] = d
    return _idx[key]


def resolve(ref):
    r = index(ref["dataset_id"], ref["version"])[ref["cut_id"]]
    return dict(path=os.path.join(ROOTS[r["root_alias"]], r["relative_path"]),
                start=ref.get("start", 0.0), dur=ref.get("duration"), ch=ref.get("channel", 0))


# ---- embedding (identical to adc-ctx-emilia/scripts/ts_benchmark/speaker_embeddings.py) ----
_embed = None


def make_embedder():
    import onnxruntime as ort, torch
    from torchaudio.compliance import kaldi
    torch.set_num_threads(1)
    so = ort.SessionOptions(); so.intra_op_num_threads = 1
    session = ort.InferenceSession(MODEL, so, providers=["CPUExecutionProvider"])
    name = session.get_inputs()[0].name

    def embed(w):
        feats = kaldi.fbank(torch.from_numpy(w).unsqueeze(0) * 32768, num_mel_bins=80, frame_length=25,
                            frame_shift=10, dither=0, sample_frequency=16000, window_type="hamming",
                            use_energy=False)
        feats = feats - feats.mean(0, keepdim=True)
        e = session.run(None, {name: feats.unsqueeze(0).numpy()})[0].reshape(-1)
        return e / np.linalg.norm(e)
    return embed


def load(spec):
    import soundfile as sf
    info = sf.info(spec["path"])
    sr = info.samplerate
    a, b = int(round(spec["start"] * sr)), None
    n = int(round(spec["dur"] * sr)) if spec["dur"] else -1
    w, _ = sf.read(spec["path"], start=a, frames=n, dtype="float32", always_2d=True)
    w = w[:, spec["ch"] if spec["ch"] < w.shape[1] else 0]
    if sr != 16000:
        import torch, torchaudio
        w = torchaudio.functional.resample(torch.from_numpy(w), sr, 16000).numpy()
    return np.ascontiguousarray(w)


def work(item):
    global _embed
    if _embed is None:
        _embed = make_embedder()
    gid, rid, e, m, extra = item
    try:
        we, wm = load(e), load(m)
        s = float(_embed(we) @ _embed(wm))
        return dict(group=gid, id=rid, sim=s, enroll_s=len(we) / 16000, mix_s=len(wm) / 16000, **extra)
    except Exception as ex:
        return dict(group=gid, id=rid, sim="", enroll_s="", mix_s="", err=repr(ex), **extra)


def slots(rec):
    s = {x["name"]: x["ref"] for x in rec["audio_slots"]}
    return resolve(s["enrollment"]), resolve(s["mixture"])


def sample(items, n, rng):
    items = list(items)
    rng.shuffle(items)
    return items[:n]


def build():
    rng = random.Random(SEED)
    jobs = []
    # 1. synthetic train_ts (all)
    p = f"{DS}/context_correction_synth_zh/versions/train-v1-20261114/records/train_ts.jsonl.gz"
    for r in jl(p):
        e, m = slots(r)
        jobs.append(("synth_train_ts", r["id"], e, m, {"target_empty": int(not r.get("target"))}))
    # 2a. real: ts_real_train (config version), 500 non-empty per split
    base = f"{DS}/ts_real_train/versions/v1-20261017-utts-punc-20261102/records"
    for sp in ("alimeeting", "aishell4", "ramc"):
        recs = [r for r in jl(f"{base}/records_{sp}.jsonl.gz") if (r.get("target") or "").strip()]
        for r in sample(recs, 500, rng):
            e, m = slots(r)
            jobs.append((f"real_ts_real_train_{sp}", r["id"], e, m, {"target_empty": 0}))
    # 2b. real: ts_scene_pool zh alt-twins (single-speaker target utt, enrollment = other utt of same speaker)
    p = f"{DS}/ts_scene_pool/versions/alt-twins-20261024-punchw-20261102/records_zh.jsonl.gz"
    recs = [r for r in jl(p) if (r.get("target") or "").strip()]
    for r in sample(recs, 500, rng):
        e, m = slots(r)
        jobs.append(("real_ts_scene_pool_zh_alt_twins", r["id"], e, m, {"target_empty": 0}))
    # 3. scene-ts (police_scene_bench@v1-20261028 samples of the ae run)
    p = "/workspace/data/ae-runs/amphionasr2-grpo-v10interp050/runs/run-20261106T021900Z-53c44b85/samples.jsonl"
    for r in jl(p):
        rec = r["record"]
        e, m = slots(rec)
        mode = "sep" if "-sep-" in r["id"] else "overlap"
        pres = "present" if (r.get("reference") or "").strip() else "absent"
        jobs.append((f"scene_ts_{mode}_{pres}", r["id"], e, m, {"target_empty": int(pres == "absent")}))
    return jobs


if __name__ == "__main__":
    out = sys.argv[1]
    jobs = build()
    print(len(jobs), "jobs", flush=True)
    with Pool(int(os.environ.get("NPROC", 24))) as pool:
        res = []
        for i, r in enumerate(pool.imap(work, jobs, chunksize=8)):
            res.append(r)
            if i % 500 == 0:
                print(i, flush=True)
    keys = ["group", "id", "sim", "enroll_s", "mix_s", "target_empty", "err"]
    with open(f"{out}/all_items.csv", "w", newline="") as f:
        w = csv.DictWriter(f, keys); w.writeheader()
        for r in res:
            w.writerow({k: r.get(k, "") for k in keys})
