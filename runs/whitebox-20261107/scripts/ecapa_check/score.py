"""ECAPA verification score per item: max over 1.5 s mixture windows (energy VAD) of cos(ECAPA(enroll), ECAPA(window)).

Audio comes through the Catalog exactly like spec_check.load_ref (int16, enrollment cut/padded to 5 s).
Speaker model: WeSpeaker ECAPA-TDNN512-LM ONNX (the one used to build the hard negatives), CPU.

  PYTHONPATH=src <env-with-onnxruntime+lhotse>/bin/python runs/whitebox-20261107/scripts/ecapa_check/score.py \
      --run-dir <AmphionEval run> --out <scores.jsonl> --workers 48
"""
import argparse, json, os, multiprocessing as mp
import numpy as np

SR = 16000
ENROLL_SECONDS = 5.0          # serving contract
WIN_S, HOP_S = 1.5, 0.5       # verification window / hop
MIN_VOICED = 0.5              # a window counts as speech if >= 50% of its 10 ms frames pass the energy VAD
MODEL = "/workspace/model/wespeaker-ecapa-tdnn512-LM/voxceleb_ECAPA512_LM.onnx"
CATALOG = "/workspace/data/registry/catalog"
ROOTS = "/workspace/data/registry/roots.json"

_res = _sess = _name = None


def _init():
    global _res, _sess, _name
    import onnxruntime as ort, torch
    torch.set_num_threads(1)
    from open_audio_llm.data.catalog_resolver import LhotseCatalogAudioResolver
    _res = LhotseCatalogAudioResolver(CATALOG, ROOTS)
    so = ort.SessionOptions(); so.intra_op_num_threads = 2; so.inter_op_num_threads = 1
    _sess = ort.InferenceSession(MODEL, sess_options=so, providers=["CPUExecutionProvider"])
    _name = _sess.get_inputs()[0].name


def load_ref(ref, seconds=None):
    from audio_data_contract import AudioRef
    cut = _res.get_cut(AudioRef(**{k: ref[k] for k in ("dataset_id", "version", "split", "cut_id")}))
    if cut.sampling_rate != SR:
        cut = cut.resample(SR)
    a = np.clip(cut.load_audio().mean(0) * 32768, -32768, 32767).astype(np.int16)
    if seconds is not None:
        n = int(seconds * SR); a = np.pad(a[:n], (0, max(0, n - len(a))))
    return a.astype(np.float32) / 32768


def fbank(w):
    import torch
    from torchaudio.compliance import kaldi
    f = kaldi.fbank(torch.from_numpy(w).unsqueeze(0) * 32768, num_mel_bins=80, frame_length=25, frame_shift=10,
                    dither=0, sample_frequency=SR, window_type="hamming", use_energy=False)
    return (f - f.mean(0, keepdim=True)).numpy()


def embed_batch(ws):
    e = _sess.run(None, {_name: np.stack([fbank(w) for w in ws])})[0].reshape(len(ws), -1)
    return e / np.linalg.norm(e, axis=1, keepdims=True)


def frames_db(w):
    hop, win = 160, 400
    n = max(1, (len(w) - win) // hop + 1)
    idx = np.arange(win)[None] + hop * np.arange(n)[:, None]
    x = w[np.minimum(idx, len(w) - 1)]
    return 10 * np.log10((x ** 2).mean(1) + 1e-12)


def vad(w):  # same energy VAD as fa_audit/feats.py
    d = frames_db(w)
    return d > max(-55.0, np.percentile(d, 95) - 25.0)


def score(line):
    s = json.loads(line)
    slots = {x["name"]: x["ref"] for x in s["record"]["audio_slots"]}
    e = load_ref(slots["enrollment"], ENROLL_SECONDS); m = load_ref(slots["mixture"])
    ea = vad(e)
    mask = np.pad(np.repeat(ea, 160), (0, max(0, len(e) - len(ea) * 160)))[:len(e)]
    ev = e[mask] if ea.sum() > 30 else e  # enrollment voiced part (drops zero padding)
    if len(ev) < int(0.3 * SR):
        ev = np.pad(ev, (0, int(0.3 * SR) - len(ev)))
    W, H = int(WIN_S * SR), int(HOP_S * SR)
    ma = vad(m)
    if len(m) <= W:
        starts = [0]; vf = [float(ma.mean())]
    else:
        starts = list(range(0, len(m) - W + 1, H))
        if starts[-1] != len(m) - W:
            starts.append(len(m) - W)
        vf = [float(ma[st // 160:(st + W) // 160].mean()) for st in starts]
    keep = [i for i, v in enumerate(vf) if v >= MIN_VOICED] or [int(np.argmax(vf))]
    wins = [m[starts[i]:starts[i] + W] for i in keep]
    if len(wins[0]) < int(0.3 * SR):
        wins = [np.pad(wins[0], (0, int(0.3 * SR) - len(wins[0])))]
    ee = embed_batch([ev])[0]
    cos = embed_batch(wins) @ ee
    return dict(id=s["id"], ecapa_max=float(cos.max()), ecapa_top3=float(np.sort(cos)[-3:].mean()),
                ecapa_whole=float(embed_batch([m])[0] @ ee), n_win=len(wins), mix_dur=len(m) / SR,
                enroll_voiced_s=float(ea.sum() * 0.01))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=48)
    a = ap.parse_args()
    done = set()
    if os.path.exists(a.out):
        done = {json.loads(l)["id"] for l in open(a.out)}
    lines = [l for l in open(f"{a.run_dir}/samples.jsonl") if json.loads(l)["id"] not in done]
    print("todo", len(lines), "done", len(done), flush=True)
    with mp.get_context("spawn").Pool(a.workers, initializer=_init) as pool, open(a.out, "a") as f:
        for i, r in enumerate(pool.imap_unordered(score, lines, chunksize=8)):
            f.write(json.dumps(r) + "\n")
            if i % 1000 == 0:
                print(i, len(lines), flush=True); f.flush()
    print("ok", flush=True)


if __name__ == "__main__":
    main()
