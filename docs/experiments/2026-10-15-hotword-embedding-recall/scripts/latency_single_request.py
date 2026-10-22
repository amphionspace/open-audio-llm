"""Single-request latency of embeddinggemma-2 audio retrieval on the selfcheck-500 audio.

One request = waveform -> processor/features -> audio encoder -> embedding -> cosine against a
10k-word pool on the same GPU -> Top-50.  Run sequentially (batch 1), so numbers are per-request
service time without queueing.  The GPU is shared with other jobs, so treat as conservative.
"""
import argparse
import glob
import json
import sys
import time
import wave

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--pkgs", required=True)
ap.add_argument("--audio-root", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--pool-size", type=int, default=10000)
args = ap.parse_args()
sys.path.insert(0, args.pkgs)
import torch  # noqa: E402
from sentence_transformers import SentenceTransformer  # noqa: E402


def read_wav(path):
    w = wave.open(path)
    assert w.getframerate() == 16000 and w.getsampwidth() == 2 and w.getnchannels() == 1
    return np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768


def pct(x, q):
    return round(float(np.percentile(x, q)), 2)


def main():
    m = SentenceTransformer("/workspace/model/embeddinggemma-2", device="cuda",
                            config_kwargs={"vision_config": None}, model_kwargs={"torch_dtype": torch.bfloat16})
    files = sorted(glob.glob(f"{args.audio_root}/audio/*/*.wav"))
    wavs = [read_wav(f) for f in files]
    dur = np.array([len(w) / 16000 for w in wavs])
    pool = m.encode([f"w{i}" for i in range(args.pool_size)], prompt_name="document", normalize_embeddings=True,
                    convert_to_tensor=True, show_progress_bar=False)
    for w in wavs[:10]:
        m.encode([{"audio": w}], normalize_embeddings=True, convert_to_tensor=True, show_progress_bar=False)
    enc_ms, tot_ms, pre_ms = [], [], []
    for w in wavs:
        t = time.perf_counter(); m.preprocess([{"audio": w}]); pre_ms.append((time.perf_counter() - t) * 1000)
        torch.cuda.synchronize(); t0 = time.perf_counter()
        q = m.encode([{"audio": w}], normalize_embeddings=True, convert_to_tensor=True, show_progress_bar=False)
        torch.cuda.synchronize(); t1 = time.perf_counter()
        torch.topk(q @ pool.T, 50); torch.cuda.synchronize(); t2 = time.perf_counter()
        enc_ms.append((t1 - t0) * 1000); tot_ms.append((t2 - t0) * 1000)
    enc_ms, tot_ms = np.array(enc_ms), np.array(tot_ms)
    buckets = [(0, 3), (3, 6), (6, 10), (10, 1e9)]
    rows = []
    for lo, hi in buckets:
        k = (dur >= lo) & (dur < hi)
        if k.any():
            rows.append({"dur_s": f"{lo}-{hi if hi < 1e8 else 'inf'}", "n": int(k.sum()),
                         "mean_dur_s": round(float(dur[k].mean()), 2), "p50_ms": pct(tot_ms[k], 50), "p95_ms": pct(tot_ms[k], 95)})
    res = {"n": len(wavs), "pool": args.pool_size, "gpu": torch.cuda.get_device_name(0), "dtype": "bfloat16",
           "mean_dur_s": round(float(dur.mean()), 2), "end_to_end_ms": {"mean": round(float(tot_ms.mean()), 2),
           "p50": pct(tot_ms, 50), "p90": pct(tot_ms, 90), "p95": pct(tot_ms, 95), "p99": pct(tot_ms, 99), "max": pct(tot_ms, 100)},
           "encode_only_ms": {"p50": pct(enc_ms, 50), "p95": pct(enc_ms, 95)},
           "cpu_preprocess_ms_inside_encode": {"p50": pct(pre_ms, 50), "p95": pct(pre_ms, 95)}, "by_duration": rows}
    # Batch scan on 3-6 s utterances: a flat per-batch time means the batch-1 cost is fixed overhead.
    mid = [w for w, d in zip(wavs, dur) if 3 <= d < 6][:64]
    scan = []
    for bs in (1, 2, 4, 8, 16, 32):
        times = []
        for i in range(0, len(mid) - bs + 1, bs):
            torch.cuda.synchronize(); t0 = time.perf_counter()
            m.encode([{"audio": w} for w in mid[i:i + bs]], normalize_embeddings=True, convert_to_tensor=True,
                     batch_size=bs, show_progress_bar=False)
            torch.cuda.synchronize(); times.append((time.perf_counter() - t0) * 1000)
        scan.append({"batch": bs, "batch_ms_p50": pct(times, 50), "ms_per_utt": round(pct(times, 50) / bs, 2)})
    res["batch_scan_3_6s"] = scan
    print(json.dumps(res, ensure_ascii=False, indent=1))
    json.dump(res, open(args.out, "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
