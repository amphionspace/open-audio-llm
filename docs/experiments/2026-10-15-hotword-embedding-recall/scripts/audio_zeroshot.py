"""Zero-shot audio -> hotword retrieval with embeddinggemma-2 on aishell test.

The deployed RAG-ASR baseline retrieves hotwords from the utterance audio, so
this is the like-for-like query modality.  Also records encode latency.
"""
import argparse
import glob
import gzip
import json
import sys
import time
import wave

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--pkgs", required=True)
ap.add_argument("--asr-json", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--extras", type=int, nargs="+", default=[0, 10000, 50000])
ap.add_argument("--audio-batch", type=int, default=16)
args = ap.parse_args()
sys.path.insert(0, args.pkgs)
sys.path.insert(0, __import__("os").path.dirname(__file__))
import torch  # noqa: E402
from common import ZH, build_pool, language_pool, load_supervisions  # noqa: E402
from sentence_transformers import SentenceTransformer  # noqa: E402

KS = [1, 5, 10, 20, 50]
REC = "/workspace/data/datasets/aishell/versions/legacy-20260804/test_rec_0/*.gz"


def read_wav(path):
    w = wave.open(path)
    assert w.getframerate() == 16000 and w.getsampwidth() == 2 and w.getnchannels() == 1
    return np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768


def recall(gold, ranked, k):
    return sum(len(g & set(r[:k])) for g, r in zip(gold, ranked)) / sum(len(g) for g in gold)


def timed(fn):
    torch.cuda.synchronize(); t = time.time(); out = fn(); torch.cuda.synchronize()
    return out, time.time() - t


def main():
    rec = {}
    for line in gzip.open(glob.glob(REC)[0], "rt"):
        d = json.loads(line); rec[d["id"]] = d["sources"][0]["source"]
    rows = [r for r in load_supervisions("aishell", "test") if r["custom"]["hotwords"]]
    gold = [set(r["custom"]["hotwords"]) for r in rows]
    asr = json.load(open(args.asr_json))
    missed = [{w for w in g if w not in asr[r["id"]]} for g, r in zip(gold, rows)]
    wavs = [read_wav(rec[r["id"]]) for r in rows]
    secs = sum(len(w) for w in wavs) / 16000
    m = SentenceTransformer("/workspace/model/embeddinggemma-2", device="cuda",
                            config_kwargs={"vision_config": None}, model_kwargs={"torch_dtype": torch.bfloat16})
    enc = lambda x, **kw: m.encode(x, normalize_embeddings=True, show_progress_bar=False, convert_to_numpy=True, **kw)
    enc([{"audio": wavs[0]}]); enc(["warmup"], prompt_name="document")
    aq, t_audio = timed(lambda: enc([{"audio": w} for w in wavs], batch_size=args.audio_batch))
    tq, t_text = timed(lambda: enc([asr[r["id"]] for r in rows], prompt_name="query", batch_size=128))
    lat = {"audio_s_per_utt": t_audio / len(rows), "audio_rtf": t_audio / secs, "text_s_per_utt": t_text / len(rows),
           "audio_batch": args.audio_batch, "mean_dur_s": secs / len(rows)}
    print(lat, flush=True)
    zh_all = language_pool(ZH)
    gold_all = {w for g in gold for w in g}
    out = {"latency": lat, "recall": []}
    for extra in args.extras:
        pool = build_pool(gold_all, zh_all, extra)
        dw, t_pool = timed(lambda: enc(pool, prompt_name="document", batch_size=256))
        for qname, q in (("audio", aq), ("asr_text", tq)):
            top = np.argsort(-(q @ dw.T), axis=1)[:, :max(KS)]
            ranked = [[pool[j] for j in row] for row in top]
            for subset, sets in (("all", gold), ("asr_missed", missed)):
                keep = [i for i, s in enumerate(sets) if s]
                row = {"pool": len(pool), "query": qname, "subset": subset,
                       **{f"R@{k}": round(recall([sets[i] for i in keep], [ranked[i] for i in keep], k), 4) for k in KS}}
                out["recall"].append(row); print(row, flush=True)
        out.setdefault("pool_encode_s_per_1k", []).append(round(t_pool / len(pool) * 1000, 3))
    json.dump(out, open(args.out, "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
