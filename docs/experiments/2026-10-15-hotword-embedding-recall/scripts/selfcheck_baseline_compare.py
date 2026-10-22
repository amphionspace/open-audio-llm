"""Same-protocol comparison against the deployed RAG on the audiollm-go selfcheck-500 set.

Protocol (audiollm-go docs/archive/evaluations/amphion-20260913/rag-recall): 500 utterances, 10k-word
pool per language, proxy labels = reference-text ∩ pool, target-pair Recall@K.  The deployed RAG ranks
(`baseline`) and the 09-13 candidate with frame-level scoring are read from its target files.

The exact en-10k/zh-10k pool files are not available.  We rebuild a pool from every word the deployed
RAG ever returned in a top-50 list (all real pool members) plus all targets, and optionally pad it to
~10k with random same-language hotwords from the local dataset.
"""
import argparse
import json
import random
import sys
import wave

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--pkgs", required=True)
ap.add_argument("--rag-recall-dir", required=True, help="audiollm-go .../amphion-20260913/rag-recall")
ap.add_argument("--audio-root", required=True, help="selfcheck dataset dir containing audio/<subset>/<id>.wav")
ap.add_argument("--out", required=True)
args = ap.parse_args()
sys.path.insert(0, args.pkgs)
sys.path.insert(0, __import__("os").path.dirname(__file__))
import torch  # noqa: E402
from common import EN, ZH, language_pool  # noqa: E402
from sentence_transformers import SentenceTransformer  # noqa: E402

KS = [1, 5, 10, 50]
POOL_TARGET = {"en": 9984, "zh": 10087}  # sizes reported by the deployed-RAG evaluation


def read_wav(path):
    w = wave.open(path)
    assert w.getframerate() == 16000 and w.getsampwidth() == 2 and w.getnchannels() == 1
    return np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768


def load_jsonl(p):
    return [json.loads(line) for line in open(p)]


def main():
    d = args.rag_recall_dir
    rows = [r for r in load_jsonl(f"{d}/targets.jsonl") if r["targets"]]
    recon = {"en": set(), "zh": set()}
    for f in ("baseline-results.jsonl", "candidate-results.jsonl", "candidate-go-local-results.jsonl"):
        for r in load_jsonl(f"{d}/{f}"):
            recon[r["language"]].update(r["top_50"])
    for r in rows:
        recon[r["language"]].update(t["word"] for t in r["targets"])
    local = {"en": language_pool(EN), "zh": language_pool(ZH)}
    rng = random.Random(0)
    m = SentenceTransformer("/workspace/model/embeddinggemma-2", device="cuda",
                            config_kwargs={"vision_config": None}, model_kwargs={"torch_dtype": torch.bfloat16})
    enc = lambda x, **kw: m.encode(x, normalize_embeddings=True, show_progress_bar=False, convert_to_numpy=True, **kw)
    wavs = [read_wav(f"{args.audio_root}/audio/{r['subset']}/{r['id']}.wav") for r in rows]
    qa = enc([{"audio": w} for w in wavs], batch_size=16)
    qt = enc([r["reference"] for r in rows], prompt_name="query", batch_size=64)
    out = []
    for variant in ("reconstructed", "padded_to_10k"):
        for lang in ("en", "zh"):
            pool = sorted(recon[lang])
            if variant == "padded_to_10k":
                extra = sorted(local[lang] - set(pool)); rng.shuffle(extra)
                pool += extra[:max(0, POOL_TARGET[lang] - len(pool))]
            idx = {w: i for i, w in enumerate(pool)}
            dw = enc(pool, prompt_name="document", batch_size=256)
            sel = [i for i, r in enumerate(rows) if r["language"] == lang]
            for qname, q in (("audio", qa), ("reference_text", qt)):
                sc = q[sel] @ dw.T
                ranks = []
                for row_i, i in enumerate(sel):
                    for t in rows[i]["targets"]:
                        ranks.append(int((sc[row_i] > sc[row_i, idx[t["word"]]]).sum()) + 1)
                n = len(ranks)
                res = {"variant": variant, "lang": lang, "query": qname, "pool": len(pool), "pairs": n,
                       **{f"R@{k}": round(sum(r <= k for r in ranks) / n, 4) for k in KS}}
                out.append(res); print(res, flush=True)
    # Reported ranks: deployed RAG from targets.jsonl; the 09-13 candidate (frame-level "Go local" scoring)
    # from go-local-targets.jsonl.
    reported = (("deployed_rag_audio", f"{d}/targets.jsonl", "baseline"),
                ("rag_20260913_candidate_go_local_audio", f"{d}/go-local-targets.jsonl", "candidate"))
    for name, path, key in reported:
        src = [r for r in load_jsonl(path) if r["targets"]]
        for lang in ("en", "zh"):
            pairs = [t[key] for r in src if r["language"] == lang for t in r["targets"]]
            res = {"variant": "reported", "lang": lang, "query": name, "pool": POOL_TARGET[lang], "pairs": len(pairs),
                   **{f"R@{k}": round(sum(1 for p in pairs if p and p <= k) / len(pairs), 4) for k in KS}}
            out.append(res); print(res, flush=True)
    json.dump(out, open(args.out, "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
