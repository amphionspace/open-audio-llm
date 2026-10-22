"""aishell test: embeddinggemma-2 vs the existing pinyin/LCS retrieve_hotwords.

Query is the Qwen3-ASR-1.7B hypothesis, matching the two-stage pipeline's phase 1.
"""
import argparse
import importlib.util
import json
import sys
from multiprocessing import Pool

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--pkgs", required=True, help="dir with sentence_transformers, pypinyin")
ap.add_argument("--asr-json", required=True)
ap.add_argument("--retrieve-py", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--extras", type=int, nargs="+", default=[0, 10000])
ap.add_argument("--workers", type=int, default=16)
args = ap.parse_args()
sys.path.insert(0, args.pkgs)
sys.path.insert(0, __import__("os").path.dirname(__file__))
from common import ZH, build_pool, hotwords_of, language_pool, load_supervisions  # noqa: E402

KS = [1, 5, 10, 20, 50]
_state = {}


def _init(path, pool):
    spec = importlib.util.spec_from_file_location("rh", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    _state["rh"], _state["pool"] = mod, pool


def _rule(hyp):
    return _state["rh"].retrieve_hotwords(_state["pool"], hyp, top_k=max(KS), language="zh")


def recall(gold_sets, ranked, k):
    hit = tot = 0
    for g, r in zip(gold_sets, ranked):
        hit += len(g & set(r[:k])); tot += len(g)
    return hit / tot


def main():
    from sentence_transformers import SentenceTransformer
    asr = json.load(open(args.asr_json))
    rows = [r for r in load_supervisions("aishell", "test") if r["custom"]["hotwords"]]
    hyps = [asr[r["id"]] for r in rows]
    gold = [set(r["custom"]["hotwords"]) for r in rows]
    # "missed": the hotword is not a verbatim substring of the ASR hypothesis, i.e. ASR got it wrong.
    missed = [{w for w in g if w not in h} for g, h in zip(gold, hyps)]
    print(f"gold={sum(map(len, gold))} asr_missed={sum(map(len, missed))}", flush=True)
    zh_all = language_pool(ZH)
    model = SentenceTransformer("/workspace/model/embeddinggemma-2", device="cuda")
    enc = lambda x, p: model.encode(x, prompt_name=p, batch_size=128, normalize_embeddings=True, show_progress_bar=False)
    q = enc(hyps, "query")
    out = []
    for extra in args.extras:
        pool = build_pool({w for g in gold for w in g}, zh_all, extra)
        sc = q @ enc(pool, "document").T
        top = np.argsort(-sc, axis=1)[:, :max(KS)]
        emb = [[pool[j] for j in row] for row in top]
        with Pool(args.workers, _init, (args.retrieve_py, pool)) as p:
            rule = p.map(_rule, hyps, chunksize=8)
        union = {k: [list(dict.fromkeys(a[:k] + b[:k])) for a, b in zip(emb, rule)] for k in KS}
        for name, ranked in (("embed", emb), ("rule", rule)):
            for subset, sets in (("all", gold), ("asr_missed", missed)):
                keep = [i for i, s in enumerate(sets) if s]
                row = {"pool": len(pool), "method": name, "subset": subset, "n_gold": sum(len(sets[i]) for i in keep)}
                row.update({f"R@{k}": round(recall([sets[i] for i in keep], [ranked[i] for i in keep], k), 4) for k in KS})
                out.append(row); print(row, flush=True)
        for subset, sets in (("all", gold), ("asr_missed", missed)):
            keep = [i for i, s in enumerate(sets) if s]
            row = {"pool": len(pool), "method": "union_of_both_top_k", "subset": subset, "n_gold": sum(len(sets[i]) for i in keep)}
            row.update({f"R@{k}": round(recall([sets[i] for i in keep], [union[k][i] for i in keep], 10**6), 4) for k in KS})
            out.append(row); print(row, flush=True)
    json.dump(out, open(args.out, "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
