"""TS-ASR 无训练探测：对已发布的 34479 在 AISHELLMix test 上改变注册音频条件。

只读：读取测试集与权重，向本地 vLLM 服务发请求，结果写到 --out。
"""
import argparse, base64, gzip, io, json, random, re, sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import requests
import soundfile as sf

REL = Path(__file__).parent / "rel"  # 已发布 v3-34479 包的本地拷贝，见 README
sys.path.insert(0, str(REL / "runtime"))
from ts_prompt import TS_CONCAT_SYSTEM  # noqa: E402

SR = 16000
BASE = Path("/workspace/data/datasets/aishellmix/versions/original-audio-record-20260910")
ROOTS = json.load(open("/workspace/data/registry/roots.json"))


def load_jsonl(split):
    with gzip.open(BASE / split / f"{split}.jsonl.gz", "rt") as f:
        return [json.loads(l) for l in f]


def load_index(cuts):
    out = {}
    with gzip.open(BASE / "audio_index" / "audio-index.jsonl.gz", "rt") as f:
        for l in f:
            r = json.loads(l)
            if r["cut_id"] in cuts:
                out[r["cut_id"]] = ROOTS[r["root_alias"]] + "/" + r["relative_path"]
    return out


def norm(s):
    return re.sub(r"[^\w]", "", s or "").replace("_", "")


def edit(a, b):
    if not a:
        return len(b)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[-1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def read(path):
    w, sr = sf.read(path, dtype="float32")
    assert sr == SR
    return w.mean(1) if w.ndim > 1 else w


def enroll_variant(w, cond, other=None):
    n = 3 * SR
    if cond == "base":
        seg = w[:n]
    elif cond == "short2":
        seg = w[: 2 * SR]
    elif cond == "short1":
        seg = w[: 1 * SR]
    elif cond == "off3":
        seg = w[3 * SR: 6 * SR]
        if len(seg) < n:
            return None
    elif cond == "mixed_noise":
        seg = w[:n]
        rng = np.random.default_rng(0)
        p = np.sqrt((seg ** 2).mean() + 1e-12)
        seg = seg + rng.standard_normal(len(seg)).astype(np.float32) * p * 10 ** (-10 / 20)  # SNR 10 dB
    elif cond == "lead_silence":
        seg = np.concatenate([np.zeros(int(1.5 * SR), np.float32), w[: int(1.5 * SR)]])
    elif cond == "lp4k":
        from scipy.signal import butter, sosfilt
        seg = sosfilt(butter(8, 4000, "low", fs=SR, output="sos"), w[:n]).astype(np.float32)
    elif cond == "overlap0":
        seg = w[:n].copy()
        o = other[:n]
        o = np.pad(o, (0, n - len(o)))
        seg = np.pad(seg, (0, n - len(seg)))
        seg = seg + o * (np.sqrt((seg ** 2).mean() + 1e-12) / np.sqrt((o ** 2).mean() + 1e-12))
    else:
        raise ValueError(cond)
    return np.pad(seg, (0, n - len(seg)))


def ask(url, model, wav):
    buf = io.BytesIO()
    sf.write(buf, wav, SR, format="WAV", subtype="PCM_16")
    r = requests.post(url + "/v1/chat/completions", timeout=300, json={
        "model": model, "temperature": 0, "max_tokens": 512,
        "messages": [{"role": "system", "content": TS_CONCAT_SYSTEM},
                     {"role": "user", "content": [{"type": "input_audio", "input_audio": {
                         "data": base64.b64encode(buf.getvalue()).decode(), "format": "wav"}}]}]})
    r.raise_for_status()
    raw = r.json()["choices"][0]["message"]["content"] or ""
    head, sep, text = raw.partition("<asr_text>")
    return (text if sep else raw).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8011")
    ap.add_argument("--model", default="amphionasr-v3-tsasr")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--splits", default="test_1spk,test_2mix,test_2mix_neg,test_3mix,test_3mix_neg")
    ap.add_argument("--conds", default="base,short2,short1,off3,mixed_noise")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rng = random.Random(0)
    recs = {}
    for sp in a.splits.split(","):
        rs = load_jsonl(sp)
        recs[sp] = rng.sample(rs, min(a.n, len(rs)))
    # 同一混合音频下其他说话人的参考文本，用来判断"转写了别人"
    sibling = {}
    for sp in ("test_2mix", "test_3mix"):
        by = {}
        for r in load_jsonl(sp):
            mix, tgt = r["id"].split(":", 1)[1].split("|")
            by.setdefault(mix, {})[tgt] = norm(r["target"])
        sibling[sp] = by
    cuts = {s["ref"]["cut_id"] for rs in recs.values() for r in rs for s in r["audio_slots"]}
    paths = load_index(cuts)
    jobs = []
    for sp, rs in recs.items():
        for r in rs:
            for c in a.conds.split(","):
                jobs.append((sp, r, c))

    def run(job):
        sp, r, c = job
        ew = read(paths[r["audio_slots"][0]["ref"]["cut_id"]])
        other = None
        if c == "overlap0":
            lst = recs[sp]
            o = lst[(lst.index(r) + 1) % len(lst)]
            other = read(paths[o["audio_slots"][0]["ref"]["cut_id"]])
        ev = enroll_variant(ew, c, other)
        if ev is None:
            return None
        mw = read(paths[r["audio_slots"][1]["ref"]["cut_id"]])
        hyp = ask(a.url, a.model, np.concatenate([ev, mw]))
        row = {"split": sp, "id": r["id"], "cond": c, "hyp": hyp, "ref": r["target"],
               "enroll_dur": len(ew) / SR, "mix_dur": len(mw) / SR}
        if sp in sibling:
            mix, tgt = r["id"].split(":", 1)[1].split("|")
            row["others"] = [t for k, t in sibling[sp][mix].items() if k != tgt]
        return row

    with ThreadPoolExecutor(24) as ex:
        rows = [x for x in ex.map(run, jobs) if x]
    Path(a.out).write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in rows))
    print("rows", len(rows))


if __name__ == "__main__":
    main()
