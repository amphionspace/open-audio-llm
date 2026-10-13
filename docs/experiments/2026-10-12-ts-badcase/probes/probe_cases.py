"""回放 badcase 并做对照：注册音频替换、源音频电平调整。只读，向本地 vLLM 发请求。"""
import base64, io, json, sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import requests
import soundfile as sf

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE / "rel" / "runtime"))
from ts_prompt import TS_CONCAT_SYSTEM  # noqa: E402

SR = 16000
ROOT = Path("/workspace/data/badcase/2026-10-10_amphionasr-target-speaker/cases")
URL = "http://127.0.0.1:8011"
CASES = ["case-01", "case-02", "case-03", "case-04/part-01", "case-04/part-02", "case-05", "case-06", "case-07"]


def read(p):
    w, sr = sf.read(p, dtype="float32")
    assert sr == SR
    return w


def ask(wav, max_tokens=512):
    buf = io.BytesIO()
    sf.write(buf, wav, SR, format="WAV", subtype="PCM_16")
    r = requests.post(URL + "/v1/chat/completions", timeout=300, json={
        "model": "amphionasr-v3-tsasr", "temperature": 0, "max_tokens": max_tokens,
        "messages": [{"role": "system", "content": TS_CONCAT_SYSTEM},
                     {"role": "user", "content": [{"type": "input_audio", "input_audio": {
                         "data": base64.b64encode(buf.getvalue()).decode(), "format": "wav"}}]}]})
    r.raise_for_status()
    c = r.json()["choices"][0]
    raw = c["message"]["content"] or ""
    t = raw.partition("<asr_text>")[2] if "<asr_text>" in raw else raw
    return t.strip(), c["finish_reason"]


def db(x):
    return 20 * np.log10(np.sqrt((x ** 2).mean()) + 1e-9)


def main():
    # 一个与测试人员无关的"错误说话人"注册：AISHELLMix test_1spk 的某条
    import gzip
    B = Path("/workspace/data/datasets/aishellmix/versions/original-audio-record-20260910")
    roots = json.load(open("/workspace/data/registry/roots.json"))
    rec = json.loads(gzip.open(B / "test_1spk/test_1spk.jsonl.gz", "rt").readline())
    cut = rec["audio_slots"][0]["ref"]["cut_id"]
    other = None
    with gzip.open(B / "audio_index/audio-index.jsonl.gz", "rt") as f:
        for l in f:
            r = json.loads(l)
            if r["cut_id"] == cut:
                other = read(roots[r["root_alias"]] + "/" + r["relative_path"])
                break
    other = other[:3 * SR]
    other = np.pad(other, (0, 3 * SR - len(other)))
    other = other * (0.17 / max(np.abs(other).max(), 1e-6))  # 与注册音频同峰值

    jobs = []
    for c in CASES:
        reg = read(ROOT / c / "registration-reference.wav")
        src = read(ROOT / c / "source-audio.wav")
        jobs.append((c, "as_is", np.concatenate([reg, src])))
        jobs.append((c, "enroll_silence", np.concatenate([np.zeros(3 * SR, np.float32), src])))
        jobs.append((c, "enroll_other_speaker", np.concatenate([other, src])))
        gain = db(reg) - db(src)  # 把源音频放大到与注册音频同 RMS
        g = np.clip(src * 10 ** (gain / 20), -1, 1)
        jobs.append((c, f"src_gain_+{gain:.0f}dB(match_reg)", np.concatenate([reg, g])))
        jobs.append((c, "src_only_no_enroll_zero_pad_short", np.concatenate([reg[: SR // 2], np.zeros(int(2.5 * SR), np.float32), src])))

    def run(j):
        t, fr = ask(j[2])
        return {"case": j[0], "cond": j[1], "text": t, "finish": fr}

    with ThreadPoolExecutor(8) as ex:
        rows = list(ex.map(run, jobs))
    Path(HERE / "cases_out.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows))
    for r in rows:
        print(f"{r['case']:16s} {r['cond']:38s} {r['finish']:6s} {r['text'][:60]!r}")


if __name__ == "__main__":
    main()
