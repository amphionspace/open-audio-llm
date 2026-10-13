"""底座 Qwen3-ASR-1.7B（官方 vLLM 方式，普通 ASR，没有注册音频）在评测集上的幻觉对照。"""
import base64, gzip, io, json, sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np, requests, soundfile as sf
sys.path.insert(0, str(Path(__file__).parent))
import probe_ts as P

ROOT = Path("/workspace/data/datasets/ts_asr_benchmark"); VER = "v1-20261013"
SPLITS = ["nonspeech-noise", "nonspeech-music", "nonspeech-traffic", "nonspeech-quiet", "nonspeech-short", "nonspeech-long", "dyn-solo"]
N = 120
idx = {}
for l in gzip.open(ROOT/"versions"/VER/"audio-index.jsonl.gz","rt"):
    r = json.loads(l); idx[r["cut_id"]] = ROOT/r["relative_path"]
def ask(url, model, wav, system):
    buf = io.BytesIO(); sf.write(buf, wav, 16000, format="WAV", subtype="PCM_16")
    r = requests.post(url + "/v1/chat/completions", timeout=300, json={"model": model, "temperature": 0, "max_tokens": 512,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": [{"type": "input_audio", "input_audio": {"data": base64.b64encode(buf.getvalue()).decode(), "format": "wav"}}]}]})
    r.raise_for_status(); raw = r.json()["choices"][0]["message"]["content"] or ""
    return (raw.partition("<asr_text>")[2] if "<asr_text>" in raw else raw).strip()
ft = {json.loads(l)["id"]: json.loads(l)["prediction"] for l in open("tsb/run2/predictions.jsonl")}
for sp in SPLITS:
    recs = [json.loads(l) for l in list(gzip.open(ROOT/"versions"/VER/"records"/f"{sp}.jsonl.gz","rt"))[:N]]
    def run(r):
        mix = idx[r["audio_slots"][1]["ref"]["cut_id"]]
        w, _ = sf.read(mix, dtype="float32")
        return ask("http://127.0.0.1:8012", "base", w, "")
    with ThreadPoolExecutor(16) as ex: base = list(ex.map(run, recs))
    if sp == "dyn-solo":
        cer = lambda hs: sum(P.edit(P.norm(h), P.norm(r["target"])) for h, r in zip(hs, recs)) / sum(len(P.norm(r["target"])) for r in recs)
        print(f"{sp:18s} 底座 CER {100*cer(base):5.1f}%   微调(34479, 带注册) CER {100*cer([ft[r['id']] for r in recs]):5.1f}%  (n={len(recs)})")
    else:
        fb = np.mean([bool(P.norm(h)) for h in base]); ff = np.mean([bool(P.norm(ft[r['id']])) for r in recs])
        print(f"{sp:18s} 底座输出文字 {100*fb:5.1f}%   微调(34479) 输出文字 {100*ff:5.1f}%  (n={len(recs)})")
