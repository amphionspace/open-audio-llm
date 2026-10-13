"""第二批 badcase（case-08~12）回放与对照：同一源音频配不同注册音频。只读，向本地 vLLM 发请求。"""
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
import probe_cases as C

CASES = ["case-08", "case-09", "case-10", "case-11", "case-12"]
SR = C.SR
reg2 = C.read(C.ROOT / "case-08" / "registration-reference.wav")   # 第二批的注册音频
reg1 = C.read(C.ROOT / "case-02" / "registration-reference.wav")   # 第一批的注册音频
rows = []
for c in CASES:
    src = C.read(C.ROOT / c / "source-audio.wav")
    rec = lambda e: C.ask(np.concatenate([e, src]).astype(np.float32), max_tokens=256)
    for name, e in (("as_is(第二批注册)", reg2), ("第一批注册音频", reg1), ("注册换成静音", np.zeros(3 * SR, np.float32))):
        text, fin = rec(e)
        rows.append({"case": c, "cond": name, "text": text, "finish": fin, "src_s": round(len(src) / SR, 2)})
        print(f"{c} {len(src)/SR:5.2f}s {name:16s} {fin:6s} {text!r}")
Path(__file__).with_name("batch2_out.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows))
