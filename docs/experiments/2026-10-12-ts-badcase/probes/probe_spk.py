"""用 WeSpeaker ECAPA 看 pad 干扰声与测试人员注册音频的说话人相似度（CPU）。"""
import gzip, json, random, sys
sys.path.insert(0, sys.argv[1] + "/pylib")
from pathlib import Path
import numpy as np, onnxruntime as ort, soundfile as sf, torch, torchaudio.compliance.kaldi as K

ROOT = Path("/workspace/data/badcase/2026-10-10_amphionasr-target-speaker/cases")
sess = ort.InferenceSession("/workspace/model/wespeaker-ecapa-tdnn512-LM/voxceleb_ECAPA512_LM.onnx", providers=["CPUExecutionProvider"])
inp = sess.get_inputs()[0].name

def emb(w):
    f = K.fbank(torch.from_numpy(w).unsqueeze(0) * 32768, num_mel_bins=80, frame_length=25, frame_shift=10, dither=0, sample_frequency=16000, window_type="hamming", use_energy=False)
    f = f - f.mean(0, keepdim=True)
    e = sess.run(None, {inp: f.unsqueeze(0).numpy()})[0].reshape(-1)
    return e / np.linalg.norm(e)

def read(p):
    w, _ = sf.read(p, dtype="float32"); return w

reg = emb(read(ROOT / "case-02/registration-reference.wav"))
print("source           dur   cos(tester_registration, source)")
for c in ["case-01", "case-02", "case-03", "case-05", "case-06", "case-07", "case-04/part-01", "case-04/part-02"]:
    w = read(ROOT / c / "source-audio.wav")
    print(f"{c:16s} {len(w)/16000:5.2f}  {float(emb(w) @ reg):.3f}")

# 对照分布：AISHELLMix test_1spk 中同一说话人（正）与不同说话人（负）的 3 秒注册 vs 句子
B = Path("/workspace/data/datasets/aishellmix/versions/original-audio-record-20260910")
roots = json.load(open("/workspace/data/registry/roots.json"))
idx = {}
rs = [json.loads(l) for l in gzip.open(B / "test_1spk/test_1spk.jsonl.gz", "rt")]
random.Random(0).shuffle(rs); rs = rs[:300]
need = {s["ref"]["cut_id"] for r in rs for s in r["audio_slots"]}
with gzip.open(B / "audio_index/audio-index.jsonl.gz", "rt") as f:
    for l in f:
        r = json.loads(l)
        if r["cut_id"] in need: idx[r["cut_id"]] = roots[r["root_alias"]] + "/" + r["relative_path"]
E3, M = [], []
for r in rs:
    E3.append(emb(read(idx[r["audio_slots"][0]["ref"]["cut_id"]])[:48000])); M.append(emb(read(idx[r["audio_slots"][1]["ref"]["cut_id"]])))
spk = [r["labels"]["target_spk"] for r in rs]
pos = [float(E3[i] @ M[i]) for i in range(len(rs))]
neg = [float(E3[i] @ M[j]) for i in range(len(rs)) for j in range(len(rs)) if spk[i] != spk[j] and (i * 7 + j) % 11 == 0]
print("AISHELLMix 同人 cos: mean %.3f p5 %.3f | 异人 cos: mean %.3f p95 %.3f p99 %.3f" % (np.mean(pos), np.percentile(pos, 5), np.mean(neg), np.percentile(neg, 95), np.percentile(neg, 99)))
