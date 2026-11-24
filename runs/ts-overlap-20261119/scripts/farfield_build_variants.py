"""Request WAVs for the far-field target-absent pack under counterfactual edits, built as speaker-enroll sends
them (enrollment first 5 s zero-padded to 5 s, then the segment, PCM16 16 kHz):
  base     as sent;
  gain     the whole request peak-normalised to -3 dBFS (an AGC on the phone channel);
  a4       the target enrollment replaced by an unrelated AISHELL-4 far-field speaker (11-16 pack);
  a3       the target enrollment replaced by an unrelated AISHELL-3 studio speaker.
usage: build_variants.py PACK FOREIGN_DIR OUT_DIR"""
import glob, hashlib, json, sys
from pathlib import Path
import numpy as np, soundfile as sf, torch, torchaudio
pack, foreign, out = map(Path, sys.argv[1:4])
REF = 5 * 16000
def a3_enroll(spk_dir):
    # three or more utterances of one AISHELL-3 test speaker, 44.1 kHz -> 16 kHz, until 5 s
    pieces, n = [], 0
    for p in sorted(glob.glob(str(spk_dir / "*.wav"))):
        w, sr = sf.read(p, dtype="float32")
        w = torchaudio.functional.resample(torch.from_numpy(w), sr, 16000).numpy()
        pieces.append(w); n += len(w)
        if n >= REF: break
    return (np.clip(np.concatenate(pieces)[:REF], -1, 1) * 32767).astype(np.int16)
A3 = Path("/workspace/data/datasets/aishell3/source/data_aishell3-fa5883f3/test/wav")
foreign_sets = {"a4": [sf.read(p, dtype="int16")[0][:REF] for p in sorted(glob.glob(str(foreign / "a4" / "*.wav")))],
                "a3": [a3_enroll(d) for d in sorted(A3.iterdir())[:4]]}
for k, v in foreign_sets.items():
    print(k, [f"{len(x)/16000:.1f}s {20*np.log10(np.sqrt(np.mean((x/32768.)**2))+1e-9):.1f}dBFS" for x in v])
cases = [json.loads(l) for l in open(pack / "cases.jsonl", encoding="utf-8")]
gains = []
for c in cases:
    enroll = sf.read(pack / c["target"]["enroll"], dtype="int16")[0]
    mix = sf.read(pack / c["mix"], dtype="int16")[0]
    h = int(hashlib.md5(c["id"].encode()).hexdigest(), 16) % 4
    for var, e in (("base", enroll), ("gain", enroll), ("a4", foreign_sets["a4"][h]), ("a3", foreign_sets["a3"][h])):
        padded = np.zeros(REF, np.int16); padded[: min(len(e), REF)] = e[:REF]
        x = np.concatenate([padded, mix])
        if var == "gain":
            g = 10 ** (-3 / 20) * 32767 / max(1, np.abs(x.astype(np.int32)).max()); gains.append(20 * np.log10(g))
            x = np.clip(np.round(x * g), -32768, 32767).astype(np.int16)
        (out / var).mkdir(parents=True, exist_ok=True)
        sf.write(out / var / f'{c["id"]}.wav', x, 16000, subtype="PCM_16")
print("cases", len(cases), "gain dB median/min/max", np.median(gains).round(1), np.min(gains).round(1), np.max(gains).round(1))
