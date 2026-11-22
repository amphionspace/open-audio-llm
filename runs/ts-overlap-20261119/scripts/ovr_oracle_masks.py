"""Oracle separation from the stems: 'clean' = target + noise (interferers removed exactly), 'ibm' = ideal binary mask
(keep STFT bins where the target exceeds interferers + noise, LC 0 dB) on the mixture, 'irm' = ideal ratio mask
(sqrt(|T|^2 / (|T|^2 + |O+N|^2))) on the mixture. STFT 32 ms Hann, 8 ms hop. Also per-sample masking statistics on the
target's energy: share of target energy in bins where it is dominated (local SNR < 0 dB) and the glimpse proportion
(bins with local SNR > 3 dB among the bins holding 99% of the target energy). usage: oracle_masks.py SUBSET_DIR"""
import json, sys
from pathlib import Path
import numpy as np, soundfile as sf
from scipy.signal import stft, istft
sub = Path(sys.argv[1]); (sub / "oracle").mkdir(exist_ok=True)
SR, NF, HOP = 16000, 512, 128
stats = {}
for line in open(sub / "manifest.jsonl", encoding="utf-8"):
    r = json.loads(line); k = r["key"]
    if not (sub / "stems" / f"{k}.target.wav").exists(): continue
    T, O, N = (sf.read(sub / "stems" / f"{k}.{n}.wav", dtype="float32")[0] for n in ("target", "other", "noise"))
    M = T + O + N
    f = lambda x: stft(x, SR, window="hann", nperseg=NF, noverlap=NF - HOP)[2]
    St, Sr, Sm = f(T), f(O + N), f(M)
    pt, pr = np.abs(St) ** 2, np.abs(Sr) ** 2
    out = {"clean": T + N,
           "ibm": istft(Sm * (pt > pr), SR, window="hann", nperseg=NF, noverlap=NF - HOP)[1][: len(M)],
           "irm": istft(Sm * np.sqrt(pt / (pt + pr + 1e-12)), SR, window="hann", nperseg=NF, noverlap=NF - HOP)[1][: len(M)]}
    for name, w in out.items():
        sf.write(sub / "oracle" / f"{k}.{name}.wav", w.astype(np.float32), SR, subtype="FLOAT")
    flat = np.sort(pt.ravel())[::-1]; cut = flat[np.searchsorted(np.cumsum(flat), 0.99 * flat.sum())]
    hold = pt >= cut
    snr = 10 * np.log10((pt + 1e-12) / (pr + 1e-12))
    stats[r["id"]] = {"dominated_energy_share": float(pt[pt < pr].sum() / pt.sum()),
                      "glimpse_proportion": float((snr[hold] > 3).mean()),
                      "sir_db": float(10 * np.log10((T ** 2).sum() / ((O ** 2).sum() + 1e-12)))}
json.dump(stats, open(sub / "oracle-stats.json", "w"), indent=0)
print(len(stats), "samples")
