"""Ways to fill a short enrollment cut to the 5 s the server splits at (tasks/24 attempt 002). The cut is the first L s
after the generator's trim_leading_silence. The floor is the 10th-percentile 20 ms frame RMS of the whole stored
enrollment (its own noise floor), as Gaussian noise.
  pad        cut + digital zeros (what the server's fixed split does today)
  noisepad   cut + floor noise
  tile       the cut repeated end to end
  tilexf     the cut repeated with a 30 ms linear cross-fade at each joint
  tilegap    the cut repeated with 0.4 s of floor noise between repeats (like the pauses of a multi-sentence enrollment)"""
import numpy as np
SR, N5 = 16000, 5 * 16000


def floor_rms(e):
    frames = e[: len(e) // 320 * 320].reshape(-1, 320)
    r = np.sqrt((frames ** 2).mean(1) + 1e-12)
    return max(float(np.percentile(r, 10)), 1e-5)


def fills(e, cut, seed=0):
    rng = np.random.default_rng(seed); fl = floor_rms(e)
    noise = lambda n: (rng.standard_normal(n) * fl).astype(np.float32)
    out = {"pad": np.pad(cut, (0, N5 - len(cut))), "noisepad": np.concatenate([cut, noise(N5 - len(cut))]),
           "tile": np.tile(cut, -(-N5 // len(cut)))[:N5]}
    xf = int(0.03 * SR); ramp = np.linspace(0, 1, xf, dtype=np.float32); y = cut.copy()
    while len(y) < N5:
        y = np.concatenate([y[:-xf], y[-xf:] * (1 - ramp) + cut[:xf] * ramp, cut[xf:]])
    out["tilexf"] = y[:N5]
    gap = int(0.4 * SR); y = cut.copy()
    while len(y) < N5:
        y = np.concatenate([y, noise(gap), cut])
    out["tilegap"] = y[:N5]
    return {k: v.astype(np.float32) for k, v in out.items()}
