"""Stem renderer for v12 scenes (see stems.py)."""
import numpy as np


def stems(plan, build, mixer):
    SR = mixer.SAMPLE_RATE
    total = int(plan["total"] * SR)
    parts = {"target": np.zeros(total + int(0.4 * SR), np.float32), "other": np.zeros(total + int(0.4 * SR), np.float32)}
    for track in plan["tracks"]:
        rir = build.read_audio(track.rir) if track.rir else None
        dest = parts["target" if track.role == "target" else "other"]
        for utt, start in track.items:
            w = mixer.normalize_rms(build.read_audio(utt.path))
            if rir is not None:
                w = mixer.apply_rir(w, rir)
            w = w * 10.0 ** (track.gain_db / 20.0)
            begin = int(start * SR); end = min(begin + len(w), total)
            if end <= begin: continue
            truncated = end - begin < len(w); w = w[: end - begin].copy()
            if truncated:
                fade = min(len(w), int(mixer.FADE_SECONDS * SR)); w[-fade:] *= np.linspace(1.0, 0.0, fade, dtype=np.float32)
            dest[begin:end] += w
    t, o = parts["target"][:total], parts["other"][:total]
    speech = t + o; noise = np.zeros(total, np.float32)
    spec = plan.get("noise")
    if spec:
        seg = mixer._fit(build._cached_noise(spec["file"]) if spec["kind"] != "quiet" else
                         np.random.default_rng(0).standard_normal(SR * 30).astype(np.float32), total, spec["offset"])
        noise = seg * (mixer.rms(speech) / mixer.rms(seg) * 10.0 ** (-spec["snr_db"] / 20.0))
    mix = speech + noise; peak = np.abs(mix).max(); k = (0.5 / peak) * 10.0 ** (plan.get("mix_gain_db", 0.0) / 20.0)
    floor = (plan.get("floor") or {}).get("mix")
    fl = mixer.floor_noise(total, floor, lambda kind, path: build._cached_noise(path)) if floor else np.zeros(total, np.float32)
    return (t * k).astype(np.float32), (o * k).astype(np.float32), (noise * k + fl).astype(np.float32)

