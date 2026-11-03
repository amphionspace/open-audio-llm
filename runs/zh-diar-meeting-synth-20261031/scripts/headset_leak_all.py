"""Crosstalk level of every material headset track: power while only others talk over power while only the owner talks,
both above the track's power when nobody talks (annotated turns, as scripts/check_synth_training_data.py). One row per
track, for deciding which AISHELL-5 tracks are too open for unlabelled speech."""
import argparse
import gzip
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent))
from check_synth_training_data import RATE, headset_tracks  # noqa: E402


def levels(job):
    corpus, meeting, turns = job
    out = []
    for speaker, path in headset_tracks(corpus, meeting, {t[0] for t in turns}).items():
        x = sf.read(path, dtype="float32", always_2d=True)[0][:, 0].astype(np.float64)
        n = len(x)
        own, others = np.zeros(n, bool), np.zeros(n, np.int16)
        for s, a, b, *_ in turns:
            a, b = max(0, round(a * RATE)), min(n, round(b * RATE))
            if s == speaker:
                own[a:b] = True
            else:
                others[a:b] += 1
        regions = {"own": own & (others == 0), "others": ~own & (others > 0), "none": ~own & (others == 0)}
        row = {"corpus": corpus, "meeting": meeting, "speaker": speaker, **{f"{k}_s": round(float(m.sum()) / RATE, 1) for k, m in regions.items()}}
        if min(m.sum() for m in regions.values()) >= 5 * RATE:
            power = {k: float(np.mean(x[m] ** 2)) for k, m in regions.items()}
            row.update({f"{k}_db": round(10 * np.log10(max(v, 1e-12)), 1) for k, v in power.items()})
            if power["own"] > power["none"] and power["others"] > power["none"]:
                row["leak_db"] = round(float(10 * np.log10((power["others"] - power["none"]) / (power["own"] - power["none"]))), 1)
        out.append(row)
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pool", type=Path, required=True)
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--corpora", nargs="+", default=["aishell5", "alimeeting"])
    args = p.parse_args()
    templates = [json.loads(line) for line in gzip.open(args.pool / "templates.jsonl.gz", "rt")]
    jobs = [(t["corpus"], t["meeting"], t["turns"]) for t in templates if t["corpus"] in args.corpora]
    with ProcessPoolExecutor(args.workers) as pool, args.output.open("w") as sink:
        for done, rows in enumerate(pool.map(levels, jobs), 1):
            sink.writelines(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
            sink.flush()
            if done % 50 == 0:
                print(done, "of", len(jobs), flush=True)


if __name__ == "__main__":
    main()
