"""Conversations to leave out of the synthesis material, from the per-track levels of scripts/headset_leak_all.py.

A conversation goes when one of its tracks is not what its labels say: the owner hardly above the track's floor (the
owner's labelled words are barely on the track, but the renderer normalizes it to speech level anyway), or the others
nearly as loud as the owner. Whole conversations, because a partner left alone is no whole conversation.
The floor is the quieter of "only others talk" and "nobody talks": some tracks hold unlabelled sound in the latter."""
import argparse
import json
from pathlib import Path

# Heuristic, set by inspection of the distributions and not validated against training: see tasks/09-material-audit.
MAX_LEAK_DB = -10.0
MIN_OWNER_OVER_FLOOR_DB = 6.0


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--levels", type=Path, nargs="+", required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    tracks = [json.loads(line) for path in args.levels for line in path.open()]
    excluded = {}
    for t in tracks:
        if "own_db" not in t:
            continue
        over_floor = t["own_db"] - min(t["others_db"], t["none_db"])
        reasons = []
        if t.get("leak_db", -99.0) > MAX_LEAK_DB:
            reasons.append(f"leak {t['leak_db']}")
        if over_floor < MIN_OWNER_OVER_FLOOR_DB:
            reasons.append(f"owner over floor {over_floor:.1f}")
        if reasons:
            excluded.setdefault(t["meeting"], []).append({"corpus": t["corpus"], "speaker": t["speaker"], "reasons": reasons,
                                                          "own_s": t["own_s"]})
    result = {"max_leak_db": MAX_LEAK_DB, "min_owner_over_floor_db": MIN_OWNER_OVER_FLOOR_DB, "tracks_measured": len(tracks),
              "exclude_meetings": sorted(excluded), "why": dict(sorted(excluded.items()))}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=1) + "\n")
    print(len(excluded), "meetings of", len({t["meeting"] for t in tracks}))


if __name__ == "__main__":
    main()
