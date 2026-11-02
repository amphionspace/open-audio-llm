"""Caption cost of the confidence gate on real speech, from partial_caption.py rows (counts only, no transcripts).

Per snapshot, without the gate = has text or rejected; with the gate = has text. Delay is the first snapshot with
text, gate on minus gate off. Flicker counts text -> no text -> text transitions. Utterances where the gate rejects
every snapshot that has text never show a partial with the gate.

    python caption_report.py rows.jsonl [rows.jsonl ...] [--json OUT]
"""
import json, statistics, sys
from collections import defaultdict


def first(flags):
    return next((end for end, on in flags if on), None)


def holes(flags):
    seq = [on for _, on in flags]
    n, seen, gap = 0, False, False
    for on in seq:
        if on:
            n += gap
            seen, gap = True, False
        elif seen:
            gap = True
    return n


def q(values, p):
    values = sorted(values)
    return values[min(len(values) - 1, int(p * len(values)))] if values else None


def summarize(rows):
    delays, lost, flick_on, flick_off, snaps, rej = [], 0, 0, 0, 0, 0
    with_text_off = 0
    for row in rows:
        off = [(c[0], bool(c[1] or c[2])) for c in row["cells"]]
        on = [(c[0], bool(c[1])) for c in row["cells"]]
        snaps += len(row["cells"])
        rej += sum(c[2] for c in row["cells"])
        a, b = first(off), first(on)
        if a is None:
            continue
        with_text_off += 1
        flick_off += holes(off) > 0
        flick_on += holes(on) > 0
        if b is None:
            lost += 1
        else:
            delays.append(round(b - a, 2))
    delayed = [d for d in delays if d > 0]
    return {"utts": len(rows), "utts_with_partial_off": with_text_off, "snapshots": snaps, "snapshots_rejected": rej,
            "snapshot_reject_rate": round(rej / snaps, 4) if snaps else None,
            "utts_delayed": len(delayed), "utts_delayed_rate": round(len(delayed) / with_text_off, 4) if with_text_off else None,
            "utts_no_partial_on": lost, "utts_no_partial_on_rate": round(lost / with_text_off, 4) if with_text_off else None,
            "delay_median_all": statistics.median(delays) if delays else None, "delay_p90_all": q(delays, .9), "delay_p99_all": q(delays, .99),
            "delay_median_delayed": statistics.median(delayed) if delayed else None, "delay_p90_delayed": q(delayed, .9),
            "utts_flicker_off": flick_off, "utts_flicker_on": flick_on,
            "flicker_off_rate": round(flick_off / with_text_off, 4) if with_text_off else None,
            "flicker_on_rate": round(flick_on / with_text_off, 4) if with_text_off else None}


def main():
    argv = sys.argv[1:]
    out = None
    if "--json" in argv:
        i = argv.index("--json")
        out = argv[i + 1]
        del argv[i:i + 2]
    groups = defaultdict(list)
    for path in argv:
        for line in open(path):
            row = json.loads(line)
            groups[row["set"]].append(row)
            groups[f"{row['set']}/{row['source']}"].append(row)
    report = {k: summarize(v) for k, v in sorted(groups.items())}
    for k, v in report.items():
        print(k, json.dumps(v, ensure_ascii=False))
    if out:
        json.dump(report, open(out, "w"), indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
