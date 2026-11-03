"""Conversation statistics of training sources vs the real meeting dev windows, from the references alone.

For each source (dataset@version:split) a fixed random sample of records is parsed from the SOT target
("[Sx][start-end] text" per turn). Per window: speakers, speech coverage, overlapped share of speech time and of
reference characters (turns overlapping another speaker, the turn_diagnostics definition), turn length, speaking rate,
backchannels (turns of <= 2 units) per minute, fillers per 100 units, gaps at speaker changes."""
import argparse
import gzip
import json
import random
import re
from pathlib import Path

import numpy as np
from audio_data_contract import load_catalog, load_roots, resolve_split

TURN = re.compile(r'^\[(S\d+)\]\[(\d+(?:\.\d+)?)-(\d+(?:\.\d+)?)\]\s*(.*)$')
UNIT = re.compile(r'[一-鿿A-Za-z0-9]')
CJK = re.compile(r'[一-鿿]')
FILLERS = set('嗯啊呃哦诶哎唉嗷噢呀吧')


def window(target, duration):
    turns = []
    for line in (target or '').splitlines():
        m = TURN.match(line.strip())
        if m:
            text = ''.join(UNIT.findall(m[4]))
            turns.append((m[1], float(m[2]), float(m[3]), text))
    if not turns:
        return None
    turns.sort(key=lambda t: t[1])
    step = 0.01
    n = int(np.ceil(max(duration, max(t[2] for t in turns)) / step)) + 1
    count = np.zeros(n, dtype=np.int16)
    for _, s, e, _ in turns:
        count[int(s / step):int(e / step)] += 1
    speech, overlap = (count >= 1).sum() * step, (count >= 2).sum() * step
    units = sum(len(t[3]) for t in turns)
    ov_units = sum(len(t[3]) for t in turns
                   if any(o[0] != t[0] and o[1] < t[2] and t[1] < o[2] for o in turns))
    gaps = [b[1] - a[2] for a, b in zip(turns, turns[1:]) if a[0] != b[0]]
    cjk = sum(len(CJK.findall(t[3])) for t in turns)
    fill = sum(sum(ch in FILLERS for ch in t[3]) for t in turns)
    return {'speakers': len({t[0] for t in turns}), 'duration': duration, 'speech_cov': speech / max(duration, 1e-6),
            'overlap_time': overlap / max(speech, 1e-6), 'overlap_chars': ov_units / max(units, 1),
            'turn_s': [t[2] - t[1] for t in turns], 'rate': units / max(sum(t[2] - t[1] for t in turns), 1e-6),
            'backchannel_per_min': sum(len(t[3]) <= 2 for t in turns) / max(duration / 60, 1e-6),
            'filler_per_100': 100 * fill / max(cjk, 1), 'gaps': gaps, 'units': units}


def target_of(record):
    """Timed SOT target; the read-speech mixtures keep times only in metadata.segments (added at load time)."""
    target = record.get('target') or ''
    if TURN.match(target.split('\n', 1)[0].strip()):
        return target
    segs = record.get('metadata', {}).get('segments') or []
    return '\n'.join(f"[{s['speaker']}][{s['start']:.2f}-{s['start'] + s['duration']:.2f}] {s['text']}" for s in segs)


def summarize(rows):
    rows = [r for r in rows if r]
    turn_s = np.concatenate([r['turn_s'] for r in rows])
    gaps = np.concatenate([r['gaps'] for r in rows if r['gaps']] or [np.array([])])
    w = np.array([r['units'] for r in rows], dtype=float)
    mean = lambda k: float(np.mean([r[k] for r in rows]))  # noqa: E731
    return {'windows': len(rows), 'duration_s': mean('duration'), 'speakers': mean('speakers'),
            'speech_cov': mean('speech_cov'), 'overlap_time': mean('overlap_time'),
            'overlap_chars': float(np.average([r['overlap_chars'] for r in rows], weights=w)),
            'turn_s_median': float(np.median(turn_s)), 'turn_s_p90': float(np.percentile(turn_s, 90)),
            'units_per_s': mean('rate'), 'backchannel_per_min': mean('backchannel_per_min'),
            'filler_per_100': mean('filler_per_100'),
            'gap_median_s': float(np.median(gaps)) if len(gaps) else None,
            'gap_negative_share': float((gaps < 0).mean()) if len(gaps) else None}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--catalog', type=Path, required=True)
    p.add_argument('--roots', type=Path, required=True)
    p.add_argument('--source', action='append', required=True, help='name=dataset@version:split[,split...]')
    p.add_argument('--sample', type=int, default=400)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    catalog, roots = load_catalog(args.catalog), load_roots(args.roots)
    out = {}
    for item in args.source:
        name, ref = item.split('=', 1)
        did, rest = ref.split('@', 1)
        ver, splits = rest.split(':', 1)
        records = []
        for split in splits.split(','):
            for path in resolve_split(catalog, did, ver, split, 'records', roots):
                with gzip.open(path, 'rt') as f:
                    records += [json.loads(line) for line in f]
        random.Random(20261031).shuffle(records)
        rows = [window(target_of(r), r['audio_slots'][0]['ref']['duration']) for r in records[:args.sample]]
        if not any(rows):
            print(name, 'no timed turns', flush=True)
            continue
        out[name] = summarize(rows)
        print(name, json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in out[name].items()},
                               ensure_ascii=False), flush=True)
    args.output.write_text(json.dumps(out, indent=2, ensure_ascii=False) + '\n')


if __name__ == '__main__':
    main()
