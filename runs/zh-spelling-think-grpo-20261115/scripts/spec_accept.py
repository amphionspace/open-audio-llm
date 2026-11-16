"""Acceptance length per suite and server from the /metrics snapshots run_mtp_check.sh saves before and after each suite.

MAL = 1 + accepted / drafts (tokens emitted per verification step); per-position rate = accepted at position i / drafts.
usage: spec_accept.py <configs/eval-dir> [...]
"""
import re
import sys
from pathlib import Path

LINE = re.compile(r'^vllm:spec_decode_(num_drafts|num_accepted_tokens|num_accepted_tokens_per_pos)_total\{([^}]*)\} (\S+)$')


def read(path):
    out = {}
    for line in Path(path).read_text().splitlines():
        m = LINE.match(line)
        if m:
            pos = re.search(r'position="(\d+)"', m.group(2))
            out[(m.group(1), int(pos.group(1)) if pos else None)] = float(m.group(3))
    return out


for d in map(Path, sys.argv[1:]):
    for after in sorted(d.glob("logs/metrics-after-*.txt")):
        key = after.name[len("metrics-after-"):-len(".txt")]
        a, b = read(after), read(after.with_name(f"metrics-before-{key}.txt"))
        delta = {k: v - b.get(k, 0.0) for k, v in a.items()}
        drafts = delta[("num_drafts", None)]
        pos = [delta[k] / drafts for k in sorted(k for k in delta if k[0] == "num_accepted_tokens_per_pos")]
        mal = 1 + delta[("num_accepted_tokens", None)] / drafts
        print(f"{d.name}\t{key}\tMAL {mal:.2f}\tper position {' / '.join(f'{p:.2f}' for p in pos)}\tdrafts {drafts:.0f}")
