"""Summarize a CUDA-only chrome trace from --torch_profile_steps.

Reports how much of the traced window the GPU had any kernel running, how much
NCCL time did not overlap computation, time per kernel family, and the largest
idle gaps with the kernels on either side (to locate host-side bubbles).
"""
import argparse
import gzip
import json
import re
from collections import defaultdict

FAMILIES = [
    ("nccl", re.compile(r"nccl", re.I)),
    ("flash_attn_varlen(text)", re.compile(r"flash_(fwd|bwd)|flash::", re.I)),
    ("mem_efficient_attn(audio)", re.compile(r"fmha|efficient_attention|mem_eff|attention_kernel", re.I)),
    ("liger_triton", re.compile(r"rms_norm|swiglu|rope|_liger|triton", re.I)),
    ("gemm", re.compile(r"gemm|cutlass|xmma|sm80_|sm90_|ampere_|cublas", re.I)),
    ("conv", re.compile(r"conv|cudnn|implicit", re.I)),
    ("optimizer", re.compile(r"multi_tensor|adam|l2norm|foreach", re.I)),
    ("softmax/ce", re.compile(r"softmax|cross_entropy|nll|log_softmax", re.I)),
    ("index/copy/cat", re.compile(r"index|gather|scatter|copy|cat|memcpy|memset|fill", re.I)),
    ("reduce", re.compile(r"reduce", re.I)),
    ("elementwise", re.compile(r"elementwise|vectorized|unrolled", re.I)),
]


def family(name):
    for label, pattern in FAMILIES:
        if pattern.search(name):
            return label
    return "other"


def union(intervals):
    total, end = 0.0, None
    merged = []
    for start, stop in sorted(intervals):
        if end is None or start > end:
            merged.append([start, stop])
            end = stop
        elif stop > end:
            merged[-1][1] = stop
            end = stop
    return merged, sum(b - a for a, b in merged)


def subtract(intervals, covered):
    """Length of ``intervals`` not covered by the merged ``covered`` list."""
    total, j = 0.0, 0
    for start, stop in sorted(intervals):
        cursor = start
        while j < len(covered) and covered[j][1] <= cursor:
            j += 1
        k = j
        while k < len(covered) and covered[k][0] < stop:
            if covered[k][0] > cursor:
                total += covered[k][0] - cursor
            cursor = max(cursor, covered[k][1])
            k += 1
        if cursor < stop:
            total += stop - cursor
    return total


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace")
    parser.add_argument("--updates", type=int, required=True, help="updates covered by the trace")
    parser.add_argument("--gaps", type=int, default=15)
    parser.add_argument("--output")
    args = parser.parse_args()
    opener = gzip.open if args.trace.endswith(".gz") else open
    with opener(args.trace, "rt") as stream:
        events = json.load(stream)["traceEvents"]
    gpu = [e for e in events if e.get("ph") == "X" and e.get("cat") in {"kernel", "gpu_memcpy", "gpu_memset"}]
    spans = [(e["ts"], e["ts"] + e["dur"], e["name"], e["args"].get("stream")) for e in gpu]
    window = (min(s[0] for s in spans), max(s[1] for s in spans))
    busy, busy_total = union([(a, b) for a, b, _, _ in spans])
    compute = [(a, b) for a, b, name, _ in spans if family(name) != "nccl"]
    compute_merged, compute_total = union(compute)
    nccl = [(a, b) for a, b, name, _ in spans if family(name) == "nccl"]
    by_family = defaultdict(float)
    for a, b, name, _ in spans:
        by_family[family(name)] += b - a
    gaps = []
    for (_, end), (start, _) in zip(busy, busy[1:]):
        gaps.append((start - end, end, start))
    gaps.sort(reverse=True)

    def neighbours(time):
        before = max((s for s in spans if s[1] <= time), key=lambda s: s[1])
        after = min((s for s in spans if s[0] >= time), key=lambda s: s[0])
        return before[2][:90], after[2][:90]

    per = lambda us: us / 1e6 / args.updates
    report = {
        "window_s_per_update": per(window[1] - window[0]),
        "gpu_busy_s_per_update": per(busy_total),
        "gpu_idle_fraction": 1 - busy_total / (window[1] - window[0]),
        "compute_busy_s_per_update": per(compute_total),
        "nccl_s_per_update": per(sum(b - a for a, b in nccl)),
        "nccl_not_overlapped_s_per_update": per(subtract(nccl, compute_merged)),
        "family_s_per_update": {k: per(v) for k, v in sorted(by_family.items(), key=lambda x: -x[1])},
        "idle_gaps_over_1ms_per_update": sum(g for g, _, _ in gaps if g > 1000) / 1e6 / args.updates,
        "largest_gaps": [
            {"ms": round(g / 1000, 2), "before": neighbours(end)[0], "after": neighbours(start)[1]}
            for g, end, start in gaps[:args.gaps]
        ],
    }
    print(json.dumps(report, indent=2))
    if args.output:
        with open(args.output, "w") as stream:
            json.dump(report, stream, indent=2)
            stream.write("\n")


if __name__ == "__main__":
    main()
