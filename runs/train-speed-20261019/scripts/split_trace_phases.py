"""Split GPU idle time in a --torch_profile_steps trace by training phase.

Phases are located by marker kernels (audio encoder: mem-efficient attention and
the conv front end; text decoder: FlashAttention fwd/bwd and Liger backward;
NCCL; fused Adam). Unmarked kernels inherit the current phase, so boundaries are
approximate. Usage: split_trace_phases.py TRACE.json.gz UPDATES
"""
import gzip, json, sys, re
from collections import defaultdict
ev = json.load(gzip.open(sys.argv[1], "rt"))["traceEvents"]
k = sorted((e["ts"], e["ts"]+e["dur"], e["name"]) for e in ev if e.get("ph")=="X" and e.get("cat") in {"kernel","gpu_memcpy","gpu_memset"})
rules = [("comm", r"nccl"), ("optimizer", r"multi_tensor|adam"), ("audio_bwd", r"fmha_cutlassB"),
         ("audio_fwd", r"fmha_cutlassF|cudnn|implicit_convolve|implicit_gemm"),
         ("text_bwd", r"flash_bwd|_backward_kernel"), ("text_fwd", r"flash_fwd")]
state = "pre"; span = defaultdict(float); idle = defaultdict(float); small = defaultdict(lambda: [0, 0.0]); busy = defaultdict(float)
end = k[0][1]; seg_start = k[0][0]
for a, b, n in k:
    new = next((s for s, p in rules if re.search(p, n)), state)
    # comm/optimizer stay sticky only until a compute marker appears
    if new != state:
        span[state] += a - seg_start; seg_start = a; state = new
    gap = a - end
    if gap > 0:
        idle[state] += gap
        if 50 <= gap <= 200: small[state][0] += 1; small[state][1] += gap
    busy[state] += b - a
    end = max(end, b)
span[state] += end - seg_start
U = int(sys.argv[2])
tot = sum(span.values())
for st in ["audio_fwd", "text_fwd", "text_bwd", "audio_bwd", "comm", "optimizer", "pre"]:
    if st in span:
        print(f"{st:10s} span {span[st]/1e3/U:6.1f} ms/upd  idle {idle[st]/1e3/U:5.1f} ms  small gaps {small[st][0]/U:4.0f} = {small[st][1]/1e3/U:4.1f} ms")
print("total", round(tot/1e3/U, 1))
