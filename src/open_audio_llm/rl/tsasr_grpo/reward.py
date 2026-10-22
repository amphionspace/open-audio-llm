"""TS-ASR reward.

- Target absent: 1 for an empty output, 0 for any text (false alarm).
- Target present: ``1 - min(error_rate, ERROR_RATE_CAP)``, so it ranges from 1
  down to -1. Output dominated by another speaker has an error rate above 1
  and scores below a miss; a plain ``max(0, 1 - CER)`` would score both 0.
- Target present but nothing output: ``miss_reward``. With 0 the miss rate
  rose with every step (8.97% -> 14.77% on ts-asr-benchmark v9 after 240
  steps). -0.5 only slows that drift: on the held-out set both runs trace the
  same false-alarm/miss curve (13.2%/11.7% at run-1 step 120, 12.9%/11.9% at
  run-2 step 180), so the miss penalty alone does not fix the trade-off.
"""

from __future__ import annotations

from .text import errors, units

# Error rates above 2 (more inserted units than twice the reference) all score -1.
ERROR_RATE_CAP = 2.0
DEFAULT_MISS_REWARD = -0.5


def reward(target: str, hypothesis: str, language: str, miss_reward: float = DEFAULT_MISS_REWARD) -> float:
    dist, n_ref = errors(target, hypothesis, language)
    if n_ref == 0:
        return 1.0 if dist == 0 else 0.0
    if not units(hypothesis, language):
        return miss_reward
    return 1.0 - min(dist / n_ref, ERROR_RATE_CAP)
