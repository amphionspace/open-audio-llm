"""ESC foreground mix policy metadata.

Training-time audio rendering can be provided by a Lhotse-backed hook. This
module owns the stochastic policy and records the sampled foreground metadata.
"""

from __future__ import annotations

from dataclasses import dataclass
import random


@dataclass
class ESCForegroundPolicy:
    p: float = 1.0
    snr_min: float = 0.0
    snr_max: float = 20.0
    overlap_prob: float = 0.8
    prepend_prob: float = 0.1
    append_prob: float = 0.1

    def sample(self, rng: random.Random) -> dict:
        if rng.random() > self.p:
            return {"enabled": False}
        total = self.overlap_prob + self.prepend_prob + self.append_prob
        x = rng.random() * total
        if x < self.overlap_prob:
            mode = "overlap"
        elif x < self.overlap_prob + self.prepend_prob:
            mode = "prepend"
        else:
            mode = "append"
        return {
            "enabled": True,
            "mode": mode,
            "snr": rng.uniform(self.snr_min, self.snr_max),
        }
