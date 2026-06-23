"""Online hotword sampling utilities."""

from __future__ import annotations

import random


def is_valid_hotword(value: str, max_len: int = 8, min_len: int = 2) -> bool:
    value = value.strip()
    return min_len <= len(value.replace(" ", "")) <= max_len


def build_hotword_pool(records, max_len: int = 8) -> list[str]:
    pool = set()
    for record in records:
        for hotword in getattr(record, "real_hotwords", []):
            if isinstance(hotword, str) and is_valid_hotword(hotword, max_len=max_len):
                pool.add(hotword)
    return sorted(pool)


def sample_hotwords(
    real_hotwords: list[str],
    hotword_pool: list[str],
    max_hotwords: int = 30,
    prompt_prob: float = 0.8,
    miss_prob: float = 0.0,
    rng: random.Random | None = None,
) -> str:
    rng = rng or random
    if rng.random() >= prompt_prob:
        return "N/A"
    real = [h for h in real_hotwords if rng.random() >= miss_prob]
    distractor_pool = [h for h in hotword_pool if h not in set(real_hotwords)]
    n_distractors = max(0, min(max_hotwords - len(real), len(distractor_pool)))
    if n_distractors:
        real.extend(rng.sample(distractor_pool, rng.randint(0, n_distractors)))
    return ",".join(real) if real else "N/A"
