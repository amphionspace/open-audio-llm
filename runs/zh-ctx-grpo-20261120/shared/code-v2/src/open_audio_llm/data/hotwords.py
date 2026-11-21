"""Online hotword sampling utilities."""

from __future__ import annotations

import random
import re

_CJK = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]")


def is_valid_hotword(value: str, max_len: int = 8, min_len: int = 2) -> bool:
    value = value.strip()
    return min_len <= len(value.replace(" ", "")) <= max_len


def build_hotword_pool(records, max_len: int = 8) -> list[str]:
    pool = set()
    for record in records:
        for hotword in record.record.hotwords:
            if isinstance(hotword, str) and is_valid_hotword(hotword, max_len=max_len):
                pool.add(hotword)
    return sorted(pool)


def hotword_language(word: str, languages) -> str:
    """Native language label of a hotword from a source of ``languages`` (catalog codes): the source's language,
    or for a zh/en source the script of the word."""
    from .qwen3_asr import native_language

    names = {native_language(language) for language in languages}
    if len(names) == 1:
        return names.pop()
    return "Chinese" if _CJK.search(word) else "English"


def build_hotword_pools(records, max_len: int = 8) -> dict[str, list[str]]:
    """``build_hotword_pool`` split by the native language of each record."""
    pools: dict[str, set[str]] = {}
    for record in records:
        for hotword in record.record.hotwords:
            if isinstance(hotword, str) and is_valid_hotword(hotword, max_len=max_len):
                pools.setdefault(hotword_language(hotword, [record.record.language]), set()).add(hotword)
    return {language: sorted(words) for language, words in pools.items()}


def sample_hotwords(
    real_hotwords: list[str],
    hotword_pool: list[str],
    max_hotwords: int = 30,
    prompt_prob: float = 0.8,
    miss_prob: float = 0.0,
    rng: random.Random | None = None,
    confusables: list[str] = (),
    confusable_rng: random.Random | None = None,
) -> str:
    """Plain-ASR hotword list. ``confusables`` (sound-alikes of the real words, chosen by the caller with
    ``confusable_rng``) join the list as distractors and the list is reshuffled with ``confusable_rng``; the
    caller's ``rng`` stream is the same with or without them."""
    rng = rng or random
    if rng.random() >= prompt_prob:
        return "N/A"
    real = [h for h in real_hotwords if rng.random() >= miss_prob]
    excluded = set(real_hotwords)
    distractor_pool = [h for h in hotword_pool if h not in excluded]
    n_distractors = max(0, min(max_hotwords - len(real), len(distractor_pool)))
    if n_distractors:
        real.extend(rng.sample(distractor_pool, rng.randint(0, n_distractors)))
    if confusables:
        # sound-alikes go before the random distractors so a full list drops random words, not them
        kept = [h for h in real if h in excluded]
        extra = [h for h in dict.fromkeys(confusables) if h not in excluded and h not in real]
        others = [h for h in real if h not in excluded]
        real = (kept + extra + others)[: max(max_hotwords, len(kept))]
        (confusable_rng or random).shuffle(real)
    return ",".join(real) if real else "N/A"


def sample_ts_hotwords(
    target_hotwords: list[str],
    interferer_hotwords: list[str],
    hotword_pool: list[str],
    max_hotwords: int = 30,
    prompt_prob: float = 0.0,
    miss_prob: float = 0.0,
    rng: random.Random | None = None,
    confusables: list[str] = (),
) -> str:
    """Hotword list for one TS-ASR sample, or "N/A" for no list.

    ``target_hotwords`` are words the enrolled speaker says in this mixture (empty when the target is absent);
    the model must write them. ``interferer_hotwords`` are words other speakers say in the same mixture: they
    are the hard negatives that teach the model not to transcribe a listed word it hears from someone else.
    ``confusables`` (sound-alikes of the target's words that nobody says) are always added as hard negatives.
    Pool words fill the rest as words nobody says. The list is shuffled so position carries no label.
    """
    rng = rng or random
    if rng.random() >= prompt_prob:
        return "N/A"
    positives = list(dict.fromkeys(h for h in target_hotwords if rng.random() >= miss_prob))
    taken = set(target_hotwords)
    hard = [h for h in dict.fromkeys(interferer_hotwords) if h not in taken]
    hard = rng.sample(hard, rng.randint(0, len(hard))) if hard else []
    hard += [h for h in dict.fromkeys(confusables) if h not in taken and h not in hard]
    words = (positives + hard)[:max_hotwords]
    taken.update(words)
    room = max_hotwords - len(words)
    if room and hotword_pool:
        # Sample then filter: building "pool minus taken" would copy the whole pool for every sample.
        count = rng.randint(0, room)
        draws = rng.sample(hotword_pool, min(len(hotword_pool), count + len(taken)))
        words += [h for h in draws if h not in taken][:count]
    rng.shuffle(words)
    return ",".join(words) if words else "N/A"
