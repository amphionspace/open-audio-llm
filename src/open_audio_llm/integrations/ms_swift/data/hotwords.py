"""Hotword pool construction and sampling utilities.

pypinyin is a **hard** dependency: ``to_pinyin`` / ``hw_to_pinyin_key``
/ :class:`PinyinIndex` produce the homophone hard-negative training
signal that teaches the model to disambiguate same-pronunciation
characters. A previous silent fallback to ``text.lower()`` would have
turned pinyin keys into raw Chinese strings — disabling the entire
homophone training pipeline without any log. We fail loudly at import
time instead.
"""

import random
import re
from functools import lru_cache
from typing import Dict, List, Optional, Tuple

import pypinyin as _pypinyin  # noqa: F401  — hard dep, see module docstring

# ---- CJK regex patterns ----
_CJK_RE = re.compile(r"[\u4e00-\u9fff]+")
_CJK_SPLIT_RE = re.compile(r"[\u4e00-\u9fff]+|[^\u4e00-\u9fff]+")


def is_char_based_script(hw):
    """Return True if *hw* is predominantly CJK or Thai (no word-level spacing)."""
    char_based = 0
    for ch in hw:
        cp = ord(ch)
        if (0x4E00 <= cp <= 0x9FFF       # CJK Unified Ideographs
                or 0x3400 <= cp <= 0x4DBF # CJK Extension A
                or 0xF900 <= cp <= 0xFAFF # CJK Compatibility
                or 0x0E00 <= cp <= 0x0E7F # Thai
                or 0x3000 <= cp <= 0x303F # CJK Symbols & Punctuation
                or 0xFF00 <= cp <= 0xFFEF # Fullwidth Forms
        ):
            char_based += 1
    alpha = sum(1 for ch in hw if ch.isalpha())
    return alpha > 0 and char_based > alpha / 2


def hotword_token_count(hw):
    """Count hotword length in semantic units.

    Character-based scripts (CJK, Thai): count characters (excluding spaces).
    Word-based scripts (Latin, etc.): count space-separated words.
    """
    if is_char_based_script(hw):
        return sum(1 for ch in hw if not ch.isspace())
    return len(hw.split())


def is_valid_hotword(hw, max_len, min_len=2, min_len_noncjk=1):
    """Filter hotwords by token count.

    *max_len* / *min_len* apply as token counts (chars for CJK/Thai,
    words for Latin/etc.).  *min_len_noncjk* overrides *min_len* for
    word-based scripts so single English words are accepted.
    """
    if not hw:
        return False
    count = hotword_token_count(hw)
    effective_min = min_len if is_char_based_script(hw) else min_len_noncjk
    return effective_min <= count <= max_len


def collect_hotword_pool(supervisions, max_hotword_len, min_hotword_len=2,
                         min_hotword_words=1):
    """Build global hotword pool from all supervisions with custom["hotwords"]."""
    pool = set()
    for sup in supervisions.values():
        custom = sup.get('custom', {}) or {}
        hw = custom.get('hotwords', [])
        if isinstance(hw, list):
            pool.update(
                h for h in hw if isinstance(h, str)
                and is_valid_hotword(h, max_hotword_len, min_hotword_len,
                                    min_hotword_words))
    return sorted(pool)


def build_hotwords_for_sample(
    real_hotwords,
    hotword_pool,
    max_hotwords,
    prompt_hotword_prob,
    hard_negatives=None,
    hard_neg_ratio=0.7,
    miss_prob=0.12,
    min_hotword_len=2,
    min_hotword_words=1,
    min_hotwords=1,
):
    """Construct hotword string for a single sample.

    Supports three distractor modes:
      1. hard_negatives provided: mix of hard negatives + random distractors
      2. max_hotwords > 0 only: random distractors
      3. max_hotwords == 0: real hotwords only

    When miss_prob > 0, each real hotword is independently dropped to
    simulate imperfect retrieval recall.

    When min_hotwords > 1, the total count of (real + distractor) after
    deduplication is guaranteed to be >= min_hotwords (up to pool size).
    max_hotwords acts as the upper bound on the same total.

    Returns comma-separated hotword string, or "N/A".
    """
    if random.random() >= prompt_hotword_prob:
        return "N/A"

    if miss_prob > 0 and real_hotwords:
        included_real = [hw for hw in real_hotwords if random.random() >= miss_prob]
    else:
        included_real = list(real_hotwords) if real_hotwords else []

    if max_hotwords <= 0:
        if not included_real:
            return "N/A"
        random.shuffle(included_real)
        return ",".join(included_real)

    if not hotword_pool and not hard_negatives:
        if not included_real:
            return "N/A"
        random.shuffle(included_real)
        return ",".join(included_real)

    # Determine distractor count so that total (real + distractor) falls in
    # [min_hotwords, max_hotwords].  real hotwords are fixed first, then we
    # fill distractors to reach the target range.
    n_real = len(included_real)
    pool_capacity = len(hotword_pool) + (len(hard_negatives) if hard_negatives else 0)
    lo = max(1, min_hotwords - n_real)
    hi = max(lo, min(max_hotwords - n_real, pool_capacity))
    n_distractor = random.randint(lo, hi)

    real_set = set(real_hotwords) if real_hotwords else set()

    n_hard = round(n_distractor * hard_neg_ratio) if hard_negatives else 0
    n_random = n_distractor - n_hard

    sampled_hard = []
    if n_hard > 0 and hard_negatives:
        avail_hard = [h for h in hard_negatives
                      if h not in real_set
                      and is_valid_hotword(h, float('inf'), min_hotword_len,
                                           min_hotword_words)]
        sampled_hard = random.sample(avail_hard, min(n_hard, len(avail_hard)))
        n_random += n_hard - len(sampled_hard)

    sampled_random = []
    if n_random > 0:
        exclude = real_set | set(sampled_hard)
        avail_random = [h for h in hotword_pool if h not in exclude]
        sampled_random = random.sample(avail_random, min(n_random, len(avail_random)))

    all_hw = set(included_real) | set(sampled_hard) | set(sampled_random)
    if not all_hw:
        return "N/A"

    hw_list = list(all_hw)
    random.shuffle(hw_list)
    return ",".join(hw_list)


def match_hotwords_in_text(prompted_hotwords, text):
    """Match hotwords against text with longest-first greedy masking.

    Longer hotwords are matched first; their positions are masked so that
    shorter substrings (e.g. "北京" inside "北京烤鸭") don't produce false
    positives.
    """
    sorted_hw = sorted(prompted_hotwords, key=len, reverse=True)
    masked = text
    detected = []
    for hw in sorted_hw:
        if hw and hw in masked:
            detected.append(hw)
            masked = masked.replace(hw, '\x00' * len(hw))
    return detected


# Backwards-compatible alias for older call-sites.
_match_hotwords_in_text = match_hotwords_in_text


# ---------------------------------------------------------------------------
# Character inverted index
# ---------------------------------------------------------------------------

def build_hotword_char_index(
    hotword_pool: List[str],
) -> Dict[str, List[int]]:
    """Build an inverted index mapping each character to hotword indices.

    Construction is O(sum of hotword lengths) and happens once at startup.
    At query time, candidate lookup is O(|text_chars|) instead of scanning
    the entire pool.
    """
    index: Dict[str, List[int]] = {}
    for i, hw in enumerate(hotword_pool):
        for ch in set(hw):
            if ch not in index:
                index[ch] = []
            index[ch].append(i)
    return index


# ---------------------------------------------------------------------------
# Pinyin helpers
# ---------------------------------------------------------------------------

def hw_to_pinyin_key(hw: str) -> Optional[Tuple[str, ...]]:
    """Convert a hotword to a tuple of tone-less pinyin syllables."""
    parts: List[str] = []
    for seg in _CJK_SPLIT_RE.findall(hw):
        if _CJK_RE.search(seg):
            parts.extend(_pypinyin.lazy_pinyin(seg, style=_pypinyin.Style.NORMAL))
        else:
            tok = seg.lower().strip()
            if tok:
                parts.append(tok)
    return tuple(parts) if parts else None


@lru_cache(maxsize=20000)
def to_pinyin(text: str) -> str:
    """Convert Chinese characters to tone-less pinyin; non-Chinese kept as-is."""
    result = []
    for seg in _CJK_SPLIT_RE.findall(text):
        if _CJK_RE.search(seg):
            result.extend(_pypinyin.lazy_pinyin(seg, style=_pypinyin.Style.NORMAL))
        else:
            result.append(seg.lower().strip())
    return " ".join(w for w in result if w)


# ---------------------------------------------------------------------------
# PinyinIndex — homophone hard-negative lookup
# ---------------------------------------------------------------------------

class PinyinIndex:
    """Pinyin inverted index for efficient homophone hard-negative lookup.

    Holds two sub-indices built once at startup:
      * ``exact`` -- pinyin-key -> list of hotword pool indices
      * ``syllable`` -- (syllable_count, position, syllable) -> list of
        pinyin keys, used for single-edit neighbour search
    """

    __slots__ = ("exact", "syllable", "pool")

    def __init__(
        self,
        exact: Dict[str, List[int]],
        syllable: Dict[tuple, List[str]],
        pool: List[str],
    ):
        self.exact = exact
        self.syllable = syllable
        self.pool = pool

    def retrieve(
        self,
        real_hotwords: List[str],
        top_k: int = 5,
    ) -> List[str]:
        """Return up to *top_k* homophone/near-homophone negatives."""
        real_set = set(real_hotwords)
        results: List[str] = []
        seen: set = set()

        for hw in real_hotwords:
            key_tuple = hw_to_pinyin_key(hw)
            if key_tuple is None:
                continue
            key = " ".join(key_tuple)

            for idx in self.exact.get(key, []):
                cand = self.pool[idx]
                if cand not in real_set and cand not in seen:
                    seen.add(cand)
                    results.append(cand)

            n = len(key_tuple)
            if n < 2:
                continue
            neighbour_keys: set = set()
            for pos in range(n):
                for nk in self.syllable.get((n, pos, key_tuple[pos]), []):
                    if nk == key:
                        continue
                    nk_parts = nk.split()
                    if sum(1 for a, b in zip(key_tuple, nk_parts) if a != b) == 1:
                        neighbour_keys.add(nk)
            for nk in neighbour_keys:
                for idx in self.exact.get(nk, []):
                    cand = self.pool[idx]
                    if cand not in real_set and cand not in seen:
                        seen.add(cand)
                        results.append(cand)

            if len(results) >= top_k:
                break

        return results[:top_k]


def build_hotword_pinyin_index(
    hotword_pool: List[str],
) -> PinyinIndex:
    """Build a :class:`PinyinIndex` from the hotword pool."""
    exact: Dict[str, List[int]] = {}
    syllable: Dict[tuple, List[str]] = {}

    for i, hw in enumerate(hotword_pool):
        key_tuple = hw_to_pinyin_key(hw)
        if key_tuple is None:
            continue
        key = " ".join(key_tuple)
        exact.setdefault(key, []).append(i)
        n = len(key_tuple)
        for pos, syl in enumerate(key_tuple):
            syllable.setdefault((n, pos, syl), []).append(key)

    return PinyinIndex(exact, syllable, hotword_pool)


# ---------------------------------------------------------------------------
# Online hard-negative retrieval
# ---------------------------------------------------------------------------

def retrieve_hard_negatives_online(
    text: str,
    hotword_pool: List[str],
    char_index: Dict[str, List[int]],
    exclude: set,
    top_k: int = 30,
    pinyin_index: Optional[PinyinIndex] = None,
    real_hotwords: Optional[List[str]] = None,
) -> List[str]:
    """Retrieve hard-negative hotwords on-the-fly.

    Combines two signals:
      1. **Character overlap + bigram Jaccard** -- fast candidate lookup
         via the inverted char->hotword index.
      2. **Homophone negatives** (optional) -- if *pinyin_index* and
         *real_hotwords* are provided, near-homophones of the real
         hotwords are appended so the model learns to disambiguate
         characters with the same pronunciation.
    """
    text_stripped = text.replace(" ", "")
    text_chars = set(text_stripped)
    if not text_chars:
        return []

    candidate_indices: set = set()
    for ch in text_chars:
        indices = char_index.get(ch)
        if indices:
            candidate_indices.update(indices)

    if not candidate_indices:
        return []

    text_bigrams: Optional[set] = None

    scored: List[Tuple[float, str]] = []
    for idx in candidate_indices:
        hw = hotword_pool[idx]
        if hw in exclude:
            continue

        hw_stripped = hw.replace(" ", "")
        hw_chars = set(hw_stripped)
        if not hw_chars:
            continue

        overlap = len(hw_chars & text_chars) / len(hw_chars)

        if text_bigrams is None:
            text_bigrams = (
                {text_stripped[j : j + 2] for j in range(len(text_stripped) - 1)}
                if len(text_stripped) >= 2
                else set()
            )
        if len(hw_stripped) >= 2:
            hw_bigrams = {hw_stripped[j : j + 2] for j in range(len(hw_stripped) - 1)}
            union = len(text_bigrams | hw_bigrams)
            jaccard = len(text_bigrams & hw_bigrams) / union if union else 0.0
        else:
            jaccard = 0.0

        score = overlap * 0.6 + jaccard * 0.4
        if score > 0:
            scored.append((score, hw))

    scored.sort(key=lambda x: x[0], reverse=True)
    char_results = [hw for _, hw in scored[:top_k]]

    if pinyin_index is not None and real_hotwords:
        homo_neg = pinyin_index.retrieve(
            real_hotwords, top_k=max(3, top_k // 3),
        )
        existing = set(char_results)
        for h in homo_neg:
            if h not in existing and h not in exclude:
                char_results.append(h)

    return char_results
