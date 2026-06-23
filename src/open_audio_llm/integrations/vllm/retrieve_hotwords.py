"""Hotword retrieval by CTC/coarse-ASR-guided text similarity.

Pure-text scoring used by the two-stage hotword evaluation pipeline
(``infer_retrieve.py``): given a coarse ASR hypothesis (phase 1) and a
candidate hotword pool, return the top-k most likely-uttered hotwords
to feed back into the hotword-aware ASR (phase 3).

The scoring is dispatched by language:

  - CJK (>= 30 % Chinese chars in the reference text):
      score = 2.0 * substring_bonus
            + 1.5 * pinyin_sliding_match
            + 0.5 * char_overlap_recall
            + 0.3 * char_bigram_jaccard
      Only hotwords with score > 1.0 are kept (low-scoring entries cause
      the LLM to hallucinate by splicing irrelevant fragments).

  - Alphabetic (English, etc.):
      score = 3.0 * substring_bonus
            + 2.0 * word_overlap (content words, stopwords filtered)
            + 1.5 * best_substring_sim (sliding SequenceMatcher)
            + 0.5 * lcs_ratio
            - length_penalty (3-char hotwords penalised most)
      Plus a global discrimination check: if top and 10th scores are too
      close (ratio < 1.30), return empty to prevent spray injection.

References:
  - GLCLAP + GRPO: arxiv.org/abs/2512.21828 (Section 2.2)
  - H-PRM:        arxiv.org/abs/2508.18295 (pre-retrieval module)

Algorithm follows ``amphion_ft/tools/retrieve_hotwords.py`` from the
sister project; kept self-contained so AmphionASR has no run-time
dependency on amphion_ft.

pypinyin is a **hard** dependency: the CJK scoring path leans on
pinyin sliding-window match (1.5x weight) to catch homophone errors
such as "富尔马诺夫" vs hyp "富尔马洛夫". A previous silent fallback
to ``text.lower()`` reduced retrieve recall on phase1-missed hotwords
from ~50% to 0.7% on commonvoice_zh_hotwords, so we fail loudly at
import time instead.
"""

import re
import unicodedata
from typing import List
from functools import lru_cache

import pypinyin  # noqa: F401  — hard dep, see module docstring
from rapidfuzz import fuzz as _rf_fuzz
from rapidfuzz.distance import LCSseq as _rf_lcs


# ---------------------------------------------------------------------------
# Text normalisation helpers
# ---------------------------------------------------------------------------

def _normalize(text: str) -> str:
    """Lowercase, strip accents, collapse whitespace."""
    text = unicodedata.normalize("NFD", text.lower())
    text = "".join(c for c in text if unicodedata.category(c) != "Mn")
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    return " ".join(text.split())


def _char_ngrams(text: str, n: int = 2) -> set:
    """Character n-gram set (spaces removed)."""
    t = _normalize(text).replace(" ", "")
    if len(t) < n:
        return {t} if t else set()
    return {t[i : i + n] for i in range(len(t) - n + 1)}


def _jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / (len(a) + len(b) - len(a & b))


def _substring_bonus(hotword: str, ctc_text: str) -> float:
    """Return 1.0 if the normalised hotword appears verbatim in CTC text."""
    hw = _normalize(hotword).replace(" ", "")
    ct = _normalize(ctc_text).replace(" ", "")
    return 1.0 if hw and hw in ct else 0.0


def _char_overlap_recall(hotword: str, ctc_text: str) -> float:
    """Fraction of hotword characters that appear in CTC text (recall)."""
    hw = _normalize(hotword).replace(" ", "")
    ct = _normalize(ctc_text).replace(" ", "")
    if not hw or not ct:
        return 0.0
    ct_chars = set(ct)
    hits = sum(1 for c in hw if c in ct_chars)
    return hits / len(hw)


# ---------------------------------------------------------------------------
# Pinyin (pronunciation) based similarity
# ---------------------------------------------------------------------------
_CHINESE_RE = re.compile(r'[\u4e00-\u9fff]')


@lru_cache(maxsize=10000)
def _to_pinyin(text: str) -> str:
    """Convert Chinese text to pinyin (no tones). Non-Chinese kept as-is."""
    result = []
    for seg in re.findall(r'[\u4e00-\u9fff]+|[^\u4e00-\u9fff]+', text):
        if _CHINESE_RE.search(seg):
            result.extend(pypinyin.lazy_pinyin(seg, style=pypinyin.Style.NORMAL))
        else:
            result.append(seg.lower().strip())
    return " ".join(w for w in result if w)


def _pinyin_similarity(hotword: str, ctc_text: str) -> float:
    """Sliding-window pronunciation match between a hotword and CTC text."""
    hw_py = _to_pinyin(hotword).split()
    ct_py = _to_pinyin(ctc_text).split()
    if not hw_py or not ct_py:
        return 0.0
    hw_len = len(hw_py)
    if hw_len > len(ct_py):
        matched = sum(1 for p in hw_py if p in ct_py)
        return matched / hw_len
    best_score = 0.0
    for start in range(len(ct_py) - hw_len + 1):
        window = ct_py[start : start + hw_len]
        matched = sum(1 for a, b in zip(hw_py, window) if a == b)
        score = matched / hw_len
        if score > best_score:
            best_score = score
            if best_score == 1.0:
                break
    return best_score


# ---------------------------------------------------------------------------
# English-specific similarity helpers
# ---------------------------------------------------------------------------

def _lcs_ratio(hotword: str, text: str) -> float:
    """LCS(hotword, text) / len(hotword).

    Backed by rapidfuzz's C-implemented ``LCSseq`` distance (Hyyrö 2004
    bit-parallel LCS). Drop-in replacement for the old pure-Python DP
    table ``_legacy_lcs_ratio``; on the EN retrieve hot loop this gives
    ~20-30x single-thread speed-up while preserving the exact LCS
    length definition (so the ``min_lcs_ratio=0.60`` threshold keeps
    its calibration).
    """
    hw = _normalize(hotword).replace(" ", "")
    txt = _normalize(text).replace(" ", "")
    if not hw or not txt:
        return 0.0
    return _rf_lcs.similarity(hw, txt) / len(hw)


def _best_substring_sim(hotword: str, text: str) -> float:
    """Score of the most similar contiguous *text* substring of len(hw).

    Backed by rapidfuzz's C-implemented ``fuzz.partial_ratio``, which
    finds the best Indel-similarity alignment of ``hw`` against any
    substring of ``txt``. Replaces the old nested-loop call to
    ``difflib.SequenceMatcher`` (kept as ``_legacy_best_substring_sim``
    for the offline benchmark).

    Semantic note: partial_ratio uses Indel distance (insertions /
    deletions only), whereas SequenceMatcher uses the Ratcliff /
    Obershelp longest-matching-block ratio. They are not bit-identical
    but agree on >95 % of top-K outputs in practice (see
    ``tmp/bench_retrieve.py`` for the Jaccard check). The downstream
    ``min_substring_sim=0.65`` threshold is unchanged because both
    metrics are normalised to [0, 1] and trigger on the same kind of
    near-match (e.g. ``"nagashino"`` vs ``"nagasheno"``).
    """
    hw = _normalize(hotword).replace(" ", "")
    txt = _normalize(text).replace(" ", "")
    if not hw or not txt:
        return 0.0
    return _rf_fuzz.partial_ratio(hw, txt) / 100.0


# ---------------------------------------------------------------------------
# Legacy (pre-rapidfuzz) implementations — retained ONLY for offline
# benchmarking via ``tmp/bench_retrieve.py``. Production code paths must
# use the canonical _best_substring_sim / _lcs_ratio above. The slow
# difflib import is local so the hot module load stays cheap.
# ---------------------------------------------------------------------------

def _legacy_lcs_length(s1: str, s2: str) -> int:
    m, n = len(s1), len(s2)
    prev = [0] * (n + 1)
    for i in range(1, m + 1):
        curr = [0] * (n + 1)
        for j in range(1, n + 1):
            if s1[i - 1] == s2[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(prev[j], curr[j - 1])
        prev = curr
    return prev[n]


def _legacy_lcs_ratio(hotword: str, text: str) -> float:
    hw = _normalize(hotword).replace(" ", "")
    txt = _normalize(text).replace(" ", "")
    if not hw or not txt:
        return 0.0
    return _legacy_lcs_length(hw, txt) / len(hw)


def _legacy_best_substring_sim(hotword: str, text: str) -> float:
    import difflib  # local: avoid paying the import cost in the hot module
    hw = _normalize(hotword).replace(" ", "")
    txt = _normalize(text).replace(" ", "")
    if not hw or not txt:
        return 0.0
    hw_len = len(hw)
    if hw_len > len(txt):
        return difflib.SequenceMatcher(None, hw, txt).ratio()
    best = 0.0
    for win in range(max(1, hw_len - 2), min(len(txt) + 1, hw_len + 3)):
        for start in range(len(txt) - win + 1):
            r = difflib.SequenceMatcher(None, hw, txt[start:start + win]).ratio()
            if r > best:
                best = r
                if best > 0.95:
                    return best
    return best


_STOPWORDS = frozenset({
    "a", "an", "the", "of", "in", "on", "at", "to", "for", "and",
    "or", "is", "was", "are", "were", "be", "been", "has", "had",
    "do", "did", "not", "no", "but", "by", "with", "from", "as",
    "it", "its", "he", "she", "we", "they", "his", "her", "my",
    "this", "that", "which", "who", "what", "how", "all", "each",
})


def _word_overlap(hotword: str, text: str) -> float:
    """Fraction of hotword's content words that appear in text."""
    hw_words = set(_normalize(hotword).split())
    txt_words = set(_normalize(text).split())
    hw_content = hw_words - _STOPWORDS
    if not hw_content:
        return 1.0 if hw_words and hw_words <= txt_words else 0.0
    return len(hw_content & txt_words) / len(hw_content)


def _hotword_char_len(hotword: str) -> int:
    """Effective character length of a hotword (spaces removed)."""
    return len(_normalize(hotword).replace(" ", ""))


# ---------------------------------------------------------------------------
# Language detection
# ---------------------------------------------------------------------------

def _is_cjk_dominant(text: str) -> bool:
    """True when >= 30 % of non-space characters are CJK."""
    chars = [c for c in text if not c.isspace()]
    if not chars:
        return True
    cjk = sum(1 for c in chars if '\u4e00' <= c <= '\u9fff')
    return cjk / len(chars) >= 0.3


# ---------------------------------------------------------------------------
# Unified hotword retrieval
# ---------------------------------------------------------------------------

def retrieve_hotwords(
    hotwords: List[str],
    ctc_text: str = "",
    *,
    no_hw_result: str = "",
    language: str = "auto",
    top_k: int = 16,
    always_keep: int = 0,
    min_substring_sim: float = 0.65,
    min_lcs_ratio: float = 0.60,
) -> List[str]:
    """Select the most relevant hotwords by comparing against CTC output.

    Automatically dispatches to CJK or alphabetic scoring based on the
    detected language of the reference text.

    **CJK scoring** (per hotword):

      1. **Substring bonus** (+2.0): hotword appears verbatim in CTC text.
      2. **Pinyin similarity** (0~1): sliding-window pronunciation match.
         Key discriminator for Chinese homophones::

           "颜妮" vs CTC "严妮" → pinyin "yan ni" vs "yan ni" → 1.0
           "田心妮" vs CTC "严妮" → pinyin "tian xin ni" vs "yan ni" → 0.33

      3. **Char-overlap recall** (0~1): fraction of hotword chars found
         in CTC text.
      4. **Bigram Jaccard** (0~1): char-bigram overlap.

      Final score = 2*substring + 1.5*pinyin_sim + 0.5*char_overlap
                    + 0.3*jaccard.
      Only hotwords with score > 1.0 are kept (low-scoring entries cause
      the LLM to hallucinate by splicing irrelevant fragments).

    **Alphabetic scoring** (per hotword):

      1. **Word overlap**: fraction of hotword content words found in text
         (stopwords excluded).
      2. **Substring similarity**: sliding-window SequenceMatcher to catch
         phonetic near-matches (e.g. ``"nagashino"`` ↔ ``"nagasheno"``).
      3. **LCS ratio**: longest common subsequence / hotword length.
      4. **Short-hotword penalty**: hotwords ≤ 4 chars require near-exact
         match to avoid false positives like ``"go"`` → ``"gho"``.
      5. **Global discrimination**: if top and 10th scores are too close
         (ratio < 1.30), return empty to prevent spray injection.

    Args:
        hotwords:         candidate hotword list.
        ctc_text:         CTC greedy-decode transcription (may be empty).
        no_hw_result:     optional no-hotword ASR result as extra reference
                          (used in alphabetic mode).
        language:         ``"zh"`` for CJK, ``"en"`` for alphabetic,
                          ``"auto"`` to detect from *ctc_text*.
        top_k:            maximum hotwords to return.
        always_keep:      number of leading hotwords to keep unconditionally
                          (CJK mode only).
        min_substring_sim: minimum substring similarity threshold
                          (alphabetic mode).
        min_lcs_ratio:    minimum LCS ratio threshold (alphabetic mode).

    Returns:
        Filtered list ordered by descending score (length ≤ *top_k*).
    """
    if top_k <= 0:
        return []
    hw = [w.strip() for w in hotwords if w and w.strip()]
    if not hw:
        return []

    # NOTE: previously short-circuited with ``if len(hw) <= top_k: return hw``
    # (returning the input order unsorted). Removed because callers now
    # pass ``top_k_max >> spec_K`` (see infer_retrieve.phase_retrieve) and
    # slice ``retrieved[:spec_K]`` themselves; an unsorted early return
    # would leak input order through the slice and break the by-score
    # invariant. For pool > top_k the behaviour is unchanged; for tiny
    # pools (pool ≤ top_k) we now still go through _retrieve_zh/_en and
    # return a score-sorted list (strictly better than input order).

    if language == "auto":
        ref = ctc_text or no_hw_result or " ".join(hw[:5])
        language = "zh" if _is_cjk_dominant(ref) else "en"

    if language == "zh":
        return _retrieve_zh(hw, ctc_text, top_k, always_keep)
    return _retrieve_en(
        hw, ctc_text, no_hw_result, top_k,
        min_substring_sim, min_lcs_ratio,
    )


# ---------------------------------------------------------------------------
# CJK scoring path
# ---------------------------------------------------------------------------

def _retrieve_zh(
    hw: List[str],
    ctc_text: str,
    top_k: int,
    always_keep: int,
) -> List[str]:
    if not ctc_text or not ctc_text.strip():
        return hw[:top_k]

    always_keep = min(always_keep, top_k)
    selected = list(hw[:always_keep])
    selected_set = set(selected)
    rest = hw[always_keep:]

    # Pre-compute CTC features ONCE (avoids 3N redundant _normalize calls)
    ct_norm = _normalize(ctc_text).replace(" ", "")
    ct_chars = set(ct_norm)
    ct_ngrams = _char_ngrams(ctc_text, n=2)

    ct_py = _to_pinyin(ctc_text).split()
    ct_py_set = set(ct_py)

    scored = []
    for w in rest:
        hw_norm = _normalize(w).replace(" ", "")

        # Substring bonus (inlined)
        substr = 1.0 if hw_norm and hw_norm in ct_norm else 0.0

        # Character overlap recall (inlined)
        if hw_norm:
            char_ovl = sum(1 for c in hw_norm if c in ct_chars) / len(hw_norm)
        else:
            char_ovl = 0.0

        # Bigram Jaccard
        hw_ng = _char_ngrams(w, n=2)
        jaccard = _jaccard(hw_ng, ct_ngrams)

        # Pinyin similarity with cheap pre-filter
        pinyin_sim = 0.0
        if ct_py:
            hw_py = _to_pinyin(w).split()
            if hw_py:
                # Cheap set-recall filter: skip sliding window when no
                # pinyin syllables overlap at all
                py_recall = sum(1 for p in hw_py if p in ct_py_set) / len(hw_py)
                if py_recall > 0:
                    hw_len = len(hw_py)
                    if hw_len > len(ct_py):
                        pinyin_sim = py_recall
                    else:
                        best = 0.0
                        for start in range(len(ct_py) - hw_len + 1):
                            matched = sum(
                                1 for a, b in zip(hw_py, ct_py[start:start + hw_len])
                                if a == b
                            )
                            score = matched / hw_len
                            if score > best:
                                best = score
                                if best == 1.0:
                                    break
                        pinyin_sim = best

        score = (substr * 2.0
                 + pinyin_sim * 1.5
                 + char_ovl * 0.5
                 + jaccard * 0.3)
        scored.append((score, w))

    scored.sort(key=lambda x: x[0], reverse=True)

    for score, w in scored:
        if len(selected) >= top_k:
            break
        # Require score > 1.0: low-scoring hotwords are largely irrelevant
        # and cause the LLM to hallucinate by splicing random hotword
        # fragments into the output (e.g. "朱圣祎若琪紫各庄村" instead of
        # "诸葛紫岐").  Do NOT backfill with near-zero entries.
        if score > 1.0 and w not in selected_set:
            selected.append(w)
            selected_set.add(w)

    return selected


# ---------------------------------------------------------------------------
# Alphabetic scoring path
# ---------------------------------------------------------------------------

def _retrieve_en(
    hw: List[str],
    ctc_text: str,
    no_hw_result: str,
    top_k: int,
    min_substring_sim: float,
    min_lcs_ratio: float,
) -> List[str]:
    if (not ctc_text or not ctc_text.strip()) and not no_hw_result:
        return hw[:top_k]

    combined = f"{ctc_text} {no_hw_result}".strip()

    ctc_text_py = ""
    no_hw_result_py = ""
    if _CHINESE_RE.search(ctc_text or ""):
        ctc_text_py = _to_pinyin(ctc_text)
    if _CHINESE_RE.search(no_hw_result or ""):
        no_hw_result_py = _to_pinyin(no_hw_result)

    scored: List[tuple] = []
    for w in hw:
        hw_clen = _hotword_char_len(w)
        w_ovl = max(_word_overlap(w, ctc_text), _word_overlap(w, no_hw_result))
        sub_sim = max(
            _best_substring_sim(w, ctc_text),
            _best_substring_sim(w, no_hw_result),
        )
        lcs = max(_lcs_ratio(w, ctc_text), _lcs_ratio(w, no_hw_result))
        substr = _substring_bonus(w, combined)

        pinyin_sim = 0.0
        hw_norm = _normalize(w).replace(" ", "")
        is_latin = bool(hw_norm) and all(c.isascii() for c in hw_norm)
        is_chinese = bool(_CHINESE_RE.search(w))

        if is_latin and (ctc_text_py or no_hw_result_py):
            if ctc_text_py:
                pinyin_sim = max(pinyin_sim, _best_substring_sim(w, ctc_text_py))
                pinyin_sim = max(pinyin_sim, _pinyin_similarity(w, ctc_text))
            if no_hw_result_py:
                pinyin_sim = max(pinyin_sim, _best_substring_sim(w, no_hw_result_py))
                pinyin_sim = max(pinyin_sim, _pinyin_similarity(w, no_hw_result))

        if is_chinese:
            hw_py = _to_pinyin(w)
            if hw_py:
                pinyin_sim = max(pinyin_sim, _best_substring_sim(hw_py, ctc_text))
                pinyin_sim = max(pinyin_sim, _best_substring_sim(hw_py, no_hw_result))
                if ctc_text_py:
                    pinyin_sim = max(pinyin_sim, _best_substring_sim(hw_py, ctc_text_py))
                if no_hw_result_py:
                    pinyin_sim = max(pinyin_sim, _best_substring_sim(hw_py, no_hw_result_py))

        if pinyin_sim > 0:
            sub_sim = max(sub_sim, pinyin_sim)
            lcs = max(lcs, pinyin_sim)

        if hw_clen <= 3:
            length_penalty = 0.35
        elif hw_clen <= 4:
            length_penalty = 0.20
        elif hw_clen <= 5:
            length_penalty = 0.10
        else:
            length_penalty = 0.0

        score = (
            substr * 3.0
            + w_ovl * 2.0
            + sub_sim * 1.5
            + lcs * 0.5
            - length_penalty
        )
        scored.append((score, sub_sim, lcs, w_ovl, substr, hw_clen, pinyin_sim, w))

    scored.sort(key=lambda x: x[0], reverse=True)

    # Global discrimination check: if top and 10th scores are too close,
    # no hotword truly matches — return empty to prevent spray injection.
    if len(scored) >= 10:
        top_score = scored[0][0]
        tenth_score = scored[9][0]
        if tenth_score > 0 and top_score / tenth_score < 1.30:
            return []

    # Multi-layer verification with length-aware thresholds
    verified = []
    for total_score, sub_sim, lcs, w_ovl, substr, hw_clen, pysim, w in scored:
        if hw_clen <= 4:
            if substr >= 1.0 or w_ovl >= 1.0:
                verified.append(w)
                if len(verified) >= top_k:
                    break
            continue
        elif hw_clen <= 5:
            if w_ovl > 0:
                effective_min_sim = min_substring_sim
                effective_min_lcs = min_lcs_ratio
            else:
                effective_min_sim = max(min_substring_sim, 0.75)
                effective_min_lcs = max(min_lcs_ratio, 0.70)
        else:
            effective_min_sim = min_substring_sim
            effective_min_lcs = min_lcs_ratio

        if w_ovl >= 0.8 or substr >= 1.0:
            verified.append(w)
            if len(verified) >= top_k:
                break
            continue

        if sub_sim >= effective_min_sim and lcs >= effective_min_lcs:
            hw_word_count = len(_normalize(w).split())
            if hw_word_count >= 2 and w_ovl < 0.01:
                if sub_sim < 0.80 or lcs < 0.75:
                    continue
            verified.append(w)
            if len(verified) >= top_k:
                break

    return verified


__all__ = ["retrieve_hotwords"]
