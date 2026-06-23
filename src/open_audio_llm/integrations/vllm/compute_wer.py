#!/usr/bin/env python3
"""
Shared WER/CER computation utilities for batch inference scripts.

Supports mixed Chinese/English text:
  - Chinese characters are treated as individual tokens (for CER-style evaluation)
  - English words are treated as whole tokens (for WER-style evaluation)
  - Text is normalized using Whisper-standard normalizers before evaluation:
      * English:     EnglishTextNormalizer  (numbers, contractions, spellings, …)
      * Chinese:     BasicTextNormalizer + OpenCC Traditional→Simplified conversion
                     + cn2an Chinese→Arabic number normalization
      * Other:       BasicTextNormalizer    (Unicode symbol/punctuation removal)
"""

import logging
import re
import warnings
from collections import Counter, defaultdict
from typing import Callable, Dict, List, Optional, Tuple

import cn2an
from opencc import OpenCC
from whisper_normalizer.basic import BasicTextNormalizer
from whisper_normalizer.english import EnglishTextNormalizer

_LOGGER = logging.getLogger(__name__)

# ─── Whisper-standard normalizers ────────────────────────────────────────────

_en_normalizer = EnglishTextNormalizer()
_basic_normalizer = BasicTextNormalizer()
_t2s = OpenCC("t2s")  # Traditional → Simplified Chinese


def _normalize_chinese_numbers(text: str) -> str:
    """Convert Chinese numerals to Arabic for Chinese ASR eval.

    Uses ``cn2an.transform(text, "cn2an")`` to canonicalize all number
    representations into Arabic form, consistent with the English normalizer
    which also normalizes to Arabic numerals.

    This handles different Chinese representations of the same number:

    Examples::

        "二零二四年"      → "2024年"
        "二千零二十四年"  → "2024年"   (same number, different Chinese form)
        "第一名"          → "第1名"
        "价格是一百元"    → "价格是100元"
        "三点一四"        → "3.14"

    Falls back to the original text if ``cn2an`` fails on any input.

    NOTE: cn2an emits a UserWarning ("不符合格式的数据：百/千/万/...") on
    isolated CJK number characters before raising; left unmuted these
    warnings flood stderr (one per malformed token, per sample) and
    overwrite the tqdm progress bar. The warning carries no information
    we don't already recover from via the ``except`` below, so we silence
    it strictly within this single call to keep cn2an warnings elsewhere
    (e.g. ad-hoc REPL use) visible.
    """
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            return cn2an.transform(text, "cn2an")
    except Exception:
        return text


def get_normalizer(language: str = "en") -> Callable[[str], str]:
    """Return the appropriate Whisper normalizer for *language*.

    - ``"en"`` → ``EnglishTextNormalizer``  (handles numbers, contractions,
      British→American spellings, filler removal, …)
    - ``"zh"`` (or any Chinese variant) → ``BasicTextNormalizer`` **+**
      Traditional → Simplified conversion via OpenCC **+**
      Chinese → Arabic number normalization via cn2an, so that WER/CER is
      not inflated by script-variant mismatches (e.g. 因為 vs 因为) or
      number-format mismatches (e.g. 二零二四 vs 二千零二十四 vs 2024).
    - anything else → ``BasicTextNormalizer``  (Unicode-aware punctuation /
      symbol removal, lowercasing, full-width → half-width)
    """
    if language.lower().startswith("en"):
        return _en_normalizer
    if language.lower().startswith("zh"):
        # Compose: Traditional→Simplified, then Chinese→Arabic numbers,
        # then basic normalize (punctuation/symbol removal)
        def _zh_normalizer(text: str) -> str:
            text = _t2s.convert(text)
            text = _normalize_chinese_numbers(text)
            return _basic_normalizer(text)
        return _zh_normalizer
    return _basic_normalizer


# ─── Text normalization & tokenization ───────────────────────────────────────

_CJK_PATTERN = re.compile(
    r"([\u3400-\u4dbf"   # CJK Unified Ideographs Extension A (rare)
    r"\u4e00-\u9fff"     # CJK Unified Ideographs (standard)
    r"\uf900-\ufaff])"   # CJK Compatibility Ideographs
)


def tokenize(text: str, normalizer: Optional[Callable[[str], str]] = None) -> List[str]:
    """
    Tokenize mixed Chinese/English text.
    - Each Chinese character → one token
    - Each English/digit word → one token

    If *normalizer* is provided it is applied first; otherwise the text is
    used as-is (caller is responsible for pre-normalizing).
    """
    if normalizer is not None:
        try:
            text = normalizer(text)
        except Exception as e:
            # whisper_normalizer.english.EnglishNumberNormalizer contains a
            # bare ``assert f is not None`` that fires on certain digit/word
            # combinations in degraded hypotheses (observed with MiMo on
            # cv_en_noise_dns). Fall back to lower() so one pathological
            # sample cannot kill an entire finalize across 22 sub-tests.
            _LOGGER.warning(
                "tokenize: normalizer raised %s (%s); falling back to lower(). "
                "text[:120]=%r",
                type(e).__name__, str(e)[:200], (text or "")[:120],
            )
            text = (text or "").lower()
    else:
        text = text.lower()

    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()

    if not text:
        return []

    tokens: List[str] = []
    parts = _CJK_PATTERN.split(text)
    for part in parts:
        part = part.strip()
        if not part:
            continue
        if _CJK_PATTERN.fullmatch(part):
            tokens.append(part)
        else:
            for word in part.split():
                if word:
                    tokens.append(word)
    return tokens


# ─── Edit distance (minimum edit distance) ──────────────────────────────────

def edit_distance(ref: List[str], hyp: List[str]) -> Tuple[int, int, int, int]:
    """
    Compute minimum edit distance between ref and hyp token lists.

    Returns:
        (substitutions, deletions, insertions, correct)
    """
    n = len(ref)
    m = len(hyp)

    # dp[i][j] = (cost, op)   op: 0=correct, 1=sub, 2=del, 3=ins
    dp = [[(0, 0)] * (m + 1) for _ in range(n + 1)]

    for i in range(1, n + 1):
        dp[i][0] = (i, 2)  # deletions
    for j in range(1, m + 1):
        dp[0][j] = (j, 3)  # insertions

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref[i - 1] == hyp[j - 1]:
                cost_sub = dp[i - 1][j - 1][0]
                op = 0  # correct
            else:
                cost_sub = dp[i - 1][j - 1][0] + 1
                op = 1  # substitution

            cost_del = dp[i - 1][j][0] + 1
            cost_ins = dp[i][j - 1][0] + 1

            min_cost = cost_sub
            min_op = op
            if cost_del < min_cost:
                min_cost = cost_del
                min_op = 2
            if cost_ins < min_cost:
                min_cost = cost_ins
                min_op = 3

            dp[i][j] = (min_cost, min_op)

    # Backtrace
    sub = dele = ins = cor = 0
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and dp[i][j][1] == 0:
            cor += 1
            i -= 1
            j -= 1
        elif i > 0 and j > 0 and dp[i][j][1] == 1:
            sub += 1
            i -= 1
            j -= 1
        elif i > 0 and dp[i][j][1] == 2:
            dele += 1
            i -= 1
        else:
            ins += 1
            j -= 1

    return sub, dele, ins, cor


def _edit_distance_aligned(
    ref: List[str], hyp: List[str],
) -> List[Tuple[str, Optional[int], Optional[int]]]:
    """Edit-distance backtrace as a list of aligned operations.

    Returned tuples ``(op, ri, hi)`` where ``op ∈ {"correct", "sub", "del",
    "ins"}``; ``ri`` indexes into *ref* (None for ``ins``), ``hi`` indexes
    into *hyp* (None for ``del``). Used by :func:`compute_hotword_metrics`
    to bucket sub/del/ins counts into biased vs unbiased ref/hyp tokens.

    Mirrors :func:`edit_distance` exactly (same ties, same DP shape) so
    overall sub/del/ins/cor sums match WER numbers byte-for-byte.
    """
    n, m = len(ref), len(hyp)
    dp = [[(0, 0)] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        dp[i][0] = (i, 2)
    for j in range(1, m + 1):
        dp[0][j] = (j, 3)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref[i - 1] == hyp[j - 1]:
                cost_sub, op = dp[i - 1][j - 1][0], 0
            else:
                cost_sub, op = dp[i - 1][j - 1][0] + 1, 1
            cost_del = dp[i - 1][j][0] + 1
            cost_ins = dp[i][j - 1][0] + 1
            min_cost, min_op = cost_sub, op
            if cost_del < min_cost:
                min_cost, min_op = cost_del, 2
            if cost_ins < min_cost:
                min_cost, min_op = cost_ins, 3
            dp[i][j] = (min_cost, min_op)

    ops: List[Tuple[str, Optional[int], Optional[int]]] = []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and dp[i][j][1] == 0:
            ops.append(("correct", i - 1, j - 1))
            i -= 1; j -= 1
        elif i > 0 and j > 0 and dp[i][j][1] == 1:
            ops.append(("sub", i - 1, j - 1))
            i -= 1; j -= 1
        elif i > 0 and dp[i][j][1] == 2:
            ops.append(("del", i - 1, None))
            i -= 1
        else:
            ops.append(("ins", None, j - 1))
            j -= 1
    ops.reverse()
    return ops


# ─── WER / CER computation ──────────────────────────────────────────────────

def compute_wer(
    references: List[str],
    hypotheses: List[str],
    language: str = "en",
    max_hyp_ratio: float = 10.0,
) -> dict:
    """
    Compute WER/CER over a list of (reference, hypothesis) pairs.

    Text is normalized with the Whisper-standard normalizer for *language*
    before tokenization:
      - ``"en"`` uses ``EnglishTextNormalizer`` (numbers, contractions, …)
      - anything else uses ``BasicTextNormalizer`` (Unicode punct removal)

    Mixed Chinese/English tokenization:
      - Chinese chars → individual tokens (≈ CER for Chinese)
      - English words → whole tokens (≈ WER for English)

    Hallucination filtering (``max_hyp_ratio``):
      If *max_hyp_ratio* > 0 and a hypothesis has more than
      ``max_hyp_ratio × len(ref_tokens)`` tokens, it is treated as an empty
      hypothesis (all ref tokens count as deletions).  This prevents
      hallucinated outputs from inflating the insertion count.

    Returns dict with:
      - wer: overall word error rate (%)
      - num_ref_tokens: total reference tokens
      - substitutions, deletions, insertions
      - num_sentences, num_err_sentences, ser (sentence error rate %)
      - num_hallucinated: number of sentences flagged as hallucination
      - per_sentence: list of per-sentence dicts
    """
    assert len(references) == len(hypotheses), \
        f"Length mismatch: {len(references)} refs vs {len(hypotheses)} hyps"

    normalizer = get_normalizer(language)

    total_ref = 0
    total_sub = 0
    total_del = 0
    total_ins = 0
    total_cor = 0
    num_err_sent = 0
    num_hallucinated = 0
    per_sentence = []

    for ref_text, hyp_text in zip(references, hypotheses):
        ref_tokens = tokenize(ref_text, normalizer)
        hyp_tokens = tokenize(hyp_text, normalizer)

        ref_len = len(ref_tokens)

        # Hallucination detection: compare raw text lengths (before
        # normalization) so that normalizer artifacts – e.g. the English
        # normalizer merging "twenty one thirty …" into "2130…" – cannot
        # cause false positives.
        is_hallucination = False
        raw_ref_len = len(ref_text.strip())
        raw_hyp_len = len(hyp_text.strip())
        if max_hyp_ratio > 0 and raw_ref_len > 0 and raw_hyp_len > max_hyp_ratio * raw_ref_len:
            is_hallucination = True
            num_hallucinated += 1
            # Treat as empty output → all ref tokens are deletions
            s, d, i, c = 0, ref_len, 0, 0
            orig_hyp_tokens = hyp_tokens
            hyp_tokens = []
        else:
            s, d, i, c = edit_distance(ref_tokens, hyp_tokens)

        err = s + d + i
        sent_wer = 100.0 * err / max(ref_len, 1)

        total_ref += ref_len
        total_sub += s
        total_del += d
        total_ins += i
        total_cor += c
        if err > 0:
            num_err_sent += 1

        sent_info = {
            "ref_tokens": ref_len,
            "sub": s, "del": d, "ins": i,
            "wer": sent_wer,
            "norm_ref": " ".join(ref_tokens),
            "norm_hyp": " ".join(hyp_tokens),
        }
        if is_hallucination:
            sent_info["hallucination"] = True
            sent_info["orig_hyp_len"] = len(orig_hyp_tokens)
            sent_info["raw_ref_len"] = raw_ref_len
            sent_info["raw_hyp_len"] = raw_hyp_len
        per_sentence.append(sent_info)

    total_err = total_sub + total_del + total_ins
    overall_wer = 100.0 * total_err / max(total_ref, 1)
    num_sent = len(references)
    ser = 100.0 * num_err_sent / max(num_sent, 1)

    return {
        "wer": overall_wer,
        "num_ref_tokens": total_ref,
        "substitutions": total_sub,
        "deletions": total_del,
        "insertions": total_ins,
        "correct": total_cor,
        "num_sentences": num_sent,
        "num_err_sentences": num_err_sent,
        "num_hallucinated": num_hallucinated,
        "ser": ser,
        "per_sentence": per_sentence,
    }


# ─── Hotword-specialised metrics ────────────────────────────────────────────
# Targets the contextual-biasing literature (2024-2025); see references:
#   * GLCLAP + GRPO  (arxiv:2512.21828)  — KER, SACC
#   * H-PRM          (arxiv:2508.18295)  — PRR, PPR, PF1, PrRR
#   * TurboBias / Trie-based ZS (arxiv:2508.07014, 2508.17796) — B-WER, U-WER
# All metrics are pure-text and operate on the same Whisper-normalised
# tokens as :func:`compute_wer`, so numbers stay consistent across sheets.

def _mark_hotword_tokens(
    tokens: List[str],
    hotword_token_seqs: List[List[str]],
) -> List[bool]:
    """Mark every token that participates in any hotword occurrence.

    *hotword_token_seqs* is a pre-tokenised list of hotword phrases
    (e.g. ``[["诸","葛","紫","岐"], ["qwen3","omni"]]``). For each phrase
    we slide it across *tokens* and any token covered by a full match is
    flagged True. Empty phrases are skipped. O(N * sum(|hw|)).
    """
    n = len(tokens)
    mask = [False] * n
    for hw_seq in hotword_token_seqs:
        hl = len(hw_seq)
        if hl == 0 or hl > n:
            continue
        for i in range(n - hl + 1):
            if tokens[i:i + hl] == hw_seq:
                for j in range(i, i + hl):
                    mask[j] = True
    return mask


def _hotword_in_text(hw_seq: List[str], tokens: List[str]) -> bool:
    """True iff the tokenised hotword phrase appears as a contiguous
    sub-sequence in *tokens* (used for KER / PRR substring match)."""
    if not hw_seq:
        return False
    hl = len(hw_seq)
    if hl > len(tokens):
        return False
    for i in range(len(tokens) - hl + 1):
        if tokens[i:i + hl] == hw_seq:
            return True
    return False


def compute_hotword_metrics(
    references: List[str],
    hypotheses: List[str],
    gt_hotwords_list: List[List[str]],
    candidate_hotwords_list: Optional[List[List[str]]] = None,
    *,
    language: str = "en",
    max_hyp_ratio: float = 10.0,
    retrieved_hotwords_list: Optional[List[List[str]]] = None,
) -> dict:
    """Hotword-specialised evaluation metrics.

    Args:
        references / hypotheses: aligned ref/hyp text per utterance.
        gt_hotwords_list:        ground-truth hotword phrases per utt
                                 (from manifest ``custom.hotwords``).
        candidate_hotwords_list: candidate hotword set fed to the model
                                 per utt (real ∪ distractor for random
                                 mode; retrieved top-k for retrieve mode;
                                 just real for none mode). Used to count
                                 false positives (distractor leaking into
                                 hyp) for PPR. Defaults to *gt_hotwords_list*.
        retrieved_hotwords_list: phase-2 retrieved top-k per utt; only
                                 used to compute Recall@K / PrRR for the
                                 retrieve pipeline. ``None`` skips both.
        language:                normaliser language code ("en" / "zh" / …).
        max_hyp_ratio:           same hallucination guard as :func:`compute_wer`;
                                 hallucinated hyps are zeroed before B-WER /
                                 U-WER attribution to avoid blowing up ins.

    Returns:
        Dict with overall metrics (suffixed with ``_pct`` already
        multiplied by 100) plus per-sentence diagnostic list under
        ``per_sentence``. Counters are micro-aggregated; macro variants
        can be computed downstream from the per-sentence list.
    """
    n = len(references)
    assert len(hypotheses) == n, (
        f"refs={n} but hyps={len(hypotheses)}"
    )
    assert len(gt_hotwords_list) == n, (
        f"refs={n} but gt_hotwords_list={len(gt_hotwords_list)}"
    )
    if candidate_hotwords_list is None:
        candidate_hotwords_list = list(gt_hotwords_list)
    else:
        assert len(candidate_hotwords_list) == n, (
            f"refs={n} but candidate_hotwords_list={len(candidate_hotwords_list)}"
        )
    if retrieved_hotwords_list is not None:
        assert len(retrieved_hotwords_list) == n, (
            f"refs={n} but retrieved_hotwords_list={len(retrieved_hotwords_list)}"
        )

    normalizer = get_normalizer(language)

    # KER / SACC counters (utterance-keyword level)
    ker_total_keywords = 0
    ker_missed_keywords = 0
    n_correct_sentences = 0
    n_sentences_with_keywords = 0  # only utts with >=1 GT hotword count for KER

    # PRR / PPR / PF1 counters (word-level)
    prr_tp = 0  # GT hotword found in hyp
    prr_fn = 0  # GT hotword NOT in hyp
    ppr_fp = 0  # candidate distractor leaked into hyp

    # B-WER / U-WER counters
    biased_ref = 0
    biased_sub = biased_del = biased_ins = 0
    unbiased_ref = 0
    unbiased_sub = unbiased_del = unbiased_ins = 0

    # Recall@K / PrRR (retrieve mode only)
    rk_total = rk_hit = 0
    n_retrieve_eval = 0
    n_retrieve_perfect = 0

    per_sentence: List[dict] = []

    for idx in range(n):
        ref_text = references[idx] or ""
        hyp_text = hypotheses[idx] or ""
        gt_hw = [h for h in (gt_hotwords_list[idx] or []) if h and h.strip()]
        cand_hw = [h for h in (candidate_hotwords_list[idx] or []) if h and h.strip()]

        ref_tokens = tokenize(ref_text, normalizer)
        hyp_tokens = tokenize(hyp_text, normalizer)

        # Hallucination guard mirrors compute_wer: clip hyp to empty so
        # spam ins doesn't dominate B-WER. Use raw lengths to skip
        # normaliser-induced false positives (e.g. EnglishTextNormalizer
        # may merge digits and shrink hyp).
        raw_ref_len = len(ref_text.strip())
        raw_hyp_len = len(hyp_text.strip())
        is_hallucination = (
            max_hyp_ratio > 0
            and raw_ref_len > 0
            and raw_hyp_len > max_hyp_ratio * raw_ref_len
        )
        if is_hallucination:
            hyp_tokens_for_align = []
        else:
            hyp_tokens_for_align = hyp_tokens

        gt_token_seqs = [tokenize(h, normalizer) for h in gt_hw]
        cand_token_seqs = [tokenize(h, normalizer) for h in cand_hw]

        # ---- SACC: exact match after normalisation ----
        sent_sacc = ref_tokens == hyp_tokens
        if sent_sacc:
            n_correct_sentences += 1

        # ---- KER + PRR/PPR ----
        sent_ker_total = 0
        sent_ker_missed = 0
        sent_tp = sent_fn = 0
        for hw_seq in gt_token_seqs:
            if not hw_seq:
                continue
            sent_ker_total += 1
            if _hotword_in_text(hw_seq, hyp_tokens):
                sent_tp += 1
            else:
                sent_ker_missed += 1
                sent_fn += 1
        if sent_ker_total > 0:
            n_sentences_with_keywords += 1
        ker_total_keywords += sent_ker_total
        ker_missed_keywords += sent_ker_missed
        prr_tp += sent_tp
        prr_fn += sent_fn

        # FP: distractor (cand \ gt) appearing in hyp
        gt_seq_set = {tuple(s) for s in gt_token_seqs if s}
        sent_fp = 0
        for hw_seq in cand_token_seqs:
            if not hw_seq:
                continue
            if tuple(hw_seq) in gt_seq_set:
                continue  # this is a real hotword, accounted for in TP/FN
            if _hotword_in_text(hw_seq, hyp_tokens):
                sent_fp += 1
        ppr_fp += sent_fp

        # ---- B-WER / U-WER ----
        # ref-token biased mask: token-level membership in any GT hotword
        # hyp-token biased mask: any candidate hotword (real ∪ distractor)
        # — ins from cand-but-not-gt counts as biased ins (it's a hotword
        # the model emitted, regardless of whether it's a true positive).
        ref_biased_mask = _mark_hotword_tokens(ref_tokens, gt_token_seqs)
        hyp_biased_mask = _mark_hotword_tokens(hyp_tokens, cand_token_seqs)

        sent_b_ref = sum(1 for x in ref_biased_mask if x)
        sent_u_ref = len(ref_tokens) - sent_b_ref
        biased_ref += sent_b_ref
        unbiased_ref += sent_u_ref

        sent_b_sub = sent_b_del = sent_b_ins = 0
        sent_u_sub = sent_u_del = sent_u_ins = 0
        for op, ri, hi in _edit_distance_aligned(ref_tokens, hyp_tokens_for_align):
            if op == "correct":
                continue
            if op == "sub":
                if ref_biased_mask[ri]:
                    sent_b_sub += 1
                else:
                    sent_u_sub += 1
            elif op == "del":
                if ref_biased_mask[ri]:
                    sent_b_del += 1
                else:
                    sent_u_del += 1
            elif op == "ins":
                # ins: classify by hyp token's biased flag (only valid
                # when hyp wasn't zeroed by hallucination guard).
                if not is_hallucination and hyp_biased_mask[hi]:
                    sent_b_ins += 1
                else:
                    sent_u_ins += 1
        if is_hallucination:
            # Hallucinated hyp: ref tokens were forced into del; the loop
            # above already attributed dels by ref bias. Nothing else to do.
            pass

        biased_sub += sent_b_sub
        biased_del += sent_b_del
        biased_ins += sent_b_ins
        unbiased_sub += sent_u_sub
        unbiased_del += sent_u_del
        unbiased_ins += sent_u_ins

        # ---- Recall@K / PrRR (retrieve mode) ----
        sent_recall = None
        sent_prrr_hit = None
        if retrieved_hotwords_list is not None:
            retrieved = [h for h in (retrieved_hotwords_list[idx] or []) if h]
            retrieved_set = {h.strip() for h in retrieved if h.strip()}
            real_set = {h.strip() for h in gt_hw if h.strip()}
            if real_set:
                hit = len(real_set & retrieved_set)
                rk_total += len(real_set)
                rk_hit += hit
                sent_recall = hit / len(real_set)
                sent_prrr_hit = (real_set <= retrieved_set)
                n_retrieve_eval += 1
                if sent_prrr_hit:
                    n_retrieve_perfect += 1

        sent_info = {
            "n_gt": sent_ker_total,
            "tp": sent_tp,
            "fn": sent_fn,
            "fp": sent_fp,
            "sacc": sent_sacc,
            "biased_ref_tokens": sent_b_ref,
            "unbiased_ref_tokens": sent_u_ref,
            "biased_sub": sent_b_sub,
            "biased_del": sent_b_del,
            "biased_ins": sent_b_ins,
            "unbiased_sub": sent_u_sub,
            "unbiased_del": sent_u_del,
            "unbiased_ins": sent_u_ins,
            "hallucination": is_hallucination,
        }
        if sent_recall is not None:
            sent_info["recall_at_k"] = sent_recall
            sent_info["prrr_hit"] = bool(sent_prrr_hit)
        per_sentence.append(sent_info)

    # ---- Aggregate ----
    ker = (
        100.0 * ker_missed_keywords / ker_total_keywords
        if ker_total_keywords > 0 else None
    )
    sacc = 100.0 * n_correct_sentences / max(n, 1)
    prr = (
        100.0 * prr_tp / (prr_tp + prr_fn)
        if (prr_tp + prr_fn) > 0 else None
    )
    ppr = (
        100.0 * prr_tp / (prr_tp + ppr_fp)
        if (prr_tp + ppr_fp) > 0 else None
    )
    if prr is not None and ppr is not None and (prr + ppr) > 0:
        pf1 = 2.0 * prr * ppr / (prr + ppr)
    else:
        pf1 = None

    b_wer = (
        100.0 * (biased_sub + biased_del + biased_ins) / biased_ref
        if biased_ref > 0 else None
    )
    u_wer = (
        100.0 * (unbiased_sub + unbiased_del + unbiased_ins) / unbiased_ref
        if unbiased_ref > 0 else None
    )

    out = {
        "ker": ker,
        "sacc": sacc,
        "prr": prr,
        "ppr": ppr,
        "pf1": pf1,
        "b_wer": b_wer,
        "u_wer": u_wer,
        "ker_total_keywords": ker_total_keywords,
        "ker_missed_keywords": ker_missed_keywords,
        "tp": prr_tp,
        "fn": prr_fn,
        "fp": ppr_fp,
        "biased_ref_tokens": biased_ref,
        "biased_sub": biased_sub,
        "biased_del": biased_del,
        "biased_ins": biased_ins,
        "unbiased_ref_tokens": unbiased_ref,
        "unbiased_sub": unbiased_sub,
        "unbiased_del": unbiased_del,
        "unbiased_ins": unbiased_ins,
        "n_sentences": n,
        "n_correct_sentences": n_correct_sentences,
        "n_sentences_with_keywords": n_sentences_with_keywords,
        "per_sentence": per_sentence,
    }
    if retrieved_hotwords_list is not None:
        recall_at_k = (
            100.0 * rk_hit / rk_total if rk_total > 0 else None
        )
        prrr = (
            100.0 * n_retrieve_perfect / n_retrieve_eval
            if n_retrieve_eval > 0 else None
        )
        out["recall_at_k"] = recall_at_k
        out["prrr"] = prrr
        out["recall_total_gt"] = rk_total
        out["recall_hit"] = rk_hit
        out["n_retrieve_eval"] = n_retrieve_eval
        out["n_retrieve_perfect"] = n_retrieve_perfect
    return out


# ─── Classification metrics (SER and other single-label tasks) ───────────────

def compute_classification_metrics(
    references: List[str],
    hypotheses: List[str],
) -> dict:
    """Compute WA, UA, macro-F1, and confusion matrix for classification.

    Labels are lowercased and stripped before comparison so cross-script
    callers (decode.py, test_vllm_inference.py) get identical numbers
    regardless of upstream casing.

    Returns dict with::

        wa, ua, macro_f1, num_samples, num_classes, labels,
        per_class_recall, per_class_precision, per_class_f1,
        per_class_total, per_class_correct, confusion
    """
    refs = [r.strip().lower() for r in references]
    hyps = [h.strip().lower() for h in hypotheses]

    all_labels = sorted(set(refs))
    label_to_idx = {l: i for i, l in enumerate(all_labels)}
    n_classes = len(all_labels)

    confusion = [[0] * n_classes for _ in range(n_classes)]
    for r, h in zip(refs, hyps):
        ri = label_to_idx.get(r)
        hi = label_to_idx.get(h)
        if ri is not None and hi is not None:
            confusion[ri][hi] += 1
        elif ri is not None:
            # hypothesis is an unseen label -> counts as wrong, no column
            pass

    ref_counts = Counter(refs)
    per_class_correct = {}
    per_class_total = {}
    per_class_pred_total = Counter(hyps)

    for label in all_labels:
        idx = label_to_idx[label]
        per_class_total[label] = ref_counts[label]
        per_class_correct[label] = confusion[idx][idx]

    total_correct = sum(per_class_correct.values())
    total_samples = len(refs)

    wa = total_correct / max(total_samples, 1)

    per_class_recall = {}
    for label in all_labels:
        n = per_class_total[label]
        per_class_recall[label] = per_class_correct[label] / n if n > 0 else 0.0
    ua = sum(per_class_recall.values()) / max(n_classes, 1)

    per_class_precision = {}
    for label in all_labels:
        n_pred = per_class_pred_total.get(label, 0)
        per_class_precision[label] = (
            per_class_correct[label] / n_pred if n_pred > 0 else 0.0
        )

    per_class_f1 = {}
    for label in all_labels:
        p = per_class_precision[label]
        r = per_class_recall[label]
        per_class_f1[label] = 2 * p * r / (p + r) if (p + r) > 0 else 0.0

    macro_f1 = sum(per_class_f1.values()) / max(n_classes, 1)

    return {
        "wa": wa,
        "ua": ua,
        "macro_f1": macro_f1,
        "num_samples": total_samples,
        "num_classes": n_classes,
        "labels": all_labels,
        "per_class_recall": per_class_recall,
        "per_class_precision": per_class_precision,
        "per_class_f1": per_class_f1,
        "per_class_total": per_class_total,
        "per_class_correct": per_class_correct,
        "confusion": confusion,
    }


def format_confusion_matrix(labels: List[str], confusion: List[List[int]]) -> str:
    """Format a confusion matrix as an aligned text table."""
    col_width = max(max((len(l) for l in labels), default=4), 5)
    header = " " * (col_width + 2) + "  ".join(l.rjust(col_width) for l in labels)
    lines = [
        "Confusion matrix (rows=ref, cols=hyp):",
        header,
    ]
    for i, label in enumerate(labels):
        row_vals = "  ".join(str(v).rjust(col_width) for v in confusion[i])
        lines.append(f"{label.rjust(col_width)}  {row_vals}")
    return "\n".join(lines)


def render_classification_report(label: str, metrics: Dict) -> str:
    """Render a wer_report.txt-style summary for a classification test set."""
    lines: List[str] = []
    lines.append(f"# Test: {label}")
    lines.append(f"WA (Weighted Accuracy) : {metrics['wa'] * 100:.2f}%")
    lines.append(f"UA (Unweighted Accuracy): {metrics['ua'] * 100:.2f}%")
    lines.append(f"Macro-F1               : {metrics['macro_f1'] * 100:.2f}%")
    lines.append(f"Samples                : {metrics['num_samples']}")
    lines.append(f"Classes                : {metrics['num_classes']}")
    lines.append(f"Labels                 : {', '.join(metrics['labels'])}")
    lines.append("")
    lines.append("Per-class breakdown:")
    lines.append(f"  {'Label':<20s}  {'Count':>6s}  {'Recall':>8s}  "
                 f"{'Prec':>8s}  {'F1':>8s}")
    for clabel in metrics["labels"]:
        lines.append(
            f"  {clabel:<20s}  "
            f"{metrics['per_class_total'][clabel]:>6d}  "
            f"{metrics['per_class_recall'][clabel] * 100:>7.2f}%  "
            f"{metrics['per_class_precision'][clabel] * 100:>7.2f}%  "
            f"{metrics['per_class_f1'][clabel] * 100:>7.2f}%"
        )
    lines.append("")
    lines.append(format_confusion_matrix(metrics["labels"], metrics["confusion"]))
    return "\n".join(lines)


# ─── ESC metrics (multi-label tag F1 + ROUGE-L) ──────────────────────────────
#
# ESC ("Environmental Sound Captioning") references in audioset_esc are
# comma-separated tag sets such as:
#
#     "indoor small room, television sounds, music"
#     "machine or tool noise"
#     "rail transport"
#
# The model's hypothesis can be tag-style or a free-form description; the
# metrics below are robust to both:
#
#   * Tag-set Micro/Macro F1: build a corpus tag vocab from refs, treat
#     containment-in-hyp as the predicted tag set per row, then aggregate
#     TP/FP/FN.  Rewards label coverage regardless of phrasing style.
#   * ROUGE-L F1 (mean over rows): word-level LCS-based F1 between the
#     normalised reference and hypothesis. Captures word overlap and order.
#
# Both are zero-dependency (pure stdlib + the existing `re`).

_ESC_TAG_SPLIT_RE = re.compile(r"[,，、;；]")
_ESC_PUNCT_RE = re.compile(r"[^\w\s]+", flags=re.UNICODE)
_ESC_WHITESPACE_RE = re.compile(r"\s+")
_ESC_CJK_RE = re.compile(r"[\u4e00-\u9fff]")


def _normalize_esc_text(text: str) -> str:
    """Lowercase + strip Unicode punctuation + collapse whitespace.

    Used by both tag-vocab construction and hyp containment checks so the
    same canonical form drives both passes. Keeps Unicode word chars
    (including CJK) intact thanks to ``re.UNICODE`` on _ESC_PUNCT_RE.
    """
    if not text:
        return ""
    s = str(text).lower()
    s = _ESC_PUNCT_RE.sub(" ", s)
    s = _ESC_WHITESPACE_RE.sub(" ", s)
    return s.strip()


def _extract_esc_tag_set(ref: str) -> List[str]:
    """Split an ESC reference into an ordered, deduplicated, normalized tag list.

    Splits on ASCII / full-width comma and semicolon (``,，、;；``), passes
    each tag through :func:`_normalize_esc_text`, and drops empties.
    """
    if not ref:
        return []
    seen: set = set()
    out: List[str] = []
    for raw in _ESC_TAG_SPLIT_RE.split(str(ref)):
        norm = _normalize_esc_text(raw)
        if not norm or norm in seen:
            continue
        seen.add(norm)
        out.append(norm)
    return out


def _esc_tag_in_hyp(tag: str, hyp_norm: str) -> bool:
    """Whether a normalized tag occurs inside a normalized hyp.

    For tags containing CJK characters, falls back to plain substring
    matching because CJK text has no inter-word whitespace. For Latin /
    space-separated tags, requires whitespace-padded boundaries so
    ``music`` does NOT spuriously match ``musical`` and ``rail transport``
    must occur as a contiguous phrase.
    """
    if not tag or not hyp_norm:
        return False
    if _ESC_CJK_RE.search(tag):
        return tag in hyp_norm
    return f" {tag} " in f" {hyp_norm} "


def _rouge_l_word(
    ref_tokens: List[str], hyp_tokens: List[str],
) -> Tuple[float, float, float]:
    """Word-level ROUGE-L (precision, recall, F1) via O(m·n) LCS DP.

    Returns ``(0.0, 0.0, 0.0)`` if either side has no tokens. Uses two
    rolling rows so memory stays at O(n) — the longest ESC hyps observed
    in practice are < 200 tokens so this is more than sufficient.
    """
    m = len(ref_tokens)
    n = len(hyp_tokens)
    if m == 0 or n == 0:
        return 0.0, 0.0, 0.0
    prev = [0] * (n + 1)
    for i in range(1, m + 1):
        curr = [0] * (n + 1)
        ri = ref_tokens[i - 1]
        for j in range(1, n + 1):
            if ri == hyp_tokens[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(prev[j], curr[j - 1])
        prev = curr
    lcs = prev[n]
    p = lcs / n
    r = lcs / m
    f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return p, r, f


def compute_esc_metrics(
    references: List[str],
    hypotheses: List[str],
) -> dict:
    """Compute ESC metrics: Tag-set Micro/Macro F1 + ROUGE-L F1.

    ``references[i]`` is expected to be a comma-separated tag set; rows
    with empty ``references[i]`` (after stripping) contribute neither to
    the tag-set nor to the ROUGE-L aggregates (they are reported under
    ``n_total - n_evaluated``).

    Returns dict with::

        task                                       # always "esc"
        n_total, n_evaluated,
        tag_micro_p, tag_micro_r, tag_micro_f1,
        tag_macro_p, tag_macro_r, tag_macro_f1,
        tp, fp, fn,
        rouge_l_p, rouge_l_r, rouge_l_f1,          # corpus-mean of per-row F1
        labels: [<tag>, ...],                      # vocab in first-seen order
        per_tag: {<tag>: {total, pred_total, hits,
                          recall, precision, f1}}
    """
    assert len(references) == len(hypotheses), (
        f"Length mismatch: {len(references)} refs vs {len(hypotheses)} hyps"
    )

    n_total = len(references)

    # Pass 1: parse refs and build a corpus tag vocab in first-seen order.
    parsed_refs: List[List[str]] = []
    ref_norms: List[str] = []
    seen_vocab: set = set()
    vocab: List[str] = []
    for ref in references:
        tags = _extract_esc_tag_set(ref)
        parsed_refs.append(tags)
        ref_norms.append(_normalize_esc_text(ref))
        for t in tags:
            if t not in seen_vocab:
                seen_vocab.add(t)
                vocab.append(t)

    # Pass 2: per-row TP/FP/FN against the vocab + per-row ROUGE-L.
    n_evaluated = 0
    tp = 0
    fp = 0
    fn = 0
    per_tag_total: Dict[str, int] = defaultdict(int)
    per_tag_pred_total: Dict[str, int] = defaultdict(int)
    per_tag_hits: Dict[str, int] = defaultdict(int)
    rouge_p_sum = 0.0
    rouge_r_sum = 0.0
    rouge_f_sum = 0.0

    for ref_tags, ref_norm, hyp in zip(parsed_refs, ref_norms, hypotheses):
        if not ref_tags:
            continue
        n_evaluated += 1
        hyp_norm = _normalize_esc_text(hyp)

        ref_set = set(ref_tags)
        pred_set = {tag for tag in vocab if _esc_tag_in_hyp(tag, hyp_norm)}
        hits = ref_set & pred_set

        tp += len(hits)
        fp += len(pred_set - ref_set)
        fn += len(ref_set - pred_set)

        for tag in ref_set:
            per_tag_total[tag] += 1
        for tag in pred_set:
            per_tag_pred_total[tag] += 1
        for tag in hits:
            per_tag_hits[tag] += 1

        ref_tokens = ref_norm.split()
        hyp_tokens = hyp_norm.split()
        p, r, f = _rouge_l_word(ref_tokens, hyp_tokens)
        rouge_p_sum += p
        rouge_r_sum += r
        rouge_f_sum += f

    # Per-tag P/R/F1.
    per_tag: Dict[str, Dict[str, float]] = {}
    for tag in vocab:
        total = per_tag_total[tag]
        pred = per_tag_pred_total[tag]
        hits = per_tag_hits[tag]
        recall = hits / total if total > 0 else 0.0
        precision = hits / pred if pred > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)
              if (precision + recall) > 0 else 0.0)
        per_tag[tag] = {
            "total": total,
            "pred_total": pred,
            "hits": hits,
            "recall": recall,
            "precision": precision,
            "f1": f1,
        }

    denom_p = tp + fp
    denom_r = tp + fn
    tag_micro_p = tp / denom_p if denom_p > 0 else 0.0
    tag_micro_r = tp / denom_r if denom_r > 0 else 0.0
    tag_micro_f1 = (2 * tag_micro_p * tag_micro_r
                    / (tag_micro_p + tag_micro_r)
                    if (tag_micro_p + tag_micro_r) > 0 else 0.0)

    # Macro is averaged over tags that appear in at least one ref so
    # vocab-only-from-pred tags don't dominate; matches the convention used
    # in compute_classification_metrics (UA = mean recall over ref classes).
    ref_supported = [t for t in vocab if per_tag_total[t] > 0]
    n_tags = len(ref_supported)
    tag_macro_p = (
        sum(per_tag[t]["precision"] for t in ref_supported) / max(n_tags, 1)
    )
    tag_macro_r = (
        sum(per_tag[t]["recall"] for t in ref_supported) / max(n_tags, 1)
    )
    tag_macro_f1 = (
        sum(per_tag[t]["f1"] for t in ref_supported) / max(n_tags, 1)
    )

    rouge_l_p = rouge_p_sum / max(n_evaluated, 1)
    rouge_l_r = rouge_r_sum / max(n_evaluated, 1)
    rouge_l_f1 = rouge_f_sum / max(n_evaluated, 1)

    return {
        "task": "esc",
        "n_total": n_total,
        "n_evaluated": n_evaluated,
        "tag_micro_p": tag_micro_p,
        "tag_micro_r": tag_micro_r,
        "tag_micro_f1": tag_micro_f1,
        "tag_macro_p": tag_macro_p,
        "tag_macro_r": tag_macro_r,
        "tag_macro_f1": tag_macro_f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "rouge_l_p": rouge_l_p,
        "rouge_l_r": rouge_l_r,
        "rouge_l_f1": rouge_l_f1,
        "labels": vocab,
        "per_tag": per_tag,
    }


def render_esc_report(label: str, metrics: Dict) -> str:
    """Render a wer_report.txt-style summary for an ESC test set."""
    lines: List[str] = []
    lines.append(f"# Test: {label}")
    lines.append(f"Total samples            : {metrics['n_total']}")
    lines.append(f"Evaluated (non-empty ref): {metrics['n_evaluated']}")
    lines.append(f"Tag vocab size           : {len(metrics.get('labels', []))}")
    lines.append("")
    lines.append("## Tag-set F1 (multi-label, hyp-containment)")
    lines.append(
        f"  Micro-F1: {metrics['tag_micro_f1'] * 100:.2f}%  "
        f"(P={metrics['tag_micro_p'] * 100:.2f}%  "
        f"R={metrics['tag_micro_r'] * 100:.2f}%)"
    )
    lines.append(
        f"  Macro-F1: {metrics['tag_macro_f1'] * 100:.2f}%  "
        f"(P={metrics['tag_macro_p'] * 100:.2f}%  "
        f"R={metrics['tag_macro_r'] * 100:.2f}%)"
    )
    lines.append(
        f"  Counts  : TP={metrics['tp']}  FP={metrics['fp']}  FN={metrics['fn']}"
    )
    lines.append("")
    lines.append("## ROUGE-L (mean over rows)")
    lines.append(
        f"  ROUGE-L F1: {metrics['rouge_l_f1'] * 100:.2f}%  "
        f"(P={metrics['rouge_l_p'] * 100:.2f}%  "
        f"R={metrics['rouge_l_r'] * 100:.2f}%)"
    )
    lines.append("")

    per_tag = metrics.get("per_tag", {})
    ref_supported = [
        (t, s) for t, s in per_tag.items() if s.get("total", 0) > 0
    ]
    ref_supported.sort(key=lambda kv: (-kv[1]["total"], kv[0]))

    lines.append(
        f"## Per-tag breakdown (top 30 of {len(ref_supported)} ref-supported tags)"
    )
    lines.append(
        f"  {'Tag':<32s}  {'Total':>5s}  {'Hits':>5s}  {'Recall':>7s}  "
        f"{'Pred':>5s}  {'Prec':>7s}  {'F1':>7s}"
    )
    for tag, stats in ref_supported[:30]:
        tag_disp = tag if len(tag) <= 32 else tag[:29] + "..."
        lines.append(
            f"  {tag_disp:<32s}  "
            f"{stats['total']:>5d}  "
            f"{stats['hits']:>5d}  "
            f"{stats['recall'] * 100:>6.2f}%  "
            f"{stats['pred_total']:>5d}  "
            f"{stats['precision'] * 100:>6.2f}%  "
            f"{stats['f1'] * 100:>6.2f}%"
        )
    return "\n".join(lines)


_LANG_NAME_MAP: Dict[str, str] = {
    "chinese": "zh", "cantonese": "zh", "english": "en",
    "japanese": "ja", "korean": "ko", "thai": "th", "vietnamese": "vi",
    "mongolian": "mn", "kazakh": "kk", "spanish": "es", "german": "de",
    "french": "fr", "italian": "it", "russian": "ru", "portuguese": "pt",
    "indonesian": "id", "hindi": "hi", "arabic": "ar", "turkish": "tr",
    "malay": "ms", "dutch": "nl", "swedish": "sv", "danish": "da",
    "finnish": "fi", "polish": "pl", "czech": "cs", "greek": "el",
    "hungarian": "hu", "romanian": "ro", "ukrainian": "uk", "persian": "fa",
    "filipino": "fil", "hebrew": "he", "norwegian": "no", "slovak": "sk",
    "slovenian": "sl", "croatian": "hr", "bulgarian": "bg", "catalan": "ca",
    "icelandic": "is", "swahili": "sw", "urdu": "ur", "telugu": "te",
    "tamil": "ta", "macedonian": "mk",
}


def normalize_lang_code(lang: Optional[str], default: str = "en") -> str:
    """Normalise a Lhotse-style language label into the short code used by
    :func:`get_normalizer` ("en", "zh", "ja", ...).

    Accepts mixed-case three-form inputs: ``"zh-CN"``, ``"zh"``, ``"Chinese"``,
    ``"english"``, etc. Empty / None values fall back to *default*.
    """
    if not lang:
        return default
    s = str(lang).strip().lower()
    if not s:
        return default
    if s.startswith("zh"):
        return "zh"
    if s in _LANG_NAME_MAP:
        return _LANG_NAME_MAP[s]
    return s


def compute_per_language_wer(
    references: List[str],
    hypotheses: List[str],
    languages: List[str],
    max_hyp_ratio: float = 10.0,
) -> dict:
    """Compute per-language WER plus a corpus-level overall WER.

    ``languages[i]`` is normalised via :func:`normalize_lang_code` before
    bucketing so callers can pass raw Lhotse fields. Empty references are
    silently dropped (kept track of via ``n_skipped`` so silence-only batches
    can still compute the overall metric without dividing by zero).

    Returns::

        {
          "per_language": {"zh": <compute_wer dict>, "en": <compute_wer dict>, ...},
          "overall": <compute_wer dict>,        # may be None if no usable refs
          "n_total": ...,
          "n_skipped": ...,                     # entries dropped (empty ref)
        }
    """
    assert len(references) == len(hypotheses) == len(languages), (
        f"Length mismatch: refs={len(references)} hyps={len(hypotheses)} "
        f"langs={len(languages)}"
    )

    buckets_ref: Dict[str, List[str]] = defaultdict(list)
    buckets_hyp: Dict[str, List[str]] = defaultdict(list)
    n_skipped = 0
    for ref, hyp, lang in zip(references, hypotheses, languages):
        if not (ref or "").strip():
            # Silence / negative samples cannot contribute to WER. They are
            # surfaced separately by ``compute_silence_metrics``.
            n_skipped += 1
            continue
        code = normalize_lang_code(lang)
        buckets_ref[code].append(ref)
        buckets_hyp[code].append(hyp)

    per_language: Dict[str, dict] = {}
    all_refs: List[str] = []
    all_hyps: List[str] = []
    overall_lang: Optional[str] = None
    for code in sorted(buckets_ref.keys()):
        per_language[code] = compute_wer(
            buckets_ref[code], buckets_hyp[code],
            language=code, max_hyp_ratio=max_hyp_ratio,
        )
        all_refs.extend(buckets_ref[code])
        all_hyps.extend(buckets_hyp[code])
        overall_lang = code  # if a single bucket, reuse its normalizer choice

    if not all_refs:
        overall = None
    else:
        # When there is more than one language bucket the corpus-level WER is
        # tokenised against whichever normalizer wins last; this is fine in
        # practice because tokens themselves are language-agnostic at the
        # whitespace / CJK-char level, but per-language buckets above are the
        # primary metric to report.
        overall = compute_wer(
            all_refs, all_hyps,
            language=overall_lang or "en", max_hyp_ratio=max_hyp_ratio,
        )

    return {
        "per_language": per_language,
        "overall": overall,
        "n_total": len(references),
        "n_skipped": n_skipped,
    }


def compute_silence_metrics(
    references: List[str],
    hypotheses: List[str],
    sample_types: Optional[List[str]] = None,
) -> dict:
    """Compute silence-aware binary metrics + per-sample-type breakdown.

    A *positive* sample is one whose reference text is non-empty. A *negative*
    sample has empty reference text (e.g. negative_silence / negative_distractor
    in TS-ASR test sets). False alarms / misses operate on the original
    (un-normalised) strings so a hypothesis containing only punctuation still
    registers as a false alarm.

    Returns::

        {
          "n_pos_ref": ..., "n_neg_ref": ...,
          "miss": ..., "false_alarm": ..., "true_silence": ...,
          "false_alarm_rate": ..., "miss_rate": ...,
          "exact_silence_match": ...,                 # 1 - false_alarm_rate
          "per_sample_type": {<type>: {...}, ...},    # only when sample_types given
        }
    """
    assert len(references) == len(hypotheses), (
        f"Length mismatch: {len(references)} refs vs {len(hypotheses)} hyps"
    )
    if sample_types is not None:
        assert len(sample_types) == len(references), (
            f"sample_types length {len(sample_types)} != refs {len(references)}"
        )

    sm = {
        "n_pos_ref": 0, "n_neg_ref": 0,
        "miss": 0, "false_alarm": 0, "true_silence": 0,
        "false_alarm_rate": None, "miss_rate": None,
        "exact_silence_match": None,
    }
    per_st: Dict[str, dict] = defaultdict(lambda: {
        "n": 0, "n_pos_ref": 0, "n_neg_ref": 0,
        "miss": 0, "false_alarm": 0, "true_silence": 0,
    })

    for i, (ref, hyp) in enumerate(zip(references, hypotheses)):
        ref_empty = not (ref or "").strip()
        hyp_empty = not (hyp or "").strip()
        st = (sample_types[i] if sample_types else "positive") or "positive"
        bucket = per_st[st]
        bucket["n"] += 1
        if ref_empty:
            sm["n_neg_ref"] += 1
            bucket["n_neg_ref"] += 1
            if hyp_empty:
                sm["true_silence"] += 1
                bucket["true_silence"] += 1
            else:
                sm["false_alarm"] += 1
                bucket["false_alarm"] += 1
        else:
            sm["n_pos_ref"] += 1
            bucket["n_pos_ref"] += 1
            if hyp_empty:
                sm["miss"] += 1
                bucket["miss"] += 1

    if sm["n_neg_ref"] > 0:
        sm["false_alarm_rate"] = sm["false_alarm"] / sm["n_neg_ref"]
        sm["exact_silence_match"] = sm["true_silence"] / sm["n_neg_ref"]
    if sm["n_pos_ref"] > 0:
        sm["miss_rate"] = sm["miss"] / sm["n_pos_ref"]

    if sample_types is not None:
        finalised: Dict[str, dict] = {}
        for st, bucket in per_st.items():
            out = {
                "n": bucket["n"],
                "n_pos_ref": bucket["n_pos_ref"],
                "n_neg_ref": bucket["n_neg_ref"],
                "miss": bucket["miss"],
                "false_alarm": bucket["false_alarm"],
                "true_silence": bucket["true_silence"],
                "false_alarm_rate": (
                    bucket["false_alarm"] / bucket["n_neg_ref"]
                    if bucket["n_neg_ref"] > 0 else None
                ),
                "exact_silence_match": (
                    bucket["true_silence"] / bucket["n_neg_ref"]
                    if bucket["n_neg_ref"] > 0 else None
                ),
                "miss_rate": (
                    bucket["miss"] / bucket["n_pos_ref"]
                    if bucket["n_pos_ref"] > 0 else None
                ),
            }
            finalised[st] = out
        sm["per_sample_type"] = finalised

    return sm


def print_wer_report(wer_result: dict, model_name: str = ""):
    """Pretty-print WER report to console."""
    header = f" WER Report: {model_name} " if model_name else " WER Report "
    print("\n" + "=" * 80)
    print(f"{header:=^80}")
    print("=" * 80)
    print(f"  Total sentences   : {wer_result['num_sentences']}")
    print(f"  Total ref tokens  : {wer_result['num_ref_tokens']}")
    print(f"  ─────────────────────────────────────")
    print(f"  Substitutions     : {wer_result['substitutions']:6d}  "
          f"({100.0 * wer_result['substitutions'] / max(wer_result['num_ref_tokens'], 1):5.2f}%)")
    print(f"  Deletions         : {wer_result['deletions']:6d}  "
          f"({100.0 * wer_result['deletions'] / max(wer_result['num_ref_tokens'], 1):5.2f}%)")
    print(f"  Insertions        : {wer_result['insertions']:6d}  "
          f"({100.0 * wer_result['insertions'] / max(wer_result['num_ref_tokens'], 1):5.2f}%)")
    print(f"  Correct           : {wer_result['correct']:6d}  "
          f"({100.0 * wer_result['correct'] / max(wer_result['num_ref_tokens'], 1):5.2f}%)")
    print(f"  ─────────────────────────────────────")
    print(f"  WER               : {wer_result['wer']:6.2f}%")
    print(f"  SER               : {wer_result['ser']:6.2f}%  "
          f"({wer_result['num_err_sentences']}/{wer_result['num_sentences']})")
    if wer_result.get('num_hallucinated', 0) > 0:
        print(f"  Hallucinated      : {wer_result['num_hallucinated']:6d}  "
              f"(treated as empty hyp)")
    print("=" * 80)


if __name__ == "__main__":
    import argparse
    import json
    import os

    parser = argparse.ArgumentParser(
        description="Evaluate WER/CER from a JSONL inference output file.",
    )
    parser.add_argument(
        "--input_jsonl", type=str, required=True,
        help="Path to JSONL file with 'reference' and 'hypothesis' fields.",
    )
    parser.add_argument(
        "--model_name", type=str, default="",
        help="Optional model name for the report header.",
    )
    parser.add_argument(
        "--language", type=str, default=None,
        help="Language code (e.g. 'en', 'zh'). "
             "If omitted, auto-detected from the first JSONL entry's 'language' "
             "field, falling back to 'en'.",
    )
    parser.add_argument(
        "--max_hyp_ratio", type=float, default=10.0,
        help="Hallucination filter: if hyp token count > ratio × ref token "
             "count, treat hyp as empty (e.g. 3.0). 0 = disabled (default).",
    )
    args = parser.parse_args()

    if not os.path.exists(args.input_jsonl):
        print(f"Error: {args.input_jsonl} not found.")
        exit(1)

    references = []
    hypotheses = []
    results = []

    with open(args.input_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line)
            results.append(data)
            references.append(data.get("reference", ""))
            hypotheses.append(data.get("hypothesis", ""))

    # Determine language
    language = args.language
    if language is None:
        # Auto-detect from first entry
        if results and "language" in results[0]:
            language = results[0]["language"]
            print(f"[INFO] Auto-detected language: {language}")
        else:
            language = "en"
            print(f"[INFO] No language field found, defaulting to: {language}")

    if language.lower().startswith("en"):
        normalizer_label = "EnglishTextNormalizer"
    elif language.lower().startswith("zh"):
        normalizer_label = "BasicTextNormalizer + OpenCC(t2s) + cn2an(cn2an)"
    else:
        normalizer_label = "BasicTextNormalizer"
    print(f"[INFO] Using Whisper normalizer: {normalizer_label}")

    if args.max_hyp_ratio > 0:
        print(f"[INFO] Hallucination filter: hyp > {args.max_hyp_ratio:.1f}× ref → empty")

    if any(ref.strip() for ref in references):
        wer_result = compute_wer(references, hypotheses, language=language,
                                 max_hyp_ratio=args.max_hyp_ratio)
        print_wer_report(wer_result, model_name=args.model_name)

        # Save WER report to file
        wer_report_path = args.input_jsonl.replace(".jsonl", "_wer.txt")
        with open(wer_report_path, "w", encoding="utf-8") as f:
            f.write(f"WER: {wer_result['wer']:.2f}%\n")
            f.write(f"SER: {wer_result['ser']:.2f}%\n")
            f.write(f"Sentences: {wer_result['num_sentences']}\n")
            f.write(f"Ref tokens: {wer_result['num_ref_tokens']}\n")
            f.write(f"Sub: {wer_result['substitutions']}  "
                    f"Del: {wer_result['deletions']}  "
                    f"Ins: {wer_result['insertions']}\n")
            f.write(f"Language: {language}\n")
            f.write(f"Normalizer: {normalizer_label}\n")
            if wer_result.get('num_hallucinated', 0) > 0:
                f.write(f"Hallucinated: {wer_result['num_hallucinated']} "
                        f"(max_hyp_ratio={args.max_hyp_ratio:.1f})\n")
            f.write(f"\nPer-sentence WER:\n")
            for i, (sample, ps) in enumerate(zip(results, wer_result['per_sentence'])):
                hall_tag = " [HALLUCINATION]" if ps.get('hallucination') else ""
                f.write(f"[{i}] WER={ps['wer']:.2f}%  "
                        f"(S={ps['sub']} D={ps['del']} I={ps['ins']} "
                        f"ref_len={ps['ref_tokens']}){hall_tag}\n")
                f.write(f"     REF: {sample.get('reference', '')}\n")
                f.write(f"     HYP: {sample.get('hypothesis', '')}\n")
                if ps.get('hallucination'):
                    f.write(f"     *** Hyp too long (raw: {ps.get('raw_hyp_len', '?')} "
                            f"chars vs {ps.get('raw_ref_len', '?')} ref chars) "
                            f"→ treated as empty\n")
                if ps.get('norm_ref') != sample.get('reference', '') or \
                   ps.get('norm_hyp') != sample.get('hypothesis', ''):
                    f.write(f"     REF(norm): {ps.get('norm_ref', '')}\n")
                    f.write(f"     HYP(norm): {ps.get('norm_hyp', '')}\n")
        print(f"\nWER report saved to {wer_report_path}")
    else:
        print("\n[INFO] No reference text provided in the JSONL, skipping WER computation.")
