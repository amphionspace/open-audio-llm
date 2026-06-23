"""Two-stage hotword retrieval pipeline for the vLLM evaluation path.

Implements the same phase split as ``amphion_ft.eval.infer_retrieve`` but
adapted to AmphionASR object types (lhotse-derived ``items``, the
existing :func:`run_batch` from ``test_vllm_inference``):

  Phase 1 — coarse transcription
    Re-uses the eval vLLM service itself as the retriever (per user
    request "self" mode); ``args.no_hotwords`` is forced True for this
    pass so the prompt has no Hotwords line at all. Output is a hyp per
    utt, never persisted as final results.

  Phase 2 — top-k hotword selection
    :func:`retrieve_hotwords` (from ``retrieve_hotwords.py``) scores
    every candidate against the phase-1 hyp and returns top-k. Pool is
    shared across the dataset.

  Phase 3 — hotword-aware ASR
    Caller injects ``item["hotwords"] = retrieved`` and runs the normal
    :func:`run_batch` path with hotwords visible in the prompt.

References:
  * GLCLAP + GRPO retrieve loop: arxiv.org/abs/2512.21828 (Sec. 2.2)
  * H-PRM pre-retrieval module: arxiv.org/abs/2508.18295

Design notes:
  - ``run_batch`` is passed in as ``run_batch_fn`` to avoid a circular
    import with ``test_vllm_inference.py``.
  - phase1 / phase2 caches default to ``<output_dir>/_retrieve_cache/``;
    pass ``cache_dir=None`` to disable. Cache key is keyed on retriever
    model + dataset label + n + (top_k for phase2), so a sweep over K
    re-uses one phase1 cache.
"""

from __future__ import annotations

import json
import logging
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

from open_audio_llm.integrations.vllm.retrieve_hotwords import retrieve_hotwords

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

_SAFE_PATH_RE = re.compile(r"[^A-Za-z0-9._-]+")

# Schema versions baked into cache filenames. Bumped independently per
# phase so that algorithm changes affecting only one phase don't waste
# the other phase's cache. Old cache files stay on disk but no longer
# collide with the new key, so a fresh run regenerates correct data
# instead of silently serving stale results.
#
# phase1 (no-hotword coarse transcription via vLLM):
#   v1: initial layout
#   v2: pypinyin made a hard dep; CJK retrieve recall on phase1-missed
#       hotwords jumped from ~0.7% to ~50%, so v1 caches are unsafe to
#       reuse (would re-inject the same poorly-retrieved hotwords).
#
# phase2 (text-similarity retrieve):
#   v2: matched phase1 v2 originally (one cache file per K).
#   v3: (1) K-decoupled — cache key drops ``__topk{K}`` so a sweep over
#       multiple K values shares one phase2 file (top_k_max worth of
#       retrieved hotwords cached, callers slice [:K] themselves);
#       (2) substring/LCS scoring switched from difflib to rapidfuzz
#       in retrieve_hotwords.py, which alters long-tail ranking enough
#       that mixing v2/v3 outputs in a sweep would smear results.
_PHASE1_VERSION = "v2"
_PHASE2_VERSION = "v3"


def _safe_for_path(s: str) -> str:
    """Make a string filesystem-safe: collapse / and unicode to ``_``."""
    return _SAFE_PATH_RE.sub("_", str(s)).strip("_") or "x"


def make_cache_paths(
    cache_dir: Optional[str | Path],
    *,
    model: str,
    dataset_label: str,
    n_items: int,
    pool_tag: str = "",
) -> tuple[Optional[Path], Optional[Path]]:
    """Return (phase1_path, phase2_path) under *cache_dir*; both None when
    *cache_dir* is falsy (cache disabled).

    Both files are K-agnostic: phase1 has always been independent of K,
    and phase2 was decoupled in v3 by having ``phase_retrieve`` cache
    ``top_k_max`` worth of retrieved hotwords once and letting callers
    slice ``retrieved[:spec_K]`` for each sweep K. Filenames embed the
    per-phase version so old/incompatible caches don't silently shadow
    new runs (e.g. an EN sweep can pick up a brand-new v3 phase2 cache
    even while a long-running v2 evaluation is still on disk).

    ``pool_tag`` is an optional suffix appended to the phase2 filename
    to distinguish caches built from different hotword pools (e.g. the
    manifest-derived pool vs. an external ``--retrieve-hotword-pool-file``).
    Phase1 is pool-independent (it is a no-hotword coarse transcription)
    so its filename is unaffected.
    """
    if not cache_dir:
        return None, None
    base = Path(cache_dir)
    safe_model = _safe_for_path(model)
    safe_ds = _safe_for_path(dataset_label)
    p1 = base / f"{safe_model}__{safe_ds}__n{n_items}.phase1.{_PHASE1_VERSION}.json"
    p2_pool = f"__{pool_tag}" if pool_tag else ""
    p2 = base / f"{safe_model}__{safe_ds}__n{n_items}{p2_pool}.phase2.{_PHASE2_VERSION}.json"
    return p1, p2


def _get_text_emb_cache_path(
    cache_dir: Optional[str | Path],
    *,
    adapter_ckpt: str,
    biasing_tsv_file: Optional[str] = None,
    hotword_pool_file: Optional[str] = None,
) -> Optional[Path]:
    """Return the text-embedding cache path (.npz); None when cache_dir is falsy.

    The cache key is hash(adapter_ckpt + pool_source) — intentionally excludes
    dataset/n_items so the same text embeddings are shared across test-clean and
    test-other when they use the same checkpoint and pool.
    """
    if not cache_dir:
        return None
    import hashlib
    key_parts = [adapter_ckpt]
    if biasing_tsv_file:
        key_parts.append(f"tsv:{biasing_tsv_file}")
    elif hotword_pool_file:
        key_parts.append(f"pool:{hotword_pool_file}")
    h = hashlib.md5("|".join(key_parts).encode()).hexdigest()[:12]
    return Path(cache_dir) / f"text_embs__{h}.npz"


def _load_text_emb_cache(path: Path) -> tuple[list[str], "torch.Tensor"]:
    """Load text embeddings from a .npz cache file.

    Returns (words, embs) where embs is a CPU float32 torch.Tensor of shape (V, D).
    """
    import numpy as np
    import torch
    data = np.load(str(path), allow_pickle=True)
    words: list[str] = data["words"].tolist()
    embs = torch.from_numpy(data["embs"].astype("float32"))
    logger.info("[neural_retrieve] text emb cache loaded: %s (%d words, dim=%d)",
                path, len(words), embs.shape[1])
    return words, embs


def _save_text_emb_cache(path: Path, words: list[str], embs: "torch.Tensor") -> None:
    """Save text embeddings to a .npz cache file."""
    import numpy as np
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        str(path),
        words=np.array(words, dtype=object),
        embs=embs.cpu().numpy().astype("float32"),
    )
    logger.info("[neural_retrieve] text emb cache saved: %s (%d words)", path, len(words))


def _load_biasing_tsv(tsv_file: str) -> dict[str, list[str]]:
    """Parse a biasing TSV file and return per-utterance candidate word lists.

    TSV format (IS21 Deep Bias style):
        utt_id  \\t  transcript  \\t  JSON[true_hotwords]  \\t  JSON[all_candidates]

    Returns {utt_id: [candidate_word, ...]} where each list contains the full
    biasing list (true hotwords + distractors) for that utterance.
    """
    import json as _json
    per_item: dict[str, list[str]] = {}
    with open(tsv_file, encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 4:
                logger.warning(
                    "[biasing_tsv] line %d has %d fields (expected >=4), skipping: %s",
                    lineno, len(parts), tsv_file,
                )
                continue
            utt_id = parts[0].strip()
            try:
                candidates = _json.loads(parts[3])
            except Exception as e:
                logger.warning(
                    "[biasing_tsv] line %d: failed to parse candidates JSON: %s", lineno, e
                )
                continue
            if not isinstance(candidates, list):
                continue
            per_item[utt_id] = [str(w) for w in candidates if w]
    logger.info(
        "[biasing_tsv] loaded %d utterances from %s", len(per_item), tsv_file
    )
    return per_item


def save_cache(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)
    logger.info("[retrieve] cache saved -> %s", path)


def load_cache(path: Path, *, label: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    logger.info(
        "[retrieve] %s cache hit (%s entries) <- %s",
        label, len(data) if hasattr(data, "__len__") else "?", path,
    )
    return data


# ---------------------------------------------------------------------------
# Phase 1: coarse transcription via the same vLLM service
# ---------------------------------------------------------------------------

def phase_transcribe(
    args,
    items: list[dict],
    *,
    run_batch_fn: Callable,
    phase1_cache: Optional[Path] = None,
    label: str = "phase1",
) -> dict[str, str]:
    """Run a no-hotword pass on *items* and return ``{utt_id: hyp}``.

    Behaviour:

    - Loads from *phase1_cache* if present (no model call).
    - Otherwise temporarily flips ``args.no_hotwords`` / ``args.no_enrollment``
      to True (phase 1 is a "what would the model say without any side
      info" probe) and re-uses :func:`run_batch` for the actual recipe-level
      concurrency / progress / WER bookkeeping. The phase1 WER is logged
      but not surfaced — phase3 results are the ones that matter.
    - Restores ``args`` exactly to its prior state on exit.

    *run_batch_fn* must be the ``run_batch`` function exported from
    ``test_vllm_inference.py``; passed in to avoid a circular import.
    """
    if phase1_cache is not None and phase1_cache.exists():
        return load_cache(phase1_cache, label=label)

    saved = {
        "no_hotwords": args.no_hotwords,
        "no_enrollment": getattr(args, "no_enrollment", False),
        "hotwords_pad_to": getattr(args, "hotwords_pad_to", 0),
        "hotword_mode": getattr(args, "hotword_mode", "none"),
    }
    try:
        args.no_hotwords = True
        if hasattr(args, "no_enrollment"):
            args.no_enrollment = True
        if hasattr(args, "hotwords_pad_to"):
            args.hotwords_pad_to = 0
        if hasattr(args, "hotword_mode"):
            args.hotword_mode = "none"

        # Ensure phase 1 doesn't see any pre-existing hotwords on items.
        # We work on a shallow-copied list of items so the caller's items
        # (which still hold real hotwords for KER/PRR scoring later) stay
        # untouched.
        p1_items = []
        for it in items:
            it2 = dict(it)
            it2["hotwords"] = []
            p1_items.append(it2)

        t0 = time.time()
        results, _summary = run_batch_fn(args, p1_items, f"{label} (phase1)")
        elapsed = time.time() - t0
        logger.info("[retrieve] phase1 done in %.1fs (%d items)",
                    elapsed, len(p1_items))

        hypotheses: dict[str, str] = {}
        for r in results:
            uid = r.get("id")
            if uid is not None:
                hypotheses[uid] = r.get("hyp", "") or ""

        if phase1_cache is not None:
            save_cache(hypotheses, phase1_cache)
        return hypotheses
    finally:
        args.no_hotwords = saved["no_hotwords"]
        if hasattr(args, "no_enrollment"):
            args.no_enrollment = saved["no_enrollment"]
        if hasattr(args, "hotwords_pad_to"):
            args.hotwords_pad_to = saved["hotwords_pad_to"]
        if hasattr(args, "hotword_mode"):
            args.hotword_mode = saved["hotword_mode"]


# ---------------------------------------------------------------------------
# Phase 2: hotword retrieval via text similarity (CPU-bound, ProcessPool)
# ---------------------------------------------------------------------------

def collect_hotword_pool(items: Iterable[dict]) -> list[str]:
    """Pool = sorted unique hotwords across the dataset (per-item ``hotwords``)."""
    pool: set[str] = set()
    for it in items:
        for h in (it.get("hotwords") or []):
            if isinstance(h, str) and h.strip():
                pool.add(h.strip())
    return sorted(pool)


def _retrieve_one(args_tuple):
    """Worker: retrieve up to ``top_k_max`` hotwords for one utt.

    The caller is responsible for slicing ``record["retrieved"][:spec_K]``
    down to the K it actually wants in the prompt (see
    :func:`inject_retrieved_into_items`). Returning the longer list lets
    a single phase2 cache file serve every K in a sweep.

    Returns a record with 'retrieved' (the cached top-k_max list) plus
    'all/real/distractor' splits so downstream metric code can compute
    Recall@K + PRR + PPR + PrRR consistently.
    """
    utt_id, hyp, real_hotwords, hotword_pool, top_k_max, language = args_tuple
    retrieved = retrieve_hotwords(
        hotwords=hotword_pool,
        ctc_text=hyp,
        top_k=top_k_max,
        language=language,
    )
    retrieved_set = set(retrieved)
    real_set = {h for h in (real_hotwords or []) if h}
    return {
        "id": utt_id,
        "retrieved": list(retrieved),
        "all": sorted(retrieved_set),
        "real": sorted(real_set),
        "distractor": sorted(retrieved_set - real_set),
    }


def phase_retrieve(
    items: list[dict],
    hypotheses: dict[str, str],
    *,
    hotword_pool: list[str],
    top_k_max: int,
    workers: int = 8,
    language: str = "auto",
    phase2_cache: Optional[Path] = None,
) -> dict[str, dict]:
    """Retrieve up to ``top_k_max`` hotwords for each item in parallel.

    ``top_k_max`` is the upper bound across whatever sweep of K values
    the caller plans to feed downstream (e.g. for
    ``retrieve_top_k_sweep: [5, 10, 15, 20]`` pass ``top_k_max=20`` or
    higher). The cached ``hw_map[uid]["retrieved"]`` is ordered by score
    so callers can take ``retrieved[:K]`` for any K ≤ top_k_max without
    re-scoring. This decoupling is the reason ``phase2`` cache lives at
    a K-free path in v3 (see :data:`_PHASE2_VERSION`).

    Returns ``{utt_id: {"retrieved": [...], "all": ..., "real": ...,
    "distractor": ...}}``. Each item's GT hotwords come from
    ``item["hotwords"]`` (read-only here — we don't mutate items).

    Falls back to the cache when *phase2_cache* exists; otherwise builds
    a process pool to parallelise the (CPU-bound) similarity scoring.
    """
    if phase2_cache is not None and phase2_cache.exists():
        return load_cache(phase2_cache, label="phase2")

    if top_k_max <= 0:
        raise ValueError(
            f"phase_retrieve: top_k_max must be > 0, got {top_k_max!r}."
        )

    work = []
    for it in items:
        uid = it["id"]
        hyp = hypotheses.get(uid, "")
        real_hw = it.get("hotwords") or []
        work.append((uid, hyp, real_hw, hotword_pool, top_k_max, language))

    hw_map: dict[str, dict] = {}
    n_total = len(work)
    t0 = time.time()
    n_done = 0
    workers = max(1, int(workers))
    if workers == 1:
        # Skip ProcessPool overhead for single-thread / debug usage.
        for item in work:
            r = _retrieve_one(item)
            hw_map[r["id"]] = r
            n_done += 1
            if n_done % 200 == 0 or n_done == n_total:
                elapsed = time.time() - t0
                speed = n_done / elapsed if elapsed > 0 else 0
                eta = (n_total - n_done) / speed if speed > 0 else 0
                sys.stdout.write(
                    f"\r  [retrieve] [{n_done}/{n_total}] "
                    f"{speed:.1f} it/s  ETA={eta:.0f}s"
                )
                sys.stdout.flush()
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_retrieve_one, item): item for item in work}
            for fut in as_completed(futures):
                r = fut.result()
                hw_map[r["id"]] = r
                n_done += 1
                if n_done % 200 == 0 or n_done == n_total:
                    elapsed = time.time() - t0
                    speed = n_done / elapsed if elapsed > 0 else 0
                    eta = (n_total - n_done) / speed if speed > 0 else 0
                    sys.stdout.write(
                        f"\r  [retrieve] [{n_done}/{n_total}] "
                        f"{speed:.1f} it/s  ETA={eta:.0f}s"
                    )
                    sys.stdout.flush()

    elapsed = time.time() - t0
    print()
    logger.info(
        "[retrieve] phase2 done in %.1fs (top_k_max=%d, %d items, pool=%d)",
        elapsed, top_k_max, n_total, len(hotword_pool),
    )

    if phase2_cache is not None:
        save_cache(hw_map, phase2_cache)
    return hw_map


# ---------------------------------------------------------------------------
# Phase 3 prep: inject retrieved hotwords into items
# ---------------------------------------------------------------------------

def inject_retrieved_into_items(
    items: list[dict],
    hw_map: dict[str, dict],
    *,
    top_k: Optional[int] = None,
) -> list[dict]:
    """Return a new items list with ``item["hotwords"]`` replaced by the
    retrieved top-k. Original real hotwords are preserved under
    ``item["hotwords_real"]`` so downstream metric code still has the
    ground truth for KER / PRR computation.

    ``top_k`` is the spec-level K (``args.retrieve_top_k``); when set we
    slice ``record["retrieved"][:top_k]`` here so that callers can share
    one ``hw_map`` (cached at ``top_k_max``) across multiple K values
    without re-running phase2. ``top_k=None`` keeps the full cached list
    (used by older callers / debug paths).
    """
    out: list[dict] = []
    for it in items:
        new_it = dict(it)
        real_hw = list(new_it.get("hotwords") or [])
        record = hw_map.get(new_it["id"])
        retrieved: list[str] = []
        if record is not None:
            retrieved = list(record.get("retrieved") or [])
        if top_k is not None and top_k > 0:
            retrieved = retrieved[:top_k]
        new_it["hotwords_real"] = real_hw
        new_it["hotwords"] = retrieved
        out.append(new_it)
    return out


__all__ = [
    "make_cache_paths",
    "save_cache",
    "load_cache",
    "collect_hotword_pool",
    "phase_transcribe",
    "phase_retrieve",
    "phase_retrieve_neural",
    "inject_retrieved_into_items",
]


# ---------------------------------------------------------------------------
# Phase 2 (neural) — dual-tower audio-text retrieval
# ---------------------------------------------------------------------------

def phase_retrieve_neural(
    items: list[dict],
    hotword_pool: list[str],
    top_k_max: int,
    *,
    adapter_ckpt: str,
    base_model_path: str,
    embed_dim: int = 512,
    adapter_hidden_dim: Optional[int] = None,
    hotword_pool_file: Optional[str] = None,
    biasing_tsv_file: Optional[str] = None,
    language: str = "auto",
    device: str = "cuda",
    batch_size: int = 16,
    batch_text: int = 512,
    num_mel_bins: int = 128,
    cache_dir: Optional[str | Path] = None,
) -> dict[str, dict]:
    """Neural dual-tower phase-2 retrieval.

    Replaces the TF-IDF / BM25 text-match ``phase_retrieve`` with an
    audio-text embedding similarity search using the trained dual-tower
    adapters.

    For each item, the audio is loaded from ``item["mixed_audio"]``,
    encoded by the frozen ``audio_encoder + projector``, passed through
    the audio adapter, and matched against all ``hotword_pool`` words
    encoded by the frozen ``embed_tokens`` + text adapter.  Top-K words
    by cosine similarity are returned in descending score order.

    Parameters
    ----------
    items : list[dict]
        Same format as produced by ``load_lhotse_cuts`` / ``cuts_to_items``.
        Each dict must contain ``"id"``, ``"mixed_audio"``, ``"start"``,
        ``"duration"``, and ``"hotwords"`` (the ground-truth, not used for
        ranking but preserved in the output for downstream scoring).
    hotword_pool : list[str]
        Full candidate vocabulary to rank for each utterance.
    top_k_max : int
        Number of top words to return per utterance (upper bound of the K
        sweep, e.g. 20 for ``retrieve_top_k_sweep: [5,10,15,20]``).
    adapter_ckpt : str
        Path to the adapter checkpoint saved by ``train_retrieval.py``
        (``best_adapter.pt`` or ``last_adapter.pt``).
    base_model_path : str
        Path to the Amphion-4B HF checkpoint directory.
    embed_dim : int
        Must match the value used during training.
    language : str
        Language hint (unused in this implementation, kept for API parity
        with ``phase_retrieve``).
    device : str
    batch_size : int
        Number of utterances encoded per GPU forward pass.
    batch_text : int
        Number of hotword strings tokenised per GPU forward pass.
    num_mel_bins : int
        Mel filter-bank bins; must match the training configuration.
    hotword_pool_file : str or None
        Path to a plain-text file (one word per line) used as the candidate
        pool instead of the ``hotword_pool`` argument.  When provided, the
        file is loaded and deduplicated; the caller-supplied ``hotword_pool``
        is ignored.  This is the recommended setting for realistic evaluation:
        use a large distractor vocabulary (e.g. ``all_rare_words.txt``, ~209k
        words) so the model must discriminate against hard negatives rather
        than only the dataset's own true hotwords (~4k).  When None the
        caller-supplied ``hotword_pool`` is used as-is.
    biasing_tsv_file : str or None
        Path to an IS21 Deep Bias style TSV file with per-utterance candidate
        lists (format: utt_id \\t transcript \\t JSON_true_hw \\t JSON_candidates).
        When provided, the model retrieves top-K hotwords from each utterance's
        own candidate list (typically 100/500/1000/2000 words) rather than a
        shared global pool.  The union of all candidates is encoded once then
        sliced per utterance.  Takes precedence over ``hotword_pool_file``.
    cache_dir : str or Path or None
        Directory for the text-embedding cache.  Text tower embeddings for the
        candidate pool are saved as ``text_embs__<hash>.npz`` keyed by
        (adapter_ckpt, pool_source).  The same cache is shared across datasets
        that use the same checkpoint and pool (e.g. test-clean and test-other
        with the same all_rare_words.txt).  When None, caching is disabled.

    Returns
    -------
    ``{utt_id: {"retrieved": list[str], "all": list[str],
                "real": list[str], "distractor": list[str]}}``
    Compatible with ``inject_retrieved_into_items``.
    """
    if top_k_max <= 0:
        raise ValueError(f"phase_retrieve_neural: top_k_max must be > 0, got {top_k_max!r}.")

    # ------------------------------------------------------------------
    # Candidate pool resolution (priority: biasing_tsv > pool_file > arg)
    # ------------------------------------------------------------------
    per_item_candidates: Optional[dict[str, list[str]]] = None
    pool_word_to_idx: Optional[dict[str, int]] = None

    if biasing_tsv_file:
        # Per-utterance candidate lists from TSV (IS21 Deep Bias style).
        if not Path(biasing_tsv_file).exists():
            raise FileNotFoundError(
                f"phase_retrieve_neural: biasing_tsv_file not found: {biasing_tsv_file}"
            )
        per_item_candidates = _load_biasing_tsv(biasing_tsv_file)
        hotword_pool = sorted({
            w for cands in per_item_candidates.values() for w in cands
        })
        logger.info(
            "[neural_retrieve] per-item biasing TSV: %d unique words across %d utts",
            len(hotword_pool), len(per_item_candidates),
        )
    elif hotword_pool_file:
        pool_path = Path(hotword_pool_file)
        if not pool_path.exists():
            raise FileNotFoundError(
                f"phase_retrieve_neural: hotword_pool_file not found: {pool_path}"
            )
        with open(pool_path, encoding="utf-8") as _f:
            hotword_pool = sorted({line.strip() for line in _f if line.strip()})
        logger.info(
            "[neural_retrieve] hotword_pool loaded from file: %s (%d words)",
            pool_path, len(hotword_pool),
        )
    else:
        logger.info(
            "[neural_retrieve] hotword_pool from items (union of true hotwords): %d words",
            len(hotword_pool),
        )

    import sys
    import numpy as np
    import torch
    from pathlib import Path as _Path

    # Ensure src/ is on sys.path so retrieval.* and open_audio_llm.* are importable.
    _src = _Path(__file__).resolve().parents[3]  # .../src
    if str(_src) not in sys.path:
        sys.path.insert(0, str(_src))

    from transformers import AutoTokenizer
    from lhotse.features import WhisperFbank, WhisperFbankConfig
    from lhotse.audio import AudioSource, Recording
    from retrieval.amphion_dual_tower import AmphionAudioTower, AmphionTextTower
    from retrieval.retrieval_dataset import tokenise_words
    from open_audio_llm.integrations.huggingface.modeling_amphion_asr import (
        AmphionASRForConditionalGeneration, AmphionASRConfig,
    )

    _device = torch.device(device)

    # ------------------------------------------------------------------
    # Load base model + tokenizer + towers
    # ------------------------------------------------------------------
    logger.info("[neural_retrieve] Loading base model from %s…", base_model_path)

    # Detect model_type from config.json to choose the right loader.
    import json as _json
    _config_path = _Path(base_model_path) / "config.json"
    with open(_config_path) as _f:
        _model_type = _json.load(_f).get("model_type", "")
    logger.info("[neural_retrieve] Detected model_type=%r", _model_type)

    _saved_path = sys.path.copy()
    while str(_src) in sys.path:
        sys.path.remove(str(_src))
    try:
        if "qwen3" in _model_type.lower():
            # src/ shadows qwen_asr's internal "import model", so strip first.
            # Import qwen_asr to resolve its internal dependencies, then load
            # the model by pointing directly at the class (avoids relying on
            # AutoModelForCausalLM's MODEL_FOR_CAUSAL_LM_MAPPING which does not
            # include Qwen3ASRForConditionalGeneration).
            import qwen_asr as _  # noqa: F401
            from qwen_asr.core.transformers_backend.modeling_qwen3_asr import (
                Qwen3ASRForConditionalGeneration as _QwenCls,
            )
            base_model = _QwenCls.from_pretrained(
                base_model_path, torch_dtype=torch.float16,
            ).to(_device)
        else:
            base_model = AmphionASRForConditionalGeneration.from_pretrained(
                base_model_path,
                config=AmphionASRConfig.from_pretrained(base_model_path),
                dtype=torch.float16,
            ).to(_device)
    finally:
        sys.path[:] = _saved_path
    for p in base_model.parameters():
        p.requires_grad_(False)

    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)

    audio_tower = AmphionAudioTower(
        base_model, embed_dim=embed_dim,
        adapter_hidden_dim=adapter_hidden_dim,
    ).to(_device)
    text_tower = AmphionTextTower(
        base_model, embed_dim=embed_dim,
        adapter_hidden_dim=adapter_hidden_dim,
    ).to(_device)

    if adapter_ckpt:
        ckpt = torch.load(adapter_ckpt, map_location="cpu")
        audio_tower.adapter.load_state_dict(ckpt["audio_adapter"])
        text_tower.adapter.load_state_dict(ckpt["text_adapter"])
        if "audio_pool" in ckpt:
            audio_tower.pool.load_state_dict(ckpt["audio_pool"])
        if "text_pool" in ckpt:
            text_tower.pool.load_state_dict(ckpt["text_pool"])
        logger.info("[neural_retrieve] Adapter loaded from %s", adapter_ckpt)
    else:
        logger.warning(
            "[neural_retrieve] No adapter_ckpt provided — using randomly initialized adapter "
            "(sanity-check / pipeline test only, results are meaningless)."
        )
    audio_tower.eval()
    text_tower.eval()

    logger.info(
        "[neural_retrieve] Towers loaded. pool=%d  top_k_max=%d  items=%d",
        len(hotword_pool), top_k_max, len(items),
    )

    # ------------------------------------------------------------------
    # Pre-encode all pool words (with .npz text-embedding cache)
    # ------------------------------------------------------------------
    fbank = WhisperFbank(WhisperFbankConfig(num_filters=num_mel_bins))

    text_cache_path = _get_text_emb_cache_path(
        cache_dir,
        adapter_ckpt=adapter_ckpt,
        biasing_tsv_file=biasing_tsv_file,
        hotword_pool_file=hotword_pool_file,
    )

    if text_cache_path is not None and text_cache_path.exists():
        cached_words, pool_embs = _load_text_emb_cache(text_cache_path)
        # Reconcile cached words with current pool (pool may differ if TSV changed)
        if cached_words == hotword_pool:
            logger.info("[neural_retrieve] text emb cache hit, skipping text encoding")
        else:
            logger.warning(
                "[neural_retrieve] text emb cache word list mismatch "
                "(%d cached vs %d current) — re-encoding",
                len(cached_words), len(hotword_pool),
            )
            text_cache_path = None  # force re-encode below
            pool_embs = None
    else:
        pool_embs = None

    if pool_embs is None:
        logger.info("[neural_retrieve] Encoding hotword pool (%d words)…", len(hotword_pool))
        pool_embs_parts: list[torch.Tensor] = []
        with torch.no_grad():
            for i in range(0, len(hotword_pool), batch_text):
                chunk = hotword_pool[i : i + batch_text]
                ids, mask = tokenise_words(chunk, tokenizer, device=_device)
                pool_embs_parts.append(text_tower(ids, mask).cpu())
        pool_embs = torch.cat(pool_embs_parts, dim=0)  # (V, D)
        if text_cache_path is not None:
            _save_text_emb_cache(text_cache_path, hotword_pool, pool_embs)

    # Build word→index mapping for per-utterance candidate slicing.
    if per_item_candidates is not None:
        pool_word_to_idx = {w: i for i, w in enumerate(hotword_pool)}

    # ------------------------------------------------------------------
    # Build FAISS index (global pool mode only; per-utterance uses slicing)
    # ------------------------------------------------------------------
    _faiss_index = None
    if per_item_candidates is None:
        try:
            import faiss  # type: ignore
            dim = pool_embs.shape[1]
            _faiss_index = faiss.IndexFlatIP(dim)
            _faiss_index.add(pool_embs.numpy().astype("float32"))
            logger.info("[neural_retrieve] FAISS IndexFlatIP built (%d words)", len(hotword_pool))
        except ImportError:
            logger.info("[neural_retrieve] faiss not installed, using torch matmul")

    # ------------------------------------------------------------------
    # Encode each utterance and retrieve top-K
    # ------------------------------------------------------------------
    hw_map: dict[str, dict] = {}
    n_total = len(items)
    t0 = time.time()

    for batch_start in range(0, n_total, batch_size):
        batch_items = items[batch_start : batch_start + batch_size]

        # Load and featurise audio
        feats_list: list[torch.Tensor] = []
        valid_items: list[dict] = []
        for it in batch_items:
            try:
                audio_path = it["mixed_audio"]
                start = float(it.get("start") or 0.0)
                duration = it.get("duration")

                import soundfile as sf
                info = sf.info(audio_path)
                sr = info.samplerate
                offset_samples = int(start * sr)
                if duration is not None:
                    num_samples = int(float(duration) * sr)
                    audio, _ = sf.read(
                        audio_path,
                        start=offset_samples,
                        stop=offset_samples + num_samples,
                        dtype="float32",
                        always_2d=False,
                    )
                else:
                    audio, _ = sf.read(
                        audio_path,
                        start=offset_samples,
                        dtype="float32",
                        always_2d=False,
                    )

                # Resample to 16 kHz if needed
                if sr != 16000:
                    import torchaudio
                    audio_t = torch.from_numpy(audio).unsqueeze(0)
                    audio_t = torchaudio.functional.resample(audio_t, sr, 16000)
                    audio = audio_t.squeeze(0).numpy()
                    sr = 16000

                feat = fbank.extract(samples=audio, sampling_rate=sr)
                feats_list.append(torch.from_numpy(feat))
                valid_items.append(it)
            except Exception as e:
                logger.warning("[neural_retrieve] Skipping %s: %s", it.get("id"), e)
                continue

        if not feats_list:
            continue

        # Pad to same length and stack
        feat_lens = torch.tensor([f.shape[0] for f in feats_list], dtype=torch.long)
        max_t = int(feat_lens.max().item())
        n_feats = feats_list[0].shape[1]
        features = torch.zeros(len(feats_list), max_t, n_feats, dtype=torch.float32)
        for i, f in enumerate(feats_list):
            features[i, : f.shape[0], :] = f

        features = features.to(_device)
        feat_lens = feat_lens.to(_device)

        with torch.no_grad():
            a_embs = audio_tower(features, feat_lens).cpu()  # (B', D)

        if _faiss_index is not None:
            # FAISS global-pool search for the whole batch at once
            import faiss  # type: ignore
            topk_k_batch = min(top_k_max, len(hotword_pool))
            _, batch_indices = _faiss_index.search(
                a_embs.numpy().astype("float32"), topk_k_batch
            )  # (B', top_k_max)
        else:
            batch_indices = None

        for i, it in enumerate(valid_items):
            uid = it["id"]
            real_hw = list(it.get("hotwords") or [])
            real_set = {h.lower() for h in real_hw}

            if per_item_candidates is not None and pool_word_to_idx is not None:
                # Per-utterance candidate pool (TSV biasing list mode) — torch matmul slice
                item_cands = per_item_candidates.get(uid)
                if item_cands:
                    g_indices = [
                        pool_word_to_idx[w] for w in item_cands
                        if w in pool_word_to_idx
                    ]
                    if g_indices:
                        item_scores = (a_embs[i].unsqueeze(0) @ pool_embs[g_indices].T).squeeze(0)
                        topk_k = min(top_k_max, len(g_indices))
                        local_idx = item_scores.topk(topk_k).indices.tolist()
                        retrieved = [item_cands[j] for j in local_idx]
                    else:
                        retrieved = []
                else:
                    # Utterance not in TSV → fallback to global pool scores
                    logger.warning(
                        "[neural_retrieve] %s not in biasing_tsv, using global pool", uid
                    )
                    scores = (a_embs[i].unsqueeze(0) @ pool_embs.T).squeeze(0)
                    topk_k = min(top_k_max, scores.shape[0])
                    retrieved = [hotword_pool[j] for j in scores.topk(topk_k).indices.tolist()]
            elif batch_indices is not None:
                # FAISS result (already sorted by score descending)
                retrieved = [hotword_pool[j] for j in batch_indices[i].tolist()]
            else:
                # torch matmul fallback (faiss not installed)
                scores = (a_embs[i].unsqueeze(0) @ pool_embs.T).squeeze(0)
                topk_k = min(top_k_max, scores.shape[0])
                retrieved = [hotword_pool[j] for j in scores.topk(topk_k).indices.tolist()]

            all_hw = sorted(set(retrieved) | set(real_hw))
            distractor = [w for w in retrieved if w.lower() not in real_set]

            hw_map[uid] = {
                "id": uid,
                "retrieved": retrieved,
                "all": all_hw,
                "real": real_hw,
                "distractor": distractor,
            }

        n_done = batch_start + len(valid_items)
        if n_done % 200 == 0 or n_done >= n_total:
            elapsed = time.time() - t0
            speed = n_done / elapsed if elapsed > 0 else 0
            eta = (n_total - n_done) / speed if speed > 0 else 0
            sys.stdout.write(
                f"\r  [neural_retrieve] [{n_done}/{n_total}] "
                f"{speed:.1f} it/s  ETA={eta:.0f}s"
            )
            sys.stdout.flush()

    print()
    elapsed = time.time() - t0
    logger.info(
        "[neural_retrieve] done in %.1fs (top_k_max=%d, %d items, pool=%d)",
        elapsed, top_k_max, n_total, len(hotword_pool),
    )
    return hw_map
