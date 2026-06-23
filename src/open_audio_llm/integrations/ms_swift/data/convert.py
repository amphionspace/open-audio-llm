"""Lhotse -> ShareGPT converter for AmphionASR training data.

Emits a single unified ShareGPT sample per supervision.  Three optional
prompt parameters are layered on top of a common template:

  * enrollment audio -- presence driven by ``custom.enrollment_audio``.
    Never masked probabilistically: a mixed-speaker recording without an
    enrollment reference is an ill-defined task for the model.
  * hotwords -- sampled from ``custom.hotwords`` and controlled by
    ``prompt_hotword_prob``.  Missing / masked hotwords omit the line.
  * language -- read directly from each supervision's ``language`` field
    (lhotse already carries it per-utterance), normalized to a canonical
    English full name via ``normalize_language`` (e.g. ``zh`` -> ``Chinese``,
    ``en`` -> ``English``), and controlled by ``prompt_language_prob``.
    Missing / ``N/A`` language omits the line.

Assistant answer is always the bare transcription text.

Usage::

    python -m open_audio_llm.integrations.ms_swift.data.convert \\
        --config config.json --out-dir output/
"""

import argparse
import logging
import os
import random
from multiprocessing import Pool
from pathlib import Path
from typing import List, Optional

import soundfile as sf
try:
    import ujson as json
except ImportError:  # pragma: no cover - optional speed-up
    import json

from .hotwords import (
    build_hotword_char_index,
    build_hotword_pinyin_index,
    build_hotwords_for_sample,
    collect_hotword_pool,
    is_valid_hotword,
    retrieve_hard_negatives_online,
)
from .lhotse_reader import (
    detect_lhotse_format,
    infer_recordings_path,
    load_cuts_gz,
    load_jsonl_gz,
    load_supervisions_auto,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Unified prompt assembly
# ---------------------------------------------------------------------------

_AUDIO_PLACEHOLDER = "<audio>"

# ---------------------------------------------------------------------------
# Language normalization
# ---------------------------------------------------------------------------
#
# Lhotse manifests carry heterogeneous language tags across datasets:
# GigaSpeech stores ``"English"`` (full name), while the TS-ASR pipeline
# and many other sources store ISO 639-1 short codes such as ``"zh"`` /
# ``"en"``.  We normalize everything to a canonical English full name
# (e.g. ``"Chinese"``) so the prompt carries a consistent signal and
# downstream filtering / stats stay clean.

# Lower-cased input token -> canonical English name (Title Case).  Unknown
# values pass through untouched so training data never silently loses a
# language tag; an unmapped value just emits a one-time worker warning.
_LANGUAGE_NORMALIZE_MAP = {
    # Chinese variants
    "zh": "Chinese",
    "zho": "Chinese",
    "chi": "Chinese",
    "cmn": "Chinese",
    "zh-cn": "Chinese",
    "zh_cn": "Chinese",
    "zh-hans": "Chinese",
    "zh-hant": "Chinese",
    "zh-tw": "Chinese",
    "zh-hk": "Chinese",
    "chinese": "Chinese",
    "mandarin": "Chinese",

    # English variants
    "en": "English",
    "eng": "English",
    "en-us": "English",
    "en_us": "English",
    "en-gb": "English",
    "en_gb": "English",
    "english": "English",

    # Japanese
    "ja": "Japanese",
    "jpn": "Japanese",
    "jp": "Japanese",
    "japanese": "Japanese",

    # Korean
    "ko": "Korean",
    "kor": "Korean",
    "korean": "Korean",

    # Common European languages
    "fr": "French",  "fra": "French",  "fre": "French",  "french": "French",
    "de": "German",  "deu": "German",  "ger": "German",  "german": "German",
    "es": "Spanish", "spa": "Spanish", "spanish": "Spanish",
    "ru": "Russian", "rus": "Russian", "russian": "Russian",
    "pt": "Portuguese", "por": "Portuguese", "portuguese": "Portuguese",
    "it": "Italian", "ita": "Italian", "italian": "Italian",
    "nl": "Dutch",   "nld": "Dutch",   "dutch": "Dutch",
    "tr": "Turkish", "tur": "Turkish", "turkish": "Turkish",

    # Other common ASR targets
    "ar": "Arabic",  "ara": "Arabic",  "arabic": "Arabic",
    "hi": "Hindi",   "hin": "Hindi",   "hindi": "Hindi",
    "vi": "Vietnamese", "vie": "Vietnamese", "vietnamese": "Vietnamese",
    "th": "Thai",    "tha": "Thai",    "thai": "Thai",
    "id": "Indonesian", "ind": "Indonesian", "indonesian": "Indonesian",
    "ms": "Malay",   "msa": "Malay",   "malay": "Malay",
}

_UNMAPPED_LANG_WARNED: set = set()


def normalize_language(raw: Optional[str]) -> Optional[str]:
    """Map a heterogeneous language tag to its canonical English name.

    Returns ``None`` when the input is empty / ``"N/A"`` (so callers can
    skip the ``Language:`` line), the mapped canonical form for known
    tokens, or the original (trimmed) string for unknown values.  An
    unknown value also triggers a one-time warning per worker so the
    mapping can be extended when new datasets arrive.
    """
    if raw is None:
        return None
    key = raw.strip()
    if not key or key.upper() == "N/A":
        return None
    mapped = _LANGUAGE_NORMALIZE_MAP.get(key.lower())
    if mapped is not None:
        return mapped
    if key not in _UNMAPPED_LANG_WARNED:
        _UNMAPPED_LANG_WARNED.add(key)
        logger.warning(
            "Unknown language tag %r; keeping original value. "
            "Extend _LANGUAGE_NORMALIZE_MAP in convert.py to normalize it.",
            key,
        )
    return key


def build_unified_instruction(
    *,
    has_enrollment: bool,
    language: Optional[str],
    hotwords_str: Optional[str],
) -> str:
    """Assemble the unified user instruction.

    Parameters
    ----------
    has_enrollment
        Whether an enrollment clip precedes the target/mixed audio.
        When ``True`` the first ``<audio>`` is enrollment, the trailing
        ``<audio>`` is the target clip.  When ``False`` only the target
        clip is present.
    language
        Language tag for the ``Language:`` line.  ``None`` / empty omits
        the entire line.
    hotwords_str
        Comma-separated hotword list for the ``Hotwords:`` line.
        ``None`` / empty / ``"N/A"`` omits the entire line.
    """
    lines: List[str] = []
    if has_enrollment:
        lines.append("Given the speaker's voice:" + _AUDIO_PLACEHOLDER)
        lines.append(
            "Transcribe what this speaker says in the following audio.")
    else:
        lines.append("Transcribe the following audio.")

    if language:
        lines.append(f"Language: {language}")

    if hotwords_str and hotwords_str != "N/A":
        lines.append(f"Hotwords: {hotwords_str}")

    return "\n".join(lines) + _AUDIO_PLACEHOLDER


def build_qwen3_asr_system(
    *,
    has_enrollment: bool,
    hotwords_str: Optional[str],
) -> str:
    """Build the system-turn content for the Qwen3-ASR training format.

    The original Qwen3-ASR ``chat_template.json`` puts **all text** in the
    system turn and **all audio tokens** in the user turn (user text is
    entirely ignored).  Language is NOT included here — the original Qwen3-ASR
    controls language via ``force_language`` on the assistant side, not via
    the input prompt.

    * Enrollment marker: "Given the speaker's voice in the first audio."
    * Hotwords:          "Hotwords: word1,word2"

    Returns an empty string when neither field is present
    (basic ASR: model transcribes without any additional context).

    User-turn content: use :func:`build_qwen3_asr_user`.
    """
    lines: List[str] = []
    if has_enrollment:
        # Tell the model that the first audio is the speaker reference.
        # All audio appears as bare tokens in user; text must live here.
        lines.append("Given the speaker's voice in the first audio.")
    if hotwords_str and hotwords_str != "N/A":
        lines.append(f"Hotwords: {hotwords_str}")
    return "\n".join(lines)


def build_qwen3_asr_user(*, has_enrollment: bool) -> str:
    """Build the user-turn content for the Qwen3-ASR training format.

    The user turn contains **only** ``<audio>`` placeholders — no text at
    all.  This matches the original ``chat_template.json`` which ignores any
    user text and renders only audio tokens in the user turn.

    For enrollment, the enrollment audio placeholder comes first, followed
    by the target / mixed audio placeholder.
    """
    if has_enrollment:
        return _AUDIO_PLACEHOLDER + _AUDIO_PLACEHOLDER   # enroll + target
    return _AUDIO_PLACEHOLDER


# ---------------------------------------------------------------------------
# Conversation format
# ---------------------------------------------------------------------------

_VALID_OUTPUT_FORMATS = ("sft", "grpo", "qwen3_asr")


def build_conversation(
    instruction,
    answer,
    audios,
    output_format="sft",
    *,
    sample_type: Optional[str] = None,
    system: Optional[str] = None,
):
    """Build a ShareGPT-format conversation sample.

    Parameters
    ----------
    instruction
        User-turn content.  For ``"sft"`` / ``"grpo"`` this is the full
        instruction string (text + ``<audio>`` placeholders).  For
        ``"qwen3_asr"`` this must contain **only** ``<audio>`` placeholders
        (no text) — see :func:`build_qwen3_asr_user`.
    answer
        Assistant-turn content (transcription text or GRPO solution).
    audios
        Ordered list of audio file paths matching the ``<audio>`` placeholders.
    output_format
        One of ``"sft"``, ``"grpo"``, ``"qwen3_asr"``.
    sample_type
        Optional metadata tag (ms-swift ignores unknown top-level keys).
    system
        System-turn content.  Required (and meaningful) only for
        ``"qwen3_asr"`` format.  For ``"sft"`` / ``"grpo"`` this parameter
        is ignored; those formats have no system turn.
    """
    if output_format not in _VALID_OUTPUT_FORMATS:
        raise ValueError(
            f"output_format must be one of {_VALID_OUTPUT_FORMATS}, "
            f"got {output_format!r}"
        )
    if isinstance(audios, str):
        audios = [audios]

    if output_format == "grpo":
        sample = {
            "messages": [{"role": "user", "content": instruction}],
            "solution": answer,
            "audios": audios,
        }
    elif output_format == "qwen3_asr":
        # Qwen3-ASR format: system has all text, user has ONLY audio tokens.
        # The system turn is always rendered (empty string for basic ASR).
        sample = {
            "messages": [
                {"role": "system",    "content": system or ""},
                {"role": "user",      "content": instruction},  # audio placeholders only
                {"role": "assistant", "content": answer},
            ],
            "audios": audios,
        }
    else:  # "sft"
        sample = {
            "messages": [
                {"role": "user", "content": instruction},
                {"role": "assistant", "content": answer},
            ],
            "audios": audios,
        }

    if sample_type:
        # Top-level metadata so downstream eval / log scripts can bucket by
        # positive vs negative_silence vs negative_distractor without having
        # to re-parse the user prompt.  ms-swift drops unknown top-level keys
        # silently, so this is a no-op for the trainer itself.
        sample["sample_type"] = sample_type
    return sample


def build_unified_sample(
    *,
    text: str,
    audios: List[str],
    instruction: str,
    output_format: str = "sft",
    sample_type: Optional[str] = None,
):
    """Wrap a unified sample as a ShareGPT conversation with bare answer.

    For hard-negative cuts (``sample_type`` starts with ``"negative"``)
    ``text`` is expected to be the empty string -- the model should learn
    to emit nothing.  We pass it through verbatim instead of dropping the
    sample, which would silently throw away the anti-hallucination signal.
    """
    return build_conversation(
        instruction=instruction,
        answer=text,
        audios=audios,
        output_format=output_format,
        sample_type=sample_type,
    )


# ---------------------------------------------------------------------------
# Text validation
# ---------------------------------------------------------------------------

def _longest_same_char_run(text):
    if not text:
        return 0
    best = cur = 1
    prev = text[0]
    for ch in text[1:]:
        if ch == prev:
            cur += 1
            if cur > best:
                best = cur
        else:
            cur = 1
            prev = ch
    return best


def is_abnormal_transcription(text, max_text_chars, max_same_char_run,
                              max_top_char_ratio):
    if not text:
        return True
    if len(text) > max_text_chars:
        return True
    if _longest_same_char_run(text) > max_same_char_run:
        return True
    if max_top_char_ratio is not None and 0 < max_top_char_ratio <= 1:
        counts = {}
        for ch in text:
            counts[ch] = counts.get(ch, 0) + 1
        if max(counts.values()) / len(text) > max_top_char_ratio:
            return True
    return False


# ---------------------------------------------------------------------------
# Audio segment extraction
# ---------------------------------------------------------------------------

def _audio_is_usable(path: str, min_dur_s: float) -> bool:
    """Header-only sanity: file exists, opens, has positive duration.

    Uses ``soundfile.info`` (no PCM decode) so the cost per call is one
    NAS read of the WAV/FLAC/etc. header. Returns ``False`` for missing
    files, unreadable formats, zero-frame files, and clips shorter than
    ``min_dur_s``. The duration floor exists to keep the audio encoder
    out of its degenerate-batch corner case (modeling_qwen3_asr.py:690
    indexes a chunk_lengths tensor whose size collapses to 0 when every
    sample in the batch has feature_lens == 0).
    """
    try:
        info = sf.info(path)
    except Exception:
        return False
    if info.frames <= 0:
        return False
    sr = max(int(info.samplerate or 0), 1)
    return (info.frames / sr) >= float(min_dur_s)


def extract_audio_segment(src_path, dst_path, start, duration,
                          *, target_sr=16000, rec_id=None,
                          min_dur_s: float = 0.0):
    """Extract ``[start, start + duration)`` from *src_path* to *dst_path*.

    When *rec_id* is provided the fully-decoded waveform is memoised per
    worker via an LRU-1 cache: consecutive supervisions that share the
    same recording reuse the in-memory buffer instead of re-opening and
    re-decoding the source file.  This is crucial for streaming codecs
    (e.g. opus) where libsndfile's ``seek`` is not random-access and has
    to decode from the previous frame, turning "open + seek + read short
    span" into an O(full-file) operation -- so a recording with N
    supervisions would be decoded O(N) times in the naive loop.  Sorting
    supervisions by ``recording_id`` upstream gives near-100% hit rate
    and reduces decode work to O(num_recordings).
    """
    global _w_audio_cache
    if os.path.exists(dst_path):
        if os.path.getsize(dst_path) > 0:
            return True
        os.remove(dst_path)
    try:
        if (rec_id is not None and _w_audio_cache is not None
                and _w_audio_cache[0] == rec_id):
            samples, sr = _w_audio_cache[1], _w_audio_cache[2]
        else:
            samples, sr = sf.read(
                src_path, dtype='float32', always_2d=False)
            if rec_id is not None:
                _w_audio_cache = (rec_id, samples, sr)
        start_frame = int(start * sr)
        n_frames = int(duration * sr)
        if start_frame >= len(samples):
            logger.warning(
                f"start ({start:.2f}s) beyond file length "
                f"({len(samples)/sr:.2f}s): {src_path}")
            return False
        data = samples[start_frame:start_frame + n_frames]
        if data.size == 0:
            return False
        if min_dur_s > 0 and (data.size / sr) < min_dur_s:
            return False
        os.makedirs(os.path.dirname(dst_path) or '.', exist_ok=True)
        sf.write(dst_path, data, sr, subtype='PCM_16')
        return True
    except Exception as e:
        logger.warning(f"Failed to extract segment: {e}")
        return False


# ---------------------------------------------------------------------------
# Multiprocess worker infrastructure
# ---------------------------------------------------------------------------

_w_recordings = None
_w_hotword_pool = None
_w_hard_neg_map = None
_w_char_index = None
_w_pinyin_index = None
_w_params = None
# LRU-1 per-worker cache of the last decoded waveform -- keyed by
# recording_id.  Populated lazily by extract_audio_segment; see that
# function for why this matters (opus seek is not O(1)).
_w_audio_cache = None

_STATUS_OK = 0
_STATUS_NO_REC = 1
_STATUS_NO_AUDIO = 2
_STATUS_BAD_TEXT = 3
_STATUS_SEG_FAIL = 4
_STATUS_SKIP = 5


def _lhotse_worker_init(seed):
    global _w_audio_cache
    random.seed(seed + os.getpid())
    _w_audio_cache = None


def _needs_segment_slicing(sup, rec):
    start = sup.get('start', 0.0)
    duration = sup.get('duration')
    rec_duration = rec.get('duration')
    if start > 0.0:
        return True
    if duration is not None and rec_duration is not None:
        if duration < rec_duration - 0.05:
            return True
    return False


def _process_one_unified(item):
    """Unified worker for one supervision.

    Independent decisions per sample:

      * enrollment: strictly driven by ``custom.enrollment_audio`` presence.
      * hotwords: sampled via ``build_hotwords_for_sample`` (which respects
        ``prompt_hotword_prob``) when the source supplies a hotword pool.
        Skipped for hard-negative samples (no candidates exist there).
      * language: taken from the supervision's own ``language`` field
        (lhotse populates it per-utterance) and kept with probability
        ``prompt_language_prob``; otherwise omitted from the prompt.

    Hard-negative supervisions (``custom.sample_type`` starting with
    ``"negative"`` -- i.e. ``negative_silence`` / ``negative_distractor``
    produced by ``prepare_tsasr_data.py``) are kept *with empty text*.
    Discarding them would throw away the anti-hallucination signal:
    they exist precisely so the model learns to stay silent when the
    enrolled speaker is absent from the mix.
    """
    sup_id, sup = item
    p = _w_params

    rec_id = sup.get('recording_id', sup_id)
    rec = _w_recordings.get(rec_id)
    if rec is None:
        return (_STATUS_NO_REC, None)

    full_audio_path = rec['sources'][0]['source']
    if p['audio_sanity_check']:
        if not _audio_is_usable(full_audio_path, p['min_audio_duration_s']):
            return (_STATUS_NO_AUDIO, None)

    custom = sup.get('custom', {}) or {}
    enrollment_path = custom.get('enrollment_audio')
    has_enrollment = bool(enrollment_path)
    if has_enrollment and p['audio_sanity_check']:
        if not _audio_is_usable(enrollment_path, p['min_audio_duration_s']):
            return (_STATUS_NO_AUDIO, None)

    sample_type = (custom.get('sample_type') or 'positive').strip()
    is_negative = sample_type.startswith('negative')

    if _needs_segment_slicing(sup, rec):
        seg_start = sup.get('start', 0.0)
        seg_duration = sup.get('duration')
        if seg_duration is None:
            return (_STATUS_SEG_FAIL, None)
        seg_path = os.path.join(p['segment_dir'], f"{sup_id}.wav")
        if not extract_audio_segment(
            src_path=full_audio_path, dst_path=seg_path,
            start=seg_start, duration=seg_duration,
            rec_id=rec_id,
            min_dur_s=p['min_audio_duration_s'],
        ):
            return (_STATUS_SEG_FAIL, None)
        mixed_path = seg_path
    else:
        mixed_path = full_audio_path

    text = sup.get('text', '').strip()
    if is_negative:
        # Anti-hallucination samples: empty text is the *target*. Force the
        # answer to "" defensively (in case the source mistakenly carries
        # whitespace) and skip both the empty-skip and abnormal-text gates.
        text = ""
    else:
        if not text:
            return (_STATUS_SKIP, None)
        if is_abnormal_transcription(
            text=text,
            max_text_chars=p['max_text_chars'],
            max_same_char_run=p['max_same_char_run'],
            max_top_char_ratio=p['max_top_char_ratio'],
        ):
            return (_STATUS_BAD_TEXT, None)

    hotwords_str: Optional[str] = None
    # Hotword injection only makes sense when there is a transcription to
    # bias toward.  Skip it entirely on hard-negative samples.
    if (not is_negative) and _w_hotword_pool:
        raw_hw = custom.get('hotwords', []) or []
        if not isinstance(raw_hw, list):
            raw_hw = []
        real_hotwords = [
            h for h in raw_hw if isinstance(h, str)
            and is_valid_hotword(
                h, p['max_hotword_len'],
                p['min_hotword_len'], p['min_hotword_words'])
        ]

        if _w_char_index is not None:
            sample_hard_neg = retrieve_hard_negatives_online(
                text=text,
                hotword_pool=_w_hotword_pool,
                char_index=_w_char_index,
                exclude=set(real_hotwords),
                top_k=p.get('online_hard_neg_top_k', 30),
                pinyin_index=_w_pinyin_index,
                real_hotwords=real_hotwords,
            )
        else:
            sample_hard_neg = _w_hard_neg_map.get(sup_id)

        sampled = build_hotwords_for_sample(
            real_hotwords=real_hotwords,
            hotword_pool=_w_hotword_pool,
            max_hotwords=p['max_hotwords'],
            prompt_hotword_prob=p['prompt_hotword_prob'],
            hard_negatives=sample_hard_neg,
            hard_neg_ratio=p['hard_neg_ratio'],
            miss_prob=p['miss_prob'],
            min_hotword_len=p['min_hotword_len'],
            min_hotword_words=p['min_hotword_words'],
            min_hotwords=p.get('min_hotwords', 1),
        )
        if sampled and sampled != "N/A":
            hotwords_str = sampled

    source_lang = normalize_language(sup.get('language'))
    if source_lang and random.random() < p['prompt_language_prob']:
        effective_lang: Optional[str] = source_lang
    else:
        effective_lang = None

    audios = ([enrollment_path, mixed_path] if has_enrollment
              else [mixed_path])

    output_format = p.get('output_format', 'sft')

    if output_format == 'qwen3_asr':
        # Qwen3-ASR format: all text → system, only audio placeholders → user.
        # Mirrors the original chat_template.json which ignores user text.
        system_content = build_qwen3_asr_system(
            has_enrollment=has_enrollment,
            hotwords_str=hotwords_str,
        )
        user_content = build_qwen3_asr_user(has_enrollment=has_enrollment)
        sample = build_conversation(
            instruction=user_content,
            answer=text,
            audios=audios,
            output_format='qwen3_asr',
            system=system_content,
            sample_type=sample_type,
        )
    else:
        # Amphion-4B format (sft / grpo): all text + audio placeholders → user.
        instruction = build_unified_instruction(
            has_enrollment=has_enrollment,
            language=effective_lang,
            hotwords_str=hotwords_str,
        )
        sample = build_unified_sample(
            text=text,
            audios=audios,
            instruction=instruction,
            output_format=output_format,
            sample_type=sample_type,
        )
    return (_STATUS_OK, json.dumps(sample, ensure_ascii=False) + '\n')


# ---------------------------------------------------------------------------
# Converter
# ---------------------------------------------------------------------------

class LhotseSharegptConverter:
    """Lhotse -> ShareGPT JSONL converter with unified prompt template.

    Both target-speaker (with enrollment) and single-speaker (optional
    hotwords) sources can be mixed in a single ``data_infos`` list; the
    worker decides the prompt shape from each supervision's ``custom``
    fields.  Train with both source types together so the model learns
    to condition on enrollment presence.
    """

    def __init__(self, config_path, out_dir):
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        self.config = config
        self.data_name = config.get("data_name")
        self.data_infos = config.get("data_infos")
        self.output_format = config.get("output_format", "sft")
        self.seed = config.get("seed", 42)
        self.num_workers = config.get("num_workers")

        self.max_hotwords = config.get("max_hotwords", 30)
        self.min_hotwords = config.get("min_hotwords", 1)
        self.prompt_hotword_prob = config.get(
            "prompt_hotword_prob", config.get("hotword_prob", 0.8))
        self.prompt_language_prob = config.get("prompt_language_prob", 0.8)
        self.answer_hotword_ratio = config.get("answer_hotword_ratio")
        self.max_hotword_len = config.get("max_hotword_len", 8)
        self.min_hotword_len = config.get("min_hotword_len", 2)
        self.min_hotword_words = config.get("min_hotword_words", 1)
        self.hard_neg_ratio = config.get("hard_neg_ratio", 0.7)
        self.miss_prob = config.get("miss_prob", 0.0)
        self.online_hard_neg = config.get("online_hard_neg", False)
        self.online_hard_neg_top_k = config.get("online_hard_neg_top_k", 30)

        if "lang_na_prob" in config:
            logger.warning(
                "`lang_na_prob` is deprecated and no longer used; "
                "set `prompt_language_prob` (probability of keeping the "
                "Language line) instead. Ignoring lang_na_prob=%s.",
                config["lang_na_prob"])
        task = config.get("task")
        if task not in (None, "", "unified"):
            logger.warning(
                "`task=%r` is no longer used; the unified converter "
                "decides enrollment / hotwords / language per-sample.",
                task)

        # ``check_audio_exist`` is the legacy switch (file-existence only);
        # ``audio_sanity_check`` is its successor and additionally enforces
        # ``min_audio_duration_s`` via ``soundfile.info``. Either key turns
        # the worker-side checks on, so old configs keep working unchanged.
        self.audio_sanity_check = config.get(
            "audio_sanity_check", config.get("check_audio_exist", False))
        self.min_audio_duration_s = float(
            config.get("min_audio_duration_s", 0.1))
        self.max_text_chars = config.get("max_text_chars", 256)
        self.max_same_char_run = config.get("max_same_char_run", 32)
        self.max_top_char_ratio = config.get("max_top_char_ratio", 0.85)

        self.out_dir = Path(out_dir).resolve()
        self.save_path = self.out_dir / f"{self.data_name}.jsonl"

    def _validate_data_infos(self):
        errors = []
        for idx, data_info in enumerate(self.data_infos):
            tag = f"data_infos[{idx}]"
            sup_path = data_info.get("supervisions")
            if not sup_path:
                errors.append(f"{tag}: 'supervisions' is missing or empty")
            elif not os.path.exists(sup_path):
                errors.append(f"{tag}: supervisions not found: {sup_path}")
            else:
                fmt = detect_lhotse_format(sup_path)
                if fmt != "cuts":
                    rec_path = data_info.get("recordings")
                    if rec_path is None:
                        try:
                            infer_recordings_path(sup_path)
                        except (FileNotFoundError, ValueError) as e:
                            errors.append(
                                f"{tag}: cannot resolve recordings: {e}")
                    elif not os.path.exists(rec_path):
                        errors.append(
                            f"{tag}: recordings not found: {rec_path}")

            extra_pool = data_info.get("extra_hotword_pool")
            if extra_pool and not os.path.exists(extra_pool):
                errors.append(
                    f"{tag}: extra_hotword_pool not found: {extra_pool}")

            hard_neg = data_info.get("hard_negatives")
            if hard_neg and not os.path.exists(hard_neg):
                errors.append(
                    f"{tag}: hard_negatives not found: {hard_neg}")

        if errors:
            for err in errors:
                logger.error(err)
            raise FileNotFoundError(
                f"Validation failed with {len(errors)} error(s) -- "
                f"fix the config before running. See errors above.")
        logger.info(
            f"Pre-flight validation passed for "
            f"{len(self.data_infos)} data_info(s)")

    def _classify_hotword_ids(self, supervisions):
        has_hw, no_hw = [], []
        for sid, sup in supervisions.items():
            custom = sup.get('custom', {}) or {}
            raw_hw = custom.get('hotwords', [])
            valid = (
                isinstance(raw_hw, list)
                and any(
                    isinstance(h, str)
                    and is_valid_hotword(
                        h, self.max_hotword_len,
                        self.min_hotword_len, self.min_hotword_words)
                    for h in raw_hw))
            (has_hw if valid else no_hw).append(sid)
        return has_hw, no_hw

    @staticmethod
    def _compute_global_quotas(scan_results, ratio):
        H = sum(len(s['has_hw_ids']) for s in scan_results)
        N = sum(len(s['no_hw_ids']) for s in scan_results)

        if H == 0 or N == 0:
            logger.warning(
                f"  Global pre-scan: {H} with hotwords, {N} without "
                f"-- answer_hotword_ratio ignored")
            for s in scan_results:
                s['hw_quota'] = len(s['has_hw_ids'])
                s['no_hw_quota'] = len(s['no_hw_ids'])
            return

        cur = H / (H + N)
        if cur > ratio:
            H_target = round(N * ratio / (1 - ratio))
            for s in scan_results:
                s['hw_quota'] = round(
                    H_target * len(s['has_hw_ids']) / H)
                s['no_hw_quota'] = len(s['no_hw_ids'])
        elif cur < ratio:
            N_target = round(H * (1 - ratio) / ratio)
            for s in scan_results:
                s['hw_quota'] = len(s['has_hw_ids'])
                s['no_hw_quota'] = round(
                    N_target * len(s['no_hw_ids']) / N)
        else:
            for s in scan_results:
                s['hw_quota'] = len(s['has_hw_ids'])
                s['no_hw_quota'] = len(s['no_hw_ids'])

        H_after = sum(s['hw_quota'] for s in scan_results)
        N_after = sum(s['no_hw_quota'] for s in scan_results)
        logger.info(
            f"  Global quotas: {H} -> {H_after} with hw, "
            f"{N} -> {N_after} without "
            f"(target ratio={ratio}, "
            f"actual={H_after / (H_after + N_after):.4f})")

    def run(self):
        global _w_recordings, _w_hotword_pool, _w_hard_neg_map
        global _w_char_index, _w_pinyin_index, _w_params, _w_audio_cache

        self._validate_data_infos()
        random.seed(self.seed)
        os.makedirs(os.path.dirname(self.save_path) or '.', exist_ok=True)

        segment_dir = str(self.out_dir / f"{self.data_name}_segments")
        n_workers = (self.num_workers
                     if self.num_workers and self.num_workers >= 1
                     else (os.cpu_count() or 4))
        logger.info(f"Lhotse parallel workers: {n_workers}")

        # -- Phase 1: global pre-scan (only with answer_hotword_ratio) --
        scan_results = None
        if self.answer_hotword_ratio is not None:
            logger.info(
                f"Phase 1: pre-scanning for "
                f"answer_hotword_ratio={self.answer_hotword_ratio}")
            scan_results = []
            for idx, data_info in enumerate(self.data_infos):
                sup_path = data_info.get("supervisions")
                num = data_info.get("num", -1)

                fmt = detect_lhotse_format(sup_path)
                if fmt == 'cuts':
                    supervisions, _ = load_cuts_gz(sup_path)
                else:
                    supervisions = load_jsonl_gz(sup_path)

                has_hw_ids, no_hw_ids = self._classify_hotword_ids(
                    supervisions)
                del supervisions

                n_total = len(has_hw_ids) + len(no_hw_ids)
                if 0 < num < n_total:
                    frac = len(has_hw_ids) / n_total
                    cap_hw = round(num * frac)
                    cap_no = num - cap_hw
                    random.shuffle(has_hw_ids)
                    random.shuffle(no_hw_ids)
                    has_hw_ids = has_hw_ids[:cap_hw]
                    no_hw_ids = no_hw_ids[:cap_no]
                else:
                    random.shuffle(has_hw_ids)
                    random.shuffle(no_hw_ids)

                scan_results.append({
                    'has_hw_ids': has_hw_ids,
                    'no_hw_ids': no_hw_ids,
                })
                logger.info(
                    f"  [{idx}] {os.path.basename(sup_path)}: "
                    f"{len(has_hw_ids)} hw + {len(no_hw_ids)} no_hw")

            self._compute_global_quotas(
                scan_results, self.answer_hotword_ratio)

        # -- Phase 2: process each source --
        with open(self.save_path, "w", encoding="utf-8") as fout:
            for idx, data_info in enumerate(self.data_infos):
                sup_path = data_info.get("supervisions")
                rec_path = data_info.get("recordings")
                extra_pool_path = data_info.get("extra_hotword_pool")
                hard_neg_path = data_info.get("hard_negatives")
                num = data_info.get("num", -1)

                fmt = detect_lhotse_format(sup_path)
                if fmt == 'cuts':
                    logger.info(f"Detected cuts format: {sup_path}")
                    supervisions, recordings = load_cuts_gz(sup_path)
                    logger.info(
                        f"  Extracted {len(supervisions)} supervisions, "
                        f"{len(recordings)} recordings")
                else:
                    if rec_path is None:
                        rec_path = infer_recordings_path(sup_path)
                    logger.info(f"Loading recordings: {rec_path}")
                    recordings = load_jsonl_gz(rec_path)
                    logger.info(f"  Loaded {len(recordings)} recordings")
                    logger.info(f"Loading supervisions: {sup_path}")
                    supervisions = load_jsonl_gz(sup_path)
                    logger.info(f"  Loaded {len(supervisions)} supervisions")

                hotword_pool = collect_hotword_pool(
                    supervisions, self.max_hotword_len,
                    self.min_hotword_len, self.min_hotword_words)
                logger.info(
                    f"  Hotword pool: {len(hotword_pool)} unique "
                    f"(len={self.min_hotword_len}~{self.max_hotword_len})")

                if extra_pool_path:
                    logger.info(
                        f"Loading extra hotword pool: {extra_pool_path}")
                    extra_sups = load_supervisions_auto(extra_pool_path)
                    extra_pool = collect_hotword_pool(
                        extra_sups, self.max_hotword_len,
                        self.min_hotword_len, self.min_hotword_words)
                    hotword_pool = sorted(
                        set(hotword_pool) | set(extra_pool))
                    logger.info(
                        f"  Combined hotword pool: "
                        f"{len(hotword_pool)} unique")
                    del extra_sups

                hard_neg_map = {}
                if hard_neg_path and not self.online_hard_neg:
                    logger.info(
                        f"Loading hard negatives: {hard_neg_path}")
                    with open(hard_neg_path, 'r', encoding='utf-8') as f:
                        hard_neg_map = json.load(f)
                    logger.info(
                        f"  Loaded hard negatives for "
                        f"{len(hard_neg_map)} samples")

                if self.online_hard_neg and hotword_pool:
                    logger.info("Building online hard-neg indices...")
                    char_idx = build_hotword_char_index(hotword_pool)
                    pinyin_idx = build_hotword_pinyin_index(hotword_pool)
                    logger.info(
                        f"  Char index: {len(char_idx)} entries, "
                        f"Pinyin index: "
                        f"{'enabled' if pinyin_idx else 'disabled'}")
                else:
                    char_idx = None
                    pinyin_idx = None

                if scan_results is not None:
                    q = scan_results[idx]
                    hw_ids = q['has_hw_ids'][:q['hw_quota']]
                    no_ids = q['no_hw_ids'][:q['no_hw_quota']]
                    keep = set(hw_ids) | set(no_ids)
                    selected_sups = {
                        sid: supervisions[sid] for sid in keep
                        if sid in supervisions}
                    logger.info(
                        f"  Quota: {len(hw_ids)} hw + {len(no_ids)} "
                        f"no_hw = {len(selected_sups)} selected")
                elif 0 < num < len(supervisions):
                    # Honour `num` up-front via random subsample instead
                    # of a post-hoc ``break written>=num``: that way the
                    # sort-by-recording_id step (needed for the waveform
                    # cache) doesn't deterministically pick the first N
                    # recordings in recording_id order, and progress
                    # counters + in-flight worker tasks reflect the real
                    # target size.
                    #
                    # ``random.sample`` is 5-10x faster than
                    # ``shuffle + slice`` when num << len(supervisions)
                    # because it only does ~num swaps on the pool copy
                    # instead of a full Fisher-Yates over the whole
                    # manifest.
                    keys = random.sample(
                        list(supervisions.keys()), num)
                    selected_sups = {k: supervisions[k] for k in keys}
                    logger.info(
                        f"  Subsampled to num={num} "
                        f"(from {len(supervisions)} total)")
                else:
                    selected_sups = supervisions

                _w_recordings = recordings
                _w_hotword_pool = hotword_pool
                _w_hard_neg_map = hard_neg_map
                _w_char_index = char_idx
                _w_pinyin_index = pinyin_idx
                _w_params = {
                    'audio_sanity_check': self.audio_sanity_check,
                    'min_audio_duration_s': self.min_audio_duration_s,
                    'max_text_chars': self.max_text_chars,
                    'max_same_char_run': self.max_same_char_run,
                    'max_top_char_ratio': self.max_top_char_ratio,
                    'max_hotwords': self.max_hotwords,
                    'min_hotwords': self.min_hotwords,
                    'prompt_hotword_prob': self.prompt_hotword_prob,
                    'prompt_language_prob': self.prompt_language_prob,
                    'hard_neg_ratio': self.hard_neg_ratio,
                    'miss_prob': self.miss_prob,
                    'min_hotword_len': self.min_hotword_len,
                    'min_hotword_words': self.min_hotword_words,
                    'max_hotword_len': self.max_hotword_len,
                    'online_hard_neg_top_k': self.online_hard_neg_top_k,
                    'segment_dir': segment_dir,
                    'output_format': self.output_format,
                }

                written = 0
                skipped = {s: 0 for s in (
                    _STATUS_NO_REC, _STATUS_NO_AUDIO,
                    _STATUS_BAD_TEXT, _STATUS_SEG_FAIL, _STATUS_SKIP)}
                total = len(selected_sups)
                processed = 0
                buf = []

                # Sort by recording_id so all supervisions of the same
                # source audio are consecutive in the input stream.  With
                # imap_unordered's default chunking this lets each worker
                # reuse the LRU-1 waveform cache in extract_audio_segment,
                # turning O(N_supervisions) full-file decodes into
                # O(N_recordings) -- a ~18x reduction for GigaSpeech
                # (~18 segments per opus podcast) and critical for any
                # streaming-codec source.
                sorted_items = sorted(
                    selected_sups.items(),
                    key=lambda kv: (
                        kv[1].get('recording_id', kv[0]),
                        kv[1].get('start', 0.0),
                    ),
                )

                with Pool(
                    n_workers, initializer=_lhotse_worker_init,
                    initargs=(self.seed,),
                ) as pool:
                    for status, line in pool.imap_unordered(
                        _process_one_unified, sorted_items,
                        chunksize=512,
                    ):
                        processed += 1
                        if processed % 10000 == 0 or processed == total:
                            logger.info(
                                f"  Processing {processed}/{total} "
                                f"({100 * processed // total}%) "
                                f"| written={written}")

                        if status == _STATUS_OK:
                            buf.append(line)
                            written += 1
                            if len(buf) >= 5000:
                                fout.writelines(buf)
                                buf.clear()
                            if (scan_results is None
                                    and 0 < num <= written):
                                break
                        elif status in skipped:
                            skipped[status] += 1

                if buf:
                    fout.writelines(buf)
                    buf.clear()

                logger.info(f"Written {written} samples from {sup_path}")
                if skipped[_STATUS_NO_REC]:
                    logger.warning(
                        f"  Skipped {skipped[_STATUS_NO_REC]} "
                        f"(no matching recording)")
                if skipped[_STATUS_NO_AUDIO]:
                    logger.warning(
                        f"  Skipped {skipped[_STATUS_NO_AUDIO]} "
                        f"(audio file not found)")
                if skipped[_STATUS_SEG_FAIL]:
                    logger.warning(
                        f"  Skipped {skipped[_STATUS_SEG_FAIL]} "
                        f"(segment extraction failed)")
                if skipped[_STATUS_BAD_TEXT]:
                    logger.warning(
                        f"  Skipped {skipped[_STATUS_BAD_TEXT]} "
                        f"(abnormal text)")
                if skipped[_STATUS_SKIP]:
                    logger.info(
                        f"  Skipped {skipped[_STATUS_SKIP]} "
                        f"(empty text)")

        _w_recordings = None
        _w_hotword_pool = None
        _w_hard_neg_map = None
        _w_char_index = None
        _w_pinyin_index = None
        _w_params = None
        # Parent-side cache is only populated inside forked workers, but
        # reset defensively here to release the last held waveform.
        _w_audio_cache = None

        logger.info(f"Done. Output: {self.save_path}")
        logger.info(
            f"ms-swift dataset ready: --dataset {self.save_path}")


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    parser = argparse.ArgumentParser(
        description="Convert lhotse manifests to ShareGPT JSONL")
    parser.add_argument("--config", required=True, help="Config JSON path")
    parser.add_argument("--out-dir", required=True, help="Output directory")
    args = parser.parse_args()
    converter = LhotseSharegptConverter(
        config_path=args.config, out_dir=args.out_dir)
    converter.run()


if __name__ == "__main__":
    main()
