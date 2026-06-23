#!/usr/bin/env python3
"""
AmphionASR vLLM API 推理测试脚本

需要先启动 vLLM 服务:
  bash examples/serve/vllm/serve.sh -m /path/to/model -p 8000

单条音频模式 (asr):
  python src/open_audio_llm/integrations/vllm/test_vllm_inference.py --audio /path/to/audio.wav
  python src/open_audio_llm/integrations/vllm/test_vllm_inference.py --audio /path/to/audio.wav --task asr_zh

Lhotse supervisions 批量 (asr):
  python src/open_audio_llm/integrations/vllm/test_vllm_inference.py --supervisions sups.jsonl.gz
  python src/open_audio_llm/integrations/vllm/test_vllm_inference.py --supervisions sups.jsonl.gz -n 50 --output results.jsonl

Lhotse cuts 批量 (支持 ts_asr 的 enrollment + mixed 双音频):
  python src/open_audio_llm/integrations/vllm/test_vllm_inference.py \
      --cuts /chenmingjie/mingdong/data/lhotse/aidatatang/magicdata_cuts_all.jsonl.gz \
      --task ts_asr -n 50 --output tsasr_magicdata.jsonl

输出 (--output) 格式 (单个 JSON):
  {
    "mode": "...", "task": "...", "model": "...", "generated_at": "...",
    "total": ..., "failed": ..., "evaluated": ...,
    "exact_match": ..., "exact_match_rate": ...,
    "per_language": {"zh-cn": {"wer": ..., "n": ..., ...}, ...},
    "overall":      {"wer": ..., "substitutions": ..., ...},
    "total_duration_s": ..., "total_inflight_s": ..., "rtf": ...,
    "records": [
      {"id": ..., "ref": ..., "hyp": ..., "wer": ..., ...},
      ...
    ]
  }
"""

import argparse
import base64
import gzip
import io
import json
import random
import sys
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import requests
import soundfile as sf
from loguru import logger
from rich.console import Console as _RichConsole
from rich.panel import Panel as _RichPanel
from tqdm.auto import tqdm

# Make ``src/`` importable so we can share registry/plan/IO modules with
# decode.py without duplicating their definitions.
_SRC_DIR = Path(__file__).resolve().parents[3]
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from open_audio_llm.integrations.vllm.compute_wer import (
    compute_classification_metrics as _cw_compute_classification_metrics,
    compute_esc_metrics as _cw_compute_esc_metrics,
    compute_hotword_metrics as _cw_compute_hotword_metrics,
    compute_per_language_wer as _cw_compute_per_language_wer,
    compute_silence_metrics as _cw_compute_silence_metrics,
    compute_wer as _cw_compute_wer,
    normalize_lang_code,
    render_classification_report as _cw_render_classification_report,
    render_esc_report as _cw_render_esc_report,
)
from open_audio_llm.integrations.vllm.eval_io import EvalRunDir, aggregate_overall
from open_audio_llm.integrations.vllm.dataset_registry import (
    _detect_language,
    _resolve_test_cuts_from_dataset_name,
    list_named_datasets,
)
from open_audio_llm.integrations.vllm.eval_plan import TestSpec, load_test_plan
from open_audio_llm.integrations.vllm.infer_retrieve import (
    collect_hotword_pool as _ir_collect_hotword_pool,
    inject_retrieved_into_items as _ir_inject_retrieved_into_items,
    make_cache_paths as _ir_make_cache_paths,
    phase_retrieve as _ir_phase_retrieve,
    phase_retrieve_neural as _ir_phase_retrieve_neural,
    phase_transcribe as _ir_phase_transcribe,
)
from open_audio_llm.integrations.vllm.triton_audio_embed import (
    DEFAULT_TRITON_MODEL,
    TritonAudioEmbedClient,
    stable_audio_embed_uuid,
    tensor_to_vllm_audio_embeds_block,
)

# Dedicated stderr console for "result/report" blocks. Bypasses the loguru
# timestamp + file-line prefix so summaries / banners stay readable. Event
# logs (run/retrieve/errors) keep using ``logger`` because their timestamps
# are useful.
_REPORT_CONSOLE = _RichConsole(
    stderr=True,
    highlight=False,
    soft_wrap=False,
)


def _report_panel(title: str, rows: list[tuple[str, str]]) -> None:
    """Render a labelled result block via ``rich.Panel`` on stderr.

    Each ``(label, text)`` row becomes one line ``"{label:<10s}  {text}"``.
    Title is left-aligned so a stream of consecutive panels (one per
    batch) is easy to scan.
    """
    body = "\n".join(f"{label:<10s}  {text}" for label, text in rows)
    _REPORT_CONSOLE.print(
        _RichPanel(body, title=title, title_align="left", expand=True)
    )


VALID_HOTWORD_MODES = {"none", "random", "retrieve"}

DEFAULT_PORT = 8000
DEFAULT_MODEL = "Amphion-4B"

# ============================================================================
# Backend routing for the vLLM eval client.
#
# 同一个 ``eval_vllm.sh`` 入口要兼容两条完全不同的 vLLM HTTP 协议:
#
# * "chat"          -> /v1/chat/completions  + JSON body w/ ``input_audio``
#   (Qwen3-Omni / Qwen3-ASR / Amphion 等 audio-LLM, base64 内嵌音频)
# * "transcription" -> /v1/audio/transcriptions + multipart/form-data
#   (Whisper / vLLM OpenAI Whisper-compat 端点, 上传 wav 二进制)
#
# ``auto`` 在 main() 健康检查后, 用 /v1/models 列表里是否含 "whisper" 关键字
# 自动选 transcription, 这样用户只需 ``vllm serve openai/whisper-large-v3``
# 就能复用同一份 eval_vllm.sh, 不用额外手动切换。显式传 ``-b chat`` /
# ``-b transcription`` 时跳过自动判定 (覆盖名字里恰好含 whisper 但实际是
# multimodal LLM 的边界情况)。
#
# TRANSCRIPTION_SUPPORTED_TASKS 是 plan-mode 下 whisper 后端允许跑的 spec
# 任务白名单 (单音频 ASR + asr_hotwords); ts_asr (enrollment+mixed 双音频) /
# ser / sec / esc 都因为 whisper 本身不支持多音频和分类/描述 prompt 而被
# run_plan 在 spec 入口处 skip 而不是真的跑出失真指标。
# ============================================================================

BACKENDS = ("auto", "chat", "transcription")
ENCODER_SOURCES = ("vllm", "triton")

_TRITON_CLIENT_LOCAL = threading.local()
WHISPER_KEYWORDS = ("whisper",)
TRANSCRIPTION_SUPPORTED_TASKS = {"asr", "asr_en", "asr_zh", "asr_hotwords"}

# Whisper 接受 ISO-639-1 两位语种码 (zh / en / ja / ...), 仓库内部用的是
# Lhotse 风格的 zh-cn / zh-tw / en 等; 这里只显式列已知映射, 未知码退化为
# "前缀截两位字母" 兜底, 再不行就传 None 让 whisper 自检测语种 —— 强行映射成
# "en" 会让中文测集 WER 直接被语种判定打废。
WHISPER_LANG_MAP = {
    "zh-cn": "zh", "zh-tw": "zh", "zh": "zh",
    "en": "en", "ja": "ja", "ko": "ko",
    "de": "de", "fr": "fr", "es": "es", "it": "it",
    "ru": "ru", "pt": "pt", "ar": "ar", "hi": "hi",
    "tr": "tr", "nl": "nl", "pl": "pl", "vi": "vi", "th": "th",
}


def build_base_url(port: int) -> str:
    """根据端口构造本机 vLLM base url。"""
    return f"http://localhost:{port}"

# ============================================================================
# 统一任务模板 (与训练端 src/train.py:TASK_PROMPTS 对齐)
#
# ASR / TS-ASR 的 user 文本由 enrollment (有/无) / language / hotwords 组合:
#
#     Given the speaker's voice:<audio>          # 仅 has_enrollment
#     Transcribe what this speaker says in the following audio.
#     Language: [lang]                           # 可选
#     Hotwords: [hw1,hw2,...]                    # 可选
#     <audio>                                    # 始终存在 (mixed)
#
# SER / SEC / ESC 在训练侧是固定 instruction + 单段音频, 不带 enrollment /
# Language / Hotwords (见 src/train.py 注释 "prompts deliberately omit
# language information"), 推理端按相同形状下发:
#
#     Classify the emotion of the following audio.<audio>           # ser
#     Describe the emotion of the following audio.<audio>           # sec
#     Describe the background acoustic scene only and ignore        # esc
#       spoken content.<audio>
#
# --task 仅作为常见组合的语法糖:
#   auto    : 完全由 item + 显式 --no-* flags 决定 (默认)
#   asr     : 强制 no_enrollment + no_hotwords   (普通 ASR)
#   asr_en  : 同 asr + 固定 language=en
#   asr_zh  : 同 asr + 固定 language=zh-cn
#   ts_asr  : 强制 no_hotwords                    (纯 TS-ASR,沿用训练默认)
#   ser     : 情感分类  (单标签 happy/sad/...)
#   sec     : 情感描述  (一句话情感 caption)
#   esc     : 环境声分类/描述  (audioset_esc)
# ============================================================================

TASK_MODES = (
    "auto", "asr", "asr_en", "asr_zh", "ts_asr", "ser", "sec", "esc",
)

# Two prompt-shape styles supported by the eval client. See
# build_unified_content + _build_content_{swift,train} for the
# byte-level differences and src/open_audio_llm/integrations/vllm/README.md for the
# selection rules per ckpt provenance.
PROMPT_STYLES = ("swift", "train", "qwen3_asr")

# Tasks whose user prompt is a fixed single-instruction sentence (no
# enrollment, no Language line, no hotwords). Sentence text mirrors
# src/train.py:TASK_PROMPTS but with the trailing ``:`` swapped for ``.``
# so vLLM's chat template inserts the input_audio block as a separate
# content item — same convention ASR uses (see ``Transcribe the
# following audio.`` below).
SINGLE_INSTRUCTION_TASKS: dict[str, str] = {
    "ser": "Classify the emotion of the following audio.",
    "sec": "Describe the emotion of the following audio.",
    "esc": (
        "Describe the background acoustic scene only and ignore "
        "spoken content."
    ),
}

# train style mirrors src/train.py:TASK_PROMPTS verbatim (colon-terminated
# instructions, audio token glued onto the colon, NO Language line). Kept
# as a literal table so the prompt shape can be diffed against
# src/train.py at a glance.
_TRAIN_SINGLE_INSTRUCTION_TASKS: dict[str, str] = {
    "ser": "Classify the emotion of the following audio:",
    "sec": "Describe the emotion of the following audio:",
    "esc": (
        "Describe the background acoustic scene only and ignore "
        "spoken content:"
    ),
}


def _resolve_task_flags(task: str, no_enrollment: bool, no_language: bool,
                        no_hotwords: bool, language_override: str | None):
    """Map --task to (no_enrollment, no_hotwords, language_override) defaults;
    explicit --no-* flags always win (they can only narrow, not widen).
    """
    lang_override = language_override
    if task == "asr":
        no_enrollment = True
        no_hotwords = True
    elif task == "asr_en":
        no_enrollment = True
        no_hotwords = True
        lang_override = lang_override or "en"
    elif task == "asr_zh":
        no_enrollment = True
        no_hotwords = True
        lang_override = lang_override or "zh-cn"
    elif task == "ts_asr":
        no_hotwords = True
    elif task in SINGLE_INSTRUCTION_TASKS:
        # Training-time SER/SEC/ESC prompts are intentionally bare: no
        # enrollment, no Language line, no hotwords (see TASK_PROMPTS in
        # src/train.py and the accompanying NOTE).
        no_enrollment = True
        no_hotwords = True
        no_language = True
    return no_enrollment, no_language, no_hotwords, lang_override


def _format_hotwords(hotwords) -> str:
    if not hotwords:
        return ""
    if isinstance(hotwords, str):
        return hotwords.strip()
    return ",".join(h.strip() for h in hotwords if h and str(h).strip())


def _collect_hotword_pool(items: list) -> list[str]:
    """Scan items and return a sorted unique pool of hotwords.

    The pool is built once per batch from the same items the workers will
    consume, so distractors never mention strings the dataset never saw.
    Mirrors :func:`src.decode._collect_hotword_pool` but operates on the
    flat item dicts produced by :func:`cuts_to_items`.
    """
    pool: set[str] = set()
    for it in items:
        for h in (it.get("hotwords") or []):
            if isinstance(h, str) and h.strip():
                pool.add(h.strip())
    return sorted(pool)


def _pad_hotwords(
    real: list[str],
    pool: list[str],
    target_size: int,
    rng: random.Random,
) -> dict:
    """Build the per-sample hotword set: real ∪ random distractors == ``target_size``.

    Behaviour:
      - real hotwords are deduplicated and kept whole;
      - if ``len(real) > target_size`` the extras are dropped (with
        ``rng.sample`` to avoid bias against later words) and a warning
        is logged — caller should bump ``target_size`` if this fires;
      - the gap is filled by sampling ``target_size - len(real)`` words
        from ``pool \\ real``;
      - if the pool is too small to fill the gap, fewer distractors are
        emitted (no fabrication of random tokens) — total may stay
        below ``target_size`` only in this edge case.

    The shuffled ``all`` list is what gets stitched into the prompt. The
    ``real`` / ``distractor`` lists are persisted to transcripts.jsonl
    so downstream slicing can re-derive recall on the real subset.
    """
    real_unique: list[str] = []
    seen: set[str] = set()
    for h in real or []:
        if not isinstance(h, str):
            continue
        h = h.strip()
        if not h or h in seen:
            continue
        seen.add(h)
        real_unique.append(h)

    if len(real_unique) > target_size:
        logger.warning(
            f"_pad_hotwords: real hotwords ({len(real_unique)}) "
            f"exceed target_size ({target_size}); truncating — "
            f"bump --hotwords-pad-to to keep them all."
        )
        real_unique = rng.sample(real_unique, target_size)

    n_distractor = target_size - len(real_unique)
    distractor: list[str] = []
    if n_distractor > 0 and pool:
        avail = [h for h in pool if h not in seen]
        if avail:
            n_take = min(n_distractor, len(avail))
            distractor = rng.sample(avail, n_take)

    all_hw = list(real_unique) + list(distractor)
    rng.shuffle(all_hw)
    return {
        "real": list(real_unique),
        "distractor": list(distractor),
        "all": all_hw,
    }


def build_unified_content(
    item: dict,
    audios_b64: dict,
    *,
    no_enrollment: bool = False,
    no_language: bool = False,
    no_hotwords: bool = False,
    language_override: str | None = None,
    custom_prompt: str | None = None,
    task: str | None = None,
    prompt_style: str = "swift",
    spec_task: str | None = None,
):
    """Dispatch the OpenAI content list based on ``prompt_style``.

    Two prompt shapes are supported (see PROMPT_STYLES):

    * ``"swift"`` (default) — matches the ms-swift unified template
      produced by :func:`open_audio_llm.integrations.ms_swift.data.convert.build_unified_instruction`.
      User content: ``Transcribe ... .\\nLanguage: ...\\nHotwords: ...``
      then audio block. Compatible with ckpts trained via
      :file:`examples/train/sft/production_swift.sh`.

    * ``"train"`` — matches :data:`src.train.TASK_PROMPTS` byte-for-byte:
      ``Hotwords:{hw}\\nTranscribe the following audio:[audio]`` with
      no Language line and a colon glued to the audio block. Compatible
      with ckpts trained via :file:`src/train.py` (icefall-style multitask).

    ``audios_b64`` must hold the relevant slots:
      - mixed audio: key ``"mixed"`` or ``"audio"``
      - enrollment audio: key ``"enrollment"`` (only when used)

    ``custom_prompt`` overrides the built-in template (single-audio tasks
    only) and is intended as an ablation knob / legacy fallback.

    ``task`` selects a fixed single-instruction template for non-ASR
    tasks (``ser`` / ``sec`` / ``esc``); when ``None`` or unset (the ASR
    family / ``auto``) the legacy enrollment + Language + Hotwords logic
    is used.

    ``spec_task`` is the original test-plan task ("asr" / "asr_hotwords"
    / "ts_asr" / ...) before being collapsed to a CLI ``task`` token.
    Train style uses it to keep ``Hotwords:N/A`` line present on
    ``asr_hotwords`` baselines (K=0 sweep) while not polluting pure
    ASR specs that pass ``--no-hotwords``. ``None`` in legacy modes.
    """
    if prompt_style not in PROMPT_STYLES:
        raise ValueError(
            f"prompt_style must be one of {PROMPT_STYLES}, got {prompt_style!r}"
        )
    if prompt_style == "train":
        return _build_content_train(
            item, audios_b64,
            no_enrollment=no_enrollment,
            no_hotwords=no_hotwords,
            custom_prompt=custom_prompt,
            task=task,
            spec_task=spec_task,
        )
    return _build_content_swift(
        item, audios_b64,
        no_enrollment=no_enrollment,
        no_language=no_language,
        no_hotwords=no_hotwords,
        language_override=language_override,
        custom_prompt=custom_prompt,
        task=task,
    )


def _build_content_swift(
    item: dict,
    audios_b64: dict,
    *,
    no_enrollment: bool = False,
    no_language: bool = False,
    no_hotwords: bool = False,
    language_override: str | None = None,
    custom_prompt: str | None = None,
    task: str | None = None,
):
    """Swift-flavour prompt builder (vLLM eval default; matches ms-swift).

    Shape: ``Transcribe the following audio.\\nLanguage: ...\\nHotwords: ...``
    then audio block. Identical to the original pre-refactor implementation
    of :func:`build_unified_content` so behaviour is byte-stable for
    ckpts trained via the ms-swift / GRPO path.
    """
    mixed_b64 = audios_b64.get("mixed") or audios_b64.get("audio")
    if mixed_b64 is None:
        raise KeyError(
            f"unified prompt requires a 'mixed'/'audio' slot, got "
            f"{list(audios_b64)}"
        )

    if custom_prompt is not None:
        return [
            {"type": "text", "text": custom_prompt},
            {
                "type": "input_audio",
                "input_audio": {"data": mixed_b64, "format": "wav"},
            },
        ]

    # SER / SEC / ESC: fixed single-sentence instruction + single audio.
    # No enrollment, Language, or Hotwords lines; ignore item-level fields
    # so prompt is byte-stable across datasets.
    if task in SINGLE_INSTRUCTION_TASKS:
        return [
            {"type": "text", "text": SINGLE_INSTRUCTION_TASKS[task]},
            {
                "type": "input_audio",
                "input_audio": {"data": mixed_b64, "format": "wav"},
            },
        ]

    has_enrollment = (
        (not no_enrollment) and bool(item.get("enrollment_audio"))
    )

    content: list = []
    if has_enrollment:
        enroll_b64 = audios_b64.get("enrollment")
        if enroll_b64 is None:
            raise KeyError(
                "enrollment audio expected but 'enrollment' slot is missing"
            )
        content.append({"type": "text",
                        "text": "Given the speaker's voice:"})
        content.append({
            "type": "input_audio",
            "input_audio": {"data": enroll_b64, "format": "wav"},
        })
        body = (
            "\nTranscribe what this speaker says in the following audio."
        )
    else:
        body = "Transcribe the following audio."

    language = (
        language_override
        if language_override is not None
        else (item.get("language") or "")
    )
    if (not no_language) and language and language.upper() != "N/A":
        body += f"\nLanguage: {language}"

    if not no_hotwords:
        hotwords_str = _format_hotwords(item.get("hotwords"))
        if hotwords_str:
            body += f"\nHotwords: {hotwords_str}"

    content.append({"type": "text", "text": body})
    content.append({
        "type": "input_audio",
        "input_audio": {"data": mixed_b64, "format": "wav"},
    })
    return content


def _build_content_train(
    item: dict,
    audios_b64: dict,
    *,
    no_enrollment: bool = False,
    no_hotwords: bool = False,
    custom_prompt: str | None = None,
    task: str | None = None,
    spec_task: str | None = None,
):
    """Train-flavour prompt builder (matches src/train.py:TASK_PROMPTS).

    Strict byte-for-byte mirror of the templates a model sees during
    training under :file:`src/train.py`:

    * ``asr``        ``Transcribe the following audio:[audio]``
    * ``asr_en``     ``Transcribe the following English audio:[audio]``
    * ``asr_zh``     ``Transcribe the following Chinese audio:[audio]``
    * ``asr_hotwords`` (auto) ``Hotwords:{hw}\\nTranscribe the following audio:[audio]``
    * ``ts_asr``     ``Given the speaker's voice:[enr]\\nTranscribe what this speaker says in the following audio:[mixed]``
    * ``ser/sec/esc`` ``<single instruction>:[audio]``

    Distinguishing features vs swift style:
      - audio block is glued onto the trailing colon (no whitespace);
      - no ``Language:`` line ever (train.py deliberately omits it);
      - Hotwords line precedes the main instruction;
      - ``Hotwords:`` has no space after the colon and uses comma-only joins.

    Hotwords-line policy:
      - real or padded hotwords present → ``Hotwords:hw1,hw2,...``;
      - asr_hotwords spec with the line otherwise dropped (e.g. K=0
        baseline that sets ``no_hotwords=True``) → ``Hotwords:N/A``,
        matching :func:`src.train._build_hotwords_prompt` when
        ``hotword_prob`` does not fire;
      - non-hotwords spec (``--no-hotwords`` on plain ASR data) → no
        Hotwords line at all, exactly the train.py ``"asr"`` template.
    """
    mixed_b64 = audios_b64.get("mixed") or audios_b64.get("audio")
    if mixed_b64 is None:
        raise KeyError(
            f"unified prompt requires a 'mixed'/'audio' slot, got "
            f"{list(audios_b64)}"
        )

    if custom_prompt is not None:
        return [
            {"type": "text", "text": custom_prompt},
            {
                "type": "input_audio",
                "input_audio": {"data": mixed_b64, "format": "wav"},
            },
        ]

    # SER / SEC / ESC: training-side prompt is a single sentence ending
    # with ``:`` immediately followed by the speech token.
    if task in _TRAIN_SINGLE_INSTRUCTION_TASKS:
        return [
            {"type": "text", "text": _TRAIN_SINGLE_INSTRUCTION_TASKS[task]},
            {
                "type": "input_audio",
                "input_audio": {"data": mixed_b64, "format": "wav"},
            },
        ]

    has_enrollment = (
        (not no_enrollment) and bool(item.get("enrollment_audio"))
    )

    # TS-ASR training prompt has NO hotwords line (train.py only renders
    # Hotwords for the asr_hotwords task), so item-level hotwords are
    # ignored here regardless of no_hotwords.
    if has_enrollment:
        enroll_b64 = audios_b64.get("enrollment")
        if enroll_b64 is None:
            raise KeyError(
                "enrollment audio expected but 'enrollment' slot is missing"
            )
        return [
            {"type": "text", "text": "Given the speaker's voice:"},
            {
                "type": "input_audio",
                "input_audio": {"data": enroll_b64, "format": "wav"},
            },
            {
                "type": "text",
                "text": (
                    "\nTranscribe what this speaker says in the "
                    "following audio:"
                ),
            },
            {
                "type": "input_audio",
                "input_audio": {"data": mixed_b64, "format": "wav"},
            },
        ]

    # Hotwords line: only emitted on asr_hotwords-flavour tests; never on
    # plain ASR tests even if the item happens to carry hotwords (that
    # combination wasn't seen during train.py training).
    is_hotwords_spec = (spec_task == "asr_hotwords")
    head = ""
    if is_hotwords_spec:
        if no_hotwords:
            head = "Hotwords:N/A\n"
        else:
            hw_str = _format_hotwords(item.get("hotwords")) or "N/A"
            head = f"Hotwords:{hw_str}\n"

    if task == "asr_en":
        body = "Transcribe the following English audio:"
    elif task == "asr_zh":
        body = "Transcribe the following Chinese audio:"
    else:
        body = "Transcribe the following audio:"

    return [
        {"type": "text", "text": head + body},
        {
            "type": "input_audio",
            "input_audio": {"data": mixed_b64, "format": "wav"},
        },
    ]


def _build_messages_qwen3_asr(
    item: dict,
    audios_b64: dict,
    *,
    no_enrollment: bool = False,
    no_hotwords: bool = False,
) -> list[dict]:
    """Build the full chat messages list for Amphion-1.7B (Qwen3-ASR format).

    Unlike the Amphion-4B ``swift``/``train`` styles (which put both text
    instructions and audio in a single user message), Qwen3-ASR keeps:

    * **system** — plain text with enrollment notice and/or hotwords
    * **user**   — only ``input_audio`` blocks, no text

    This mirrors the training data format produced by
    :func:`open_audio_llm.integrations.ms_swift.data.convert.build_qwen3_asr_system`
    and :func:`build_qwen3_asr_user`.

    Wire format sent to ``/v1/chat/completions``::

        messages = [
            {"role": "system", "content": "Given the speaker's voice in the first audio.\\nHotwords: w1,w2"},
            {"role": "user",   "content": [
                {"type": "input_audio", "input_audio": {"data": "<enroll_b64>", "format": "wav"}},
                {"type": "input_audio", "input_audio": {"data": "<mixed_b64>",  "format": "wav"}},
            ]},
        ]

    If neither enrollment nor hotwords are present, ``system`` is an empty
    string (the model was trained with empty system turns for plain ASR).
    Language is intentionally omitted — Qwen3-ASR's original design does
    not use a Language field and our fine-tune follows suit.
    """
    mixed_b64 = audios_b64.get("mixed") or audios_b64.get("audio")
    if mixed_b64 is None:
        raise KeyError(
            "qwen3_asr prompt requires a 'mixed'/'audio' audio slot, "
            f"got keys: {list(audios_b64)}"
        )

    has_enrollment = (not no_enrollment) and bool(item.get("enrollment_audio"))

    # ── system message ─────────────────────────────────────────────────
    sys_lines: list[str] = []
    if has_enrollment:
        sys_lines.append("Given the speaker's voice in the first audio.")
    if not no_hotwords:
        hotwords_str = _format_hotwords(item.get("hotwords"))
        if hotwords_str:
            sys_lines.append(f"Hotwords: {hotwords_str}")
    system_content = "\n".join(sys_lines)

    # ── user message (audio tokens only) ───────────────────────────────
    user_content: list[dict] = []
    if has_enrollment:
        enroll_b64 = audios_b64.get("enrollment")
        if enroll_b64 is None:
            raise KeyError(
                "enrollment audio expected but 'enrollment' slot is missing"
            )
        user_content.append({
            "type": "input_audio",
            "input_audio": {"data": enroll_b64, "format": "wav"},
        })
    user_content.append({
        "type": "input_audio",
        "input_audio": {"data": mixed_b64, "format": "wav"},
    })

    return [
        {"role": "system", "content": system_content},
        {"role": "user",   "content": user_content},
    ]


# ======================================================================
# Qwen3-ASR 原始输出后处理
#   官方 Qwen3-ASR 的 raw 输出形如:
#       "language Chinese<asr_text>儿子几点接你？"
#   其中 "language <Lang>" 由模型内部做语种检测后附带, 不属于真实转写内容。
#   若直接送进 jiwer 计算 WER, "language" / "chinese" / "asr_text" 会被
#   误判为插入项, 严重拉高指标。
#   以下工具与 https://github.com/QwenLM/Qwen3-ASR 的
#   qwen_asr/inference/utils.py:parse_asr_output 保持一致, 负责:
#     1. 剥离 language/<asr_text> 元信息前缀, 仅保留实际转写
#     2. 修复模型偶发的字符/模式级重复 (>20 次) hallucination
#   对于 Amphion 等不带 <asr_text> tag 的模型, 函数退化为 pass-through,
#   不改变原有行为。
# ======================================================================

_ASR_TEXT_TAG = "<asr_text>"
_LANG_PREFIX = "language "


def _normalize_language_name(language: str) -> str:
    """首字母大写、其余小写的规范化 (与 Qwen3-ASR 一致)."""
    s = (language or "").strip()
    if not s:
        return ""
    return s[:1].upper() + s[1:].lower()


def detect_and_fix_repetitions(text: str, threshold: int = 20) -> str:
    """折叠字符级 / 模式级重复超过 ``threshold`` 次的片段.

    端口自 Qwen3-ASR utils.detect_and_fix_repetitions, 用于消除 decode 退化
    (例如连续 50 个 "啊" 或重复 30 次 "abab"). 正常转写不受影响。
    """

    def fix_char_repeats(s: str, thresh: int) -> str:
        res: list[str] = []
        i, n = 0, len(s)
        while i < n:
            count = 1
            while i + count < n and s[i + count] == s[i]:
                count += 1
            if count > thresh:
                res.append(s[i])
            else:
                res.append(s[i:i + count])
            i += count
        return "".join(res)

    def fix_pattern_repeats(s: str, thresh: int, max_len: int = 20) -> str:
        n = len(s)
        min_repeat_chars = thresh * 2
        if n < min_repeat_chars:
            return s

        i = 0
        result: list[str] = []
        found = False
        while i <= n - min_repeat_chars:
            found = False
            for k in range(1, max_len + 1):
                if i + k * thresh > n:
                    break
                pattern = s[i:i + k]
                valid = True
                for rep in range(1, thresh):
                    start_idx = i + rep * k
                    if s[start_idx:start_idx + k] != pattern:
                        valid = False
                        break
                if valid:
                    end_index = i + thresh * k
                    while (
                        end_index + k <= n
                        and s[end_index:end_index + k] == pattern
                    ):
                        end_index += k
                    result.append(pattern)
                    result.append(
                        fix_pattern_repeats(s[end_index:], thresh, max_len)
                    )
                    i = n
                    found = True
                    break
            if found:
                break
            else:
                result.append(s[i])
                i += 1
        if not found:
            result.append(s[i:])
        return "".join(result)

    text = fix_char_repeats(text, threshold)
    text = fix_pattern_repeats(text, threshold)
    return text


def parse_asr_output(
    raw: str, user_language: str | None = None,
) -> tuple[str, str]:
    """解析 Qwen3-ASR raw 输出为 ``(language, text)``.

    支持的格式:
      - ``language Chinese<asr_text>你好`` -> ("Chinese", "你好")
      - ``<asr_text>你好``                   -> ("", "你好")
      - ``language None<asr_text>``          -> ("", "")  (模型判定静音)
      - ``你好``                              -> ("", "你好")  (无 tag, 其他模型)

    当调用方已在 prompt 中 force language (此时模型不再输出 meta 前缀),
    传入 ``user_language`` 即可直接把 raw 当纯文本返回。
    """
    if raw is None:
        return "", ""
    s = str(raw).strip()
    if not s:
        return "", ""

    s = detect_and_fix_repetitions(s)

    if user_language:
        return user_language, s

    if _ASR_TEXT_TAG not in s:
        return "", s

    meta_part, text_part = s.split(_ASR_TEXT_TAG, 1)

    if "language none" in meta_part.lower():
        return "", text_part.strip()

    lang = ""
    for line in meta_part.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.lower().startswith(_LANG_PREFIX):
            val = line[len(_LANG_PREFIX):].strip()
            if val:
                lang = _normalize_language_name(val)
            break

    return lang, text_part.strip()


# ======================================================================
# Audio helpers
# ======================================================================

def load_audio(path: str, offset: float = 0.0, duration: float = None):
    """Load audio file (or segment), convert to mono 16 kHz."""
    info = sf.info(path)
    sr = info.samplerate
    start_sample = int(offset * sr)
    stop_sample = int((offset + duration) * sr) if duration else None
    audio, sr = sf.read(path, start=start_sample, stop=stop_sample)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != 16000:
        import librosa
        audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
        sr = 16000
    return audio, sr


def audio_to_wav_bytes(audio: np.ndarray, sr: int = 16000) -> bytes:
    """Encode numpy audio array as raw PCM-16 WAV bytes (no base64).

    Shared building block: ``audio_to_base64_wav`` simply base64-encodes
    the output. Pulled out so the transcription backend can upload the
    bytes via multipart without paying a round-trip base64 encode/decode.
    """
    buf = io.BytesIO()
    sf.write(buf, audio, sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def audio_to_base64_wav(audio: np.ndarray, sr: int = 16000) -> str:
    """Encode numpy audio array as base64 WAV string."""
    return base64.b64encode(audio_to_wav_bytes(audio, sr)).decode("utf-8")


def _get_triton_embed_client(url: str, model_name: str) -> TritonAudioEmbedClient:
    """Thread-local Triton client; batch eval uses many worker threads."""

    key = (url, model_name)
    cached = getattr(_TRITON_CLIENT_LOCAL, "audio_embed_client", None)
    if cached is not None and getattr(_TRITON_CLIENT_LOCAL, "audio_embed_key", None) == key:
        return cached
    client = TritonAudioEmbedClient(url=url, model_name=model_name)
    _TRITON_CLIENT_LOCAL.audio_embed_key = key
    _TRITON_CLIENT_LOCAL.audio_embed_client = client
    return client


def _content_with_triton_audio_embeds(
    content,
    *,
    triton_url: str,
    triton_model: str,
    triton_top_k: int = 0,
):
    """Replace OpenAI ``input_audio`` blocks with vLLM ``audio_embeds`` blocks."""

    if not isinstance(content, list):
        return content

    client = _get_triton_embed_client(triton_url, triton_model)
    converted = []
    for block in content:
        if not isinstance(block, dict) or block.get("type") != "input_audio":
            converted.append(block)
            continue
        payload = block.get("input_audio") or {}
        audio_b64 = payload.get("data")
        if not audio_b64:
            converted.append(block)
            continue
        embedding = client.infer_base64_wav(audio_b64, top_k=triton_top_k)
        converted.append(
            tensor_to_vllm_audio_embeds_block(
                embedding.frames,
                uuid=stable_audio_embed_uuid(audio_b64),
            )
        )
    return converted


def _messages_with_triton_audio_embeds(
    messages: list[dict],
    *,
    triton_url: str,
    triton_model: str,
    triton_top_k: int = 0,
) -> list[dict]:
    out = []
    for message in messages:
        new_message = dict(message)
        new_message["content"] = _content_with_triton_audio_embeds(
            message.get("content"),
            triton_url=triton_url,
            triton_model=triton_model,
            triton_top_k=triton_top_k,
        )
        out.append(new_message)
    return out


def _log_triton_hotword_snapshot(args) -> None:
    """Record a small mutable-state snapshot before Triton-bypass eval."""

    if getattr(args, "encoder_source", "vllm") != "triton":
        return
    try:
        client = _get_triton_embed_client(args.triton_url, args.triton_model)
        snapshot = client.list_hotwords(limit=5)
        logger.info(
            "[triton] hotword snapshot: url={} model={} count={} sample={}",
            args.triton_url,
            args.triton_model,
            snapshot.get("hotword_count"),
            snapshot.get("hotwords"),
        )
    except Exception as exc:
        logger.warning("[triton] failed to read hotword snapshot: {}", exc)


def _to_whisper_lang(code: str | None) -> str | None:
    """Map a repo-local lang code (zh-cn / en / ...) to whisper ISO-639-1.

    Returns ``None`` when the code is missing or unmapped so the whisper
    request omits ``language`` and lets the model auto-detect — strictly
    safer than guessing ``"en"`` (which would force English decoding on
    Chinese audio and tank WER).
    """
    if not code:
        return None
    s = str(code).strip().lower()
    if not s:
        return None
    if s in WHISPER_LANG_MAP:
        return WHISPER_LANG_MAP[s]
    head = s.split("-", 1)[0]
    return head if head and len(head) == 2 else None


def _resolve_backend(requested: str, served_names: list[str]) -> str:
    """Pick a backend given an explicit request and the /v1/models list.

    - ``chat`` / ``transcription`` are honoured verbatim (used to override
      auto-detect when a model id misleadingly contains "whisper" but the
      server actually exposes chat-completions, or vice versa).
    - ``auto`` flips to ``transcription`` iff at least one served model id
      contains a :data:`WHISPER_KEYWORDS` keyword (case-insensitive); else
      stays on ``chat``.
    """
    requested = (requested or "auto").lower()
    if requested in ("chat", "transcription"):
        return requested
    if requested != "auto":
        raise ValueError(
            f"unknown backend {requested!r}; expected one of {BACKENDS}"
        )
    for n in (served_names or []):
        nl = (n or "").lower()
        if any(kw in nl for kw in WHISPER_KEYWORDS):
            return "transcription"
    return "chat"


# Model name substrings that imply prompt_style="qwen3_asr"
# (Amphion-1.7B fine-tunes served under names like "Amphion-1.7B").
_QWEN3_ASR_STYLE_KEYWORDS = ("1.7b", "1.7", "qwen3-asr", "qwen3_asr")


def _resolve_prompt_style(requested: str, served_names: list[str]) -> str:
    """Auto-detect prompt style from served model names when unspecified.

    Only triggers when ``requested == "swift"`` (the CLI default) and at
    least one served model name contains a :data:`_QWEN3_ASR_STYLE_KEYWORDS`
    keyword.  Explicit ``"train"`` or ``"qwen3_asr"`` values are returned
    unchanged, so the user can always override.

    The auto-detection is symmetric with the shell-side logic in
    ``eval_vllm.sh`` (which tracks whether ``-s`` was explicitly passed).
    """
    if requested != "swift":
        return requested
    for n in (served_names or []):
        nl = (n or "").lower()
        if any(kw in nl for kw in _QWEN3_ASR_STYLE_KEYWORDS):
            return "qwen3_asr"
    return requested


# ======================================================================
# Lhotse readers
# ======================================================================

def _opener(path: str):
    return gzip.open if str(path).endswith(".gz") else open


def read_jsonl_gz(path: str, limit: int = None) -> list[dict]:
    out = []
    with _opener(path)(path, "rt", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            out.append(json.loads(line))
            if limit is not None and len(out) >= limit:
                break
    return out


def infer_recordings_path(supervisions_path: str) -> str:
    """Infer recordings manifest path from supervisions path."""
    p = Path(supervisions_path)
    name = p.name.replace("supervisions", "recordings")
    candidate = p.parent / name
    if candidate.exists():
        return str(candidate)

    stem = p.name
    for ext in (".gz", ".jsonl"):
        stem = stem.removesuffix(ext)
    base = stem.replace("supervisions", "recordings")
    for suffix in ("_punc", "_raw", "_norm"):
        fallback = p.parent / (base.removesuffix(suffix) + "".join(p.suffixes))
        if fallback.exists():
            return str(fallback)

    raise FileNotFoundError(
        f"无法自动推断 recordings 文件，已尝试: {candidate}\n"
        f"请通过 --recordings 手动指定"
    )


def load_lhotse_pairs(recordings_path: str, supervisions_path: str,
                      limit: int = None):
    """Return list of dicts from lhotse supervisions/recordings manifests.

    Each dict: {id, audio, ref, start, duration}. 当传入 limit 时,流式读取
    supervisions,在累计到 limit 条即停止,避免全量载入。
    """
    # supervisions 决定采样数量,先把需要的 recording_id 收集起来
    sups = read_jsonl_gz(supervisions_path, limit=limit)
    needed_rids = {s.get("recording_id", s["id"]) for s in sups}

    rec_by_id = {}
    with _opener(recordings_path)(recordings_path, "rt", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            if r["id"] in needed_rids:
                rec_by_id[r["id"]] = r["sources"][0]["source"]
                if len(rec_by_id) == len(needed_rids):
                    break

    pairs = []
    for s in sups:
        rid = s.get("recording_id", s["id"])
        audio_path = rec_by_id.get(rid)
        if audio_path is None:
            continue
        custom = s.get("custom") or {}
        pairs.append({
            "id": s["id"],
            "audio": audio_path,
            "ref": s.get("text", ""),
            "start": s.get("start", 0.0),
            "duration": s.get("duration", None),
            "language": s.get("language", ""),
            "task": custom.get("task", "asr"),
            "sample_type": custom.get("sample_type", "positive"),
            "hotwords": custom.get("hotwords") or [],
            "enrollment_audio": custom.get("enrollment_audio"),
        })
    return pairs


def load_lhotse_cuts(cuts_path: str, limit: int = None):
    """Load Lhotse cuts (.jsonl[.gz]) into a flat list.

    Each cut's supervision may carry custom task metadata (e.g. ts_asr
    with enrollment_audio). 当传入 limit 时,读取到 limit 条有效 cut 即停止,
    避免大文件被一次性载入内存。
    """
    items = []
    with _opener(cuts_path)(cuts_path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            cut = json.loads(line)
            rec = cut.get("recording") or {}
            mixed_path = None
            if rec:
                srcs = rec.get("sources") or []
                if srcs:
                    mixed_path = srcs[0].get("source")
            if not mixed_path:
                continue

            sups = cut.get("supervisions") or []
            if not sups:
                continue
            sup = sups[0]
            custom = sup.get("custom") or {}
            items.append({
                "id": cut.get("id", sup.get("id")),
                "mixed_audio": mixed_path,
                "enrollment_audio": custom.get("enrollment_audio"),
                "ref": sup.get("text", ""),
                "start": cut.get("start", 0.0),
                "duration": cut.get("duration", None),
                "language": sup.get("language", ""),
                "task": custom.get("task", "asr"),
                "sample_type": custom.get("sample_type", "positive"),
                "hotwords": custom.get("hotwords") or [],
            })
            if limit is not None and len(items) >= limit:
                break
    return items


# Cuts whose duration is shorter than this are skipped (mirrors
# decode.py::_MIN_DECODE_DURATION). Most encoders need at least a couple
# of frames of audio to produce a non-degenerate fbank.
_MIN_DECODE_DURATION = 0.04


def _trim_and_filter_cuts(cuts, *, keep_empty_text: bool):
    """Trim each cut to its supervisions and drop too-short / empty-ref cuts.

    For raw ``recordings + supervisions`` datasets (gigaspeech, wenetspeech,
    aishell, ...) one cut from ``CutSet.from_manifests`` spans the entire
    long recording and carries N supervisions. Without this trim we'd only
    test the first supervision per recording and slice the wrong audio
    range, throwing away most of the test data.

    For pre-trimmed datasets (e.g. ts_hw_test) this is a no-op.

    Mirrors ``src/decode.py::_trim_and_filter`` so both eval paths see the
    exact same per-utterance set when running the same plan.
    """
    cuts = cuts.trim_to_supervisions(keep_overlapping=False).filter(
        lambda c: c.duration > _MIN_DECODE_DURATION
    )
    if not keep_empty_text:
        cuts = cuts.filter(
            lambda c: len(c.supervisions) > 0
            and (c.supervisions[0].text or "").strip() != ""
        )
    return cuts


def cuts_to_items(cuts, limit: int = None) -> list[dict]:
    """Convert a Lhotse ``CutSet`` (in-memory or lazy) into the same
    item-dict format that :func:`load_lhotse_cuts` yields.

    Used by the plan-mode entry point so cuts loaded via
    :func:`dataset_registry._resolve_test_cuts_from_dataset_name`
    flow through the existing :func:`run_batch` pipeline unchanged.
    """
    items: list[dict] = []
    for cut in cuts:
        rec = getattr(cut, "recording", None)
        if rec is None:
            continue
        srcs = list(getattr(rec, "sources", []) or [])
        if not srcs:
            continue
        mixed_path = getattr(srcs[0], "source", None)
        if not mixed_path:
            continue
        sups = list(cut.supervisions or [])
        if not sups:
            continue
        sup = sups[0]
        custom = (sup.custom or {}) if sup is not None else {}
        items.append({
            "id": cut.id,
            "mixed_audio": mixed_path,
            "enrollment_audio": custom.get("enrollment_audio"),
            "ref": sup.text or "",
            "start": float(cut.start or 0.0),
            "duration": float(cut.duration) if cut.duration is not None else None,
            "language": sup.language or "",
            "task": custom.get("task", "asr"),
            "sample_type": custom.get("sample_type", "positive"),
            "hotwords": custom.get("hotwords") or [],
        })
        if limit is not None and len(items) >= limit:
            break
    return items


# ======================================================================
# WER / silence metrics (delegates to src/compute_wer.py for byte-identical
# alignment with src/decode.py — no jiwer, no per-script normalizer drift)
# ======================================================================


def resolve_lang_code(lang_str: str) -> str:
    """Map a Lhotse-style language label to the short code expected by
    :func:`src.compute_wer.compute_wer`.

    Accepts ``"zh-CN"`` / ``"zh"`` / ``"Chinese"`` / ``"english"`` / ... and
    returns ``"zh"`` / ``"en"`` / ... matching the rest of the codebase.
    """
    return normalize_lang_code(lang_str, default="en")


def compute_wer(refs: list[str], hyps: list[str], language: str = "en") -> dict:
    """Corpus-level WER via :mod:`src.compute_wer`.

    Returns the legacy schema used throughout this module (WER on a 0–100
    scale matching ``src/compute_wer.compute_wer``):

        {wer, substitutions, deletions, insertions, hits, ref_tokens,
         num_sentences}
    """
    res = _cw_compute_wer(refs, hyps, language=language)
    return {
        "wer": res["wer"],
        "substitutions": res["substitutions"],
        "deletions": res["deletions"],
        "insertions": res["insertions"],
        "hits": res["correct"],
        "ref_tokens": res["num_ref_tokens"],
        "num_sentences": res["num_sentences"],
    }


def normalize_text(text: str, lang_code: str) -> str:
    """Backwards-compat shim: returns the canonical token sequence used by
    :func:`src.compute_wer.compute_wer`. Kept so downstream tooling that
    reads transcripts.jsonl can still see the normalised string.
    """
    from open_audio_llm.integrations.vllm.compute_wer import get_normalizer, tokenize  # local import is fine
    tokens = tokenize(text or "", get_normalizer(lang_code))
    return " ".join(tokens)


# ======================================================================
# vLLM chat-completion client
# ======================================================================

# Process-wide latch: only log the "vLLM rejected hotwords field" warning
# once per run instead of spamming the log line per failed request, the
# usual cause is a vLLM build that predates the hotwords extra_body PR and
# the user only needs to see the upgrade hint once.
_HOTWORDS_UNSUPPORTED_WARNED = False


def call_vllm_transcribe(
    audios_b64: dict,
    item: dict,
    *,
    no_hotwords: bool = False,
    language_override: str | None = None,
    base_url: str | None = None,
    model: str = DEFAULT_MODEL,
    max_tokens: int = 200,
    default_lang: str | None = None,
) -> tuple[str, dict]:
    """Send a single-audio transcription request to vLLM's whisper endpoint.

    Speaks the OpenAI Whisper-compatible ``/v1/audio/transcriptions`` API:
    multipart/form-data with the wav blob as ``file`` and the model name +
    optional ``language`` / ``hotwords`` as form fields. Returns the same
    ``(text, usage)`` shape as the chat path so :func:`_process_one` can
    consume both uniformly; whisper does not report token usage so the
    second tuple slot is best-effort ({} if absent).

    Field policy:
      - ``language``: only sent when :func:`_to_whisper_lang` returns a
        valid ISO-639-1 code; missing / unmapped codes are dropped so
        whisper falls back to its own language detector.
      - ``hotwords``: comma-joined from ``item['hotwords']`` (whatever
        the random-padding / retrieve pipeline put there) and sent as a
        top-level form field — this is the vLLM-specific extra_body
        extension; older vLLM versions return 4xx with the message
        ``unexpected ... hotwords``. We retry once without the field
        and log a single big-letter warning so the run continues but
        the user sees the upgrade hint.

    ``max_tokens`` is not honoured by the whisper endpoint (its decoder
    cap is hard-coded to 448 in vLLM); accepted as a signature-compatible
    no-op so :func:`call_vllm_api` can forward all chat kwargs blindly.
    """
    del max_tokens, default_lang  # kept for signature compatibility

    mixed_b64 = audios_b64.get("mixed") or audios_b64.get("audio")
    if mixed_b64 is None:
        raise KeyError(
            "transcription backend requires a 'mixed' / 'audio' slot, got "
            f"{list(audios_b64)}"
        )
    wav_bytes = base64.b64decode(mixed_b64)

    data: dict[str, str] = {
        "model": model,
        "response_format": "json",
        "temperature": "0",
    }
    lang = _to_whisper_lang(
        language_override if language_override is not None
        else item.get("language")
    )
    if lang:
        data["language"] = lang

    hotwords_str = "" if no_hotwords else _format_hotwords(item.get("hotwords"))
    if hotwords_str:
        data["hotwords"] = hotwords_str

    base_url = base_url or build_base_url(DEFAULT_PORT)
    url = f"{base_url}/v1/audio/transcriptions"

    def _files():
        return {"file": ("audio.wav", io.BytesIO(wav_bytes), "audio/wav")}

    resp = requests.post(url, files=_files(), data=dict(data), timeout=120)

    # vLLM versions older than the hotwords PR reject the field with 400/422;
    # retry once without hotwords to keep the eval flowing instead of dropping
    # every sample, but surface the cause loud-and-clear the first time.
    if (
        not resp.ok
        and resp.status_code in (400, 422)
        and "hotwords" in data
        and "hotword" in (resp.text or "").lower()
    ):
        global _HOTWORDS_UNSUPPORTED_WARNED
        if not _HOTWORDS_UNSUPPORTED_WARNED:
            _HOTWORDS_UNSUPPORTED_WARNED = True
            logger.error(
                "[transcription] vLLM 拒绝 'hotwords' 字段 (status=%d, body=%s); "
                "已退化为无热词请求 — 升级 vLLM 到含 transcription hotwords "
                "支持的版本才能让 whisper 接入仓库的热词链路。",
                resp.status_code, (resp.text or "")[:200],
            )
        data_retry = {k: v for k, v in data.items() if k != "hotwords"}
        resp = requests.post(url, files=_files(), data=data_retry, timeout=120)

    if not resp.ok:
        raise RuntimeError(
            f"vLLM transcription 失败: status={resp.status_code} "
            f"url={url} body={(resp.text or '')[:500]}"
        )

    payload = resp.json()
    text = payload.get("text", "")
    usage = payload.get("usage", {}) or {}
    return text, usage


# ----------------------------------------------------------------------
# Transcription-endpoint adapter (Fun-ASR-Nano-2512 / Kimi-Audio etc.)
# ----------------------------------------------------------------------
#
# 上游 vLLM 把 `Fun-ASR-Nano-2512` 这类模型标成 ``supports_transcription_only``,
# 只挂 ``/v1/audio/transcriptions``, 不挂 ``/v1/chat/completions``. 客户端
# 这边需要分流: 凡是模型名含 "fun-asr" 就改走 transcription endpoint, 用
# multipart 上传 WAV bytes, prompt 字段塞我们的 ``Language:``/``Hotwords:``
# 文本 (server 侧 plugin ``open_audio_llm.integrations.vllm.plugin.funasr`` 会解析并翻成 Fun-ASR 中文 prompt).
#
# Kimi-Audio (MoonshotKimiaForCausalLM) 也走这里: 它是 audio+text 双 stream
# 模型, tokenizer_config.json 故意不带 chat_template (jinja 无法表达双流),
# 走 ``/v1/chat/completions`` 必崩 (transformers>=4.44 抛 ChatTemplateResolutionError).
# vLLM 通过 ``SupportsTranscription`` 接口为它写死了 ASR prompt 拼装,
# 所以客户端只要改走 ``/v1/audio/transcriptions`` 即可正常推理.
#
# 其它 (Amphion-4B / Qwen-Audio 等) 不动, 继续走 chat completions.

_TRANSCRIPTION_ONLY_MODEL_SUBSTRS = (
    "fun-asr-nano",
    "fun-asr-mlt-nano",
    "kimi-audio",
)


def _is_transcription_only_model(model: str) -> bool:
    """True iff vLLM 必须通过 ``/v1/audio/transcriptions`` 而非 chat 端点访问.

    含义有两类:
      1) vLLM 上游显式标 ``supports_transcription_only`` (如 Fun-ASR-Nano);
      2) chat 端点存在但因 tokenizer 缺 chat_template 而走不通
         (如 Kimi-Audio 这类多 stream 模型, 见上文注释).
    两种情况客户端的应对都是: 强制走 transcription endpoint.
    """
    m = (model or "").lower()
    return any(s in m for s in _TRANSCRIPTION_ONLY_MODEL_SUBSTRS)


def _extract_prompt_text_from_content(content) -> str:
    """Concat all ``type=='text'`` chunks in a chat-style content list into a
    single string. Drops ``input_audio`` blocks (they're hauled out
    separately as multipart files).
    """
    if isinstance(content, str):
        return content
    parts: list[str] = []
    for c in content:
        if isinstance(c, dict) and c.get("type") == "text":
            t = c.get("text") or ""
            if t:
                parts.append(t)
    return "\n".join(parts).strip()


def _extract_mixed_audio_b64(content) -> str | None:
    """Return the *last* ``input_audio`` data field — by AmphionASR
    convention the enrollment audio comes first and the mixed/target
    audio last (see :func:`_build_content_swift`). Fun-ASR-Nano-2512 has
    no enrollment input, so we transcribe only the mixed audio.
    """
    if isinstance(content, str):
        return None
    last: str | None = None
    for c in content:
        if isinstance(c, dict) and c.get("type") == "input_audio":
            data = (c.get("input_audio") or {}).get("data")
            if data:
                last = data
    return last


def _call_vllm_transcription(
    *,
    content,
    base_url: str,
    model: str,
    temperature: float = 0.0,
) -> tuple[str, dict]:
    """Send a single transcription request to the vLLM OpenAI-compatible
    ``/v1/audio/transcriptions`` endpoint. Audio bytes are uploaded as
    multipart ``file``; ``Language:`` / ``Hotwords:`` metadata travels in
    the ``prompt`` field for the server to parse.

    We *intentionally do not* set the OpenAI ``language`` field: upstream
    ``SupportsTranscription.validate_language`` would otherwise fall back
    to ``"en"`` whenever the caller omits it, which would mask the
    in-prompt ``Language:`` annotation. Our server-side plugin treats the
    in-prompt line as ground truth.
    """
    audio_b64 = _extract_mixed_audio_b64(content)
    if not audio_b64:
        raise ValueError(
            "transcription-only model selected but the prompt content "
            "has no input_audio block"
        )
    prompt_text = _extract_prompt_text_from_content(content)
    audio_bytes = base64.b64decode(audio_b64)
    files = {"file": ("audio.wav", audio_bytes, "audio/wav")}
    data = {
        "model": model,
        "prompt": prompt_text,
        "temperature": str(temperature),
        "response_format": "json",
    }
    url = f"{base_url}/v1/audio/transcriptions"
    resp = requests.post(url, files=files, data=data, timeout=120)
    resp.raise_for_status()
    body = resp.json()
    text = body.get("text") or ""
    return text, {}


def call_vllm_api(
    audios_b64: dict,
    item: dict,
    *,
    no_enrollment: bool = False,
    no_language: bool = False,
    no_hotwords: bool = False,
    language_override: str | None = None,
    custom_prompt: str | None = None,
    task: str | None = None,
    base_url: str | None = None,
    model: str = DEFAULT_MODEL,
    max_tokens: int = 200,
    prompt_style: str = "swift",
    spec_task: str | None = None,
    backend: str = "chat",
    encoder_source: str = "vllm",
    triton_url: str = "localhost:8000",
    triton_model: str = DEFAULT_TRITON_MODEL,
    triton_top_k: int = 0,
) -> tuple[str, dict]:
    """Dispatch a request to the right vLLM endpoint based on ``backend``.

    ``backend``:
      - ``"chat"`` (default) — assemble :func:`build_unified_content` and
        POST to ``/v1/chat/completions``. Used for Qwen3-Omni / Qwen3-ASR
        / AmphionASR; ``prompt_style`` (swift / train) and the three
        ``no_*`` flags govern the chat content layout exactly as before.
        Models flagged as ``transcription_only`` upstream
        (e.g. Fun-ASR-Nano-2512; see :func:`_is_transcription_only_model`)
        auto-route to :func:`_call_vllm_transcription` so the multipart
        ``prompt`` field carries the ``Language:`` / ``Hotwords:`` lines
        for the server-side ``open_audio_llm.integrations.vllm.plugin.funasr`` plugin to parse.
      - ``"transcription"`` — delegate to :func:`call_vllm_transcribe`
        which speaks the OpenAI Whisper-compatible
        ``/v1/audio/transcriptions`` API. Chat-only knobs
        (``custom_prompt`` / ``task`` / ``prompt_style`` / ``spec_task``
        / enrollment / Language line) are intentionally ignored because
        whisper cannot consume them; the unsupported-task spec skip
        happens upstream in :func:`run_plan`.

    ``backend == "auto"`` is treated as ``"chat"`` here — by the time a
    request is sent ``main()`` should have already resolved ``auto`` via
    :func:`_resolve_backend`. We keep ``auto`` valid as a defensive
    default so callers like single-audio mode don't crash if backend
    resolution gets skipped.

    Chat-path parameters: the three ``no_*`` flags plus
    ``language_override`` give ablation-style control equivalent to
    zeroing out the corresponding training probabilities. ``task``
    selects the single-instruction SER / SEC / ESC templates; falsy /
    ``auto`` means the legacy ASR-family logic is used.
    ``prompt_style`` (``"swift"`` default / ``"train"`` / ``"qwen3_asr"``)
    selects between the prompt-shape families; see
    :func:`build_unified_content` and :func:`_build_messages_qwen3_asr`.
    ``"qwen3_asr"`` targets Amphion-1.7B fine-tunes: text instructions go
    into a ``system`` message and the ``user`` message contains only
    ``input_audio`` blocks (no text), matching the training template.
    ``spec_task`` is the original test-plan task name (used by train style
    to keep the ``Hotwords:`` line on K=0 hotwords baselines).
    """
    encoder_source = (encoder_source or "vllm").lower()
    if encoder_source not in ENCODER_SOURCES:
        raise ValueError(
            f"unknown encoder_source {encoder_source!r}; "
            f"expected one of {ENCODER_SOURCES}"
        )
    if encoder_source == "triton" and backend == "transcription":
        raise ValueError("encoder_source=triton is only supported for chat backend")

    if backend == "transcription":
        return call_vllm_transcribe(
            audios_b64, item,
            no_hotwords=no_hotwords,
            language_override=language_override,
            base_url=base_url, model=model, max_tokens=max_tokens,
        )
    if backend not in ("chat", "auto"):
        raise ValueError(
            f"unknown backend {backend!r}; expected one of {BACKENDS}"
        )

    base_url = base_url or build_base_url(DEFAULT_PORT)

    # ── Amphion-1.7B / Qwen3-ASR format: system + audio-only user ─────
    if prompt_style == "qwen3_asr":
        messages = _build_messages_qwen3_asr(
            item, audios_b64,
            no_enrollment=no_enrollment,
            no_hotwords=no_hotwords,
        )
        if encoder_source == "triton":
            messages = _messages_with_triton_audio_embeds(
                messages,
                triton_url=triton_url,
                triton_model=triton_model,
                triton_top_k=triton_top_k,
            )
        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": 0,
        }
        url = f"{base_url}/v1/chat/completions"
        resp = requests.post(url, json=payload, timeout=120)
        resp.raise_for_status()
        data = resp.json()
        text = data["choices"][0]["message"]["content"]
        usage = data.get("usage", {})
        return text, usage

    # ── Amphion-4B / swift / train format ──────────────────────────────
    content = build_unified_content(
        item, audios_b64,
        no_enrollment=no_enrollment,
        no_language=no_language,
        no_hotwords=no_hotwords,
        language_override=language_override,
        custom_prompt=custom_prompt,
        task=task,
        prompt_style=prompt_style,
        spec_task=spec_task,
    )

    if _is_transcription_only_model(model):
        if encoder_source == "triton":
            raise ValueError(
                "encoder_source=triton cannot be used with transcription-only models"
            )
        return _call_vllm_transcription(
            content=content, base_url=base_url, model=model,
        )

    if encoder_source == "triton":
        content = _content_with_triton_audio_embeds(
            content,
            triton_url=triton_url,
            triton_model=triton_model,
            triton_top_k=triton_top_k,
        )

    messages = [{"role": "user", "content": content}]
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0,
    }
    url = f"{base_url}/v1/chat/completions"
    resp = requests.post(url, json=payload, timeout=120)
    resp.raise_for_status()
    data = resp.json()
    text = data["choices"][0]["message"]["content"]
    usage = data.get("usage", {})
    return text, usage


# ======================================================================
# Single-audio mode
# ======================================================================

def run_single(args):
    audio, sr = load_audio(args.audio)
    dur = len(audio) / sr
    audio_b64 = audio_to_base64_wav(audio, sr)

    no_enroll, no_lang, no_hw, lang_override = _resolve_task_flags(
        args.task,
        no_enrollment=args.no_enrollment,
        no_language=args.no_language,
        no_hotwords=args.no_hotwords,
        language_override=None,
    )
    # Single-audio mode has no enrollment reference by construction.
    no_enroll = True
    item = {"id": "single", "language": args.default_lang}

    t0 = time.time()
    text, usage = call_vllm_api(
        {"audio": audio_b64}, item,
        no_enrollment=no_enroll,
        no_language=no_lang,
        no_hotwords=no_hw,
        language_override=lang_override,
        custom_prompt=args.prompt,
        task=args.task,
        base_url=args.base_url,
        model=args.model,
        max_tokens=args.max_tokens,
        prompt_style=getattr(args, "prompt_style", "swift"),
        spec_task=getattr(args, "spec_task", None),
        backend=getattr(args, "backend", "chat"),
        encoder_source=getattr(args, "encoder_source", "vllm"),
        triton_url=getattr(args, "triton_url", "localhost:8000"),
        triton_model=getattr(args, "triton_model", DEFAULT_TRITON_MODEL),
        triton_top_k=getattr(args, "triton_top_k", 0),
    )
    elapsed = time.time() - t0
    detected_lang, clean_text = parse_asr_output(text)

    rows: list[tuple[str, str]] = [
        (
            "input",
            f"file={args.audio}  dur={dur:.2f}s  "
            f"api={args.base_url}  base64={len(audio_b64)} chars",
        ),
    ]
    timing_parts = [f"elapsed={elapsed:.2f}s"]
    if usage:
        timing_parts.append(
            f"prompt_tokens={usage.get('prompt_tokens', '?')}"
        )
        timing_parts.append(
            f"completion_tokens={usage.get('completion_tokens', '?')}"
        )
    rows.append(("timing", "  ".join(timing_parts)))

    if clean_text != text.strip():
        if detected_lang:
            rows.append(("lang", detected_lang))
        rows.append(("hyp", clean_text))
        rows.append(("raw", text))
    else:
        rows.append(("result", text))

    _report_panel("single", rows)


# ======================================================================
# Batch helpers (shared by supervisions & cuts paths)
# ======================================================================

def _load_item_audios(item: dict, *, has_enrollment: bool):
    """Load mixed (+ optional enrollment) audios, returning ``(audios_b64,
    eval_duration_s)``.  ``eval_duration_s`` is the *mixed* clip duration
    only so that RTF stays comparable regardless of enrollment presence.
    """
    mixed_path = item.get("mixed_audio") or item.get("audio")
    if not mixed_path:
        raise FileNotFoundError("缺少 mixed/目标音频")
    mix_audio, mix_sr = load_audio(
        mixed_path, item.get("start", 0.0), item.get("duration"),
    )
    audios = {"mixed": audio_to_base64_wav(mix_audio, mix_sr)}
    eval_dur = len(mix_audio) / mix_sr

    if has_enrollment:
        enroll_path = item.get("enrollment_audio")
        if not enroll_path:
            raise FileNotFoundError("要求 enrollment 但缺少 enrollment_audio")
        # enrollment 通常是完整文件,不需要 start/duration 切片。
        enr_audio, enr_sr = load_audio(enroll_path)
        audios["enrollment"] = audio_to_base64_wav(enr_audio, enr_sr)

    return audios, eval_dur


def _compute_summary(
    results: list,
    total_duration: float,
    total_time: float,
) -> dict:
    """Aggregate WER / silence metrics from per-utt records.

    Records carry the raw ref/hyp + per-utt language and sample_type, so we
    delegate the heavy lifting to ``src.compute_wer.compute_per_language_wer``
    and ``compute_silence_metrics`` for byte-identical alignment with
    decode.py's metric path.

    Returns a structured summary dict with:
      - total / failed / evaluated: counts (failed = error during inference)
      - exact_match{,_rate}: case-insensitive string equality after strip.
        Training lowercases every supervision text (see src/train.py
        ``texts = [t.lower() for t in batch[...]]``) so models emit lower
        case; case-sensitive comparison would mark every SER label
        ("Happy" vs "happy") wrong despite the model being correct.
      - per_language[lang]: ``{wer, S, D, I, hits, ref_tokens, n, duration}``
      - overall: cross-language aggregate (None when no usable refs)
      - silence_metrics / per_sample_type: silence-aware binary metrics
      - total_duration_s / total_inflight_s / rtf: throughput info
    """
    n = len(results)
    errors = [r for r in results if "error" in r]
    good = [r for r in results if "error" not in r]
    evaluated = len(good)
    correct = sum(
        1 for r in good
        if (r.get("hyp", "") or "").strip().lower()
           == (r.get("ref", "") or "").strip().lower()
    )

    refs = [r.get("ref", "") or "" for r in good]
    hyps = [r.get("hyp", "") or "" for r in good]
    langs = [r.get("language") or "en" for r in good]
    sample_types = [r.get("sample_type", "positive") or "positive" for r in good]
    eval_durs = [float(r.get("eval_dur", 0.0) or 0.0) for r in good]

    # ---- Per-language WER + overall ----
    wer_pkg = _cw_compute_per_language_wer(refs, hyps, langs)
    per_language_raw = wer_pkg["per_language"]
    overall_raw = wer_pkg["overall"]

    # Tally per-language counts + duration on the same buckets that
    # compute_per_language_wer used (drops empty refs).
    per_lang_n: dict[str, int] = defaultdict(int)
    per_lang_dur: dict[str, float] = defaultdict(float)
    for r, lng, du in zip(refs, langs, eval_durs):
        if not r.strip():
            continue
        code = normalize_lang_code(lng)
        per_lang_n[code] += 1
        per_lang_dur[code] += du

    per_language: dict[str, dict] = {}
    for lang in sorted(per_language_raw.keys()):
        m = per_language_raw[lang]
        per_language[lang] = {
            "wer": m["wer"],
            "substitutions": m["substitutions"],
            "deletions": m["deletions"],
            "insertions": m["insertions"],
            "hits": m["correct"],
            "ref_tokens": m["num_ref_tokens"],
            "n": per_lang_n.get(lang, m["num_sentences"]),
            "duration": per_lang_dur.get(lang, 0.0),
        }

    overall = None
    if overall_raw is not None:
        overall = {
            "wer": overall_raw["wer"],
            "substitutions": overall_raw["substitutions"],
            "deletions": overall_raw["deletions"],
            "insertions": overall_raw["insertions"],
            "hits": overall_raw["correct"],
            "ref_tokens": overall_raw["num_ref_tokens"],
            "n": overall_raw["num_sentences"],
        }

    # ---- Silence-aware metrics ----
    silence_full = _cw_compute_silence_metrics(refs, hyps, sample_types)
    per_sample_type = silence_full.pop("per_sample_type", {}) or {}
    # Per-sample-type WER on raw refs/hyps (compute_wer rebuckets per type).
    refs_by_type: dict[str, list[str]] = defaultdict(list)
    hyps_by_type: dict[str, list[str]] = defaultdict(list)
    langs_by_type: dict[str, list[str]] = defaultdict(list)
    for r, h, lng, st in zip(refs, hyps, langs, sample_types):
        if not r.strip():
            continue
        refs_by_type[st].append(r)
        hyps_by_type[st].append(h)
        langs_by_type[st].append(lng)
    for st, bucket in per_sample_type.items():
        if refs_by_type.get(st):
            wpkg = _cw_compute_per_language_wer(
                refs_by_type[st], hyps_by_type[st], langs_by_type[st],
            )
            ov = wpkg.get("overall")
            bucket["wer"] = ov["wer"] if ov else None
            bucket["wer_n"] = len(refs_by_type[st])
        else:
            bucket["wer"] = None
            bucket["wer_n"] = 0

    # ---- Hotword-specialised metrics ----
    # Triggered automatically when at least one record carries the
    # ``hotwords_real`` field (set by random padding, retrieve injection,
    # or plain-mode persistence). 'none' mode without any per-item GT
    # hotwords leaves the block absent; xlsx columns degrade gracefully.
    hotword_overall = _compute_hotword_extension(good, refs, hyps, langs)
    if hotword_overall is not None and overall is not None:
        # Merge into the same overall block so xlsx / metrics consumers
        # see one cross-language headline number per metric. Skip merging
        # when ``overall`` itself is None (silence-only batch).
        for k, v in hotword_overall.items():
            if k != "per_sentence" and k not in overall:
                overall[k] = v

    return {
        "total": n,
        "failed": len(errors),
        "evaluated": evaluated,
        "exact_match": correct,
        "exact_match_rate": correct / max(evaluated, 1),
        "per_language": per_language,
        "overall": overall,
        "silence_metrics": silence_full,
        "per_sample_type": per_sample_type,
        "total_duration_s": total_duration,
        "total_inflight_s": total_time,
        "rtf": (total_time / total_duration) if total_duration > 0 else None,
        "hotword_metrics": hotword_overall,
    }


def _compute_hotword_extension(
    records: list,
    refs: list[str],
    hyps: list[str],
    langs: list[str],
) -> Optional[dict]:
    """Compute KER / SACC / B-WER / U-WER / PRR / PPR / PF1 (plus
    Recall@K + PrRR for retrieve runs) when hotword metadata is present.

    Reads ``hotwords_real`` (GT), ``hotwords_all`` (candidates fed to the
    model), ``hotwords_retrieved`` (top-k from phase 2, retrieve mode
    only) off each record. Returns None when no record carries any
    hotword metadata, so legacy non-hotword tasks see no hotword block
    in summary.json (and xlsx degrades gracefully).

    Multi-language mixed eval: we run :func:`compute_hotword_metrics` once
    over the union with the dominant language code, since the hotword
    matching is token-based and the same Whisper normaliser handles
    English / CJK by tokenising at char vs word granularity.
    """
    has_meta = any(
        r.get("hotwords_real") is not None
        or r.get("hotwords_all") is not None
        for r in records
    )
    if not has_meta:
        return None

    has_retrieve = any(
        r.get("hotwords_retrieved") is not None for r in records
    )

    gt_list = [list(r.get("hotwords_real") or []) for r in records]
    cand_list = [
        list(r.get("hotwords_all") or r.get("hotwords_real") or [])
        for r in records
    ]
    retrieved_list = (
        [list(r.get("hotwords_retrieved") or []) for r in records]
        if has_retrieve else None
    )

    # Pick the dominant language across this batch so KER tokenises with
    # a sensible default (zh tokenises CJK by char; en handles digits).
    if langs:
        lang_counts: defaultdict = defaultdict(int)
        for lng in langs:
            lang_counts[normalize_lang_code(lng)] += 1
        primary_lang = max(lang_counts.items(), key=lambda kv: kv[1])[0]
    else:
        primary_lang = "en"

    return _cw_compute_hotword_metrics(
        references=refs,
        hypotheses=hyps,
        gt_hotwords_list=gt_list,
        candidate_hotwords_list=cand_list,
        language=primary_lang,
        retrieved_hotwords_list=retrieved_list,
    )


def _print_summary(summary: dict, label: str = "") -> None:
    """渲染 ``_compute_summary`` 的结果为一个 rich Panel.

    ``overview / wer / wer:<lang> / silence / sample_type:<name> / perf``
    每段一行，由 ``_report_panel`` 包成无 loguru 前缀的盒子，便于多
    batch 连跑时区分。指标方向: ↑ 越大越好, ↓ 越小越好.
    """
    rows: list[tuple[str, str]] = []

    n = summary["total"]
    failed = summary["failed"]
    evaluated = summary["evaluated"]
    correct = summary["exact_match"]
    rate = summary["exact_match_rate"] * 100
    rows.append((
        "overview",
        f"total={n}  failed={failed}  "
        f"exact_match(↑)={rate:.2f}% ({correct}/{evaluated})",
    ))

    overall = summary.get("overall")
    if overall:
        rows.append((
            "wer",
            f"overall(↓)={overall['wer']:.2f}%  "
            f"S={overall['substitutions']}  D={overall['deletions']}  "
            f"I={overall['insertions']}  N={overall['n']}",
        ))
    pl = summary.get("per_language") or {}
    for lang, m in pl.items():
        rows.append((
            f"wer:{lang}",
            f"wer(↓)={m['wer']:.2f}%  "
            f"S={m['substitutions']}  D={m['deletions']}  "
            f"I={m['insertions']}  N={m['n']}  "
            f"dur={m['duration']:.1f}s",
        ))

    sm = summary.get("silence_metrics") or {}
    if sm.get("n_pos_ref") or sm.get("n_neg_ref"):
        parts = [
            f"pos_ref={sm['n_pos_ref']}",
            f"neg_ref={sm['n_neg_ref']}",
        ]
        if sm.get("false_alarm_rate") is not None:
            parts.append(
                f"false_alarm(↓)={sm['false_alarm_rate'] * 100:.2f}% "
                f"({sm['false_alarm']}/{sm['n_neg_ref']})"
            )
            parts.append(
                f"silence_match(↑)={sm['exact_silence_match'] * 100:.2f}% "
                f"({sm['true_silence']}/{sm['n_neg_ref']})"
            )
        if sm.get("miss_rate") is not None:
            parts.append(
                f"miss(↓)={sm['miss_rate'] * 100:.2f}% "
                f"({sm['miss']}/{sm['n_pos_ref']})"
            )
        rows.append(("silence", "  ".join(parts)))

    per_st = summary.get("per_sample_type") or {}
    if len(per_st) > 1 or any(k != "positive" for k in per_st):
        for st in sorted(per_st):
            m = per_st[st]
            wer = (
                f"{m['wer']:.2f}%" if m.get("wer") is not None else "-"
            )
            far = (
                f"{m['false_alarm_rate'] * 100:.2f}%"
                if m.get("false_alarm_rate") is not None else "-"
            )
            sil = (
                f"{m['exact_silence_match'] * 100:.2f}%"
                if m.get("exact_silence_match") is not None else "-"
            )
            miss = (
                f"{m['miss_rate'] * 100:.2f}%"
                if m.get("miss_rate") is not None else "-"
            )
            rows.append((
                f"sample_type:{st}",
                f"n={m['n']}  wer(↓)={wer}  far(↓)={far}  "
                f"silence(↑)={sil}  miss(↓)={miss}",
            ))

    perf_parts = [
        f"audio={summary['total_duration_s']:.1f}s",
        f"inflight={summary['total_inflight_s']:.1f}s",
    ]
    if summary.get("wall_clock_s") is not None:
        perf_parts.append(f"wall={summary['wall_clock_s']:.1f}s")
    if summary.get("rtf") is not None:
        perf_parts.append(f"rtf(↓)={summary['rtf']:.3f}")
    if summary.get("speedup") is not None:
        perf_parts.append(f"speedup(↑)={summary['speedup']:.2f}x")
    rows.append(("perf", "  ".join(perf_parts)))

    title = f"summary  {label}" if label else "summary"
    _report_panel(title, rows)


def _process_one(
    idx: int,
    item: dict,
    flags: dict,
    default_lang: str,
    args,
) -> dict:
    """Run one item: 加载音频 + 调 vLLM + 归一化 + 单句 WER。

    ``flags`` must contain ``no_enrollment`` / ``no_language`` /
    ``no_hotwords`` / ``language_override`` (see ``_resolve_task_flags``).
    线程安全: 不修改外部共享状态; 失败时也返回带 ``error`` 字段的 record。
    """
    utt_id = item["id"]
    ref_text = item.get("ref", "")
    lang_code = resolve_lang_code(item.get("language", "") or default_lang)
    if not lang_code:
        lang_code = default_lang

    has_enrollment = (
        bool(item.get("enrollment_audio")) and not flags["no_enrollment"]
    )

    record: dict = {
        "idx": idx,
        "id": utt_id,
        "language": lang_code,
        "ref": ref_text,
        "hyp": "",
        "eval_dur": 0.0,
        "elapsed": 0.0,
        "sample_type": item.get("sample_type", "positive"),
        "has_enrollment": has_enrollment,
    }
    if has_enrollment:
        record["mixed_audio"] = item.get("mixed_audio") or item.get("audio")
        record["enrollment_audio"] = item.get("enrollment_audio")
    else:
        record["audio"] = item.get("mixed_audio") or item.get("audio")

    # Optional hotword padding / retrieve injection: replace
    # ``item["hotwords"]`` with the final candidate list before the prompt
    # is assembled. Three sources, mutually exclusive:
    #
    #   1. Retrieve mode (item already carries ``hotwords_real`` set by
    #      ``infer_retrieve.inject_retrieved_into_items``): real list is
    #      preserved for KER/PRR scoring; ``item["hotwords"]`` is the
    #      retrieved top-k that the model actually sees.
    #   2. Random padding (``flags.hotwords_pad_to > 0``): ``real ∪
    #      distractor`` to fixed size, draws from the dataset's own pool.
    #   3. Plain (no padding, hotwords visible): real == all.
    #
    # Per-sample seed = utt_id keeps the distractor draw reproducible
    # across re-runs even under multi-thread concurrency.
    pad_to = int(flags.get("hotwords_pad_to", 0) or 0)
    explicit_real = item.get("hotwords_real")
    if explicit_real is not None:
        retrieved = list(item.get("hotwords") or [])
        record["hotwords_real"] = sorted(
            {h.strip() for h in explicit_real
             if isinstance(h, str) and h.strip()}
        )
        record["hotwords_retrieved"] = retrieved
        record["hotwords_all"] = retrieved
    elif pad_to > 0:
        rng = random.Random(f"hotwords:{utt_id}")
        hw_meta = _pad_hotwords(
            real=list(item.get("hotwords") or []),
            pool=flags.get("hotword_pool") or [],
            target_size=pad_to,
            rng=rng,
        )
        item = dict(item)
        item["hotwords"] = hw_meta["all"]
        record["hotwords_real"] = hw_meta["real"]
        record["hotwords_distractor"] = hw_meta["distractor"]
        record["hotwords_all"] = hw_meta["all"]
    elif item.get("hotwords") and (
        not flags.get("no_hotwords")
        or flags.get("spec_task") == "asr_hotwords"
    ):
        # When no_hotwords=True but spec_task=="asr_hotwords" (K=0 baseline),
        # still record hotwords_real so KER/B-WER/U-WER/PRR/SACC are computed.
        # hotwords_all == real only (no distractors), so PPR/PF1 trivially
        # 100% / equal to PRR for K=0 — which is expected and documented.
        real_unique = sorted(
            {h.strip() for h in (item.get("hotwords") or [])
             if isinstance(h, str) and h.strip()}
        )
        record["hotwords_real"] = real_unique
        record["hotwords_all"] = real_unique

    try:
        audios_b64, eval_dur = _load_item_audios(
            item, has_enrollment=has_enrollment,
        )
    except Exception as e:
        record["error"] = f"加载失败: {e}"
        return record
    record["eval_dur"] = eval_dur

    t0 = time.time()
    try:
        hyp, _usage = call_vllm_api(
            audios_b64, item,
            no_enrollment=flags["no_enrollment"],
            no_language=flags["no_language"],
            no_hotwords=flags["no_hotwords"],
            language_override=flags["language_override"],
            custom_prompt=args.prompt,
            task=flags.get("task"),
            base_url=args.base_url,
            model=args.model,
            max_tokens=args.max_tokens,
            prompt_style=flags.get("prompt_style", "swift"),
            spec_task=flags.get("spec_task"),
            backend=getattr(args, "backend", "chat"),
            encoder_source=getattr(args, "encoder_source", "vllm"),
            triton_url=getattr(args, "triton_url", "localhost:8000"),
            triton_model=getattr(args, "triton_model", DEFAULT_TRITON_MODEL),
            triton_top_k=getattr(args, "triton_top_k", 0),
        )
    except Exception as e:
        record["error"] = f"API 调用失败: {e}"
        return record
    record["elapsed"] = time.time() - t0

    # Qwen3-ASR raw 输出自带 "language <Lang><asr_text>" 前缀, 先剥离; 其他模型
    # 无 tag 时 parse_asr_output 相当于 pass-through, 不影响 Amphion 等路径。
    detected_lang, hyp_text = parse_asr_output(hyp)
    record["hyp_raw"] = hyp
    record["hyp"] = hyp_text
    if detected_lang:
        record["detected_language"] = detected_lang

    # Persist the human-readable normalised forms for downstream tooling /
    # transcripts.jsonl. compute_wer below operates on RAW strings + lang so
    # the language-specific normalizer is applied exactly once.
    record["norm_ref"] = normalize_text(ref_text, lang_code)
    record["norm_hyp"] = normalize_text(hyp_text, lang_code)

    if (ref_text or "").strip():
        sm = compute_wer([ref_text], [hyp_text], language=lang_code)
    else:
        sm = {"wer": 0.0, "substitutions": 0, "deletions": 0,
              "insertions": 0, "hits": 0, "ref_tokens": 0,
              "num_sentences": 0}
    record.update({
        "wer": sm["wer"],
        "substitutions": sm["substitutions"],
        "deletions": sm["deletions"],
        "insertions": sm["insertions"],
        "ref_tokens": sm["ref_tokens"],
    })
    return record


def _running_wer_postfix(
    wer_num: int,
    wer_den: int,
    err_count: int,
    inflight_sum: float,
    dur_sum: float,
) -> str:
    """Compose tqdm postfix string with running corpus WER + RTF + err count.

    WER uses (S+D+I) / ref_tokens to match :func:`_compute_summary`; RTF is
    总并发 in-flight 时间 / 总音频时长 (>1 表示比实时慢)。统计皆为成功 record
    的累积值,失败样本不会污染分母。
    """
    wer_pct = (100.0 * wer_num / wer_den) if wer_den > 0 else 0.0
    rtf = inflight_sum / dur_sum if dur_sum > 1e-6 else 0.0
    return f"wer={wer_pct:.2f}% err={err_count} rtf={rtf:.2f}"


def _log_error_summary(buf: list[tuple[str, str]], top: int = 20) -> None:
    """Log a one-shot summary of failed samples after a batch finishes.

    Per-sample logger.error 在跑长批次时会把 tqdm 进度条挤垮; 把错误延迟到
    批次末统一吐出更友好。完整错误信息已写入 transcripts.jsonl 的 ``error``
    字段, 这里只做高亮提醒。
    """
    if not buf:
        return
    n = len(buf)
    logger.warning(
        f"[errors] failed={n} (showing first {min(n, top)})"
    )
    for utt_id, msg in buf[:top]:
        logger.warning(f"[errors]   {utt_id}: {msg}")
    if n > top:
        logger.warning(
            f"[errors]   ... and {n - top} more "
            f"(see transcripts.jsonl 'error' field for the full list)"
        )


def _dump_first_prompt_preview(item: dict, flags: dict, args) -> None:
    """Pretty-print the assembled chat content for one sample (debug aid).

    Builds the prompt with placeholder audio payloads (so the dump is
    cheap and JSON-readable) and logs the resulting OpenAI-style content
    list. Use to sanity-check that ``--prompt-style train`` matches the
    expected ``src/train.py`` shape token-for-token.
    """
    audios_b64 = {"mixed": "<MIXED_AUDIO_BASE64>"}
    if item.get("enrollment_audio") and not flags["no_enrollment"]:
        audios_b64["enrollment"] = "<ENROLLMENT_AUDIO_BASE64>"

    pad_to = int(flags.get("hotwords_pad_to", 0) or 0)
    preview_item = dict(item)
    if pad_to > 0:
        rng = random.Random(f"hotwords:{item.get('id', '?')}:dump")
        hw_meta = _pad_hotwords(
            real=list(item.get("hotwords") or []),
            pool=flags.get("hotword_pool") or [],
            target_size=pad_to,
            rng=rng,
        )
        preview_item["hotwords"] = hw_meta["all"]

    content = build_unified_content(
        preview_item, audios_b64,
        no_enrollment=flags["no_enrollment"],
        no_language=flags["no_language"],
        no_hotwords=flags["no_hotwords"],
        language_override=flags["language_override"],
        custom_prompt=getattr(args, "prompt", None),
        task=flags.get("task"),
        prompt_style=flags.get("prompt_style", "swift"),
        spec_task=flags.get("spec_task"),
    )
    item_id = item.get("id", "?")
    logger.info(f"[prompt-preview:start] id={item_id}")
    for blk in content:
        kind = blk.get("type")
        if kind == "text":
            for line in blk["text"].split("\n"):
                logger.info(f"[prompt-preview]   [text] {line}")
        elif kind == "input_audio":
            logger.info(
                f"[prompt-preview]   [input_audio] "
                f"{blk['input_audio'].get('data', '')}"
            )
        else:
            logger.info(f"[prompt-preview]   [{kind}] {blk!r}")
    logger.info(f"[prompt-preview:end]   id={item_id}")


def run_batch(args, items: list, mode_label: str):
    """Generic batch runner for supervisions / cuts items.

    调用方已按 ``args.num_samples`` 进行流式截断; 这里再兜底一次,
    防止传入的 items 来自全量加载。
    """
    if args.num_samples and args.num_samples < len(items):
        items = items[: args.num_samples]
    n = len(items)

    no_enroll, no_lang, no_hw, lang_override = _resolve_task_flags(
        args.task,
        no_enrollment=args.no_enrollment,
        no_language=args.no_language,
        no_hotwords=args.no_hotwords,
        language_override=None,
    )

    # Optional: pad each sample's hotwords up to a fixed K via random
    # distractors drawn from the hotword pool. Pool source priority:
    #   1. --random-hotword-pool-file (external file, harder evaluation)
    #   2. manifest true-hotword union (default, easy — distractors come
    #      from the same dataset so the model has seen them all).
    # Pool is built once on the main thread and handed read-only to workers.
    pad_to = max(0, int(getattr(args, "hotwords_pad_to", 0) or 0))
    hotword_pool: list[str] = []
    if pad_to > 0 and not no_hw:
        _random_pool_file = (
            getattr(args, "random_hotword_pool_file", "") or ""
        ).strip()
        if _random_pool_file:
            if not Path(_random_pool_file).exists():
                raise FileNotFoundError(
                    f"[random] random_hotword_pool_file not found: "
                    f"{_random_pool_file}"
                )
            with open(_random_pool_file, encoding="utf-8") as _pf:
                hotword_pool = sorted({
                    ln.strip() for ln in _pf if ln.strip()
                })
            logger.info(
                "[random] external pool_file loaded: %d words from %s",
                len(hotword_pool), _random_pool_file,
            )
        else:
            hotword_pool = _collect_hotword_pool(items)
        if not hotword_pool:
            logger.warning(
                f"--hotwords-pad-to={pad_to} set but no real hotwords "
                f"in this dataset — pad disabled for this batch."
            )
            pad_to = 0
    elif pad_to > 0 and no_hw:
        logger.info(
            f"--hotwords-pad-to={pad_to} ignored because "
            f"task={args.task} has no_hotwords=True."
        )
        pad_to = 0

    prompt_style = getattr(args, "prompt_style", "swift")
    spec_task = getattr(args, "spec_task", None)
    flags = {
        "no_enrollment": no_enroll,
        "no_language": no_lang,
        "no_hotwords": no_hw,
        "language_override": lang_override,
        "task": args.task,
        "hotwords_pad_to": pad_to,
        "hotword_pool": hotword_pool,
        "prompt_style": prompt_style,
        "spec_task": spec_task,
    }

    n_enroll = sum(
        1 for it in items
        if bool(it.get("enrollment_audio")) and not no_enroll
    )
    n_hw_items = sum(
        1 for it in items
        if (it.get("hotwords") and not no_hw)
    )

    default_lang = resolve_lang_code(args.default_lang)
    concurrency = max(1, int(getattr(args, "concurrency", 1) or 1))

    _prompt_extra = (
        f"  spec_task={spec_task!r}" if spec_task else ""
    )
    _banner_rows: list[tuple[str, str]] = [
        (
            "task",
            f"{args.task}  n={n}  api={args.base_url}  "
            f"default_lang={default_lang}  concurrency={concurrency}  "
            f"metric=WER(CJK)",
        ),
        (
            "encoder",
            f"source={getattr(args, 'encoder_source', 'vllm')}  "
            f"triton={getattr(args, 'triton_url', '')}/"
            f"{getattr(args, 'triton_model', '')}",
        ),
        (
            "flags",
            f"no_enroll={no_enroll}  no_lang={no_lang}  "
            f"no_hw={no_hw}  lang_override={lang_override!r}",
        ),
        (
            "prompt",
            f"enrollment={n_enroll}/{n}  hotwords={n_hw_items}/{n}  "
            f"style={prompt_style}{_prompt_extra}",
        ),
    ]
    if pad_to > 0:
        _banner_rows.append((
            "hotwords",
            f"pad_to={pad_to}  pool={len(hotword_pool)}",
        ))
    _report_panel(f"batch  {mode_label}", _banner_rows)

    # Optional one-shot dump of the assembled prompt for the first item
    # in this batch — invaluable when tuning prompt_style or diffing the
    # eval client against training prompts.
    if getattr(args, "dump_first_prompt", False) and items:
        try:
            _dump_first_prompt_preview(items[0], flags, args)
        except Exception as e:
            logger.warning(f"[batch:dump-prompt] failed: {e}")

    verbose_each = bool(getattr(args, "verbose_each", False))

    print_lock = threading.Lock()
    records_by_idx: dict[int, dict] = {}
    done_count = 0
    wall_t0 = time.time()

    # Running aggregates feeding the tqdm postfix; only successful records
    # contribute to wer_num / wer_den / dur_sum / inflight_sum so failures
    # don't skew the live WER readout.
    wer_num = 0
    wer_den = 0
    dur_sum = 0.0
    inflight_sum = 0.0
    errors_buf: list[tuple[str, str]] = []

    pbar = None
    if not verbose_each:
        pbar = tqdm(
            total=n,
            desc=f"{mode_label} task={args.task}",
            unit="utt",
            dynamic_ncols=True,
            mininterval=0.5,
            smoothing=0.1,
            file=sys.stderr,
            leave=True,
        )

    def _on_done(rec: dict):
        nonlocal done_count, wer_num, wer_den, dur_sum, inflight_sum
        done_count += 1
        with print_lock:
            if "error" in rec:
                errors_buf.append((str(rec.get("id", "")), str(rec["error"])))
            else:
                wer_num += int(rec.get("substitutions", 0)) \
                    + int(rec.get("deletions", 0)) \
                    + int(rec.get("insertions", 0))
                wer_den += int(rec.get("ref_tokens", 0))
                dur_sum += float(rec.get("eval_dur", 0.0))
                inflight_sum += float(rec.get("elapsed", 0.0))

            if verbose_each:
                idx = rec["idx"]
                utt_id = rec["id"]
                eval_dur = rec.get("eval_dur", 0.0)
                elapsed = rec.get("elapsed", 0.0)
                prefix = f"[{done_count}/{n} idx={idx + 1}]"
                if "error" in rec:
                    logger.error(f"{prefix} {utt_id}  *** {rec['error']}")
                    return
                logger.info(
                    f"{prefix} {utt_id}  ({eval_dur:.1f}s, {elapsed:.2f}s)  "
                    f"[{rec['language']}] WER={rec['wer']:.1f}%"
                )
                logger.info(f"  REF: {rec['ref']}")
                logger.info(f"  HYP: {rec['hyp']}")
                return

            if pbar is not None:
                pbar.update(1)
                pbar.set_postfix_str(
                    _running_wer_postfix(
                        wer_num, wer_den, len(errors_buf),
                        inflight_sum, dur_sum,
                    ),
                    refresh=False,
                )

    try:
        if concurrency == 1:
            for i, item in enumerate(items):
                rec = _process_one(i, item, flags, default_lang, args)
                records_by_idx[i] = rec
                _on_done(rec)
        else:
            with ThreadPoolExecutor(max_workers=concurrency) as pool:
                futures = {
                    pool.submit(_process_one, i, item, flags, default_lang, args): i
                    for i, item in enumerate(items)
                }
                for fut in as_completed(futures):
                    i = futures[fut]
                    try:
                        rec = fut.result()
                    except Exception as e:
                        rec = {
                            "idx": i,
                            "id": items[i].get("id", f"idx_{i}"),
                            "ref": items[i].get("ref", ""),
                            "hyp": "",
                            "error": f"worker 异常: {e}",
                            "eval_dur": 0.0,
                            "elapsed": 0.0,
                        }
                    records_by_idx[i] = rec
                    _on_done(rec)
    finally:
        if pbar is not None:
            pbar.close()

    if not verbose_each:
        _log_error_summary(errors_buf)

    wall_elapsed = time.time() - wall_t0

    results = [records_by_idx[i] for i in range(n)]
    total_duration = sum(r.get("eval_dur", 0.0) for r in results)
    total_inflight = sum(r.get("elapsed", 0.0) for r in results)

    summary = _compute_summary(results, total_duration, total_inflight)
    if concurrency > 1:
        summary["wall_clock_s"] = wall_elapsed
        summary["speedup"] = total_inflight / max(wall_elapsed, 1e-6)
    _print_summary(summary, label=mode_label)

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        records = [
            {k: v for k, v in r.items() if k != "idx"} for r in results
        ]
        out_obj = {
            "mode": mode_label,
            "task": args.task,
            "flags": flags,
            "model": args.model,
            "base_url": args.base_url,
            "encoder_source": getattr(args, "encoder_source", "vllm"),
            "triton_url": getattr(args, "triton_url", ""),
            "triton_model": getattr(args, "triton_model", ""),
            "default_lang": default_lang,
            "concurrency": concurrency,
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            **summary,
            "records": records,
        }
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(out_obj, f, ensure_ascii=False, indent=2)
            f.write("\n")
        logger.info(f"\n结果已保存至: {out_path}")
        logger.info(f"  - 顶层字段: 汇总指标 (overall / per_language / rtf / ...)")
        logger.info(f"  - records:  {len(records)} 条逐条样本结果")

    return results, summary


# ======================================================================
# Entry
# ======================================================================

def main():
    parser = argparse.ArgumentParser(
        description="AmphionASR vLLM API 推理测试 (asr / ts_asr)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    grp = parser.add_argument_group("输入源 (五选一)")
    grp.add_argument("--audio", type=str, help="单条音频文件路径 (wav/flac/mp3)")
    grp.add_argument("--supervisions", type=str,
                     help="Lhotse supervisions manifest (jsonl/jsonl.gz)，"
                          "自动推断 recordings")
    grp.add_argument("--cuts", type=str,
                     help="Lhotse cuts manifest (jsonl/jsonl.gz)，"
                          "自带 recording 与 custom 字段; ts_asr 测试首选")
    grp.add_argument(
        "--test-dataset", type=str, default=None,
        help="单数据集快捷入口（与 decode.py 共用注册表）。"
             f"内置: {list_named_datasets()[:5]} ... + 多语种 <lang>:<name>",
    )
    grp.add_argument(
        "--test-plan-file", type=str, default=None,
        help="YAML test plan 文件路径，跑多个 (dataset, task) 组合; "
             "与 --test-dataset 互斥。",
    )
    parser.add_argument("--recordings", type=str, default=None,
                        help="Lhotse recordings manifest (可选，"
                             "默认从 supervisions 推断)")
    parser.add_argument(
        "--manifest-dir", type=Path,
        default=Path("/ai_sds_wuzz/DATA_ASR/LHOTSE"),
        help="Lhotse 通用 manifest 根目录 (供 --test-dataset / --test-plan-file 使用)",
    )
    parser.add_argument(
        "--ser-manifest-dir", type=Path,
        default=Path("/ai_sds_wuzz/DATA_SER/LHOTSE"),
        help="SER/SEC 数据 manifest 目录 (供 --test-dataset / --test-plan-file 使用)",
    )
    parser.add_argument(
        "--tsasr-test-manifest-dir", type=Path,
        default=Path("/chenmingjie/mingdong/data/lhotse/ts_hw_test"),
        help="TS-ASR 测试集 manifest 目录 (含 ts_hw_test_cuts_all.jsonl.gz)",
    )
    parser.add_argument(
        "--esc-manifest-dir", type=Path,
        default=Path("/ai_sds_wuzz/MULTILINGUAL_DATA/sound/AudioSet/manifest_v1_esc"),
        help="AudioSet ESC manifest 目录 (供 audioset_esc_test 等 _esc 测试集使用; "
             "与 src/decode.py 的 EscDataModule.add_arguments 默认保持一致)",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("exp/eval_vllm"),
        help="plan 模式 / --test-dataset 模式的产物根目录: "
             "<output-dir>/<run_id>/per_test/.../",
    )
    parser.add_argument(
        "--run-name", type=str, default=None,
        help="覆盖 run_id (默认: <model>-<YYYYmmdd_HHMMSS>)。"
             "plan 文件中的 run_name 优先级更高。",
    )

    parser.add_argument("--port", type=int, default=DEFAULT_PORT,
                        help=f"vLLM 服务端口 (默认: {DEFAULT_PORT}, "
                             f"主机固定为 localhost)")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL,
                        help=f"模型名称 (默认: {DEFAULT_MODEL})")
    parser.add_argument("--task", type=str, default="auto",
                        choices=list(TASK_MODES),
                        help="任务语法糖: auto=数据驱动 (默认); "
                             "asr/asr_en/asr_zh 强制屏蔽 enrollment+hotwords; "
                             "ts_asr 仅屏蔽 hotwords")
    parser.add_argument("--prompt", type=str, default=None,
                        help="自定义 prompt (覆盖整套模板; 仅建议搭配单音频)")
    parser.add_argument("--no-enrollment", action="store_true",
                        help="屏蔽 enrollment 字段 (baseline / ablation 用)")
    parser.add_argument("--no-language", action="store_true",
                        help="屏蔽 Language 提示行")
    parser.add_argument("--no-hotwords", action="store_true",
                        help="屏蔽 Hotwords 提示行")
    parser.add_argument(
        "--hotwords-pad-to", type=int, default=0,
        help="把每条样本的 hotwords 列表用随机干扰词填到固定大小 K "
             "(默认 0=禁用)。real ⊂ all，distractor 从该数据集 hotwords 全集采。"
             "仅在 task 携带 hotwords 时生效 (no_hotwords=False)。"
             "注入元数据 (hotwords_real / hotwords_distractor / hotwords_all) "
             "会写进 transcripts.jsonl 便于后续 recall 切片分析。",
    )
    parser.add_argument(
        "--hotword-mode",
        choices=sorted(VALID_HOTWORD_MODES),
        default="none",
        help="热词供给模式 (默认 none): "
             "none = 用 manifest 自带 hotwords (或 --no-hotwords 屏蔽); "
             "random = 用 --hotwords-pad-to N 随机 distractor 填到 K 条; "
             "retrieve = 两阶段 retrieve, phase1 用本服务 --no-hotwords 跑一遍, "
             "phase2 retrieve_hotwords 选 top-k, phase3 带 retrieved hotwords 推理。"
             "扩展指标 (KER/SACC/B-WER/U-WER/PRR/PPR/PF1) 会在所有非 none 模式自动计算。",
    )
    parser.add_argument(
        "--retrieve-top-k", type=int, default=10,
        help="retrieve 模式下每条样本保留的热词数 (默认 10)。"
             "仅 --hotword-mode retrieve 生效; ablation 用 hotwords_retrieve.yaml "
             "的 retrieve_top_k_sweep 一次扫多个 K。",
    )
    parser.add_argument(
        "--retrieve-cache-dir", type=str, default="",
        help="retrieve phase1/phase2 落盘缓存目录, 跨 run 共享。"
             "默认 <output-dir>/_retrieve_cache; 传空字符串禁用缓存。"
             "phase1 cache 与 K 无关, sweep 共享; phase2 按 K 分桶。",
    )
    parser.add_argument(
        "--retrieve-workers", type=int, default=8,
        help="retrieve phase2 (CPU 文本相似度) 进程池大小 (默认 8)",
    )
    parser.add_argument(
        "--tower-adapter-ckpt", type=str, default="",
        help="tower_retrieve 模式: adapter checkpoint 路径 (必填)。",
    )
    parser.add_argument(
        "--tower-base-model-path", type=str, default="",
        help="tower_retrieve 模式: AmphionASR 基座模型目录 (必填, 用于加载 "
             "audio encoder / LLM embed_tokens)。",
    )
    parser.add_argument(
        "--tower-embed-dim", type=int, default=512,
        help="tower_retrieve 模式: adapter 输出嵌入维度 (默认 512)。",
    )
    parser.add_argument(
        "--tower-adapter-hidden-dim", type=int, default=None,
        help="tower_retrieve 模式: MLP adapter 隐层维度, "
             "None 表示与 embed_dim 相同 (默认 None)。",
    )
    parser.add_argument(
        "--tower-biasing-tsv-file", type=str, default="",
        help="tower_retrieve 模式: IS21 Deep Bias 格式 TSV 文件路径 "
             "(格式: utt_id\\ttranscript\\tJSON_true_hw\\tJSON_candidates)。"
             "指定后每条 utterance 使用 TSV 中预定义的候选列表 (如 biasing_100/500/1000/2000), "
             "优先级高于 --tower-hotword-pool-file。",
    )
    parser.add_argument(
        "--random-hotword-pool-file", type=str, default="",
        help="random 模式: 外部干扰词词典文件路径 (每行一个词)。"
             "若指定, 则替代从 manifest 收集真实热词作为干扰词池的默认行为, "
             "使用外部大词典 (如 zh-50k.txt) 提供干扰词, 评测更真实。"
             "不指定时退化为 manifest 真实热词并集 (所有干扰词来自同数据集)。",
    )
    parser.add_argument(
        "--retrieve-hotword-pool-file", type=str, default="",
        help="retrieve 模式: 外部候选热词词典文件路径 (每行一个词)。"
             "若指定, 则替代从 manifest 收集真实热词的默认行为, 使用外部大词典 "
             "作为检索候选池 (含干扰词, 评测更真实)。"
             "不指定时退化为 manifest 真实热词并集 (无干扰词)。",
    )
    parser.add_argument(
        "--tower-hotword-pool-file", type=str, default="",
        help="tower_retrieve 模式: 候选热词词典文件路径 (每行一个词)。"
             "若指定, 则替代从 manifest 收集真实热词的默认行为, 使用外部大词典 "
             "(如 all_rare_words.txt ~209k 词) 作为检索候选池, 评测更真实。"
             "不指定时退化为 manifest 真实热词并集 (~4k 词, 无干扰词)。",
    )
    parser.add_argument("--max-tokens", type=int, default=200)
    parser.add_argument("--default-lang", type=str, default="zh-cn",
                        help="当 supervision 没有 language 字段时的回退语言 "
                             "(默认: zh-cn)")

    parser.add_argument("-n", "--num-samples", type=int, default=None,
                        help="仅处理前 N 条 (批量模式); 传 -1 (或 <=0) 表示跑全集")
    parser.add_argument("-j", "--concurrency", type=int, default=16,
                        help="批量推理的并发线程数 (默认 1, 即串行)")
    parser.add_argument("--output", type=str, default=None,
                        help="结果输出文件路径 (单个 JSON); "
                             "顶层字段为汇总指标 (overall/per_language/rtf/...), "
                             "逐条样本结果存于 \"records\" 字段")
    parser.add_argument(
        "--verbose-each", action="store_true", default=False,
        help="逐条样本打印 [i/N] id WER + REF/HYP (旧版行为)。"
             "默认关闭, 仅显示一行 tqdm 进度条 + running WER/RTF/err; "
             "完整逐条结果以 transcripts.jsonl 为准。",
    )
    parser.add_argument(
        "--prompt-style", choices=list(PROMPT_STYLES), default="swift",
        help="评测端 prompt 形状: "
             "'swift' (默认, 与 ms-swift convert.py / GRPO ckpt 对齐, "
             "适用于 Amphion-4B) | "
             "'train' (严格复刻 src/train.py:TASK_PROMPTS, "
             "用于评测 train.py 训练的 ckpt, 例如 multitask-92000) | "
             "'qwen3_asr' (Amphion-1.7B 专用: 文本指令放 system, "
             "user 仅含音频 token, 无 Language 字段, 对齐 Qwen3-ASR "
             "原始 chat_template)。选错会出现 hotwords 复读 / WER 暴涨。",
    )
    parser.add_argument(
        "--backend", choices=list(BACKENDS), default="auto",
        help="vLLM API 后端: auto (默认) 在 /v1/models 列表中含 'whisper' 时"
             "自动切到 transcription, 否则保持 chat; "
             "chat=POST /v1/chat/completions (Qwen3-Omni/Qwen3-ASR/Amphion 等"
             "音频 LLM); "
             "transcription=POST /v1/audio/transcriptions (vLLM 部署的 whisper-"
             "large-v3 等 OpenAI Whisper 兼容模型, 仅支持单音频 ASR; ts_asr / "
             "ser / sec / esc 任务会被 plan-mode 跳过, 见 run_plan)。",
    )
    parser.add_argument(
        "--encoder-source", choices=list(ENCODER_SOURCES), default="vllm",
        help="音频 encoder 来源: vllm=原始 input_audio 路径 (默认); "
             "triton=先调用 RAG-ASR Triton 取得 PROJECTOR_OUT, 再以 "
             "audio_embeds 发送给 vLLM。triton 模式要求 vLLM 启动时开启 "
             "--enable-mm-embeds。",
    )
    parser.add_argument(
        "--triton-url", type=str, default="localhost:8000",
        help="encoder-source=triton 时的 RAG-ASR Triton HTTP 地址。",
    )
    parser.add_argument(
        "--triton-model", type=str, default=DEFAULT_TRITON_MODEL,
        help=f"encoder-source=triton 时调用的 Triton 模型名 "
             f"(默认 {DEFAULT_TRITON_MODEL})。",
    )
    parser.add_argument(
        "--triton-top-k", type=int, default=0,
        help="encoder-source=triton 时传给 rag_asr_retrieve 的 TOP_K。"
             "默认 0, 因为 vLLM bypass 只消费 PROJECTOR_OUT。",
    )
    parser.add_argument(
        "--dump-first-prompt", action="store_true", default=False,
        help="每个 (sub-)spec 启动前 dump 第一条样本拼装出的 chat content "
             "(audio payload 用占位符), 便于人工 diff 训练 prompt 形状。",
    )

    args = parser.parse_args()
    args.base_url = build_base_url(args.port)
    # spec_task is set per-spec by run_plan; absent in legacy modes.
    args.spec_task = None

    # 约定: num_samples <= 0 (常见传 -1) 表示使用全部数据。
    # 内部一律用 None 表示无上限, 避免 limit=-1 让 readers 立即停在 0 条,
    # 或 run_batch 误把 items 切成 items[:-1] 丢掉最后一条。
    if args.num_samples is not None and args.num_samples <= 0:
        args.num_samples = None

    # 健康检查
    try:
        health = requests.get(f"{args.base_url}/health", timeout=5)
        health.raise_for_status()
        logger.info(f"vLLM 服务已就绪: {args.base_url}\n")
    except Exception as e:
        logger.warning(f"⚠ 无法连接 vLLM 服务 ({args.base_url}): {e}")
        logger.warning("请确认已运行: bash examples/serve/vllm/serve.sh -m <model> -p <port>\n")
        return

    # 模型校验: 健康检查通过不代表 model id 也对得上 (常见坑:
    # vLLM 启动时 served-model-name 与请求里的 --model 不一致, 此时
    # /v1/chat/completions 会全部返回 404, 但 /health 仍然 200)。
    try:
        resp = requests.get(f"{args.base_url}/v1/models", timeout=5)
        resp.raise_for_status()
        served = [m.get("id") for m in (resp.json().get("data") or [])]
    except Exception as e:
        served = []
        logger.warning(f"⚠ 无法获取 /v1/models 列表 ({e}), 跳过 model 校验。\n")

    if served and args.model not in served:
        logger.warning(f"⚠ 当前 vLLM 服务可用模型: {served}")
        logger.warning(f"  但你传入了 --model {args.model!r} (不在列表中),"
              f" 所有请求会被 vLLM 返回 404。")
        if len(served) == 1:
            logger.warning(f"  自动切换为唯一可用模型: {served[0]!r}")
            args.model = served[0]
        else:
            logger.warning(f"  请用 --model 指定上面任一可用名称, 或重启 vLLM "
                  f"用 -n {args.model} 设置 served-model-name。\n")
            return
    elif served:
        logger.info(f"模型校验通过: {args.model!r} (服务可用模型: {served})\n")

    # Backend resolution: auto -> chat / transcription.
    # Done after the served-model-name check so the keyword match operates on
    # the *actually*-served id (covers users who pass --model qwen3-omni against
    # a whisper service: auto-switch above will already have aligned args.model
    # to the real one, so we vote on that). When /v1/models was unreachable the
    # fallback is args.model itself — explicit -b transcription / -b chat still
    # works because _resolve_backend honours them verbatim.
    requested_backend = args.backend
    args.backend = _resolve_backend(args.backend, served or [args.model])
    if args.backend != requested_backend:
        logger.info(
            f"[backend] auto -> {args.backend} "
            f"(detected whisper in served={served or [args.model]})"
            if args.backend == "transcription"
            else f"[backend] auto -> {args.backend} (served={served or [args.model]})"
        )
    else:
        logger.info(f"[backend] {args.backend}")

    if args.encoder_source == "triton" and args.backend == "transcription":
        parser.error("--encoder-source triton 只支持 chat backend, 不支持 transcription")
    logger.info(
        f"[encoder_source] {args.encoder_source}"
        + (
            f"  triton={args.triton_url}/{args.triton_model}"
            if args.encoder_source == "triton" else ""
        )
    )
    _log_triton_hotword_snapshot(args)

    requested_style = args.prompt_style
    args.prompt_style = _resolve_prompt_style(args.prompt_style, served or [args.model])
    if args.prompt_style != requested_style:
        logger.info(
            f"[prompt_style] auto -> {args.prompt_style} "
            f"(detected 1.7B/qwen3-asr in served={served or [args.model]})"
        )
    else:
        logger.info(f"[prompt_style] {args.prompt_style}")

    # 分发
    inputs = [args.cuts, args.supervisions, args.audio,
              args.test_dataset, args.test_plan_file]
    n_inputs = sum(1 for v in inputs if v)
    if n_inputs == 0:
        parser.error("请指定 --audio / --supervisions / --cuts / "
                     "--test-dataset / --test-plan-file 之一")
    if n_inputs > 1:
        parser.error("--audio / --supervisions / --cuts / --test-dataset / "
                     "--test-plan-file 互斥，请只传一个")

    if args.test_dataset or args.test_plan_file:
        run_plan(args, parser)
        return

    if args.cuts:
        items = load_lhotse_cuts(args.cuts, limit=args.num_samples)
        run_batch(args, items, "Lhotse cuts 批量推理")
    elif args.supervisions:
        if not args.recordings:
            args.recordings = infer_recordings_path(args.supervisions)
            logger.info(f"自动推断 recordings: {args.recordings}")
        items = load_lhotse_pairs(
            args.recordings, args.supervisions, limit=args.num_samples,
        )
        run_batch(args, items, "Lhotse supervisions 批量推理")
    elif args.audio:
        if args.task == "auto":
            args.task = "asr"
        run_single(args)


# ======================================================================
# Plan-mode entry point (mirrors src/decode.py for cross-source diff)
# ======================================================================

def _resolve_spec_task_to_args_task(spec: TestSpec) -> str:
    """Map a registry-side task name into the vLLM script's TASK_MODES.

    Built-in registry tasks: asr / asr_hotwords / ts_asr / ser / sec / esc.
    Tasks with a fixed prompt template (asr / ts_asr / ser / sec / esc) are
    forwarded as-is; anything else (currently asr_hotwords) falls back to
    ``auto`` and the item-level enrollment / hotwords flags drive prompt
    construction.
    """
    if spec.task in ("asr", "ts_asr", "ser", "sec", "esc"):
        return spec.task
    return "auto"


def _format_spec_report(label: str, summary: dict) -> str:
    """Render a per-test wer_report.txt for the EvalRunDir layout."""
    lines = [
        f"# Test: {label}",
        f"Total: {summary['total']}    Failed: {summary['failed']}    "
        f"Evaluated: {summary['evaluated']}    "
        f"Exact match: {summary['exact_match']} "
        f"({summary['exact_match_rate'] * 100:.2f}%)",
        "",
    ]
    pl = summary.get("per_language") or {}
    if pl:
        lines.append("## Per-language WER")
        for lang in sorted(pl):
            m = pl[lang]
            lines.append(
                f"  {lang}: WER={m['wer']:.2f}%  "
                f"S={m['substitutions']} D={m['deletions']} "
                f"I={m['insertions']}  N={m['n']}  "
                f"duration={m['duration']:.1f}s"
            )
        ov = summary.get("overall")
        if ov:
            lines.append(
                f"  OVERALL: WER={ov['wer']:.2f}%  "
                f"S={ov['substitutions']} D={ov['deletions']} "
                f"I={ov['insertions']}  N={ov['n']}"
            )
        lines.append("")
    sm = summary.get("silence_metrics") or {}
    if sm.get("n_pos_ref") or sm.get("n_neg_ref"):
        lines.append("## Silence-aware metrics")
        if sm.get("false_alarm_rate") is not None:
            lines.append(
                f"  false_alarm_rate: {sm['false_alarm_rate'] * 100:.2f}%"
                f" ({sm['false_alarm']}/{sm['n_neg_ref']})"
            )
        if sm.get("miss_rate") is not None:
            lines.append(
                f"  miss_rate: {sm['miss_rate'] * 100:.2f}%"
                f" ({sm['miss']}/{sm['n_pos_ref']})"
            )
        if sm.get("exact_silence_match") is not None:
            lines.append(
                f"  exact_silence_match: "
                f"{sm['exact_silence_match'] * 100:.2f}%"
            )
    perf = summary.get("rtf"), summary.get("total_duration_s"), summary.get("total_inflight_s")
    if any(v is not None for v in perf):
        lines.append("")
        lines.append("## Performance")
        lines.append(
            f"  audio={summary.get('total_duration_s', 0):.1f}s  "
            f"inflight={summary.get('total_inflight_s', 0):.1f}s  "
            f"rtf={summary.get('rtf')}"
        )
        if summary.get("wall_clock_s") is not None:
            lines.append(
                f"  wall={summary['wall_clock_s']:.1f}s  "
                f"speedup={summary.get('speedup')}"
            )
    return "\n".join(lines)


_HOTWORD_OVERALL_KEYS = (
    # Headline rates surfaced in xlsx + summary.json (already in %)
    "ker", "sacc", "prr", "ppr", "pf1", "b_wer", "u_wer",
    # Retrieve-mode extras
    "recall_at_k", "prrr",
    # Counters useful for cross-spec aggregation / debugging
    "ker_total_keywords", "ker_missed_keywords",
    "tp", "fn", "fp",
    "biased_ref_tokens", "biased_sub", "biased_del", "biased_ins",
    "unbiased_ref_tokens", "unbiased_sub", "unbiased_del", "unbiased_ins",
    "n_sentences_with_keywords",
    "recall_total_gt", "recall_hit",
    "n_retrieve_eval", "n_retrieve_perfect",
)


def _spec_summary_for_run(spec: TestSpec, summary: dict) -> dict:
    """Adapt run_batch's summary into the cross-source spec block schema
    consumed by EvalRunDir.write_summary / aggregate_overall.
    """
    overall = summary.get("overall")
    overall_block: Optional[dict] = None
    if overall:
        overall_block = {
            "wer": overall["wer"],
            "substitutions": overall["substitutions"],
            "deletions": overall["deletions"],
            "insertions": overall["insertions"],
            "num_ref_tokens": overall["ref_tokens"],
            "num_sentences": overall["n"],
        }
        # Hotword-specialised metrics (see _compute_hotword_extension).
        # Persist whichever subset exists; keep raw counters so downstream
        # tooling (xlsx Aggregate sheet, paper-style table) can re-derive
        # macro variants without re-running compute_hotword_metrics.
        for k in _HOTWORD_OVERALL_KEYS:
            if k in overall:
                overall_block[k] = overall[k]
    return {
        "spec": spec.to_dict(),
        "n_total": summary.get("total", 0),
        "n_evaluated": summary.get("evaluated", 0),
        "n_failed": summary.get("failed", 0),
        "exact_match_rate": summary.get("exact_match_rate"),
        "per_language": {
            lang: {
                "wer": m["wer"],
                "substitutions": m["substitutions"],
                "deletions": m["deletions"],
                "insertions": m["insertions"],
                "num_ref_tokens": m["ref_tokens"],
                "num_sentences": m["n"],
                "duration": m["duration"],
            }
            for lang, m in (summary.get("per_language") or {}).items()
        },
        "overall": overall_block,
        "silence_metrics": summary.get("silence_metrics"),
        "per_sample_type": summary.get("per_sample_type"),
        "perf": {
            "total_duration_s": summary.get("total_duration_s"),
            "total_inflight_s": summary.get("total_inflight_s"),
            "rtf": summary.get("rtf"),
            "wall_clock_s": summary.get("wall_clock_s"),
            "speedup": summary.get("speedup"),
        },
    }


def _make_perf_block(summary: dict) -> dict:
    """Extract the perf sub-block (matching the one written by
    :func:`_spec_summary_for_run`) from ``run_batch``'s summary dict."""
    return {
        "total_duration_s": summary.get("total_duration_s"),
        "total_inflight_s": summary.get("total_inflight_s"),
        "rtf": summary.get("rtf"),
        "wall_clock_s": summary.get("wall_clock_s"),
        "speedup": summary.get("speedup"),
    }


def _records_for_disk(results: list) -> list:
    """Strip the in-memory-only ``idx`` field before persisting to disk."""
    return [{k: v for k, v in r.items() if k != "idx"} for r in results]


def _persist_ser_to_run(
    spec: TestSpec,
    label: str,
    results: list,
    batch_summary: dict,
    run_dir: EvalRunDir,
) -> dict:
    """SER: classification metrics (WA / UA / Macro-F1 / per-class P-R-F1 /
    confusion matrix). Mirrors :func:`src.decode.save_ser_results_to_run`
    so both eval paths produce byte-identical metrics.json + wer_report.txt
    for the same test set.
    """
    sorted_results = sorted(results, key=lambda r: r.get("id", ""))
    good = [r for r in sorted_results if "error" not in r]
    refs = [r.get("ref", "") or "" for r in good]
    hyps = [r.get("hyp", "") or "" for r in good]
    metrics = _cw_compute_classification_metrics(refs, hyps)
    report_text = _cw_render_classification_report(label, metrics)

    run_dir.write_test_transcripts(label, _records_for_disk(sorted_results))
    run_dir.write_test_metrics(label, metrics)
    run_dir.write_test_wer_report(label, report_text)

    _ser_summary_line = (
        f"WA={metrics['wa'] * 100:.2f}%  "
        f"UA={metrics['ua'] * 100:.2f}%  "
        f"F1={metrics['macro_f1'] * 100:.2f}%  "
        f"n={metrics['num_samples']}"
    )
    _REPORT_CONSOLE.print(
        _RichPanel(
            report_text.rstrip() + "\n\n" + _ser_summary_line,
            title=f"ser report  {label}",
            title_align="left",
            expand=True,
        )
    )

    spec_block = dict(metrics)
    spec_block["perf"] = _make_perf_block(batch_summary)
    return spec_block


def _persist_sec_or_esc_to_run(
    spec: TestSpec,
    label: str,
    results: list,
    batch_summary: dict,
    run_dir: EvalRunDir,
) -> dict:
    """SEC: pure generation task — persist transcripts only and record
    an n_total + note. Mirrors :func:`src.decode.save_sec_results_to_run`
    byte-for-byte so metrics.json stays identical between the two eval
    paths.

    ESC has its own metric path now (:func:`_persist_esc_to_run`); the
    function name is kept so external callers that still target it for
    SEC continue to work.
    """
    sorted_results = sorted(results, key=lambda r: r.get("id", ""))
    run_dir.write_test_transcripts(label, _records_for_disk(sorted_results))

    metrics = {
        "task": "sec",
        "n_total": len(sorted_results),
        "note": "SEC is a generation task — compute BLEU/ROUGE externally",
    }
    run_dir.write_test_metrics(label, metrics)
    run_dir.write_test_wer_report(
        label,
        f"# Test: {label}\nSEC is a generation task. "
        f"{len(sorted_results)} predictions saved to transcripts.jsonl. "
        "Use external scripts for BLEU / ROUGE.\n",
    )
    logger.info(
        f"[{label}] {len(sorted_results)} predictions saved "
        "(SEC: compute BLEU/ROUGE externally)"
    )

    spec_block = dict(metrics)
    spec_block["perf"] = _make_perf_block(batch_summary)
    return spec_block


def _persist_esc_to_run(
    spec: TestSpec,
    label: str,
    results: list,
    batch_summary: dict,
    run_dir: EvalRunDir,
) -> dict:
    """ESC: Tag-set Micro/Macro F1 + ROUGE-L F1.

    Mirrors :func:`src.decode.save_esc_results_to_run`: both paths route
    through :func:`compute_wer.compute_esc_metrics` and
    :func:`compute_wer.render_esc_report` so metrics.json + wer_report.txt
    are byte-identical for the same inputs (same regression-check
    invariant as SER's dual-path setup).
    """
    sorted_results = sorted(results, key=lambda r: r.get("id", ""))
    good = [r for r in sorted_results if "error" not in r]
    refs = [r.get("ref", "") or "" for r in good]
    hyps = [r.get("hyp", "") or "" for r in good]
    metrics = _cw_compute_esc_metrics(refs, hyps)
    report_text = _cw_render_esc_report(label, metrics)

    run_dir.write_test_transcripts(label, _records_for_disk(sorted_results))
    run_dir.write_test_metrics(label, metrics)
    run_dir.write_test_wer_report(label, report_text)

    _esc_summary_line = (
        f"tag_f1_micro={metrics['tag_micro_f1'] * 100:.2f}%  "
        f"tag_f1_macro={metrics['tag_macro_f1'] * 100:.2f}%  "
        f"rouge_l={metrics['rouge_l_f1'] * 100:.2f}%  "
        f"n={metrics['n_evaluated']}/{metrics['n_total']}"
    )
    _REPORT_CONSOLE.print(
        _RichPanel(
            report_text.rstrip() + "\n\n" + _esc_summary_line,
            title=f"esc report  {label}",
            title_align="left",
            expand=True,
        )
    )

    spec_block = dict(metrics)
    spec_block["perf"] = _make_perf_block(batch_summary)
    return spec_block


def _persist_asr_to_run(
    spec: TestSpec,
    label: str,
    results: list,
    batch_summary: dict,
    run_dir: EvalRunDir,
) -> dict:
    """ASR / TS-ASR / asr_hotwords: existing WER + silence path.

    Writes transcripts, wer_report.txt and metrics.json (the bare
    spec block produced by :func:`_spec_summary_for_run`, without the
    ``spec`` field which the run_plan caller re-attaches in summary.json).
    """
    run_dir.write_test_transcripts(label, _records_for_disk(results))
    run_dir.write_test_wer_report(
        label, _format_spec_report(label, batch_summary),
    )
    spec_block = _spec_summary_for_run(spec, batch_summary)
    spec_block.pop("spec", None)
    run_dir.write_test_metrics(label, spec_block)
    return spec_block


def _persist_spec_to_run(
    spec: TestSpec,
    label: str,
    results: list,
    batch_summary: dict,
    run_dir: EvalRunDir,
) -> dict:
    """Dispatch per-test persistence to the right per-task helper.

    Mirrors :func:`src.decode._persist_test_metrics` so both eval paths
    produce on-disk artefacts with identical schema for SER / SEC / ESC
    and the legacy ASR shape for everything else. Each helper writes
    transcripts.jsonl + metrics.json + wer_report.txt and returns the
    spec block that the caller merges into spec_summary for summary.json.
    """
    if spec.task == "ser":
        return _persist_ser_to_run(spec, label, results, batch_summary, run_dir)
    if spec.task == "esc":
        return _persist_esc_to_run(spec, label, results, batch_summary, run_dir)
    if spec.task == "sec":
        return _persist_sec_or_esc_to_run(
            spec, label, results, batch_summary, run_dir,
        )
    return _persist_asr_to_run(spec, label, results, batch_summary, run_dir)


def run_plan(args, parser):
    """Plan-mode entry: load TestPlan -> resolve cuts via registry ->
    run_batch per spec -> persist EvalRunDir layout. Mirrors decode.py."""
    plan = load_test_plan(
        plan_file=args.test_plan_file,
        test_dataset=args.test_dataset,
        cli_task=None,
        cli_overrides={
            "max_tokens": args.max_tokens,
            "num_samples": args.num_samples,
            "concurrency": args.concurrency,
        },
        args=args,
        validate_existence=True,
    )

    # Whisper backend can only do single-audio ASR. Pre-scan the plan and
    # fail loudly if NOTHING in it is runnable — saves the user from
    # spinning up retrieve caches and then finding every spec was skipped.
    # Mixed plans (ts_asr + asr) are fine: the unsupported ones get skipped
    # per-spec below and the supported ones still produce real metrics.
    if getattr(args, "backend", "chat") == "transcription":
        runnable = [
            t for t in plan.tests
            if t.task in TRANSCRIPTION_SUPPORTED_TASKS
        ]
        if not runnable:
            unsupported = sorted({t.task for t in plan.tests})
            parser.error(
                f"backend=transcription (whisper) 不支持 plan 里的任何任务 "
                f"({unsupported}); 仅支持: "
                f"{sorted(TRANSCRIPTION_SUPPORTED_TASKS)}. "
                f"请改用纯 ASR 的 plan, 或显式 -b chat 评测 audio-LLM。"
            )

    run_dir = EvalRunDir.create(
        base_dir=Path(args.output_dir),
        ckpt_label=args.model,
        run_name=plan.run_name or args.run_name,
    )
    run_dir.attach_file_logger()
    logger.info(f"[run:start] run_dir={run_dir.run_dir}")
    _tests_desc = ", ".join(f"{t.dataset}({t.task})" for t in plan.tests)
    logger.info(f"[run:plan] tests={len(plan.tests)} -> {_tests_desc}")
    run_dir.write_config({
        "args": {k: str(v) for k, v in sorted(vars(args).items())},
        "test_plan": plan.to_dict(),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    })

    summaries: list[dict] = []
    saved_args = {
        "task": args.task, "max_tokens": args.max_tokens,
        "num_samples": args.num_samples, "concurrency": args.concurrency,
        "default_lang": args.default_lang,
        "no_enrollment": args.no_enrollment, "no_language": args.no_language,
        "no_hotwords": args.no_hotwords,
        "hotwords_pad_to": getattr(args, "hotwords_pad_to", 0),
        "hotword_mode": getattr(args, "hotword_mode", "none"),
        "retrieve_top_k": getattr(args, "retrieve_top_k", 0),
        "retrieve_cache_dir": getattr(args, "retrieve_cache_dir", ""),
        "retrieve_workers": getattr(args, "retrieve_workers", 8),
        "random_hotword_pool_file": getattr(args, "random_hotword_pool_file", ""),
        "retrieve_hotword_pool_file": getattr(args, "retrieve_hotword_pool_file", ""),
        "tower_adapter_ckpt": getattr(args, "tower_adapter_ckpt", ""),
        "tower_base_model_path": getattr(args, "tower_base_model_path", ""),
        "tower_embed_dim": getattr(args, "tower_embed_dim", 512),
        "tower_adapter_hidden_dim": getattr(args, "tower_adapter_hidden_dim", None),
        "tower_hotword_pool_file": getattr(args, "tower_hotword_pool_file", ""),
        "tower_biasing_tsv_file": getattr(args, "tower_biasing_tsv_file", ""),
        "encoder_source": getattr(args, "encoder_source", "vllm"),
        "triton_url": getattr(args, "triton_url", ""),
        "triton_model": getattr(args, "triton_model", DEFAULT_TRITON_MODEL),
        "triton_top_k": getattr(args, "triton_top_k", 0),
        "spec_task": getattr(args, "spec_task", None),
    }
    # Default retrieve cache dir = <output_dir>/_retrieve_cache (cross-run
    # shared); empty string still means "disabled".
    default_cache_dir = (
        str(Path(args.output_dir) / "_retrieve_cache")
        if args.output_dir is not None else ""
    )

    # Pre-compute per-base_label max K for phase2 cache decoupling.
    # ``base_label`` = spec.label with the ``__retK{K}`` suffix stripped, so
    # every spec from one ``retrieve_top_k_sweep`` entry shares the same
    # base. We retrieve ``top_k_max`` hotwords once into the (K-agnostic)
    # phase2.v3 cache, then each spec slices ``retrieved[:spec_K]`` for
    # its own prompt — turning the previous N-cache-file sweep into one.
    # The 64 floor gives headroom for ad-hoc single-K specs without a
    # sweep companion (so a future K=32 run can reuse a K=20 sweep cache).
    base_label_to_max_K: dict[str, int] = {}
    for _s in plan.tests:
        if _s.overrides.get("hotword_mode") != "retrieve":
            continue
        _bl = _s.label.split("__retK", 1)[0]
        _k = int(_s.overrides.get("retrieve_top_k") or 0)
        if _k > 0:
            base_label_to_max_K[_bl] = max(
                base_label_to_max_K.get(_bl, 0), _k,
            )
    for _bl in list(base_label_to_max_K):
        base_label_to_max_K[_bl] = max(base_label_to_max_K[_bl], 64)

    for spec in plan.tests:
        spec_summary: dict = {
            "spec": spec.to_dict(),
            "started_at": datetime.now().isoformat(timespec="seconds"),
        }
        # Backend-level task gate: whisper transcription cannot service
        # ts_asr (needs enrollment+mixed) / ser / sec / esc (need
        # classification or generation prompts). Skip such specs cleanly
        # — write a stub metrics.json so aggregate_overall can count the
        # spec as skipped (not failed), and do NOT touch retrieve cache.
        if (
            getattr(args, "backend", "chat") == "transcription"
            and spec.task not in TRANSCRIPTION_SUPPORTED_TASKS
        ):
            skip_reason = (
                f"unsupported_by_backend (whisper transcription; "
                f"task='{spec.task}' not in "
                f"{sorted(TRANSCRIPTION_SUPPORTED_TASKS)})"
            )
            logger.warning(
                f"[run:skip] label='{spec.label}' task='{spec.task}' "
                f"-> {skip_reason}"
            )
            spec_summary["skipped"] = skip_reason
            spec_summary["finished_at"] = (
                datetime.now().isoformat(timespec="seconds")
            )
            # Persist a minimal metrics.json so the per_test/ directory is
            # consistent across the run; downstream xlsx renderers see
            # the ``skipped`` field and can omit the row.
            stub = {k: v for k, v in spec_summary.items() if k != "spec"}
            run_dir.write_test_metrics(spec.label, stub)
            summaries.append(spec_summary)
            continue
        try:
            test_cuts, _ = _resolve_test_cuts_from_dataset_name(
                dataset_name=spec.dataset, args=args,
            )
            # ts_asr keeps empty-ref negative samples (silence /
            # distractor); other tasks drop them so empty refs don't
            # accidentally influence WER.
            keep_empty = spec.task == "ts_asr"
            if isinstance(test_cuts, dict):
                # Multi-split entries (LibriSpeech / WenetSpeech) return a
                # dict-of-CutSet. Trim each then concat for a single batch
                # run; per-sub label info is lost here but per-language WER
                # below recovers the breakdown.
                from lhotse import CutSet
                merged = CutSet.from_cuts([
                    c
                    for sub in test_cuts.values()
                    for c in _trim_and_filter_cuts(
                        sub, keep_empty_text=keep_empty,
                    )
                ])
                items = cuts_to_items(
                    merged, limit=spec.overrides.get("num_samples"),
                )
            else:
                trimmed = _trim_and_filter_cuts(
                    test_cuts, keep_empty_text=keep_empty,
                )
                items = cuts_to_items(
                    trimmed, limit=spec.overrides.get("num_samples"),
                )

            # Override args for this spec only (restored after).
            args.task = _resolve_spec_task_to_args_task(spec)
            # Preserve the original (un-collapsed) spec task so train style
            # can keep ``Hotwords:N/A`` on K=0 baselines of asr_hotwords
            # specs (see _build_content_train).
            args.spec_task = spec.task
            if "max_tokens" in spec.overrides and spec.overrides["max_tokens"]:
                args.max_tokens = int(spec.overrides["max_tokens"])
            if "num_samples" in spec.overrides:
                args.num_samples = (
                    int(spec.overrides["num_samples"])
                    if spec.overrides["num_samples"] is not None
                    and int(spec.overrides["num_samples"]) > 0
                    else None
                )
            if "concurrency" in spec.overrides and spec.overrides["concurrency"]:
                args.concurrency = int(spec.overrides["concurrency"])
            if "hotwords_pad_to" in spec.overrides:
                args.hotwords_pad_to = int(
                    spec.overrides["hotwords_pad_to"] or 0
                )
            if "no_hotwords" in spec.overrides:
                args.no_hotwords = bool(spec.overrides["no_hotwords"])
            # Hotword mode: per-spec override > CLI default. Retrieve-only
            # knobs (retrieve_top_k / retrieve_cache_dir / retrieve_workers)
            # follow the same pattern.
            if "hotword_mode" in spec.overrides:
                args.hotword_mode = (
                    spec.overrides["hotword_mode"] or "none"
                )
            if "retrieve_top_k" in spec.overrides:
                args.retrieve_top_k = int(
                    spec.overrides["retrieve_top_k"] or 0
                )
            if "retrieve_cache_dir" in spec.overrides:
                args.retrieve_cache_dir = (
                    spec.overrides["retrieve_cache_dir"] or ""
                )
            if "retrieve_workers" in spec.overrides:
                args.retrieve_workers = int(
                    spec.overrides["retrieve_workers"] or 8
                )
            if "tower_adapter_ckpt" in spec.overrides:
                args.tower_adapter_ckpt = (
                    spec.overrides["tower_adapter_ckpt"] or ""
                )
            if "tower_base_model_path" in spec.overrides:
                args.tower_base_model_path = (
                    spec.overrides["tower_base_model_path"] or ""
                )
            if "tower_embed_dim" in spec.overrides:
                args.tower_embed_dim = int(
                    spec.overrides["tower_embed_dim"] or 512
                )
            if "tower_adapter_hidden_dim" in spec.overrides:
                v = spec.overrides["tower_adapter_hidden_dim"]
                args.tower_adapter_hidden_dim = int(v) if v else None
            if "random_hotword_pool_file" in spec.overrides:
                args.random_hotword_pool_file = (
                    spec.overrides["random_hotword_pool_file"] or ""
                )
            if "retrieve_hotword_pool_file" in spec.overrides:
                args.retrieve_hotword_pool_file = (
                    spec.overrides["retrieve_hotword_pool_file"] or ""
                )
            if "tower_hotword_pool_file" in spec.overrides:
                args.tower_hotword_pool_file = (
                    spec.overrides["tower_hotword_pool_file"] or ""
                )
            if "tower_biasing_tsv_file" in spec.overrides:
                args.tower_biasing_tsv_file = (
                    spec.overrides["tower_biasing_tsv_file"] or ""
                )
            args.output = None     # never write the legacy single-file dump

            if args.hotword_mode == "retrieve":
                if args.retrieve_top_k <= 0:
                    raise ValueError(
                        f"spec '{spec.label}': hotword_mode=retrieve "
                        f"requires retrieve_top_k > 0, got "
                        f"{args.retrieve_top_k}."
                    )

                # Cache directory resolution:
                #   "none" / "off" / "disabled" -> explicitly off
                #   "" / None                   -> default <output>/_retrieve_cache
                #   else                        -> use the supplied path
                raw_cache = (args.retrieve_cache_dir or "").strip()
                if raw_cache.lower() in {"none", "off", "disabled"}:
                    cache_dir_for_call: Optional[str] = None
                elif raw_cache == "":
                    cache_dir_for_call = default_cache_dir or None
                else:
                    cache_dir_for_call = raw_cache

                # Both phase1 and phase2 caches are K-agnostic in v3:
                # phase1 was always K-independent (no-hotword pass), and
                # phase2 in v3 caches ``top_k_max`` retrieved hotwords
                # once so sibling specs from the same
                # ``retrieve_top_k_sweep`` (e.g. __retK5 / __retK20)
                # collapse to one cache file under base_label.
                base_label = spec.label.split("__retK", 1)[0]
                # External pool file takes priority over manifest hotwords.
                # Resolved here (before make_cache_paths) so pool_tag can
                # be embedded in the phase2 cache filename, preventing stale
                # cache hits when the pool changes between runs.
                _retrieve_pool_file = (
                    getattr(args, "retrieve_hotword_pool_file", "") or ""
                ).strip()
                _pool_tag = (
                    Path(_retrieve_pool_file).stem if _retrieve_pool_file else ""
                )
                p1_cache, p2_cache = _ir_make_cache_paths(
                    cache_dir=cache_dir_for_call,
                    model=args.model,
                    dataset_label=base_label,
                    n_items=len(items),
                    pool_tag=_pool_tag,
                )
                # top_k_max comes from the pre-scan above; for single-K
                # specs without a sweep, fall back to max(64, spec_K).
                top_k_max = base_label_to_max_K.get(
                    base_label, max(64, args.retrieve_top_k),
                )
                logger.info(
                    f"[retrieve:start] label='{spec.label}'  "
                    f"top_k={args.retrieve_top_k}  "
                    f"top_k_max={top_k_max}  "
                    f"pool_file={_retrieve_pool_file or '(manifest)'}"
                    f"  cache={cache_dir_for_call or '(disabled)'}"
                )
                hypotheses = _ir_phase_transcribe(
                    args, items,
                    run_batch_fn=run_batch,
                    phase1_cache=p1_cache,
                    label=spec.label,
                )
                if _retrieve_pool_file:
                    if not Path(_retrieve_pool_file).exists():
                        raise FileNotFoundError(
                            f"[retrieve] retrieve_hotword_pool_file not found: "
                            f"{_retrieve_pool_file}"
                        )
                    with open(_retrieve_pool_file, encoding="utf-8") as _pf:
                        hotword_pool = sorted({
                            ln.strip() for ln in _pf if ln.strip()
                        })
                    logger.info(
                        "[retrieve] pool_file loaded: %d words from %s",
                        len(hotword_pool), _retrieve_pool_file,
                    )
                else:
                    hotword_pool = _ir_collect_hotword_pool(items)
                if not hotword_pool:
                    logger.warning(
                        f"[retrieve:skip] label='{spec.label}' "
                        f"reason='hotword pool empty' -> "
                        f"running plain phase3."
                    )
                    items_p3 = items
                else:
                    hw_map = _ir_phase_retrieve(
                        items, hypotheses,
                        hotword_pool=hotword_pool,
                        top_k_max=top_k_max,
                        workers=args.retrieve_workers,
                        language="auto",
                        phase2_cache=p2_cache,
                    )
                    items_p3 = _ir_inject_retrieved_into_items(
                        items, hw_map, top_k=args.retrieve_top_k,
                    )

                results, batch_summary = run_batch(args, items_p3, spec.label)
            elif args.hotword_mode == "tower_retrieve":
                if args.retrieve_top_k <= 0:
                    raise ValueError(
                        f"spec '{spec.label}': hotword_mode=tower_retrieve "
                        f"requires retrieve_top_k > 0, got "
                        f"{args.retrieve_top_k}."
                    )
                if not args.tower_base_model_path:
                    raise ValueError(
                        f"spec '{spec.label}': hotword_mode=tower_retrieve "
                        f"requires --tower-base-model-path (or YAML field "
                        f"tower_base_model_path) to be set."
                    )

                raw_cache = (args.retrieve_cache_dir or "").strip()
                if raw_cache.lower() in {"none", "off", "disabled"}:
                    tower_cache_dir: Optional[str] = None
                elif raw_cache == "":
                    tower_cache_dir = default_cache_dir or None
                else:
                    tower_cache_dir = raw_cache

                base_label = spec.label.split("__retK", 1)[0]
                top_k_max = base_label_to_max_K.get(
                    base_label, max(64, args.retrieve_top_k),
                )
                logger.info(
                    f"[tower_retrieve:start] label='{spec.label}'  "
                    f"top_k={args.retrieve_top_k}  "
                    f"top_k_max={top_k_max}  "
                    f"adapter={args.tower_adapter_ckpt}  "
                    f"cache={tower_cache_dir or '(disabled)'}"
                )
                # Candidate pool: prefer biasing TSV (per-utterance list) >
                # external pool file (hard distractors) >
                # manifest's true-hotword union (easy, no distractors).
                _tsv_file = (args.tower_biasing_tsv_file or "").strip()
                _pool_file = (args.tower_hotword_pool_file or "").strip()
                if _tsv_file or _pool_file:
                    hotword_pool_for_neural: list[str] = []
                else:
                    hotword_pool_for_neural = _ir_collect_hotword_pool(items)

                if not _tsv_file and not _pool_file and not hotword_pool_for_neural:
                    logger.warning(
                        f"[tower_retrieve:skip] label='{spec.label}' "
                        f"reason='hotword pool empty' -> "
                        f"running plain phase3."
                    )
                    items_p3 = items
                else:
                    hw_map = _ir_phase_retrieve_neural(
                        items, hotword_pool_for_neural,
                        top_k_max=top_k_max,
                        adapter_ckpt=args.tower_adapter_ckpt,
                        base_model_path=args.tower_base_model_path,
                        embed_dim=args.tower_embed_dim,
                        adapter_hidden_dim=args.tower_adapter_hidden_dim,
                        hotword_pool_file=_pool_file or None,
                        biasing_tsv_file=_tsv_file or None,
                        cache_dir=tower_cache_dir or None,
                    )
                    items_p3 = _ir_inject_retrieved_into_items(
                        items, hw_map, top_k=args.retrieve_top_k,
                    )

                results, batch_summary = run_batch(args, items_p3, spec.label)
            else:
                results, batch_summary = run_batch(args, items, spec.label)
            spec_block = _persist_spec_to_run(
                spec=spec,
                label=spec.label,
                results=results,
                batch_summary=batch_summary,
                run_dir=run_dir,
            )
            spec_summary.update(spec_block)
        except Exception as e:
            logger.exception("Test '%s' failed; continuing.", spec.label)
            spec_summary["error"] = str(e)
            run_dir.write_test_metrics(spec.label, spec_summary)
        finally:
            # Restore defaults for the next spec.
            args.task = saved_args["task"]
            args.max_tokens = saved_args["max_tokens"]
            args.num_samples = saved_args["num_samples"]
            args.concurrency = saved_args["concurrency"]
            args.no_enrollment = saved_args["no_enrollment"]
            args.no_language = saved_args["no_language"]
            args.no_hotwords = saved_args["no_hotwords"]
            args.hotwords_pad_to = saved_args["hotwords_pad_to"]
            args.hotword_mode = saved_args["hotword_mode"]
            args.retrieve_top_k = saved_args["retrieve_top_k"]
            args.retrieve_cache_dir = saved_args["retrieve_cache_dir"]
            args.retrieve_workers = saved_args["retrieve_workers"]
            args.random_hotword_pool_file = saved_args["random_hotword_pool_file"]
            args.retrieve_hotword_pool_file = saved_args["retrieve_hotword_pool_file"]
            args.tower_adapter_ckpt = saved_args["tower_adapter_ckpt"]
            args.tower_base_model_path = saved_args["tower_base_model_path"]
            args.tower_embed_dim = saved_args["tower_embed_dim"]
            args.tower_adapter_hidden_dim = saved_args["tower_adapter_hidden_dim"]
            args.tower_hotword_pool_file = saved_args["tower_hotword_pool_file"]
            args.tower_biasing_tsv_file = saved_args["tower_biasing_tsv_file"]
            args.encoder_source = saved_args["encoder_source"]
            args.triton_url = saved_args["triton_url"]
            args.triton_model = saved_args["triton_model"]
            args.triton_top_k = saved_args["triton_top_k"]
            args.spec_task = saved_args["spec_task"]

        spec_summary["finished_at"] = datetime.now().isoformat(timespec="seconds")
        summaries.append(spec_summary)

    aggregate = aggregate_overall(summaries)
    summary_obj = {
        "run_id": run_dir.run_id,
        "source": "vllm",
        "model": {
            "served_model_name": args.model,
            "vllm_url": args.base_url,
        },
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "test_plan_file": args.test_plan_file,
        "tests": summaries,
        "aggregate": aggregate,
    }
    json_path, md_path = run_dir.write_summary(summary_obj)
    logger.info(f"[run:summary] json={json_path}  md={md_path}")
    failed = [t for t in summaries if t.get("error") is not None]
    if failed:
        _failed_labels = ", ".join(t["spec"]["label"] for t in failed)
        logger.warning(
            f"[run:failed] {len(failed)}/{len(summaries)} tests failed: "
            f"{_failed_labels}"
        )
    skipped = [t for t in summaries if t.get("skipped") is not None]
    if skipped:
        _skipped_labels = ", ".join(t["spec"]["label"] for t in skipped)
        logger.warning(
            f"[run:skipped] {len(skipped)}/{len(summaries)} tests skipped "
            f"(backend={getattr(args, 'backend', '?')}): {_skipped_labels}"
        )


if __name__ == "__main__":
    main()
