"""vLLM-compatible Fun-ASR-Nano-2512 wrapper.

Design (transcription-endpoint route)
-------------------------------------
vLLM 上游 (`vllm/model_executor/models/funasr.py`) 已经为同名
`FunASRForConditionalGeneration` arch 给出了完整的、能跑的实现 —— 模型本体
(encoder + adaptor + Qwen3 decoder)、多模态 processor、PromptReplacement
都齐了。我们 plugin 不重新搭模型 / processor, 只做 **一件事**:

    复写 `get_generation_prompt`, 把 AmphionASR 客户端通过 OpenAI
    `/v1/audio/transcriptions` 的 `prompt` 字段送进来的元数据
    (`Language: zh` / `Hotwords: a,b,c` 这两行) 翻译成 Fun-ASR 训练时使用的
    中文 prompt 模板, 这样热词才会真的进 attention context.

为什么走 transcription endpoint 而非 chat completions?

- 上游 `FunASRForConditionalGeneration.supports_transcription_only = True`,
  vLLM OpenAI server 只挂 `/v1/audio/transcriptions`, 这是设计好的入口.
- 之前尝试把它翻成 False + 在 multimodal processor 里改写 chat-prompt 不
  靠谱 — chat completions 走的多模态数据流跟 transcription 不同, 我们曾经
  得到一连串 500/dtype/shape 错. transcription 路径是 vLLM 团队真正测过的.
- 客户端 (`test_vllm_inference.py:call_vllm_api`) 这边 cost 极低: 同样的
  base64 WAV 解出 bytes 走 multipart `file`, content 文本拼好填进 `prompt`
  字段, response 拿 `text` 就完事.

热词不是模型 ABI
------------------
看 `funasr.AutoModel(...).generate(hotwords=[...])` 的源码
(`funasr/models/fun_asr_nano/model.py:FunASRNano.get_prompt` & `.inference`),
那个 `hotwords=` 参数在 SDK 入口立刻就被拼成 prompt 文本, 底层 model 并不
存在 "hotwords" 这个张量参数. 所以无论用哪种 serving 方式, **热词只能通
过 prompt 文本传**, 我们做的只是把 prompt 文本拼对.
"""

from __future__ import annotations

import re
from typing import Literal, cast

import numpy as np

from vllm.config import ModelConfig, SpeechToTextConfig
from vllm.inputs.data import PromptType
from vllm.model_executor.models.funasr import FunASRForConditionalGeneration

# ---------------------------------------------------------------------------
# Prompt assembly (byte-identical to funasr.AutoModel SDK)
# ---------------------------------------------------------------------------

# vLLM tokenizer 把 funasr 原始 `<|startofspeech|>!!<|endofspeech|>` 替换为
# 单个 `<|AUDIO|>` token, 然后 `FunASRMultiModalProcessor._get_prompt_updates`
# 在 PromptReplacement 里再用 audio embed 替换之. 所以 user-content 末尾必
# 须出现 `<|AUDIO|>`, 别用 funasr 训练时的原始 marker.
_AUDIO_PLACEHOLDER_TOKEN = "<|AUDIO|>"

# AmphionASR client (swift / train 风格) -> Fun-ASR 训练用语言中文名.
# Fun-ASR-Nano-2512 训练时只见过这三种语言, MLT 版另外支持 31 种, 这里
# 用一个超集映射, miss 的语言走 fall-through (prompt 不写语言, 让模型自
# 己判断).
_LANG_MAP: dict[str, str] = {
    "zh": "中文", "zh-cn": "中文", "zh-hans": "中文", "cmn": "中文",
    "chinese": "中文", "mandarin": "中文",
    "en": "英文", "en-us": "英文", "en-gb": "英文", "english": "英文",
    "ja": "日文", "ja-jp": "日文", "jp": "日文", "japanese": "日文",
}

# AmphionASR swift style 写: `Language: zh` (冒号后有空格)
# AmphionASR train style 写:  `Language: zh` 同样格式 (train.py 里
# 实际没有 Language 行, 但 client `_build_content_swift` 一定有)
_LANG_RE = re.compile(r"^Language:\s*(\S+)\s*$", re.M)

# Swift style: `Hotwords: w1,w2,w3` (逗号无空格, 跟 _format_hotwords 一致)
# Train style: `Hotwords:w1,w2,w3` (冒号后也无空格)
# 两者都允许, 也允许 `Hotwords: [w1, w2, ...]` 这种用户自己手写的方括号
# 写法 — 我们一律 strip 掉外层 `[]`, 然后 comma-split.
_HOTWORDS_RE = re.compile(r"^Hotwords:\s*(.*?)\s*$", re.M)


def _parse_hotwords(s: str) -> list[str]:
    """Accept ``a,b,c`` / ``a, b, c`` / ``[a, b, c]`` / ``N/A`` and return a
    cleaned list. ``N/A`` (case-insensitive) maps to empty list — that
    matches AmphionASR's K=0 baseline convention.
    """
    s = s.strip()
    if not s or s.upper() == "N/A":
        return []
    if s.startswith("[") and s.endswith("]"):
        s = s[1:-1]
    return [w.strip() for w in s.split(",") if w.strip()]


def _build_funasr_user_prompt(
    hotwords: list[str],
    language: str | None,
    itn: bool = True,
) -> str:
    """Byte-identical replica of
    ``funasr/models/fun_asr_nano/model.py:FunASRNano.get_prompt``.
    Must stay byte-stable because the model was SFT'd against exactly
    this wording.
    """
    if hotwords:
        hw_str = ", ".join(hotwords)
        p = (
            "请结合上下文信息，更加准确地完成语音转写任务。"
            "如果没有相关信息，我们会留空。\n\n\n"
            "**上下文信息：**\n\n\n"
            f"热词列表：[{hw_str}]\n"
        )
    else:
        p = ""
    p += f"语音转写成{language}" if language else "语音转写"
    if not itn:
        p += "，不进行文本规整"
    p += "："
    return p


def _funasr_chat_template(user_prompt: str) -> str:
    """Wrap a single user-content prompt into the Qwen3 chat template that
    Fun-ASR-Nano-2512 was trained with. `<|AUDIO|>` is appended *inside*
    the user content (right after the colon), mirroring funasr's
    ``generate_chatml`` ({prompt}<|startofspeech|>!!<|endofspeech|>) and
    matching upstream vLLM's hardcoded template (`语音转写：<|AUDIO|>`).
    """
    return (
        "<|im_start|>system\n"
        "You are a helpful assistant."
        "<|im_end|>\n"
        "<|im_start|>user\n"
        f"{user_prompt}{_AUDIO_PLACEHOLDER_TOKEN}"
        "<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


# ---------------------------------------------------------------------------
# Model: inherit upstream verbatim, only override prompt assembly
# ---------------------------------------------------------------------------


class FunASRForVLLM(FunASRForConditionalGeneration):
    """Fun-ASR-Nano-2512 wrapper that propagates AmphionASR-style hotwords
    & language metadata (sent via OpenAI transcription `prompt` field)
    into the Fun-ASR-native Chinese prompt template the model expects.

    Everything else — weight layout, encoder/adaptor/decoder forward,
    multimodal processor, PromptReplacement on `<|AUDIO|>` — is inherited
    from `vllm.model_executor.models.funasr.FunASRForConditionalGeneration`.
    """

    # Upstream default = True; keep it. This is what makes the OpenAI
    # server mount `/v1/audio/transcriptions` for us.
    supports_transcription_only = True

    @classmethod
    def get_generation_prompt(
        cls,
        audio: np.ndarray,
        model_config: ModelConfig,
        stt_config: SpeechToTextConfig,
        language: str | None,
        task_type: Literal["transcribe", "translate"],
        request_prompt: str,
        to_language: str | None,
    ) -> PromptType:
        """Assemble the model prompt with hotwords / language extracted
        from the OpenAI `prompt` field.

        `request_prompt` is exactly what the client put in the
        transcription request's `prompt` parameter — for AmphionASR's
        eval client that's the AmphionASR-style content block, e.g.::

            Transcribe the following audio.
            Language: zh
            Hotwords: 开放时间,运营商

        We don't care about the leading "Transcribe ..." line (Fun-ASR
        was never trained on that English instruction), only the
        ``Language:`` and ``Hotwords:`` annotations. Anything we can't
        parse is silently dropped — the model then sees the bare
        ``语音转写：`` prompt, equivalent to upstream behaviour.
        """
        rp = request_prompt or ""

        # 1. Language: prefer the explicit `Language:` line in the request
        #    prompt (that's what AmphionASR's client actually writes); fall
        #    back to the OpenAI `language` field. Important: upstream
        #    `validate_language()` defaults `language=None` to "en", so the
        #    OpenAI field is almost never None and would mask the
        #    in-prompt annotation if we trusted it first.
        lang_code: str | None = None
        m = _LANG_RE.search(rp)
        if m:
            lang_code = m.group(1)
        if not lang_code:
            lang_code = language
        lang_zh = _LANG_MAP.get((lang_code or "").lower())

        # 2. Hotwords: only from the request prompt.
        hotwords: list[str] = []
        m = _HOTWORDS_RE.search(rp)
        if m:
            hotwords = _parse_hotwords(m.group(1))

        # 3. Build the funasr-native user prompt and wrap in chat template.
        user_prompt = _build_funasr_user_prompt(hotwords, lang_zh)
        full_prompt = _funasr_chat_template(user_prompt)

        prompt: PromptType = {
            "prompt": full_prompt,
            "multi_modal_data": {
                "audio": (audio, stt_config.sample_rate),
            },
        }
        return cast(PromptType, prompt)
