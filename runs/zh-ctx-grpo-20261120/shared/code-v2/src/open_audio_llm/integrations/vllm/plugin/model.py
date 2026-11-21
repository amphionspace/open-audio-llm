"""Initial vLLM model class.

This class keeps the registration target stable. A version-specific optimized
PagedAttention implementation can later replace the inheritance layer without
changing checkpoint names or plugin entry points.
"""

from __future__ import annotations

from open_audio_llm.modeling_audio_llm import AudioLLMForConditionalGeneration


class AudioLLMForVLLM(AudioLLMForConditionalGeneration):
    pass
