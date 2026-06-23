"""Register Open Audio-LLM with vLLM ModelRegistry.

This file is intentionally importable as an ms-swift ``--external_plugins``
entry so rollout workers register the architecture before constructing vLLM.
"""

from open_audio_llm.integrations.vllm.plugin import register

register()
