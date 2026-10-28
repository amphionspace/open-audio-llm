"""Register AudioLLM with ms-swift.

Use with:

    swift sft --external_plugins path/to/register_audio_llm.py --model <model_dir>
"""

from __future__ import annotations

from open_audio_llm.integrations.ms_swift.template import AudioLLMTemplate


try:
    from swift.model import ModelLoader
except ImportError:  # pragma: no cover - newer ms-swift API path
    ModelLoader = object


class AudioLLMLoader(ModelLoader):
    def get_model(self, model_dir: str, *args, **kwargs):
        from transformers import AutoModelForCausalLM

        self.auto_model_cls = self.auto_model_cls or AutoModelForCausalLM
        return super().get_model(model_dir, *args, **kwargs)


def register():
    loader = None
    try:
        from swift.llm import (
            Model,
            ModelGroup,
            ModelMeta,
            get_model_tokenizer_with_flash_attn,
            register_model,
        )
        from swift.llm.model.model_arch import MultiModelKeys, register_model_arch
    except ImportError:  # pragma: no cover - ms-swift API compatibility
        from swift.model import (
            Model,
            ModelGroup,
            ModelMeta,
            MultiModelKeys,
            register_model,
            register_model_arch,
        )

        loader = AudioLLMLoader
    from swift.template import TemplateMeta, register_template

    try:
        arch_keys = MultiModelKeys(
            "audio_llm",
            language_model="language_model",
            aligner="connector",
            vision_tower="audio_tower",
        )
    except TypeError:
        arch_keys = MultiModelKeys(
            "audio_llm",
            language_model="language_model",
            connector="connector",
            tower_model="audio_tower",
        )

    register_model_arch(arch_keys)

    register_model(
        ModelMeta(
            "audio_llm",
            [ModelGroup([Model("audio_llm", "audio_llm")], "audio_llm")],
            loader or get_model_tokenizer_with_flash_attn,
            template="audio_llm",
            model_arch="audio_llm",
            architectures=["AudioLLMForConditionalGeneration"],
            is_multimodal=True,
            requires=["librosa"],
        )
    )

    register_template(
        TemplateMeta(
            "audio_llm",
            prefix=["<|im_start|>system\n{{SYSTEM}}<|im_end|>\n"],
            prompt=["<|im_start|>user\n{{QUERY}}<|im_end|>\n<|im_start|>assistant\n"],
            chat_sep=["<|im_end|>\n"],
            suffix=["<|im_end|>"],
            default_system=None,
            template_cls=AudioLLMTemplate,
        )
    )


register()
