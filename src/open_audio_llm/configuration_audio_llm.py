"""Hugging Face configuration for composable Audio-LLM models."""

from __future__ import annotations

from transformers import AutoConfig, PretrainedConfig


class AudioTowerConfig(PretrainedConfig):
    model_type = "audio_llm_audio_tower"

    def __init__(
        self,
        type: str = "identity",
        input_dim: int = 128,
        output_dim: int = 128,
        feature_extractor_type: str = "whisper",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.type = type
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.feature_extractor_type = feature_extractor_type


class ConnectorConfig(PretrainedConfig):
    model_type = "audio_llm_connector"

    def __init__(
        self,
        type: str = "mlp_downsample",
        input_dim: int = 128,
        output_dim: int = 128,
        downsample_rate: int = 1,
        dropout: float = 0.1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.type = type
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.downsample_rate = downsample_rate
        self.dropout = dropout


class MergeConfig(PretrainedConfig):
    model_type = "audio_llm_merge"

    def __init__(
        self,
        type: str = "replace_slots",
        modality_token: str = "<speech>",
        max_audio_slots: int = 4,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.type = type
        self.modality_token = modality_token
        self.max_audio_slots = max_audio_slots


class AudioLLMConfig(PretrainedConfig):
    """Configuration for `AudioTower + Connector + CausalLM` composition."""

    model_type = "audio_llm"

    def __init__(
        self,
        audio_tower_config: dict | AudioTowerConfig | None = None,
        connector_config: dict | ConnectorConfig | None = None,
        text_config: dict | PretrainedConfig | None = None,
        merge_config: dict | MergeConfig | None = None,
        num_prompt_tokens: int = 4,
        default_speech_token_id: int | None = None,
        start_text_token_id: int | None = None,
        end_text_token_id: int | None = None,
        start_speech_token_id: int | None = None,
        end_speech_token_id: int | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.audio_tower_config = self._coerce_audio_tower(audio_tower_config)
        self.connector_config = self._coerce_connector(connector_config)
        self.merge_config = self._coerce_merge(merge_config)
        self.text_config = self._coerce_text(text_config)

        self.num_prompt_tokens = num_prompt_tokens
        self.default_speech_token_id = default_speech_token_id
        self.start_text_token_id = start_text_token_id
        self.end_text_token_id = end_text_token_id
        self.start_speech_token_id = start_speech_token_id
        self.end_speech_token_id = end_speech_token_id

        self.architectures = kwargs.get(
            "architectures",
            ["AudioLLMForConditionalGeneration"],
        )
        self.auto_map = {
            "AutoConfig": "configuration_audio_llm.AudioLLMConfig",
            "AutoModel": "modeling_audio_llm.AudioLLMForConditionalGeneration",
            "AutoModelForCausalLM": "modeling_audio_llm.AudioLLMForConditionalGeneration",
        }

    @staticmethod
    def _coerce_audio_tower(value):
        if value is None:
            return AudioTowerConfig()
        return value if isinstance(value, AudioTowerConfig) else AudioTowerConfig(**value)

    @staticmethod
    def _coerce_connector(value):
        if value is None:
            return ConnectorConfig()
        return value if isinstance(value, ConnectorConfig) else ConnectorConfig(**value)

    @staticmethod
    def _coerce_merge(value):
        if value is None:
            return MergeConfig()
        return value if isinstance(value, MergeConfig) else MergeConfig(**value)

    @staticmethod
    def _coerce_text(value):
        if value is None:
            return {}
        if isinstance(value, dict) and "model_type" in value:
            return AutoConfig.for_model(**value)
        return value

    def to_dict(self):
        output = super().to_dict()
        for key in ("audio_tower_config", "connector_config", "merge_config", "text_config"):
            value = getattr(self, key)
            if hasattr(value, "to_dict"):
                output[key] = value.to_dict()
        return output
