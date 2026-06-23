"""Feature processor factories.

The core package keeps processors optional so importing the model does not
force audio IO libraries into every environment.
"""

from __future__ import annotations


def build_feature_processor(feature_extractor_type: str, model_dir: str | None = None):
    if feature_extractor_type == "whisper":
        from transformers import WhisperFeatureExtractor

        return (
            WhisperFeatureExtractor.from_pretrained(model_dir)
            if model_dir
            else WhisperFeatureExtractor()
        )
    if feature_extractor_type == "kaldi_fbank":
        return None
    raise ValueError(f"Unsupported feature processor: {feature_extractor_type}")
