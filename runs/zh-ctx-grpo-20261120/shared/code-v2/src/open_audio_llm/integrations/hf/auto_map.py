"""Utilities for making saved checkpoints loadable with `trust_remote_code`."""

from __future__ import annotations

import json
from pathlib import Path


AUTO_MAP = {
    "AutoConfig": "configuration_audio_llm.AudioLLMConfig",
    "AutoModel": "modeling_audio_llm.AudioLLMForConditionalGeneration",
    "AutoModelForCausalLM": "modeling_audio_llm.AudioLLMForConditionalGeneration",
}


def patch_config_auto_map(model_dir: str | Path) -> None:
    config_path = Path(model_dir) / "config.json"
    with config_path.open("r", encoding="utf-8") as f:
        config = json.load(f)
    config["auto_map"] = AUTO_MAP
    config["architectures"] = ["AudioLLMForConditionalGeneration"]
    with config_path.open("w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
        f.write("\n")
