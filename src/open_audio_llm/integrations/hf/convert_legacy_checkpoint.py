"""Legacy upstream checkpoint conversion entry point.

This module defines the target config and file-copy contract for the new
project. Full key remapping from historical `.pt` checkpoints remains isolated
here so the runtime model does not carry legacy branches. It supports checkpoint
formats inherited from AmphionASR migration work.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, GenerationConfig

from open_audio_llm.constants import (
    DEFAULT_SPEECH_TOKEN,
    END_SPEECH_TOKEN,
    END_TEXT_TOKEN,
    SPECIAL_TOKENS,
    START_SPEECH_TOKEN,
    START_TEXT_TOKEN,
)
from open_audio_llm.configuration_audio_llm import (
    AudioLLMConfig,
    AudioTowerConfig,
    ConnectorConfig,
)
from .auto_map import patch_config_auto_map

logger = logging.getLogger(__name__)

CHAT_TEMPLATE_JINJA = """\
{%- for message in messages %}
    {%- set content = message.content if message.content is string else '' %}
    {%- if message.role == "user" %}
        {{- '<|im_start|>user\n' + content.replace('<audio>', '<start_speech><speech><end_speech>') + '<|im_end|>\n' }}
    {%- elif message.role == "assistant" %}
        {{- '<|im_start|>assistant\n' + content + '<|im_end|>\n' }}
    {%- elif message.role == "system" %}
        {{- '<|im_start|>system\n' + content + '<|im_end|>\n' }}
    {%- endif %}
{%- endfor %}
{%- if add_generation_prompt %}
    {{- '<|im_start|>assistant\n' }}
{%- endif %}
"""


def _load_state(path: str | None) -> dict[str, torch.Tensor]:
    if not path:
        return {}
    state = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(state, dict):
        return state.get("model", state.get("state_dict", state))
    raise TypeError(f"Unsupported checkpoint type at {path}: {type(state)!r}")


def _legacy_key_mapping(encoder_type: str) -> list[tuple[str, str]]:
    if encoder_type == "zipformer":
        return [
            ("encoder_embed.", "audio_tower.encoder.encoder_embed."),
            ("encoder.", "audio_tower.encoder.encoder."),
            ("encoder_projector.", "connector."),
            ("llm.", "language_model."),
            ("prompt_embedding.", "slot_merger.prompt_embedding."),
        ]
    return [
        ("audio_encoder.encoder.", "audio_tower.encoder."),
        ("multi_modal_projector.", "connector."),
        ("language_model.", "language_model."),
        ("prompt_embedding.", "slot_merger.prompt_embedding."),
        ("encoder.", "audio_tower.encoder."),
        ("encoder_projector.", "connector."),
        ("llm.", "language_model."),
        ("prompt_embedding.", "slot_merger.prompt_embedding."),
    ]


def remap_legacy_key(key: str, encoder_type: str = "qwen3asr") -> str:
    """Map AmphionASR checkpoint keys to Open Audio-LLM module keys."""

    for old_prefix, new_prefix in _legacy_key_mapping(encoder_type):
        if key.startswith(old_prefix):
            return new_prefix + key[len(old_prefix) :]
    return key


def _strip_lora_wrapper(key: str) -> str:
    prefix = "base_model.model."
    return key[len(prefix) :] if key.startswith(prefix) else key


def _merge_lora_into_combined(model_state: dict, combined: dict, alpha: float) -> None:
    lora_a: dict[str, torch.Tensor] = {}
    lora_b: dict[str, torch.Tensor] = {}
    for key, value in model_state.items():
        if "lora_A" in key and key.startswith("llm."):
            base_key = key.replace("base_model.model.", "").split(".lora_A")[0]
            lora_a[base_key.replace("llm.", "")] = value
        elif "lora_B" in key and key.startswith("llm."):
            base_key = key.replace("base_model.model.", "").split(".lora_B")[0]
            lora_b[base_key.replace("llm.", "")] = value

    if not lora_a:
        return

    rank = next(iter(lora_a.values())).shape[0]
    scale = alpha / rank
    merged = 0
    for base_key, a_weight in lora_a.items():
        b_weight = lora_b.get(base_key)
        if b_weight is None:
            logger.warning("LoRA A without B for %s; skipping", base_key)
            continue
        full_key = f"language_model.{base_key}.weight"
        if full_key not in combined:
            logger.warning("Base weight %s not found; skipping LoRA merge", full_key)
            continue
        delta = (b_weight.float() @ a_weight.float()) * scale
        if delta.shape != combined[full_key].shape:
            raise RuntimeError(
                f"LoRA merge shape mismatch for {full_key}: "
                f"{tuple(delta.shape)} vs {tuple(combined[full_key].shape)}"
            )
        combined[full_key] = combined[full_key].float() + delta
        merged += 1
    logger.info("Merged %d LoRA matrices with alpha=%s", merged, alpha)


def _build_combined_state(args: argparse.Namespace) -> dict:
    if args.legacy_hf_dir:
        return _load_legacy_hf_state(Path(args.legacy_hf_dir), args.audio_tower_type)

    model_state = _load_state(args.checkpoint)
    encoder_state = _load_state(args.encoder_weights)
    combined: dict[str, torch.Tensor] = {}

    if not args.skip_base_lm_weights:
        logger.info("Loading base LLM weights from %s", args.llm_path)
        llm = AutoModelForCausalLM.from_pretrained(
            args.llm_path,
            torch_dtype=torch.float32,
            trust_remote_code=True,
        )
        for key, value in llm.state_dict().items():
            combined[f"language_model.{key}"] = value

    has_lora = any("lora_" in key for key in model_state)
    if has_lora and args.merge_lora:
        _merge_lora_into_combined(model_state, combined, args.lora_alpha)

    for key, value in model_state.items():
        if "lora_" in key:
            continue
        mapped = remap_legacy_key(key, args.audio_tower_type)
        if mapped.startswith("language_model."):
            lm_key = _strip_lora_wrapper(mapped[len("language_model.") :])
            mapped = f"language_model.{lm_key}"
        combined[mapped] = value

    if encoder_state and not any(key.startswith("audio_tower.encoder.") for key in combined):
        logger.info("Filling audio tower weights from standalone encoder state")
        for key, value in encoder_state.items():
            combined[f"audio_tower.encoder.{key}"] = value

    projection_fragments = (".proj1.", ".act.", ".proj2.")
    for key in [key for key in combined if any(fragment in key for fragment in projection_fragments)]:
        if key.startswith("audio_tower.encoder."):
            del combined[key]

    return combined


def _load_legacy_hf_state(model_dir: Path, encoder_type: str) -> dict:
    """Load and remap an already-converted AmphionASR HF checkpoint."""

    index_path = model_dir / "model.safetensors.index.json"
    if index_path.exists():
        index = json.loads(index_path.read_text(encoding="utf-8"))
        files = sorted(set(index["weight_map"].values()))
    else:
        files = [path.name for path in sorted(model_dir.glob("*.safetensors"))]
    if not files:
        raise FileNotFoundError(f"No safetensors files found in {model_dir}")

    combined = {}
    for filename in files:
        shard_path = model_dir / filename
        logger.info("Loading legacy HF shard %s", shard_path)
        shard = load_file(shard_path)
        for key, value in shard.items():
            mapped = remap_legacy_key(key, encoder_type)
            combined[mapped] = value
    return combined


def _write_remote_code_wrappers(out_dir: Path) -> None:
    (out_dir / "configuration_audio_llm.py").write_text(
        "from open_audio_llm.configuration_audio_llm import AudioLLMConfig\n",
        encoding="utf-8",
    )
    (out_dir / "modeling_audio_llm.py").write_text(
        "from open_audio_llm.modeling_audio_llm import AudioLLMForConditionalGeneration\n",
        encoding="utf-8",
    )


def build_config(args: argparse.Namespace) -> AudioLLMConfig:
    legacy_config = {}
    if args.legacy_hf_dir:
        legacy_config_path = Path(args.legacy_hf_dir) / "config.json"
        legacy_config = json.loads(legacy_config_path.read_text(encoding="utf-8"))
        text_config = legacy_config.get("text_config", {})

        class _TextConfigProxy:
            hidden_size = text_config.get("hidden_size", args.llm_dim)

        text_config_obj = _TextConfigProxy()
    else:
        text_config_obj = AutoConfig.from_pretrained(args.llm_path, trust_remote_code=True)
        text_config = text_config_obj.to_dict()

    if args.encoder_config:
        with Path(args.encoder_config).open("r", encoding="utf-8") as handle:
            encoder_config = json.load(handle)
    elif legacy_config:
        encoder_config = legacy_config.get("audio_encoder_config", {})
    else:
        encoder_config = {}
    projector_config = legacy_config.get("projector_config", {}) if legacy_config else {}

    audio_input_dim = args.audio_input_dim
    audio_output_dim = args.audio_output_dim
    if encoder_config:
        audio_input_dim = int(
            encoder_config.get("num_mel_bins", encoder_config.get("feature_dim", audio_input_dim))
        )
        audio_output_dim = int(
            encoder_config.get("output_dim", encoder_config.get("d_model", audio_output_dim))
        )
    if projector_config:
        audio_output_dim = int(projector_config.get("encoder_dim", audio_output_dim))
        args.llm_dim = int(projector_config.get("llm_dim", args.llm_dim))
        args.downsample_rate = int(
            projector_config.get("downsample_rate", args.downsample_rate)
        )
    audio_tower_kwargs = dict(encoder_config)
    for duplicate_key in ("type", "input_dim", "output_dim", "feature_extractor_type"):
        audio_tower_kwargs.pop(duplicate_key, None)

    return AudioLLMConfig(
        audio_tower_config=AudioTowerConfig(
            type=args.audio_tower_type,
            input_dim=audio_input_dim,
            output_dim=audio_output_dim,
            feature_extractor_type=args.feature_extractor_type,
            **audio_tower_kwargs,
        ),
        connector_config=ConnectorConfig(
            type=args.connector_type,
            input_dim=audio_output_dim,
            output_dim=getattr(
                text_config_obj,
                "hidden_size",
                args.llm_dim,
            ),
            downsample_rate=args.downsample_rate,
        ),
        text_config=text_config,
        default_speech_token_id=args.default_speech_token_id,
        start_text_token_id=args.start_text_token_id,
        end_text_token_id=args.end_text_token_id,
        start_speech_token_id=args.start_speech_token_id,
        end_speech_token_id=args.end_speech_token_id,
    )


def convert_checkpoint(args: argparse.Namespace) -> Path:
    args.audio_tower_type = args.audio_tower_type or "qwen3asr"
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer_source = args.llm_path or args.legacy_hf_dir
    if not tokenizer_source:
        raise ValueError("--llm-path or --legacy-hf-dir is required")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, trust_remote_code=True)
    tokenizer.add_special_tokens({"additional_special_tokens": SPECIAL_TOKENS})
    if args.default_speech_token_id is None:
        args.default_speech_token_id = tokenizer.convert_tokens_to_ids(DEFAULT_SPEECH_TOKEN)
    if args.start_text_token_id is None:
        args.start_text_token_id = tokenizer.convert_tokens_to_ids(START_TEXT_TOKEN)
    if args.end_text_token_id is None:
        args.end_text_token_id = tokenizer.convert_tokens_to_ids(END_TEXT_TOKEN)
    if args.start_speech_token_id is None:
        args.start_speech_token_id = tokenizer.convert_tokens_to_ids(START_SPEECH_TOKEN)
    if args.end_speech_token_id is None:
        args.end_speech_token_id = tokenizer.convert_tokens_to_ids(END_SPEECH_TOKEN)

    config = build_config(args)
    config.save_pretrained(out_dir)
    patch_config_auto_map(out_dir)

    combined = _build_combined_state(args)
    if combined:
        save_file(combined, out_dir / "model.safetensors")

    tokenizer.save_pretrained(out_dir)
    tokenizer_config = out_dir / "tokenizer_config.json"
    if tokenizer_config.exists():
        token_config = json.loads(tokenizer_config.read_text(encoding="utf-8"))
        token_config["chat_template"] = CHAT_TEMPLATE_JINJA
        tokenizer_config.write_text(
            json.dumps(token_config, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    (out_dir / "chat_template.jinja").write_text(CHAT_TEMPLATE_JINJA, encoding="utf-8")

    legacy_preprocessor = (
        Path(args.legacy_hf_dir) / "preprocessor_config.json"
        if args.legacy_hf_dir
        else None
    )
    if legacy_preprocessor and legacy_preprocessor.exists():
        shutil.copy2(legacy_preprocessor, out_dir / "preprocessor_config.json")
    elif args.feature_extractor_type == "whisper":
        from transformers import WhisperFeatureExtractor

        WhisperFeatureExtractor(
            feature_size=args.feature_size,
            sampling_rate=args.sampling_rate,
        ).save_pretrained(out_dir)

    GenerationConfig(
        pad_token_id=tokenizer.pad_token_id,
        max_new_tokens=200,
        do_sample=False,
    ).save_pretrained(out_dir)

    package_root = Path(__file__).resolve().parents[2]
    dst_pkg = out_dir / "open_audio_llm"
    if dst_pkg.exists():
        shutil.rmtree(dst_pkg)
    shutil.copytree(
        package_root,
        dst_pkg,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "tests"),
    )
    _write_remote_code_wrappers(out_dir)
    return out_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", "--amphion-ckpt", dest="checkpoint", default=None)
    parser.add_argument("--legacy-hf-dir", default=None)
    parser.add_argument("--encoder-weights", default=None)
    parser.add_argument("--encoder-config", default=None)
    parser.add_argument("--llm-path", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--audio-tower-type", default="qwen3asr")
    parser.add_argument("--encoder-type", dest="audio_tower_type")
    parser.add_argument("--connector-type", default="mlp_downsample")
    parser.add_argument("--audio-input-dim", type=int, default=128)
    parser.add_argument("--audio-output-dim", type=int, default=1280)
    parser.add_argument("--llm-dim", type=int, default=2048)
    parser.add_argument("--downsample-rate", "--ds-rate", dest="downsample_rate", type=int, default=1)
    parser.add_argument("--feature-extractor-type", default="whisper")
    parser.add_argument("--feature-size", type=int, default=128)
    parser.add_argument("--sampling-rate", type=int, default=16000)
    parser.add_argument("--merge-lora", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--lora-alpha", type=float, default=16.0)
    parser.add_argument("--skip-base-lm-weights", action="store_true")
    parser.add_argument("--default-speech-token-id", type=int, default=None)
    parser.add_argument("--start-text-token-id", type=int, default=None)
    parser.add_argument("--end-text-token-id", type=int, default=None)
    parser.add_argument("--start-speech-token-id", type=int, default=None)
    parser.add_argument("--end-speech-token-id", type=int, default=None)
    return parser


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = build_parser().parse_args(argv)
    out_dir = convert_checkpoint(args)
    print(f"Saved AudioLLM checkpoint scaffold to {out_dir}")


if __name__ == "__main__":
    main()
