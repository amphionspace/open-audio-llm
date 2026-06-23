#!/usr/bin/env python3
"""Convert an Amphion-ASR ``.pt`` checkpoint to a HuggingFace-format
directory that can be loaded with::

    AutoModelForCausalLM.from_pretrained(output_dir, trust_remote_code=True)

Minimal usage
-------------
    python convert_amphion_to_hf.py --amphion-ckpt exp/<run>/checkpoint-N.pt

When ``train.yaml`` exists next to the checkpoint (the trainer always
writes one), the script reads ``encoder_type``, ``speech_encoder_path``,
``encoder_config_path``, ``llm_path``, ``encoder_projector_ds_rate`` and
``zipformer_model_type`` from it.  Anything passed on the CLI takes
precedence over the yaml; ``--train-config`` lets you point at a yaml in
a different location.

Optional flags
--------------
* ``--output-dir``        : destination directory
                            (default ``<ckpt-dir>/hf-<ckpt-stem>``).
* ``--train-config``      : custom path to a training yaml.
* ``--no-merge-lora``     : keep LoRA deltas separate
                            (default is to merge into the base LLM).
* ``--encoder-type / --encoder-weights / --encoder-config /
   --llm-path / --ds-rate / --zipformer-model-type``
                          : per-field overrides for what the yaml says.

Output
------
A HuggingFace model directory with:
  config.json, model.safetensors (or model-*.safetensors shards),
  configuration_amphion_asr.py, modeling_amphion_asr.py,
  processing_amphion_asr.py, tokenizer files, preprocessor_config.json,
  and generation_config.json.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from pathlib import Path

# Training checkpoints may pickle objects whose classes live as
# top-level modules under ``src/`` (e.g. ``hotword_utils.PinyinIndex``,
# ``esc_mixing._cut_task``).  Make those importable so ``torch.load``
# can rebuild them.  ``__file__`` now lives at
# ``src/open_audio_llm/integrations/hf/convert_amphion_to_hf.py``; four
# parent hops bring us to the ``src/`` root.
_SRC_ROOT = Path(__file__).resolve().parents[3]
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

if __package__:
    from .constants import (
        DEFAULT_SPEECH_TOKEN,
        END_SPEECH_TOKEN,
        END_TEXT_TOKEN,
        START_SPEECH_TOKEN,
        START_TEXT_TOKEN,
    )
else:
    from constants import (
        DEFAULT_SPEECH_TOKEN,
        END_SPEECH_TOKEN,
        END_TEXT_TOKEN,
        START_SPEECH_TOKEN,
        START_TEXT_TOKEN,
    )

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# ---- Chat template (Jinja2) -----------------------------------------------
# Extends the standard Qwen3 template to handle structured content with
# audio parts (OpenAI-style content arrays).  When vLLM encounters
# {"type": "audio"} in the content list, it renders the special tokens
# automatically — users never have to write them manually.

CHAT_TEMPLATE_JINJA = """\
{%- if tools %}
    {{- '<|im_start|>system\\n' }}
    {%- if messages[0].role == 'system' %}
        {{- messages[0].content + '\\n\\n' }}
    {%- endif %}
    {{- "# Tools\\n\\nYou may call one or more functions to assist with the user query.\\n\\nYou are provided with function signatures within <tools></tools> XML tags:\\n<tools>" }}
    {%- for tool in tools %}
        {{- "\\n" }}
        {{- tool | tojson }}
    {%- endfor %}
    {{- "\\n</tools>\\n\\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\\n<tool_call>\\n{\\"name\\": <function-name>, \\"arguments\\": <args-json-object>}\\n</tool_call><|im_end|>\\n" }}
{%- else %}
    {%- if messages[0].role == 'system' %}
        {{- '<|im_start|>system\\n' + messages[0].content + '<|im_end|>\\n' }}
    {%- endif %}
{%- endif %}
{%- for message in messages %}
    {%- if message.content is string %}
        {%- set content = message.content %}
    {%- else %}
        {%- set ns = namespace(content='') %}
        {%- for part in message.content %}
            {%- if part.type == 'text' %}
                {%- set ns.content = ns.content + part.text %}
            {%- elif part.type == 'audio' %}
                {%- set ns.content = ns.content + '<start_speech><speech><end_speech>' %}
            {%- endif %}
        {%- endfor %}
        {%- set content = ns.content %}
    {%- endif %}
    {%- if (message.role == "user") or (message.role == "system" and not loop.first) %}
        {{- '<|im_start|>' + message.role + '\\n' + content + '<|im_end|>' + '\\n' }}
    {%- elif message.role == "assistant" %}
        {{- '<|im_start|>' + message.role + '\\n' + content }}
        {%- if message.tool_calls %}
            {%- for tool_call in message.tool_calls %}
                {%- if (loop.first and content) or (not loop.first) %}
                    {{- '\\n' }}
                {%- endif %}
                {%- if tool_call.function %}
                    {%- set tool_call = tool_call.function %}
                {%- endif %}
                {{- '<tool_call>\\n{"name": "' }}
                {{- tool_call.name }}
                {{- '", "arguments": ' }}
                {%- if tool_call.arguments is string %}
                    {{- tool_call.arguments }}
                {%- else %}
                    {{- tool_call.arguments | tojson }}
                {%- endif %}
                {{- '}\\n</tool_call>' }}
            {%- endfor %}
        {%- endif %}
        {{- '<|im_end|>\\n' }}
    {%- elif message.role == "tool" %}
        {%- if loop.first or (messages[loop.index0 - 1].role != "tool") %}
            {{- '<|im_start|>user' }}
        {%- endif %}
        {{- '\\n<tool_response>\\n' }}
        {{- content }}
        {{- '\\n</tool_response>' }}
        {%- if loop.last or (messages[loop.index0 + 1].role != "tool") %}
            {{- '<|im_end|>\\n' }}
        {%- endif %}
    {%- endif %}
{%- endfor %}
{%- if add_generation_prompt %}
    {{- '<|im_start|>assistant\\n' }}
{%- endif %}
"""

# ---- Special tokens (imported from constants.py) --------------------------


def _key_mapping(encoder_type: str = "qwen3asr"):
    """Return a list of (old_prefix, new_prefix) tuples.

    The training checkpoint stores keys under the ``Amphion_LLM``
    namespace.  With the AudioEncoderWrapper, encoder weights gain an
    extra ``.encoder.`` level::

        encoder.*             -> audio_encoder.encoder.*
        encoder_projector.*   -> multi_modal_projector.*
        llm.*                 -> language_model.*
        prompt_embedding.*    -> prompt_embedding.*

    For ``zipformer``, training keys are ``encoder_embed.*`` and
    ``encoder.*`` (no top-level ``encoder.`` prefix collision with
    ``encoder_projector.``).
    """
    if encoder_type == "zipformer":
        return [
            ("encoder_embed.", "audio_encoder.encoder.encoder_embed."),
            ("encoder.", "audio_encoder.encoder.encoder."),
            ("encoder_projector.", "multi_modal_projector."),
            ("llm.", "language_model."),
            ("prompt_embedding.", "prompt_embedding."),
        ]
    return [
        ("encoder.", "audio_encoder.encoder."),
        ("encoder_projector.", "multi_modal_projector."),
        ("llm.", "language_model."),
        ("prompt_embedding.", "prompt_embedding."),
    ]


def _remap_key(key: str, encoder_type: str = "qwen3asr") -> str:
    for old, new in _key_mapping(encoder_type):
        if key.startswith(old):
            return new + key[len(old):]
    return key


def _strip_lora_wrapper(key: str) -> str:
    """Remove PEFT wrapper prefixes from keys.

    PEFT wraps the LLM as ``base_model.model.<original_key>`` and adds
    ``lora_A`` / ``lora_B`` parameters.  After merging, we only keep
    the base model keys.
    """
    peft_prefix = "base_model.model."
    if key.startswith(peft_prefix):
        return key[len(peft_prefix):]
    return key


# Mapping from CLI dest name to the corresponding key in train.yaml.
# Anything in this dict is auto-filled from yaml when the CLI flag is
# omitted; CLI always wins when explicitly set.
_YAML_FALLBACKS: dict[str, str] = {
    "encoder_type":         "encoder_type",
    "encoder_weights":      "speech_encoder_path",
    "encoder_config":       "encoder_config_path",
    "llm_path":             "llm_path",
    "ds_rate":              "encoder_projector_ds_rate",
    "zipformer_model_type": "zipformer_model_type",
}


def _load_train_yaml(ckpt_path: Path, explicit: str | None) -> dict:
    """Look for ``train.yaml`` next to the checkpoint (or use *explicit*).

    Returns ``{}`` when no yaml is found; the caller must then either
    rely entirely on CLI flags or raise.
    """
    cfg_path = Path(explicit) if explicit else ckpt_path.parent / "train.yaml"
    if not cfg_path.is_file():
        if explicit:
            raise FileNotFoundError(f"--train-config not found: {cfg_path}")
        return {}
    import yaml

    with open(cfg_path) as f:
        cfg = yaml.safe_load(f) or {}
    logger.info("Loaded training config from %s", cfg_path)
    return cfg


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--amphion-ckpt", type=str, required=True,
                        help="Path to the Amphion .pt training checkpoint")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output dir (default: <ckpt-dir>/hf-<ckpt-stem>)")
    parser.add_argument("--train-config", type=str, default=None,
                        help="Override path to train.yaml "
                             "(default: <ckpt-dir>/train.yaml)")
    parser.add_argument("--merge-lora", action=argparse.BooleanOptionalAction,
                        default=True,
                        help="Merge LoRA weights into base model")

    # The following are normally auto-filled from train.yaml; CLI values
    # override the yaml when given.
    parser.add_argument("--encoder-type", type=str, default=None,
                        choices=["qwen3asr", "qwen3omni_captioner", "qwen3omni", "zipformer"])
    parser.add_argument("--encoder-weights", type=str, default=None,
                        help="Path to the encoder .pth state_dict "
                             "(yaml: speech_encoder_path)")
    parser.add_argument("--encoder-config", type=str, default=None,
                        help="Path to the encoder JSON config "
                             "(yaml: encoder_config_path; not needed for zipformer)")
    parser.add_argument("--llm-path", type=str, default=None,
                        help="Path to the base LLM HF directory (yaml: llm_path)")
    parser.add_argument("--ds-rate", type=int, default=None,
                        help="Projector downsample rate "
                             "(yaml: encoder_projector_ds_rate; default 1)")
    parser.add_argument("--zipformer-model-type", type=str, default=None,
                        choices=["custom", "custom_noncausal"],
                        help="Zipformer preset (yaml: zipformer_model_type; "
                             "only used when encoder_type=zipformer)")
    args = parser.parse_args()

    ckpt_path = Path(args.amphion_ckpt)
    if not ckpt_path.is_file():
        parser.error(f"--amphion-ckpt not found: {ckpt_path}")

    # ----- Auto-fill missing CLI values from train.yaml ----------------
    try:
        yaml_cfg = _load_train_yaml(ckpt_path, args.train_config)
    except FileNotFoundError as e:
        parser.error(str(e))
    for arg_name, yaml_key in _YAML_FALLBACKS.items():
        if getattr(args, arg_name) is None and yaml_key in yaml_cfg:
            setattr(args, arg_name, yaml_cfg[yaml_key])

    # ----- Defaults & required-field validation ------------------------
    if args.ds_rate is None:
        args.ds_rate = 1
    if args.zipformer_model_type is None:
        args.zipformer_model_type = "custom_noncausal"

    missing = []
    if args.encoder_type is None:
        missing.append("--encoder-type / encoder_type")
    if args.encoder_weights is None:
        missing.append("--encoder-weights / speech_encoder_path")
    if args.llm_path is None:
        missing.append("--llm-path / llm_path")
    if args.encoder_type != "zipformer" and args.encoder_config is None:
        missing.append("--encoder-config / encoder_config_path")
    if missing:
        parser.error(
            "Missing required parameters (not on CLI nor in train.yaml): "
            + ", ".join(missing)
        )

    if __package__:
        from .configuration_amphion_asr import (
            AmphionASRAudioEncoderConfig,
            AmphionASRConfig,
            AmphionASRProjectorConfig,
            ZipformerAudioEncoderConfig,
        )
    else:
        from configuration_amphion_asr import (
            AmphionASRAudioEncoderConfig,
            AmphionASRConfig,
            AmphionASRProjectorConfig,
            ZipformerAudioEncoderConfig,
        )

    import torch
    from safetensors.torch import save_file
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        GenerationConfig,
        WhisperFeatureExtractor,
    )

    if args.output_dir is None:
        args.output_dir = str(ckpt_path.parent / f"hf-{ckpt_path.stem}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ----- Resolved-config summary -------------------------------------
    # Print everything we will use before any heavy I/O (torch.load,
    # AutoModelForCausalLM.from_pretrained), so a misconfigured run can
    # be aborted within seconds rather than after minutes of LLM load.
    logger.info("=" * 70)
    logger.info("Conversion plan:")
    logger.info("  amphion_ckpt        : %s", args.amphion_ckpt)
    logger.info("  output_dir          : %s", args.output_dir)
    logger.info("  encoder_type        : %s", args.encoder_type)
    logger.info("  encoder_weights     : %s", args.encoder_weights)
    if args.encoder_type == "zipformer":
        logger.info("  zipformer_model_type: %s", args.zipformer_model_type)
    else:
        logger.info("  encoder_config      : %s", args.encoder_config)
    logger.info("  llm_path            : %s", args.llm_path)
    logger.info("  ds_rate             : %d", args.ds_rate)
    logger.info("  merge_lora          : %s", args.merge_lora)
    logger.info("=" * 70)

    # ------------------------------------------------------------------
    # 1. Load the training checkpoint
    # ------------------------------------------------------------------
    logger.info("Loading Amphion checkpoint from %s", args.amphion_ckpt)
    ckpt = torch.load(args.amphion_ckpt, map_location="cpu", weights_only=False)
    model_state = ckpt.get("model", ckpt.get("state_dict", ckpt))

    # ------------------------------------------------------------------
    # 2. Load base LLM (for frozen-parameter recovery and config)
    # ------------------------------------------------------------------
    logger.info("Loading base LLM from %s", args.llm_path)
    tokenizer = AutoTokenizer.from_pretrained(args.llm_path, trust_remote_code=True)
    llm = AutoModelForCausalLM.from_pretrained(
        args.llm_path, dtype=torch.float32, trust_remote_code=True
    )
    text_config = llm.config

    # Add special tokens (matches training code)
    tokenizer.add_special_tokens({
        "additional_special_tokens": [
            DEFAULT_SPEECH_TOKEN, START_TEXT_TOKEN, END_TEXT_TOKEN,
            START_SPEECH_TOKEN, END_SPEECH_TOKEN,
        ]
    })
    llm.resize_token_embeddings(len(tokenizer))

    # ------------------------------------------------------------------
    # 3. Load encoder config
    # ------------------------------------------------------------------
    if args.encoder_type == "zipformer":
        logger.info("Building Zipformer encoder config for preset: %s", args.zipformer_model_type)
        # For zipformer, the config is derived from the preset, not a JSON file
        if __package__:
            from .zipformer_inference import get_model_params
        else:
            from zipformer_inference import get_model_params
        zf_params = get_model_params(args.zipformer_model_type)
        enc_cfg_dict = {
            k: getattr(zf_params, k) if hasattr(zf_params, k) else zf_params[k]
            for k in [
                "feature_dim", "num_encoder_layers", "downsampling_factor",
                "feedforward_dim", "num_heads", "encoder_dim", "query_head_dim",
                "value_head_dim", "pos_head_dim", "pos_dim",
                "encoder_unmasked_dim", "cnn_module_kernel",
                "causal", "chunk_size", "left_context_frames",
            ]
        }
    else:
        logger.info("Loading encoder config from %s", args.encoder_config)
        with open(args.encoder_config, "r") as f:
            enc_cfg_dict = json.load(f)

    # ------------------------------------------------------------------
    # 4. Load encoder weights (may be missing from training ckpt)
    # ------------------------------------------------------------------
    logger.info("Loading encoder weights from %s", args.encoder_weights)
    enc_state = torch.load(args.encoder_weights, map_location="cpu", weights_only=False)
    if isinstance(enc_state, dict) and "model" in enc_state:
        enc_state = enc_state["model"]

    # ------------------------------------------------------------------
    # 5. Determine encoder dimension from config
    # ------------------------------------------------------------------
    if args.encoder_type == "zipformer":
        encoder_dim_str = enc_cfg_dict.get("encoder_dim", "256,384,512,768,512,384")
        encoder_dim = max(int(x) for x in encoder_dim_str.split(","))
    else:
        encoder_dim = enc_cfg_dict.get("d_model", 1280)
    llm_dim = text_config.hidden_size

    # ------------------------------------------------------------------
    # 6. Build the combined state dict
    # ------------------------------------------------------------------
    logger.info("Building combined state dict ...")

    # Start with base LLM state dict
    combined = {}
    for k, v in llm.state_dict().items():
        combined[f"language_model.{k}"] = v

    # Overlay checkpoint LLM weights (which may include LoRA-merged or
    # partial updates).  We need to handle the LoRA case.
    has_lora = any("lora_" in k for k in model_state)
    if has_lora and args.merge_lora:
        logger.info("LoRA weights detected — merging into base LLM ...")
        _merge_lora_into_combined(model_state, combined)
    else:
        for k, v in model_state.items():
            mapped = _remap_key(k, args.encoder_type)
            if mapped.startswith("language_model."):
                lm_key = _strip_lora_wrapper(mapped[len("language_model."):])
                full_key = f"language_model.{lm_key}"
                combined[full_key] = v

    # Encoder weights: prefer checkpoint, fall back to standalone file.
    # For zipformer, training keys are encoder_embed.* and encoder.*
    if args.encoder_type == "zipformer":
        enc_prefixes = ("encoder_embed.", "encoder.")
        encoder_keys_in_ckpt = {
            k: v for k, v in model_state.items()
            if any(k.startswith(p) for p in enc_prefixes)
            and not k.startswith("encoder_projector")
        }
    else:
        encoder_keys_in_ckpt = {
            k: v for k, v in model_state.items()
            if k.startswith("encoder.") and not k.startswith("encoder_projector")
        }

    if encoder_keys_in_ckpt:
        logger.info("Using encoder weights from training checkpoint (%d keys)", len(encoder_keys_in_ckpt))
        for k, v in encoder_keys_in_ckpt.items():
            combined[_remap_key(k, args.encoder_type)] = v
    else:
        logger.info("Encoder weights missing from checkpoint, using standalone file (%d keys)", len(enc_state))
        for k, v in enc_state.items():
            combined[f"audio_encoder.encoder.{k}"] = v

    # Remove proj1/act/proj2 from encoder (not needed for Zipformer or Amphion)
    if args.encoder_type != "zipformer":
        proj_keys = [k for k in combined if any(
            f"audio_encoder.encoder.{p}" in k for p in ("proj1.", "act.", "proj2.")
        )]
        for k in proj_keys:
            del combined[k]

    # Projector weights
    proj_keys_in_ckpt = {k: v for k, v in model_state.items() if k.startswith("encoder_projector.")}
    if proj_keys_in_ckpt:
        logger.info("Found projector weights in checkpoint (%d keys)", len(proj_keys_in_ckpt))
        for k, v in proj_keys_in_ckpt.items():
            combined[_remap_key(k, args.encoder_type)] = v
    else:
        logger.warning("No projector weights found in checkpoint!")

    # Prompt embedding
    prompt_keys = {k: v for k, v in model_state.items() if k.startswith("prompt_embedding.")}
    if prompt_keys:
        for k, v in prompt_keys.items():
            combined[_remap_key(k, args.encoder_type)] = v
    else:
        logger.warning("No prompt_embedding weights found in checkpoint")

    # ------------------------------------------------------------------
    # 6b. Reconcile embedding / lm_head vocab sizes
    # ------------------------------------------------------------------
    embed_key = "language_model.model.embed_tokens.weight"
    lm_head_key = "language_model.lm_head.weight"

    if embed_key in combined and lm_head_key in combined:
        embed_vocab = combined[embed_key].shape[0]
        head_vocab = combined[lm_head_key].shape[0]
        if embed_vocab != head_vocab:
            target_vocab = max(embed_vocab, head_vocab)
            hidden = combined[embed_key].shape[1]
            logger.info(
                "Reconciling vocab mismatch: embed_tokens=%d, lm_head=%d → %d",
                embed_vocab, head_vocab, target_vocab,
            )
            if embed_vocab < target_vocab:
                padded = torch.zeros(target_vocab, hidden, dtype=combined[embed_key].dtype)
                padded[:embed_vocab] = combined[embed_key]
                combined[embed_key] = padded
            if head_vocab < target_vocab:
                padded = torch.zeros(target_vocab, hidden, dtype=combined[lm_head_key].dtype)
                padded[:head_vocab] = combined[lm_head_key]
                combined[lm_head_key] = padded
    elif embed_key in combined:
        target_vocab = combined[embed_key].shape[0]
    elif lm_head_key in combined:
        target_vocab = combined[lm_head_key].shape[0]
    else:
        target_vocab = len(tokenizer)

    actual_vocab_size = target_vocab

    # ------------------------------------------------------------------
    # 7. Build AmphionASRConfig
    # ------------------------------------------------------------------
    logger.info("Building AmphionASRConfig ...")

    text_config_dict = text_config.to_dict()
    if text_config_dict.get("vocab_size") != actual_vocab_size:
        logger.info(
            "Updating text_config.vocab_size: %d → %d",
            text_config_dict.get("vocab_size"), actual_vocab_size,
        )
        text_config_dict["vocab_size"] = actual_vocab_size

    if args.encoder_type == "zipformer":
        audio_enc_config = ZipformerAudioEncoderConfig(**enc_cfg_dict)
    else:
        audio_enc_config = AmphionASRAudioEncoderConfig(**enc_cfg_dict)

    proj_config = AmphionASRProjectorConfig(
        encoder_dim=encoder_dim,
        llm_dim=llm_dim,
        downsample_rate=args.ds_rate,
    )

    config = AmphionASRConfig(
        audio_encoder_config=audio_enc_config.to_dict(),
        projector_config=proj_config.to_dict(),
        text_config=text_config_dict,
        num_prompt_tokens=4,
        default_speech_token_id=tokenizer.convert_tokens_to_ids(DEFAULT_SPEECH_TOKEN),
        start_text_token_id=tokenizer.convert_tokens_to_ids(START_TEXT_TOKEN),
        end_text_token_id=tokenizer.convert_tokens_to_ids(END_TEXT_TOKEN),
        start_speech_token_id=tokenizer.convert_tokens_to_ids(START_SPEECH_TOKEN),
        end_speech_token_id=tokenizer.convert_tokens_to_ids(END_SPEECH_TOKEN),
        encoder_type=args.encoder_type,
    )

    # ------------------------------------------------------------------
    # 8. Save everything
    # ------------------------------------------------------------------
    logger.info("Saving to %s ...", output_dir)

    # config.json
    config.save_pretrained(output_dir)

    # model weights (safetensors)
    save_file(combined, output_dir / "model.safetensors")
    logger.info("Saved %d tensors to model.safetensors", len(combined))

    # Tokenizer
    tokenizer.save_pretrained(output_dir)

    # Chat template (enables automatic special-token insertion for audio)
    jinja_path = output_dir / "chat_template.jinja"
    jinja_path.write_text(CHAT_TEMPLATE_JINJA)
    logger.info("Saved chat_template.jinja")

    tok_cfg_path = output_dir / "tokenizer_config.json"
    with open(tok_cfg_path) as f:
        tok_cfg = json.load(f)
    tok_cfg["chat_template"] = CHAT_TEMPLATE_JINJA

    # transformers >=5.x expects extra_special_tokens as dict, not list
    _EST_NAME_MAP = {
        DEFAULT_SPEECH_TOKEN: "speech_token",
        START_TEXT_TOKEN: "start_text_token",
        END_TEXT_TOKEN: "end_text_token",
        START_SPEECH_TOKEN: "start_speech_token",
        END_SPEECH_TOKEN: "end_speech_token",
    }
    est = tok_cfg.get("extra_special_tokens")
    if isinstance(est, list):
        tok_cfg["extra_special_tokens"] = {
            _EST_NAME_MAP.get(t, t): t for t in est
        }
        logger.info("Patched extra_special_tokens from list to dict")

    with open(tok_cfg_path, "w") as f:
        json.dump(tok_cfg, f, indent=2, ensure_ascii=False)
    logger.info("Updated tokenizer_config.json with chat_template")

    # Feature extractor config — only save WhisperFeatureExtractor for
    # Whisper-based encoders.  Zipformer uses Kaldi fbank (via torchaudio)
    # at inference time; the feature_extractor_type field in config.json
    # tells the processor which backend to use.
    if args.encoder_type != "zipformer":
        mel_bins = 128
        fe = WhisperFeatureExtractor(feature_size=mel_bins, sampling_rate=16000)
        fe.save_pretrained(output_dir)
    else:
        logger.info(
            "Zipformer uses Kaldi fbank — skipping WhisperFeatureExtractor "
            "(feature_extractor_type='kaldi_fbank' is set in config.json)"
        )

    # Generation config
    gen_config = GenerationConfig(
        bos_token_id=tokenizer.convert_tokens_to_ids("<|im_start|>"),
        eos_token_id=tokenizer.convert_tokens_to_ids("<|im_end|>"),
        pad_token_id=tokenizer.pad_token_id,
        max_new_tokens=200,
        do_sample=False,
    )
    gen_config.save_pretrained(output_dir)

    # Copy Python source files for trust_remote_code
    src_dir = Path(__file__).parent
    source_files = [
        "configuration_amphion_asr.py",
        "modeling_amphion_asr.py",
        "processing_amphion_asr.py",
        "constants.py",
        "zipformer_inference.py",
    ]
    for fname in source_files:
        src_file = src_dir / fname
        if src_file.exists():
            shutil.copy2(src_file, output_dir / fname)
            logger.info("Copied %s", fname)

    logger.info("Conversion complete!  Load with:")
    logger.info('  AutoModelForCausalLM.from_pretrained("%s", trust_remote_code=True)', output_dir)


def _merge_lora_into_combined(model_state: dict, combined: dict):
    """Merge LoRA A/B deltas into the base weights already in *combined*.

    LoRA parameterises ``W' = W + (alpha/r) * B @ A`` where
    ``lora_A.weight`` is ``(r, in)`` and ``lora_B.weight`` is
    ``(out, r)``.

    Keys in *model_state* that contain LoRA parameters look like::

        llm.base_model.model.<module>.lora_A.default.weight
        llm.base_model.model.<module>.lora_B.default.weight
    """
    lora_a = {}
    lora_b = {}

    for k, v in model_state.items():
        if "lora_A" in k and k.startswith("llm."):
            base_key = k.replace("base_model.model.", "").split(".lora_A")[0]
            base_key = base_key.replace("llm.", "")
            lora_a[base_key] = v
        elif "lora_B" in k and k.startswith("llm."):
            base_key = k.replace("base_model.model.", "").split(".lora_B")[0]
            base_key = base_key.replace("llm.", "")
            lora_b[base_key] = v

    if not lora_a:
        logger.info("No LoRA weights found to merge")
        return

    # Auto-detect rank and compute scaling factor (alpha=16 by default)
    first_a = next(iter(lora_a.values()))
    rank = first_a.shape[0]
    alpha = 16.0
    scale = alpha / rank
    logger.info("Detected LoRA rank=%d, using scale=alpha/r=%.4f (alpha=%.0f)", rank, scale, alpha)

    merged_count = 0
    skipped_count = 0
    for base_key in lora_a:
        if base_key not in lora_b:
            logger.warning("LoRA A without B for %s — skipping", base_key)
            skipped_count += 1
            continue
        full_key = f"language_model.{base_key}.weight"
        if full_key not in combined:
            logger.warning("Base weight %s not found — skipping LoRA merge", full_key)
            skipped_count += 1
            continue

        A = lora_a[base_key].float()  # (r, in)
        B = lora_b[base_key].float()  # (out, r)
        base_w = combined[full_key].float()

        delta = (B @ A) * scale

        # Dimension check: LoRA delta must match the base weight
        if delta.shape != base_w.shape:
            raise RuntimeError(
                f"LoRA merge dimension mismatch for {full_key}: "
                f"base weight shape {tuple(base_w.shape)}, "
                f"LoRA delta shape {tuple(delta.shape)}.  "
                f"This usually means --llm-path points to a different "
                f"LLM architecture than the one used during training."
            )

        combined[full_key] = base_w + delta
        merged_count += 1

    # Copy non-LoRA LLM weights from checkpoint (e.g. resized
    # embeddings / lm_head after adding special tokens).
    extra_count = 0
    for k, v in model_state.items():
        if "lora_" in k:
            continue
        mapped = _remap_key(k)
        if not mapped.startswith("language_model."):
            continue
        lm_key = _strip_lora_wrapper(mapped[len("language_model."):])
        full_key = f"language_model.{lm_key}"

        if full_key in combined and combined[full_key].shape != v.shape:
            # Checkpoint has a different shape (e.g. vocab was resized
            # after adding special tokens).  Use the checkpoint version.
            logger.info(
                "Overwriting %s with checkpoint version "
                "(shape %s → %s)",
                full_key, tuple(combined[full_key].shape), tuple(v.shape),
            )
            combined[full_key] = v
        elif full_key not in combined:
            combined[full_key] = v
            extra_count += 1

    logger.info(
        "Merged %d LoRA adapters (scale=%.4f), skipped %d, "
        "copied %d extra LLM keys",
        merged_count, scale, skipped_count, extra_count,
    )


if __name__ == "__main__":
    main()
