"""Training checkpoint of LongEnrollTSASR -> inference directory for vLLM (see frontend.py).

    python -m open_audio_llm.longenroll.export --checkpoint <output_dir>/checkpoint-N \
        --model <Qwen3-ASR-1.7B> --ecapa_model <ECAPA dir> --model_config <longenroll_v5.yaml> \
        --output_dir <new dir>

Reads both checkpoint layouts of train.py: ``model.pt`` (trainable parameters, --deepspeed none)
and DeepSpeed's ``mp_rank_00_model_states.pt`` (the whole bf16 module, frozen parts included).
Writes bf16 weights; the frozen ECAPA stays fp32. After writing, the directory is read back the
way the frontend and vLLM read it and every tensor is compared with the source.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch
import yaml

from ..audio.target_sot import checkpoint_tensors
from .backbone import load_qwen3_asr
from .config import Config
from .ecapa import load_ecapa
from .frontend import EXTRA_CONFIG, EXTRA_DIR, EXTRA_MODULES, EXTRA_WEIGHTS, LongEnrollFrontend
from .model import LongEnrollTSASR

# Base-model files copied as they are. The architecture is unchanged, so the official config is
# kept (save_pretrained rewrites model_type and dtype fields that vLLM and qwen-asr were not
# validated with); then tokenizer, processor and the generation config (eos ids).
BASE_FILES = ("config.json", "generation_config.json", "chat_template.json", "merges.txt", "preprocessor_config.json",
              "tokenizer_config.json", "vocab.json", "tokenizer.json", "special_tokens_map.json",
              "added_tokens.json")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True, help="checkpoint-N directory written by train.py")
    p.add_argument("--model", required=True, help="Qwen3-ASR-1.7B directory the run started from")
    p.add_argument("--ecapa_model", required=True)
    p.add_argument("--model_config", required=True, help="YAML with the long-enrollment `model:` section")
    p.add_argument("--output_dir", required=True, help="must not exist")
    return p.parse_args(argv)


def load_training_checkpoint(model: LongEnrollTSASR, checkpoint: Path) -> str:
    if (checkpoint / "model.pt").exists():
        model.load_trainable_state_dict(torch.load(checkpoint / "model.pt", map_location="cpu", mmap=True))
        return "torch"
    states = checkpoint / "mp_rank_00_model_states.pt"
    if states.exists():
        # The whole module as trained, including the frozen parts DeepSpeed held in bf16
        # (ECAPA was cast back to fp32 from those bf16 values).
        module = torch.load(states, map_location="cpu", weights_only=False)["module"]
        model.load_state_dict(module, strict=True)
        return "deepspeed"
    raise FileNotFoundError(f"{checkpoint} has neither model.pt nor mp_rank_00_model_states.pt")


def export(args) -> dict:
    out = Path(args.output_dir)
    if out.exists():
        raise FileExistsError(f"{out} exists; exports never overwrite")
    checkpoint = Path(args.checkpoint)
    model_section = yaml.safe_load(Path(args.model_config).read_text())["model"]
    cfg = Config({"model": model_section, "train": {"gradient_checkpointing": False}})
    asr, _ = load_qwen3_asr(args.model, dtype=torch.float32, device="cpu")
    model = LongEnrollTSASR(asr, load_ecapa(args.ecapa_model), cfg)
    layout = load_training_checkpoint(model, checkpoint)

    tmp = out.with_name(out.name + ".partial")
    if tmp.exists():
        raise FileExistsError(f"{tmp} is left from an interrupted export; inspect and remove it first")
    # Transformers refuses to write the base generation config (a temperature without sampling);
    # the base files are copied over the written ones below.
    model.asr.generation_config.temperature = None
    model.asr.to(torch.bfloat16).save_pretrained(str(tmp), safe_serialization=True)
    for name in BASE_FILES:
        if (Path(args.model) / name).exists():
            shutil.copy2(Path(args.model) / name, tmp / name)
    from safetensors.torch import save_file

    extra = {}
    for name, value in model.state_dict().items():
        if name.split(".", 1)[0] in EXTRA_MODULES:
            if value.is_floating_point() and not name.startswith("ecapa."):
                value = value.to(torch.bfloat16)
            extra[name] = value.contiguous()
    (tmp / EXTRA_DIR).mkdir()
    save_file(extra, str(tmp / EXTRA_DIR / EXTRA_WEIGHTS))
    info = {"model": model_section, "checkpoint": str(checkpoint.resolve()), "checkpoint_layout": layout,
            "base_model": str(Path(args.model).resolve()), "ecapa_model": str(Path(args.ecapa_model).resolve())}
    (tmp / EXTRA_DIR / EXTRA_CONFIG).write_text(json.dumps(info, indent=2, ensure_ascii=False))

    report = verify(model, tmp)
    tmp.rename(out)
    report.update(output_dir=str(out), checkpoint_layout=layout)
    return report


@torch.no_grad()
def verify(model: LongEnrollTSASR, folder: Path) -> dict:
    """Every tensor of the inference directory equals the source cast to the export dtype, the
    Qwen3-ASR checkpoint holds every backbone tensor, and the frontend loads strictly."""
    source = model.asr.state_dict()
    saved = dict(checkpoint_tensors(folder, ""))
    # Qwen3-ASR ties the LM head to the input embeddings; vLLM ties them the same way.
    tied = bool(model.asr.config.thinker_config.text_config.tie_word_embeddings)
    missing = sorted(k for k in source if k not in saved and not (tied and k == "thinker.lm_head.weight"))
    if missing:
        raise RuntimeError(f"backbone tensors missing from the export: {missing[:8]}")
    for name, value in saved.items():
        if not torch.equal(value, source[name].to(value.dtype)):
            raise RuntimeError(f"exported {name} differs from the checkpoint")
    frontend = LongEnrollFrontend.from_pretrained(folder, device="cpu")
    reference = model.state_dict()
    for name, value in frontend.state_dict().items():
        if not torch.equal(value, reference[name].to(value.dtype)):
            raise RuntimeError(f"frontend {name} differs from the checkpoint")
    return {"backbone_tensors": len(saved), "frontend_tensors": len(frontend.state_dict())}


def main(argv=None):
    print(json.dumps(export(parse_args(argv)), ensure_ascii=False))


if __name__ == "__main__":
    main()
