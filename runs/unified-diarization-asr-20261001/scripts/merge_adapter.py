"""Merge a saved LoRA adapter into its base on CPU for vLLM evaluation.

Same merge as local_checkpoint_eval.py, for checkpoints evaluated outside the
training callback (e.g. LoRA 002 checkpoint-8000, whose in-training evaluation failed).
"""
import argparse
import shutil
from pathlib import Path

import torch
import yaml
from peft import PeftModel
from qwen_asr import Qwen3ASRModel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    settings = yaml.safe_load(parser.parse_args().config.read_text())['parameters']
    checkpoint, base, output = (Path(settings[k]) for k in ('adapter', 'base', 'output'))
    model = Qwen3ASRModel.from_pretrained(str(base), dtype=torch.bfloat16, device_map='cpu').model
    model = PeftModel.from_pretrained(model, str(checkpoint)).merge_and_unload(safe_merge=True)
    model.save_pretrained(output, safe_serialization=True)
    for source in base.iterdir():
        if source.suffix in {'.json', '.txt', '.jinja'} and source.name not in {
            'config.json', 'args.json', 'trainer_state.json', 'catalog_sampler.json',
            'model.safetensors.index.json'} and not (output / source.name).exists():
            shutil.copy2(source, output / source.name)


if __name__ == '__main__':
    main()
