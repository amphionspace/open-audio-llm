"""Per-dimension statistics of the frozen enrollment features, for ``feature_standardize``.

    python -m open_audio_llm.longenroll.feature_stats --data_config <recipe> \
        --model <Qwen3-ASR-1.7B> --model_config <model yaml> --samples 3000 --output stats.pt

Draws ``--samples`` training records spread over the sources (the recipe's validation sampling
applied to the train split), keeps the target-speaker ones, and runs their enrollments, without
augmentation, through the frozen copy of the original encoder layers in 30 s windows exactly as
the model does. Writes the mean and standard deviation over all valid frames at every depth the
enrollment encoder reads. The statistics depend only on the base encoder and the data.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import torch
import yaml

from ..data.catalog_dataset import read_data_config
from .backbone import PromptBuilder, load_qwen3_asr
from .config import Config
from .data import LongEnrollCatalogDataset
from .ecapa import ECAPATDNN
from .model import LongEnrollTSASR
from .train import loader_for, validation_batches

# Standard deviations below this are clamped (a constant dimension would otherwise blow up).
MIN_STD = 1e-3


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data_config", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--model_config", required=True)
    p.add_argument("--samples", type=int, default=3000, help="training records drawn before keeping TS ones")
    p.add_argument("--output", required=True)
    p.add_argument("--workers", type=int, default=8)
    return p.parse_args(argv)


@torch.no_grad()
def main(argv=None):
    args = parse_args(argv)
    out = Path(args.output)
    if out.exists():
        raise FileExistsError(f"{out} exists; statistics are never overwritten")
    device = torch.device("cuda", 0)
    cfg = Config({"model": Config(yaml.safe_load(Path(args.model_config).read_text())).model,
                  "train": {"gradient_checkpointing": False}})
    asr, processor = load_qwen3_asr(args.model, dtype=torch.float32, device="cpu")
    model = LongEnrollTSASR(asr, ECAPATDNN(), cfg)
    model.to(device=device, dtype=torch.bfloat16).eval()
    data = copy.deepcopy(read_data_config(args.data_config))
    data["validation"], data["validation_samples"] = data["train"], args.samples  # unaugmented train records
    dataset = LongEnrollCatalogDataset(data, processor, training=False)
    loader = loader_for(dataset, validation_batches(dataset, 0, 1), args.workers, 2,
                        PromptBuilder(processor.tokenizer).pad_id, persistent=False)
    depths = model.enroll_encoder.feature_depths
    total = {d: torch.zeros(model.enroll_encoder.hidden_size, dtype=torch.float64, device=device) for d in depths}
    square = {d: torch.zeros_like(total[d]) for d in depths}
    frames, enrollments, seconds = 0, 0, 0.0
    for batch in loader:
        rows = [i for i, task in enumerate(batch["task"]) if task in ("ts", "ts_absent")]
        if not rows:
            continue
        with torch.autocast("cuda", dtype=torch.bfloat16):
            feats, lengths, _ = model.enrollment_features([batch["enroll_mels"][i] for i in rows])
        valid = torch.arange(feats[depths[0]].size(1), device=device)[None] < lengths[:, None]
        for d in depths:
            x = feats[d].double()[valid]
            total[d] += x.sum(0)
            square[d] += x.pow(2).sum(0)
        frames += int(valid.sum())
        enrollments += len(rows)
        seconds += float(batch["enroll_sec"][rows].sum())
    mean = torch.stack([total[d] / frames for d in depths])
    std = torch.stack([(square[d] / frames - (total[d] / frames).pow(2)).clamp_min(0).sqrt() for d in depths])
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"depths": depths, "mean": mean.float().cpu(), "std": std.clamp_min(MIN_STD).float().cpu(),
                "frames": frames, "enrollments": enrollments, "enrollment_seconds": seconds,
                "data_config": str(Path(args.data_config).resolve()), "split": "train"}, out)
    common = [float(mean[i].pow(2).sum() / (square[d] / frames).sum()) for i, d in enumerate(depths)]
    print(json.dumps({"output": str(out), "enrollments": enrollments, "frames": frames,
                      "enrollment_hours": round(seconds / 3600, 2),
                      "mean_energy_share": {str(d): round(c, 4) for d, c in zip(depths, common)}}), flush=True)


if __name__ == "__main__":
    main()
