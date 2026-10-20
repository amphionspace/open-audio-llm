"""One forward+backward of the full Qwen3-ASR training path, padded vs packed (no optimizer).

Real samples from the training plan go through the ms-swift template collator; the model
is loaded as training loads it, with audio-encoder batching, Liger and the supervised head.
The same batch runs padded on SDPA, then packed on FlashAttention-2 varlen.
"""
import argparse
import json
import random
from itertools import islice

import torch
import torch.nn.functional as F
from swift.model import get_model_processor
from swift.template import get_template

import open_audio_llm.integrations.ms_swift.register_qwen3_asr  # noqa: F401
from open_audio_llm.data.catalog_dataset import CatalogSwiftDataset, read_data_config
from open_audio_llm.data.catalog_sampler import CatalogBatchSampler
from open_audio_llm.integrations.ms_swift.audio_batching import enable_batched_audio
from open_audio_llm.integrations.ms_swift.liger import enable_liger_kernels
from open_audio_llm.integrations.ms_swift.packed_text import enable_packed_text, pack_text
from open_audio_llm.integrations.ms_swift.retention import supervised_forward


def run(model, batch):
    model.zero_grad(set_to_none=True)
    inputs = {k: v.cuda() if torch.is_tensor(v) else v for k, v in batch.items()}
    inputs["input_features"] = inputs["input_features"].to(model.thinker.lm_head.weight.dtype)
    labels = inputs.pop("labels")
    outputs, targets, _, _ = supervised_forward(model.forward, model.thinker.lm_head, inputs, labels)
    loss = F.cross_entropy(outputs.logits.float(), targets)
    loss.backward()
    grads = {n: p.grad.float().norm().item() for n, p in model.named_parameters() if p.grad is not None}
    return loss.item(), int(targets.numel()), grads


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_config", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--samples", type=int, default=4)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    model, processor = get_model_processor(args.model, model_type="amphion_asr_1.7b",
                                           torch_dtype=torch.bfloat16, attn_impl="sdpa", device_map="cuda:0")
    model.train()
    template = get_template(processor, max_length=2048)
    template.set_mode("train")
    config = read_data_config(args.data_config)
    dataset = CatalogSwiftDataset(config, encode=template.encode, message_format="qwen3_asr")
    sampler = CatalogBatchSampler(dataset, batch_size=12, world_size=8)
    sampler.set_epoch(0)
    keys = list(islice(sampler._mux(random.Random(f"{dataset.seed}:0:mux")), args.samples))
    batch = template.data_collator([dataset[key] for key in keys])
    enable_batched_audio(model)
    patched, restore_liger = enable_liger_kernels(model)
    padded = run(model, dict(batch))
    restore_text = enable_packed_text(model)
    padded_flash = run(model, dict(batch))
    packed_batch = pack_text(dict(batch))
    packed = run(model, packed_batch)
    restore_text()
    # FP32 reference on the padded SDPA path (FlashAttention-2 has no FP32 kernels).
    model.float()
    reference = run(model, dict(batch))
    restore_liger()
    names = sorted(padded[2])
    relative = {n: abs(packed[2][n] - padded[2][n]) / max(padded[2][n], 1e-12) for n in names}
    error = lambda arm: sorted(abs(arm[2][n] - reference[2][n]) / max(reference[2][n], 1e-12) for n in names)
    report = {
        "liger": patched,
        "tokens": int(batch["attention_mask"].sum()), "padded_capacity": int(batch["attention_mask"].numel()),
        "packed_row": list(packed_batch["input_ids"].shape),
        "supervised": {"padded": padded[1], "packed": packed[1]},
        "loss": {"fp32_padded_sdpa": reference[0], "padded_sdpa": padded[0],
                 "padded_flash": padded_flash[0], "packed_flash": packed[0]},
        "grad_norm_error_vs_fp32_median": {name: error(arm)[len(names) // 2] for name, arm in
                                           (("padded_sdpa", padded), ("padded_flash", padded_flash),
                                            ("packed_flash", packed))},
        "packed_vs_padded_flash_grad_norm_relative_diff_median": sorted(
            abs(packed[2][n] - padded_flash[2][n]) / max(padded_flash[2][n], 1e-12) for n in names)[len(names) // 2],
        "grad_norm_relative_diff_median": sorted(relative.values())[len(relative) // 2],
        "grad_norm_relative_diff_max": max(relative.values()),
        "audio_tower_grads": sum(n.startswith("thinker.audio_tower") for n in names),
        "same_parameters_with_grads": names == sorted(packed[2]),
        "peak_gib": torch.cuda.max_memory_allocated() / 1024**3,
    }
    print(json.dumps(report, indent=2), flush=True)
    with open(args.output, "w") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    main()
