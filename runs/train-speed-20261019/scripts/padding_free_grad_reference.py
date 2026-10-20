"""Per-parameter gradient error of BF16 SDPA (padded) and BF16 FlashAttention-2 (packed)
against an FP32 SDPA reference, on one real microbatch's token lengths.

Packing is acceptable if its error per parameter stays at the level BF16 SDPA already has.
"""
import argparse
import copy
import json
import random
import statistics
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from qwen_asr import Qwen3ASRModel

sys.path.insert(0, str(Path(__file__).parent))
from length_key_sim import batches  # noqa: E402
from padding_free_bench import make_inputs  # noqa: E402


def run(text, head, inputs, positions, targets, attention):
    text.config._attn_implementation = attention
    for module in (text, head):
        module.zero_grad(set_to_none=True)
    dtype = head.weight.dtype
    inputs = {k: v.to(dtype) if k == "inputs_embeds" else v for k, v in inputs.items()}
    hidden = text(**inputs).last_hidden_state[positions]
    loss = F.cross_entropy(head(hidden).float(), targets)
    loss.backward()
    return loss.item(), hidden.detach().float(), {n: p.grad.float().clone() for n, p in text.named_parameters()
                                                  if p.grad is not None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--lengths", required=True)
    parser.add_argument("--max_tokens", type=int, default=2000)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    rows = sorted((json.loads(line) for line in open(args.lengths)), key=lambda r: r["number"])
    plan = batches(rows[:2000], lambda r: r["cost"], 12, 480)
    # A real batch whose padded size keeps the FP32 copy within a shared GPU.
    candidates = [b for b in plan if len(b) == 12 and len(b) * max(r["tokens"] for r in b) <= args.max_tokens
                  and min(r["tokens"] for r in b) < 0.8 * max(r["tokens"] for r in b)]
    batch = random.Random(0).choice(candidates)
    lengths = [r["tokens"] for r in batch]
    supervised = [max(1, r["supervised"]) for r in batch]
    torch.manual_seed(0)
    full = Qwen3ASRModel.from_pretrained(args.model, dtype=torch.bfloat16, device_map="cuda:0",
                                         attn_implementation="sdpa").model
    text, head = full.thinker.model.train(), full.thinker.lm_head
    del full.thinker.audio_tower
    scale = text.embed_tokens.weight.float().std().item()
    inputs = make_inputs(lengths, supervised, text.config.hidden_size, scale, head.out_features)
    padded, padded_positions = inputs["padded"]
    packed, packed_positions = inputs["packed"]
    targets = inputs["targets"]
    padded = {k: v.float() if k == "inputs_embeds" else v for k, v in padded.items()}
    packed = {k: v.float() if k == "inputs_embeds" else v for k, v in packed.items()}

    reference_text, reference_head = copy.deepcopy(text).float(), copy.deepcopy(head).float()
    ref_loss, ref_hidden, ref_grads = run(reference_text, reference_head, padded, padded_positions, targets, "sdpa")
    del reference_text, reference_head
    torch.cuda.empty_cache()
    arms = {"sdpa_padded_bf16": run(text, head, padded, padded_positions, targets, "sdpa"),
            "flash_packed_bf16": run(text, head, packed, packed_positions, targets, "flash_attention_2")}
    report = {"lengths": lengths, "padding": 1 - sum(lengths) / (len(lengths) * max(lengths)),
              "loss": {"fp32_sdpa_padded": ref_loss, **{k: v[0] for k, v in arms.items()}}, "arms": {}}
    errors = {}
    for name, (_, hidden, grads) in arms.items():
        errors[name] = {n: (grads[n] - g).norm().item() / max(g.norm().item(), 1e-30) for n, g in ref_grads.items()}
        report["arms"][name] = {
            "hidden_rel_error": ((hidden - ref_hidden).norm() / ref_hidden.norm()).item(),
            "grad_rel_error_median": statistics.median(errors[name].values()),
            "grad_rel_error_max": max(errors[name].values()),
        }
    ratio = [errors["flash_packed_bf16"][n] / max(errors["sdpa_padded_bf16"][n], 1e-30) for n in ref_grads]
    report["packed_over_sdpa_error_ratio"] = {"median": statistics.median(ratio), "max": max(ratio)}
    print(json.dumps(report, indent=2), flush=True)
    with open(args.output, "w") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    main()
