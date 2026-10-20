"""Per-parameter gradient error of native-BF16 and Liger-BF16 against an FP32 reference.

The same real-weight text decoder and synthetic microbatch run three times: FP32 native
(reference), BF16 native and BF16 Liger. Liger is acceptable if its error per parameter
stays at the level BF16 already has.
"""
import argparse
import copy
import json

import torch
import torch.nn.functional as F
from qwen_asr import Qwen3ASRModel
from qwen_asr.core.transformers_backend import modeling_qwen3_asr as qwen3_asr

from open_audio_llm.integrations.ms_swift.liger import enable_liger_kernels


def grads(text, head, embeds, mask, positions, targets):
    for module in (text, head):
        module.zero_grad(set_to_none=True)
    hidden = text(inputs_embeds=embeds.to(head.weight.dtype), attention_mask=mask).last_hidden_state
    loss = F.cross_entropy(head(hidden[positions]).float(), targets)
    loss.backward()
    return loss.item(), {n: p.grad.float().clone() for n, p in text.named_parameters()
                         if p.grad is not None and n != "embed_tokens.weight"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--batch", type=int, default=12)
    parser.add_argument("--length", type=int, default=120)
    parser.add_argument("--supervised", type=int, default=120)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    torch.manual_seed(0)
    full = Qwen3ASRModel.from_pretrained(args.model, dtype=torch.bfloat16, device_map="cuda:0",
                                         attn_implementation="sdpa").model
    native = full.thinker.model.train()
    head = full.thinker.lm_head
    lengths = torch.randint(args.length // 2, args.length + 1, (args.batch,))
    mask = (torch.arange(args.length)[None] < lengths[:, None]).long().cuda()
    scale = native.embed_tokens.weight.float().std().item()
    embeds = torch.randn(args.batch, args.length, native.config.hidden_size, device="cuda") * scale
    valid = mask.nonzero()
    positions = tuple(valid[torch.randperm(len(valid))[:args.supervised]].T)
    targets = torch.randint(0, head.out_features, (len(positions[0]),), device="cuda")

    reference_text, reference_head = copy.deepcopy(native).float(), copy.deepcopy(head).float()
    reference_loss, reference = grads(reference_text, reference_head, embeds, mask, positions, targets)
    del reference_text, reference_head
    native_loss, native_grads = grads(native, head, embeds, mask, positions, targets)
    holder = torch.nn.Module()
    holder.thinker = torch.nn.Module()
    holder.thinker.model = native
    _, restore = enable_liger_kernels(holder)
    try:
        liger_loss, liger_grads = grads(native, head, embeds, mask, positions, targets)
    finally:
        restore()

    rows = []
    for name, ref in reference.items():
        norm = ref.norm().item()
        rows.append({"name": name, "ref_norm": norm,
                     "native_error": (native_grads[name] - ref).norm().item() / max(norm, 1e-30),
                     "liger_error": (liger_grads[name] - ref).norm().item() / max(norm, 1e-30)})
    rows.sort(key=lambda row: row["liger_error"] - row["native_error"], reverse=True)
    ratio = [row["liger_error"] / max(row["native_error"], 1e-30) for row in rows]
    report = {"loss": {"fp32": reference_loss, "native_bf16": native_loss, "liger_bf16": liger_loss},
              "parameters": len(rows),
              "native_error_max": max(r["native_error"] for r in rows),
              "liger_error_max": max(r["liger_error"] for r in rows),
              "liger_over_native_error_ratio_median": sorted(ratio)[len(ratio) // 2],
              "liger_over_native_error_ratio_max": max(ratio),
              "worst_liger_minus_native": rows[:12]}
    print(json.dumps(report, indent=2), flush=True)
    with open(args.output, "w") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    main()
