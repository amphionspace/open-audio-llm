"""Time the Qwen3-ASR text decoder forward+backward with and without Liger kernels.

Real weights, synthetic embeddings at real microbatch shapes. Native and Liger copies
alternate every iteration on the same GPU, so slow drifts in shared load cancel.
The head projects only supervised positions, as --supervised_head_loss does.
"""
import argparse
import copy
import json
import statistics
import time

import torch
import torch.nn.functional as F
from qwen_asr import Qwen3ASRModel
from qwen_asr.core.transformers_backend import modeling_qwen3_asr as qwen3_asr

from open_audio_llm.integrations.ms_swift.liger import enable_liger_kernels


def step(text, head, embeds, mask, positions, targets):
    hidden = text(inputs_embeds=embeds, attention_mask=mask).last_hidden_state
    loss = F.cross_entropy(head(hidden[positions]).float(), targets)
    loss.backward()
    return loss.detach()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--shapes", nargs="+", required=True, help="BATCHxLENGTH")
    parser.add_argument("--padding", type=float, default=0.24)
    parser.add_argument("--supervised", type=int, default=262)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    torch.manual_seed(0)
    full = Qwen3ASRModel.from_pretrained(args.model, dtype=torch.bfloat16, device_map="cuda:0",
                                         attn_implementation="sdpa").model
    native = full.thinker.model.train()
    head = full.thinker.lm_head
    holder = torch.nn.Module()
    holder.thinker = torch.nn.Module()
    holder.thinker.model = copy.deepcopy(native)
    patched, restore = enable_liger_kernels(holder)
    liger_rope = qwen3_asr.apply_rotary_pos_emb
    restore()
    native_rope = qwen3_asr.apply_rotary_pos_emb
    arms = {"native": (native, native_rope), "liger": (holder.thinker.model, liger_rope)}
    scale = native.embed_tokens.weight.float().std().item()
    report = {"model": args.model, "patched": patched, "padding": args.padding,
              "supervised": args.supervised, "shapes": {}}
    for shape in args.shapes:
        batch, length = map(int, shape.split("x"))
        lengths = torch.randint(int(length * (1 - 2 * args.padding)), length + 1, (batch,))
        lengths[0] = length
        mask = (torch.arange(length)[None] < lengths[:, None]).long().cuda()
        embeds = (torch.randn(batch, length, native.config.hidden_size) * scale).to("cuda", torch.bfloat16)
        valid = mask.nonzero()
        positions = tuple(valid[torch.randperm(len(valid))[:args.supervised]].T)
        targets = torch.randint(0, head.out_features, (len(positions[0]),), device="cuda")
        times = {name: [] for name in arms}
        losses, grads = {}, {}
        for iteration in range(5 + args.iterations):
            order = list(arms) if iteration % 2 else list(arms)[::-1]
            for name in order:
                text, rope = arms[name]
                qwen3_asr.apply_rotary_pos_emb = rope
                for module in (text, head):
                    module.zero_grad(set_to_none=True)
                torch.cuda.synchronize()
                started = time.perf_counter()
                loss = step(text, head, embeds, mask, positions, targets)
                torch.cuda.synchronize()
                if iteration >= 5:
                    times[name].append(time.perf_counter() - started)
                if iteration == 0:
                    losses[name] = loss.item()
                    grads[name] = {n: p.grad.float().norm().item() for n, p in text.named_parameters()
                                   if p.grad is not None}
        qwen3_asr.apply_rotary_pos_emb = native_rope
        # lm_head shares the native embed_tokens weight; the deep copy does not.
        relative = [abs(grads["liger"][n] - g) / max(g, 1e-12) for n, g in grads["native"].items()
                    if n in grads["liger"]]
        result = {name: {"median_s": statistics.median(v), "mean_s": statistics.fmean(v)}
                  for name, v in times.items()}
        result.update(speedup=result["native"]["median_s"] / result["liger"]["median_s"],
                      saved_s=result["native"]["median_s"] - result["liger"]["median_s"],
                      loss=losses, grad_norm_relative_diff_max=max(relative),
                      grad_norm_relative_diff_median=statistics.median(relative),
                      tokens=int(mask.sum()), capacity=batch * length,
                      peak_gib=torch.cuda.max_memory_allocated() / 1024**3)
        report["shapes"][shape] = result
        print(shape, json.dumps(result), flush=True)
    with open(args.output, "w") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    main()
