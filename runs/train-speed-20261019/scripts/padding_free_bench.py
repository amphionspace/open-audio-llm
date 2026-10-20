"""Time the Qwen3-ASR text decoder on real microbatch lengths: padded vs padding-free.

Batches are cut from realized token lengths (04-length-key) with the current sort key.
Arms run the same embeddings through SDPA with padding (current), FlashAttention-2 with
padding, and FlashAttention-2 on one packed row (cu_seqlens computed once per batch, as a
collator would), each with and without Liger. Supervised positions are each sample's
last tokens, as transcripts are; the head projects only those (--supervised_head_loss).
"""
import argparse
import json
import random
import statistics
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from qwen_asr import Qwen3ASRModel
from qwen_asr.core.transformers_backend import modeling_qwen3_asr as qwen3_asr

from open_audio_llm.integrations.ms_swift.liger import enable_liger_kernels

sys.path.insert(0, str(Path(__file__).parent))
from length_key_sim import batches  # noqa: E402


def make_inputs(lengths, supervised, hidden, scale, vocab):
    embeds = [torch.randn(n, hidden, device="cuda") * scale for n in lengths]
    targets = [torch.randint(0, vocab, (s,), device="cuda") for s in supervised]
    width = max(lengths)
    padded = torch.zeros(len(lengths), width, hidden, device="cuda")
    mask = torch.zeros(len(lengths), width, dtype=torch.long, device="cuda")
    for row, e in enumerate(embeds):
        padded[row, :len(e)] = e
        mask[row, :len(e)] = 1
    rows = torch.cat([torch.full((s,), r) for r, s in enumerate(supervised)])
    cols = torch.cat([torch.arange(n - s, n) for n, s in zip(lengths, supervised)])
    starts = torch.tensor([0, *torch.tensor(lengths).cumsum(0)[:-1].tolist()])
    flat = cols + starts[rows]
    cu = torch.tensor([0, *torch.tensor(lengths).cumsum(0).tolist()], dtype=torch.int32, device="cuda")
    packed = {
        "inputs_embeds": torch.cat(embeds)[None].to(torch.bfloat16),
        "position_ids": torch.cat([torch.arange(n) for n in lengths])[None].cuda(),
        "cu_seq_lens_q": cu, "cu_seq_lens_k": cu, "max_length_q": width, "max_length_k": width,
    }
    return {
        "padded": ({"inputs_embeds": padded.to(torch.bfloat16), "attention_mask": mask},
                   (rows.cuda(), cols.cuda())),
        "packed": (packed, (torch.zeros_like(flat).cuda(), flat.cuda())),
        "targets": torch.cat(targets),
    }


def step(text, head, inputs, positions, targets):
    hidden = text(**inputs).last_hidden_state
    loss = F.cross_entropy(head(hidden[positions]).float(), targets)
    loss.backward()
    return loss.detach(), hidden.detach()[positions]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--lengths", required=True)
    parser.add_argument("--batches", type=int, default=24)
    parser.add_argument("--repeats", type=int, default=6)
    parser.add_argument("--kernels", choices=("native", "liger"), required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    rows = sorted((json.loads(line) for line in open(args.lengths)), key=lambda r: r["number"])
    plan = [b for start in range(0, len(rows) - 1999, 2000)
            for b in batches(rows[start:start + 2000], lambda r: r["cost"], 12, 480)]
    chosen = random.Random(0).sample(plan, args.batches)
    torch.manual_seed(0)
    full = Qwen3ASRModel.from_pretrained(args.model, dtype=torch.bfloat16, device_map="cuda:0",
                                         attn_implementation="sdpa").model
    head = full.thinker.lm_head
    text = full.thinker.model.train()
    del full.thinker.audio_tower  # Only the text decoder is timed; keep memory for activations.
    torch.cuda.empty_cache()
    rope = qwen3_asr.apply_rotary_pos_emb
    if args.kernels == "liger":
        holder = torch.nn.Module()
        holder.thinker = torch.nn.Module()
        holder.thinker.model = text
        _, restore = enable_liger_kernels(holder)
        rope = qwen3_asr.apply_rotary_pos_emb
        restore()
    native_rope = qwen3_asr.apply_rotary_pos_emb
    scale = text.embed_tokens.weight.float().std().item()
    arms = {f"{args.kernels}/{attention}/{layout}": (text, rope, attention, layout)
            for attention, layout in (("sdpa", "padded"), ("flash_attention_2", "padded"),
                                      ("flash_attention_2", "packed"))}
    times = {name: [] for name in arms}
    agreement = []
    capacity = real = 0
    for number, batch in enumerate(chosen):
        lengths = [r["tokens"] for r in batch]
        supervised = [max(1, r["supervised"]) for r in batch]
        capacity += len(lengths) * max(lengths)
        real += sum(lengths)
        inputs = make_inputs(lengths, supervised, text.config.hidden_size, scale, head.out_features)
        outputs = {}
        names = list(arms)
        for repeat in range(args.repeats + 1):
            random.Random(f"{number}:{repeat}").shuffle(names)
            for name in names:
                text, rope, attention, layout = arms[name]
                qwen3_asr.apply_rotary_pos_emb = rope
                text.config._attn_implementation = attention
                batch_inputs, positions = inputs[layout]
                for module in (text, head):
                    module.zero_grad(set_to_none=True)
                torch.cuda.synchronize()
                started = time.perf_counter()
                loss, hidden = step(text, head, batch_inputs, positions, inputs["targets"])
                torch.cuda.synchronize()
                if repeat:
                    times[name].append(time.perf_counter() - started)
                else:
                    outputs[name] = (loss.item(), hidden.float())
        reference = outputs[f"{args.kernels}/sdpa/padded"]
        agreement.append({name: {"loss_diff": abs(loss - reference[0]),
                                 "hidden_rel_diff": ((hidden - reference[1]).norm() / reference[1].norm()).item()}
                          for name, (loss, hidden) in outputs.items()})
        print(number, lengths, flush=True)
    qwen3_asr.apply_rotary_pos_emb = native_rope
    base = statistics.fmean(times[f"{args.kernels}/sdpa/padded"])
    report = {"batches": args.batches, "repeats": args.repeats,
              "token_padding_of_chosen_batches": 1 - real / capacity,
              "kernels": args.kernels, "arms": {}}
    for name, values in times.items():
        mean = statistics.fmean(values)
        report["arms"][name] = {
            "mean_s": mean, "median_s": statistics.median(values), "speedup_vs_current": base / mean,
            "loss_diff_max": max(a[name]["loss_diff"] for a in agreement),
            "hidden_rel_diff_max": max(a[name]["hidden_rel_diff"] for a in agreement),
        }
        print(name, json.dumps(report["arms"][name]), flush=True)
    report["peak_gib"] = torch.cuda.max_memory_allocated() / 1024**3
    with open(args.output, "w") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    main()
