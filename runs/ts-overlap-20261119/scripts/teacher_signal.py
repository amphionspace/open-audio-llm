"""tasks/05: was the clean-teacher signal of task 03 right? Teacher-forced label scoring on training mixtures.

Draws target-present online TS mixtures from the training pipeline: CatalogSwiftDataset on
configs/teacher-signal-data.yaml (the clean-target sources of train-data-cleankl.yaml), records drawn by source
weight, at epoch EPOCH so the mixtures are fresh draws. Each sample's training label is teacher-forced under four
audio inputs, the enrollment first as in training:
  mix          the mixture: the student's input (no SpecAugment)
  clean_zero   the clean target on the mixture timeline, digital silence elsewhere: the task-03 teacher input
  clean_floor  clean_zero plus a white or pink noise floor FLOOR_BELOW_DB under the target's level
  clean_crop   the clean target from its first to its last sample (no lead or trail)
For each input and answer part (prefix: up to and including <asr_text>; text; end: <|im_end|>) one JSON line per
sample keeps the per-token -log p(label), entropy and argmax == label, KL(input || mix) on the same label tokens, and
log P(" None") where the label says " Chinese" / " English" (does the input make the model doubt the target is there).
Aggregate with teacher_signal_report.py.
Usage: teacher_signal.py --checkpoint DIR --data-config YAML --out OUT.jsonl --samples N [--shard i --num-shards n]
"""
import argparse
import json
import random
import time

import numpy as np
import torch

SR = 16000
EPOCH = 1000  # task 03 trained 6000 steps inside its first epochs; epoch 1000 gives fresh mixtures of the same recipe
CANDIDATES_PER_SAMPLE = 3  # absent-target and nonspeech draws carry no clean target and are skipped
FLOOR_BELOW_DB = (30.0, 50.0)  # floor level under the target's RMS: audible only in pauses, like a quiet room
VIEWS = ("mix", "clean_zero", "clean_floor", "clean_crop")
DROP = ("audio_augmentation", "twin_enroll_wav", "ts_activity")  # SpecAugment and losses this check does not use
LAST = {}  # info of the last mixture built in this process (set by the patched builders)


def patch_builders():
    import open_audio_llm.data.catalog_dataset as cd

    build_mixture, build_scene = cd.build_mixture, cd.build_scene

    def mixture(*args, **kwargs):
        out = build_mixture(*args, **kwargs)
        info = kwargs.get("info")
        LAST.update(kind="single", target_spans=[list(s) for s in info["target_spans"]],
                    other_spans=[list(s) for s in info["other_spans"]], window=len(out[0]))
        return out

    def scene(*args, **kwargs):
        out = build_scene(*args, **kwargs)
        info = kwargs.get("info")
        LAST.update(kind=f"scene-{args[1]}", target_spans=[list(s) for s in info["target_spans"]],
                    other_spans=[list(s) for s in info["other_spans"]], window=len(out))
        return out

    cd.build_mixture, cd.build_scene = mixture, scene


def overlap_ratio(target_spans, other_spans, length):
    target, other = np.zeros(length, bool), np.zeros(length, bool)
    for a, b in target_spans:
        target[a:b] = True
    for a, b in other_spans:
        other[max(0, a):max(0, b)] = True
    return float((target & other).sum() / max(1, target.sum()))


def hop_pad(x, hop):
    return np.pad(x, (0, -len(x) % hop)).astype(np.float32)


class Views(torch.utils.data.Dataset):
    """Training sample -> template encodings of the four inputs (or None without a clean target)."""

    def __init__(self, dataset, candidates, template):
        self.dataset, self.candidates, self.template = dataset, candidates, template

    def __len__(self):
        return len(self.candidates)

    def __getitem__(self, i):
        from open_audio_llm.data.qwen3_asr import MEL_HOP_SECONDS
        from open_audio_llm.data.ts_online import _synthetic_noise

        record_idx = self.candidates[i]
        LAST.clear()
        sample = self.dataset[(EPOCH, record_idx, None)]
        clean = sample.pop("clean_mix_wav", None)
        if clean is None or not LAST.get("target_spans"):
            return i, None, None
        record = self.dataset.records[record_idx].record
        for key in DROP:
            sample.pop(key, None)
        hop = int(SR * MEL_HOP_SECONDS)
        spans = LAST["target_spans"]
        start, end = min(a for a, _ in spans), max(b for _, b in spans)
        rng = random.Random(f"teacher-signal:{EPOCH}:{record_idx}")
        active = clean[start:end]
        rms = float(np.sqrt(np.mean(active[np.abs(active) > 0] ** 2))) if np.any(active) else 0.0
        noise = _synthetic_noise(rng, rng.choice(("white", "pink")), len(clean))
        below = rng.uniform(*FLOOR_BELOW_DB)
        noise = noise / max(float(np.sqrt(np.mean(noise ** 2))), 1e-12) * rms * 10 ** (-below / 20)
        inputs = {"mix": sample["mix_wav"], "clean_zero": clean, "clean_floor": hop_pad(clean + noise, hop),
                  "clean_crop": hop_pad(clean[start:end], hop)}
        encoded = {}
        for name in VIEWS:
            row = dict(sample, mix_wav=inputs[name])
            encoded[name] = self.template.encode(row)
        meta = {"record": record.id, "source": f"{record.slot('mixture').ref.dataset_id}:{record.slot('mixture').ref.split}",
                "language": record.language, "kind": LAST["kind"], "window_s": round(LAST["window"] / SR, 3),
                "target_s": round(sum(b - a for a, b in spans) / SR, 3),
                "lead_s": round(start / SR, 3), "trail_s": round((LAST["window"] - end) / SR, 3),
                "overlap": round(overlap_ratio(spans, LAST["other_spans"], LAST["window"]), 4),
                "zero_frac": round(float(np.mean(clean == 0)), 4), "floor_below_db": round(below, 2)}
        return i, meta, encoded


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-config", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--samples", type=int, required=True, help="kept samples over all shards")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    from swift.model import get_model_processor
    from swift.template import get_template

    import open_audio_llm.integrations.ms_swift.register_qwen3_asr  # noqa: F401  registers the model/template
    from open_audio_llm.data.catalog_dataset import CatalogSwiftDataset, read_data_config, source_weights

    patch_builders()
    dataset = CatalogSwiftDataset(read_data_config(args.data_config), training=True, encode=None,
                                  message_format="qwen3_asr")
    weights = source_weights(dataset.sources)
    rng = random.Random(args.seed)
    picks = rng.choices(range(len(dataset.sources)), weights=weights, k=args.samples * CANDIDATES_PER_SAMPLE)
    candidates = [dataset.source_ranges[s][rng.randrange(len(dataset.source_ranges[s]))] for s in picks]
    candidates = candidates[args.shard::args.num_shards]
    quota = len(range(args.shard, args.samples, args.num_shards))
    print(f"shard {args.shard}/{args.num_shards}: {len(candidates)} candidates for {quota} samples", flush=True)

    model, processor = get_model_processor(args.checkpoint, model_type="amphion_asr_1.7b", torch_dtype=torch.bfloat16,
                                           attn_impl="sdpa", device_map=args.device)
    model.eval()
    tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor
    template = get_template(processor, max_length=None, template_type="amphion_asr_1.7b")
    template.set_mode("train")
    asr_text = tokenizer.convert_tokens_to_ids("<asr_text>")
    im_end = tokenizer.convert_tokens_to_ids("<|im_end|>")
    none_id = tokenizer.encode(" None", add_special_tokens=False)
    assert len(none_id) == 1
    none_id = none_id[0]

    def forward(encoded):
        batch = template.data_collator([encoded])
        labels = batch.pop("labels")[0].to(args.device)
        batch = {k: v.to(args.device) if torch.is_tensor(v) else v for k, v in batch.items()}
        batch["input_features"] = batch["input_features"].to(torch.bfloat16)
        with torch.inference_mode():
            logits = model(**batch, use_cache=False).logits[0]
        ids = batch["input_ids"][0]
        pos = (labels != -100).nonzero().flatten()
        pos = pos[pos > 0]
        return ids[pos], torch.log_softmax(logits[pos - 1].float(), -1)

    out = open(args.out, "w")
    loader = torch.utils.data.DataLoader(Views(dataset, candidates, template), batch_size=None,
                                         num_workers=args.workers, prefetch_factor=4 if args.workers else None)
    start, kept = time.time(), 0
    for i, meta, encoded in loader:
        if meta is None:
            continue
        scored = {name: forward(encoded[name]) for name in VIEWS}
        target, mix_lp = scored["mix"]
        text_start = int((target == asr_text).nonzero().flatten()[0]) + 1
        part = torch.full_like(target, 1)  # 0 prefix, 1 text, 2 end
        part[:text_start] = 0
        part[target == im_end] = 2
        rec = dict(meta, n_tokens=[int((part == p).sum()) for p in range(3)],
                   label_head=tokenizer.decode(target[:text_start].tolist()))
        for name in VIEWS:
            t, lp = scored[name]
            if not torch.equal(t, target):
                raise ValueError(f"{meta['record']}: {name} labels differ from the mixture's")
            nll = -lp.gather(1, t[:, None])[:, 0]
            entropy = -(lp.exp() * lp).sum(-1)
            correct = lp.argmax(-1) == t
            kl = (lp.exp() * (lp - mix_lp)).sum(-1)
            rec[name] = {
                "nll": [round(float(x), 4) for x in nll.tolist()],
                "entropy": [round(float(entropy[part == p].sum()), 4) for p in range(3)],
                "correct": [int(correct[part == p].sum()) for p in range(3)],
                "kl_to_mix": [round(float(kl[part == p].sum()), 4) for p in range(3)],
                "none_logp": round(float(lp[1, none_id]), 4),  # label position 1 is the language word
            }
        out.write(json.dumps(rec, ensure_ascii=False) + "\n")
        kept += 1
        if kept % 50 == 0:
            out.flush()
            print(f"{kept}/{quota} {kept / (time.time() - start):.2f} samples/s", flush=True)
        if kept >= quota:
            break
    out.close()
    print(f"done {kept} samples in {time.time() - start:.0f}s", flush=True)
    if kept < quota:
        raise SystemExit(f"only {kept} of {quota} samples had a clean target")


if __name__ == "__main__":
    main()
