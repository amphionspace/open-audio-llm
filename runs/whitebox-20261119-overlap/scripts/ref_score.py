"""Where are the target's words lost in overlapped speech? Teacher-forced reference scoring + enrollment swap.

For every target-present Chinese TS-ASR sample of an AmphionEval run (same input construction as
``runs/whitebox-20261107/scripts/extract_states.py``: ms-swift template, 5 s enrollment, global audio attention):

1. own pass: the model's vLLM output (``raw_text``) is teacher-forced; the share of assistant tokens whose HF
   argmax equals the next emitted token checks that HF reproduces the served model.
2. reference pass: ``language Chinese<asr_text>`` + reference is teacher-forced. For each reference token we keep
   its log-prob and rank among non-punctuation tokens (the reference has no punctuation, the model writes it), the
   rank of ``<|im_end|>`` (would the model stop here), and whether the top competitor is a word the interferer
   said (and, as a chance control, a word some other sample's interferer said).
3. swap pass: the same mixture with another speaker's enrollment (same 5 s length, so every position lines up).
   Per audio-tower layer and per LLM layer, cosine between the true- and swapped-enrollment states at the mixture
   tokens (and, in the LLM, at the transcript positions): where does the enrollment change how the mixture is
   represented. Also the reference log-prob and P(None) at the language token under both enrollments.

Output: one JSON line per sample (``--out``); aggregate with ``analyze_ref_score.py``.
Usage: ref_score.py --checkpoint DIR --run-dir AE_RUN --out OUT.jsonl [--shard i --num-shards n] [--limit N]
"""
import argparse
import json
import random
import re
import time
from pathlib import Path

import numpy as np
import torch

CATALOG = "/workspace/data/registry/catalog"
ROOTS = "/workspace/data/registry/roots.json"
SR = 16000
ENROLL_SECONDS = 5.0  # serving contract (AMPHION_TSASR_ENROLL_SEC), as in extract_states.py
RANK_EDGES = (1, 5, 20, 100)  # rank buckets: 1, 2-5, 6-20, 21-100, >100
# whitespace, ASCII punctuation, CJK symbols and fullwidth punctuation (fullwidth digits and letters are kept)
_WIDE = [(0x3000, 0x303F), (0xFF01, 0xFF0F), (0xFF1A, 0xFF20), (0xFF3B, 0xFF40), (0xFF5B, 0xFF65)]
PUNCT = re.compile("[" + r"\s!-/:-@\[-`{-~" + "".join(f"{chr(a)}-{chr(b)}" for a, b in _WIDE) + "…—·“”‘’" + "]+")


def load_ref(resolver, ref, seconds=None):
    from audio_data_contract import AudioRef

    cut = resolver.get_cut(AudioRef(**{k: ref[k] for k in ("dataset_id", "version", "split", "cut_id")}))
    if cut.sampling_rate != SR:
        cut = cut.resample(SR)
    audio = np.clip(cut.load_audio().mean(0) * 32768, -32768, 32767).astype(np.int16)
    if seconds is not None:
        count = int(seconds * SR)
        audio = np.pad(audio[:count], (0, max(0, count - len(audio))))
    return audio


class Encoded(torch.utils.data.Dataset):
    """Catalog audio + template encoding of the three rows (own output, reference, swapped enrollment)."""

    def __init__(self, samples, swap, predictions, template):
        self.samples, self.swap, self.predictions, self.template = samples, swap, predictions, template
        self.resolver = None

    def __len__(self):
        return len(self.samples)

    def row(self, sample, enroll, mix, content):
        return {"messages": [{"role": "system", "content": sample["context"]}, {"role": "user", "content": "<audio>"},
                             {"role": "assistant", "content": content}], "audios": [enroll], "mix_wav": mix}

    def __getitem__(self, index):
        from open_audio_llm.data.catalog_resolver import LhotseCatalogAudioResolver
        from open_audio_llm.data.qwen3_asr import prepare_ts_clips

        if self.resolver is None:
            self.resolver = LhotseCatalogAudioResolver(CATALOG, ROOTS)
        sample = self.samples[index]
        slots = {s["name"]: s["ref"] for s in sample["record"]["audio_slots"]}
        other = {s["name"]: s["ref"] for s in self.swap[index]["record"]["audio_slots"]}
        mix = load_ref(self.resolver, slots["mixture"]).astype(np.float32) / 32768
        enroll = load_ref(self.resolver, slots["enrollment"], ENROLL_SECONDS).astype(np.float32) / 32768
        enroll_swap = load_ref(self.resolver, other["enrollment"], ENROLL_SECONDS).astype(np.float32) / 32768
        e1, m1 = prepare_ts_clips([enroll, mix], SR)
        e2, m2 = prepare_ts_clips([enroll_swap, mix], SR)
        assert len(e1) == len(e2) and np.array_equal(m1, m2)
        reference = "language Chinese<asr_text>" + sample["reference"]
        own = self.predictions[sample["id"]]["raw_text"]
        return index, [self.template.encode(self.row(sample, e1, m1, own)),
                       self.template.encode(self.row(sample, e1, m1, reference)),
                       self.template.encode(self.row(sample, e2, m2, reference))]


def interferer_text(sample):
    meta = sample["record"].get("metadata") or {}
    return "".join(u.get("text", "") for it in (meta.get("interferers") or []) for u in it.get("utts", []))


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--limit", type=int, default=0, help="first N selected samples of this shard (smoke tests)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    from swift.model import get_model_processor
    from swift.template import get_template

    import open_audio_llm.integrations.ms_swift.register_qwen3_asr  # noqa: F401  registers the model/template

    run = Path(args.run_dir)
    predictions = {p["id"]: p for p in map(json.loads, open(run / "predictions.jsonl")) if p["status"] == "success"}
    rows = [json.loads(line) for line in open(run / "samples.jsonl")]
    rows = [s for s in rows if s["id"] in predictions and s.get("task") == "ts_asr" and s.get("language") == "zh"
            and s["reference"].strip()]
    rows.sort(key=lambda s: s["id"])
    rng = random.Random(args.seed)
    swap = []
    for s in rows:  # another sample whose target speaker differs (deterministic)
        spk = (s["record"].get("metadata") or {}).get("target_speaker")
        while True:
            o = rows[rng.randrange(len(rows))]
            if o["id"] != s["id"] and (o["record"].get("metadata") or {}).get("target_speaker") != spk:
                swap.append(o)
                break
    chance_texts = [interferer_text(rows[rng.randrange(len(rows))]) for _ in rows]
    keep = [i for i in range(len(rows)) if i % args.num_shards == args.shard]
    if args.limit:
        keep = keep[:args.limit]
    samples, swap, chance_texts = [rows[i] for i in keep], [swap[i] for i in keep], [chance_texts[i] for i in keep]
    print(f"shard {args.shard}/{args.num_shards}: {len(samples)} target-present samples", flush=True)

    model, processor = get_model_processor(args.checkpoint, model_type="amphion_asr_1.7b", torch_dtype=torch.bfloat16,
                                           attn_impl="eager", device_map=args.device)
    model.eval()
    thinker = model.thinker
    tower, text = thinker.audio_tower, thinker.model
    tower.config._attn_implementation = "sdpa"  # one global window over [enrollment, SEP, mixture], as training/serving
    tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor
    template = get_template(processor, max_length=None, template_type="amphion_asr_1.7b")
    template.set_mode("train")

    pad_id = tokenizer.convert_tokens_to_ids("<|audio_pad|>")
    asr_text_id = tokenizer.convert_tokens_to_ids("<asr_text>")
    im_end = tokenizer.convert_tokens_to_ids("<|im_end|>")
    none_id = tokenizer.encode(" None", add_special_tokens=False)
    chinese_id = tokenizer.encode(" Chinese", add_special_tokens=False)
    assert len(none_id) == len(chinese_id) == 1
    none_id, chinese_id = none_id[0], chinese_id[0]
    vocab = len(tokenizer)
    punct = torch.zeros(thinker.lm_head.out_features, dtype=torch.bool)
    for i in range(min(vocab, punct.numel())):
        if i == im_end:
            continue
        piece = tokenizer.decode([i])
        if piece and not PUNCT.sub("", piece):
            punct[i] = True
    punct = punct.to(args.device)

    captured = {"llm": [None] * (len(text.layers) + 1), "audio": [None] * (len(tower.layers) + 1)}

    def save(store, index, pre):
        def hook(module, inputs, output=None):
            captured[store][index] = inputs[0] if pre else (output[0] if isinstance(output, tuple) else output)
        return hook

    text.layers[0].register_forward_pre_hook(save("llm", 0, True))
    for i, layer in enumerate(text.layers):
        layer.register_forward_hook(save("llm", i + 1, False))
    tower.layers[0].register_forward_pre_hook(save("audio", 0, True))
    for i, layer in enumerate(tower.layers):
        layer.register_forward_hook(save("audio", i + 1, False))

    def forward(encoded):
        batch = template.data_collator([encoded])
        labels = batch.pop("labels")[0]
        batch = {k: v.to(args.device) if torch.is_tensor(v) else v for k, v in batch.items()}
        batch["input_features"] = batch["input_features"].to(torch.bfloat16)
        with torch.inference_mode():
            logits = model(**batch, use_cache=False).logits[0].float()
        ids = batch["input_ids"][0]
        llm = [h[0] for h in captured["llm"]]
        audio = [a for a in captured["audio"]]
        return ids, labels.to(args.device), logits, llm, audio

    def cos(a, b):
        return torch.nn.functional.cosine_similarity(a.float(), b.float(), dim=-1).mean().item()

    out = open(args.out, "w")
    loader = torch.utils.data.DataLoader(Encoded(samples, swap, predictions, template), batch_size=None,
                                         num_workers=args.workers, prefetch_factor=4 if args.workers else None)
    start = time.time()
    for done, (index, (own, ref, swp)) in enumerate(loader):
        sample = samples[index]
        rec = {"id": sample["id"], "source": sample["source"], "labels": sample["record"].get("labels") or {},
               "swap_id": swap[index]["id"]}

        # 1. own output: does HF greedy reproduce vLLM token by token?
        ids, labels, logits, _, _ = forward(own)
        pos = (labels != -100).nonzero().flatten()
        pos = pos[pos > 0]
        pred_next = logits[pos - 1].argmax(-1)
        rec["own_agree"] = [int((pred_next == ids[pos]).sum()), int(pos.numel())]

        # 2. reference teacher forcing
        ids, labels, logits, llm_ref, audio_ref = forward(ref)
        assistant = (labels != -100).nonzero().flatten()
        decision = int(assistant[0])  # "language"
        start_text = int((ids == asr_text_id).nonzero().flatten()[-1]) + 1
        text_pos = torch.arange(start_text, int(assistant[-1]), device=ids.device)  # reference tokens, without <|im_end|>
        lp = torch.log_softmax(logits[text_pos - 1].masked_fill(punct, float("-inf")), -1)
        tgt = ids[text_pos]
        tgt_lp = lp.gather(1, tgt[:, None])[:, 0]
        is_punct_tgt = punct[tgt]
        rank = (lp > tgt_lp[:, None]).sum(-1) + 1
        eos_rank = (lp > lp[:, im_end][:, None]).sum(-1) + 1
        top1 = lp.argmax(-1)
        itext, ctext = interferer_text(sample), chance_texts[index]
        wrong = (rank > 1) & ~is_punct_tgt
        comp = [tokenizer.decode([int(t)]).strip() for t in top1[wrong].tolist()]
        hist = np.histogram(rank[~is_punct_tgt].cpu().numpy(), bins=[0, *[e + 0.5 for e in RANK_EDGES], 1e9])[0]
        rec["ref"] = {
            "n_tokens": int((~is_punct_tgt).sum()), "logp_sum": float(tgt_lp[~is_punct_tgt].sum()),
            "rank_hist": hist.tolist(),
            "wrong_logp": [round(float(x), 3) for x in tgt_lp[wrong].tolist()],
            "wrong_rank": rank[wrong].tolist(),
            "eos_top1": int(((top1 == im_end) & ~is_punct_tgt).sum()),
            "competitor_in_interferer": sum(1 for c in comp if c and itext and c in itext),
            "competitor_in_chance": sum(1 for c in comp if c and ctext and c in ctext),
            "competitor_in_reference": sum(1 for c in comp if c and c in sample["reference"]),
        }
        none_lp = torch.log_softmax(logits[decision], -1)
        rec["decision_true"] = {"none": float(none_lp[none_id]), "chinese": float(none_lp[chinese_id])}

        # 3. swapped enrollment, same mixture and reference
        audio_ref_mix = None
        n_tower = audio_ref[-1].shape[0]
        pads = (ids == pad_id).nonzero().flatten()
        ids2, labels2, logits2, llm_swp, audio_swp = forward(swp)
        assert torch.equal(ids, ids2), "swapped input must line up token by token"
        n_enroll = None
        # mixture tokens in the tower output: after enrollment tokens and SEP; the LLM sees them at the matching pads
        from open_audio_llm.tsasr.sep_token import conv_token_count
        n_enroll = conv_token_count(int(ref["enroll_n_frames"]))
        mix_tower = slice(n_enroll + 1, n_tower)
        mix_llm = pads[n_enroll + 1:]
        rec["swap"] = {
            "audio_mix_cos": [round(cos(a[mix_tower], b[mix_tower]), 5) for a, b in zip(audio_ref, audio_swp)],
            "audio_enroll_cos": [round(cos(a[:n_enroll], b[:n_enroll]), 5) for a, b in zip(audio_ref, audio_swp)],
            "llm_mix_cos": [round(cos(a[mix_llm], b[mix_llm]), 5) for a, b in zip(llm_ref, llm_swp)],
            "llm_text_cos": [round(cos(a[text_pos], b[text_pos]), 5) for a, b in zip(llm_ref, llm_swp)],
        }
        # Same cosines split by who speaks at each mixture token (metadata times; token k covers
        # [k, k+1) * duration / n_tokens): a target-selective representation changes most where the target speaks.
        meta = sample["record"].get("metadata") or {}
        dur = float(rec["labels"].get("duration_s") or 0)
        spans_t = [(u["start"], u["start"] + u["duration"]) for u in meta.get("target_utts") or []]
        spans_i = [(u["start"], u["start"] + u["duration"]) for it in meta.get("interferers") or [] for u in it.get("utts", [])]
        n_mix = n_tower - n_enroll - 1
        if dur > 0 and spans_t and n_mix > 0:
            t = (torch.arange(n_mix, dtype=torch.float32) + 0.5) * dur / n_mix
            in_t = torch.zeros(n_mix, dtype=torch.bool)
            in_i = torch.zeros(n_mix, dtype=torch.bool)
            for a, b in spans_t:
                in_t |= (t >= a) & (t < b)
            for a, b in spans_i:
                in_i |= (t >= a) & (t < b)
            regions = {"target_only": in_t & ~in_i, "overlap": in_t & in_i, "interferer_only": ~in_t & in_i,
                       "silence": ~in_t & ~in_i}
            rec["swap"]["regions"] = {}
            for name, mask in regions.items():
                if int(mask.sum()) < 3:
                    continue
                idx = mask.nonzero().flatten().to(ids.device)
                rec["swap"]["regions"][name] = {
                    "n": int(mask.sum()),
                    "audio": [round(cos(a[mix_tower][idx], b[mix_tower][idx]), 5) for a, b in zip(audio_ref, audio_swp)],
                    "llm": [round(cos(a[mix_llm][idx], b[mix_llm][idx]), 5) for a, b in zip(llm_ref, llm_swp)],
                }
        lp2 = torch.log_softmax(logits2[text_pos - 1].masked_fill(punct, float("-inf")), -1)
        rec["swap"]["ref_logp_sum"] = float(lp2.gather(1, tgt[:, None])[:, 0][~is_punct_tgt].sum())
        none2 = torch.log_softmax(logits2[decision], -1)
        rec["decision_swap"] = {"none": float(none2[none_id]), "chinese": float(none2[chinese_id])}
        del audio_ref_mix
        out.write(json.dumps(rec, ensure_ascii=False) + "\n")
        if (done + 1) % 100 == 0:
            out.flush()
            print(f"{done + 1}/{len(samples)} {(done + 1) / (time.time() - start):.2f} samples/s", flush=True)
    out.close()
    print(f"done {len(samples)} in {time.time() - start:.0f}s", flush=True)


if __name__ == "__main__":
    main()
