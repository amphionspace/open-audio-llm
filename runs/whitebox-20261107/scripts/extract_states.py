"""White-box states of AmphionASR (Qwen3-ASR-1.7B TS-ASR) at the language decision position.

For each TS-ASR sample of an AmphionEval run, the exact training/inference input is built through
the ms-swift template (``Qwen3ASRTemplate._encode``: enrollment and mixture Mel extracted separately,
learned SEP between them) and the model's own vLLM output (``raw_text``) is teacher-forced after the
prompt. One forward then yields, at the decision position (the ``language`` token of the assistant
turn, whose logits predict the language name, `` None`` for "target absent"):

- LLM residual stream: embedding output + the output of each of the 28 decoder layers (pre final norm);
- audio-tower states per layer (conv+PE output, then each of the 24 encoder layers), mean-pooled over
  enrollment tokens and mixture tokens, plus the SEP token state;
- per LLM layer and head, attention mass from the decision position onto enrollment / SEP / mixture /
  system-turn / other tokens;
- logit lens (final norm + lm_head on every residual state): log-probs of `` None``, `` Chinese``,
  `` English`` and of the language token the model emitted, plus the per-layer argmax.

Attention: the LLM runs eager attention, and the weights are taken by a hook on each layer's
``self_attn`` (qwen_asr's ``_can_record_outputs`` names ``Qwen3ASRThinkerTextAttention`` while the layers
use ``Qwen3ASRTextAttention``, so ``output_attentions`` returns None). The audio tower runs SDPA with
no mask, i.e. one global window over [enrollment, SEP, mixture]: that is what training does (the
encoder never applies its ``cu_seqlens`` windows outside FlashAttention-2) and what the TS-ASR vLLM
service does (``tsasr/global_audio_attn.py`` overrides ``n_window_infer``). FlashAttention-2 here would
impose 800-frame windows that neither path uses.

Audio is read through the Catalog as int16 like ``joint-mtp-spec-20261107/scripts/spec_check.py``;
the enrollment is cut/zero-padded to ``ENROLL_SECONDS`` (the serving contract), the mixture is used
whole. Both are then hop-padded by ``prepare_ts_clips`` exactly as in training.
"""
import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

CATALOG = "/workspace/data/registry/catalog"
ROOTS = "/workspace/data/registry/roots.json"
SR = 16000
ENROLL_SECONDS = 5.0  # AMPHION_TSASR_ENROLL_SEC of the TS-ASR service and AmphionEval enroll_seconds
LANGUAGE_CANDIDATES = (" None", " Chinese", " English")
ATTN_BUCKETS = ("enrollment", "sep", "mixture", "system", "other")
MIN_AGREEMENT = 0.95


def load_ref(resolver, ref, seconds=None):
    """16 kHz mono int16 through the Catalog, cut or zero-padded to ``seconds`` (as spec_check.load_ref)."""
    from audio_data_contract import AudioRef

    cut = resolver.get_cut(AudioRef(**{k: ref[k] for k in ("dataset_id", "version", "split", "cut_id")}))
    if cut.sampling_rate != SR:
        cut = cut.resample(SR)
    audio = np.clip(cut.load_audio().mean(0) * 32768, -32768, 32767).astype(np.int16)
    if seconds is not None:
        count = int(seconds * SR)
        audio = np.pad(audio[:count], (0, max(0, count - len(audio))))
    return audio


def select(samples, predictions, args):
    """All ids from --ids-file, else a seed-balanced draw of --limit (half target-absent), else all."""
    rows = [s for s in samples if s["id"] in predictions]
    if args.ids_file:
        wanted = [line.strip() for line in open(args.ids_file) if line.strip()]
        by_id = {s["id"]: s for s in rows}
        return [by_id[i] for i in wanted]
    if not args.limit or args.limit >= len(rows):
        return rows
    rng = random.Random(args.seed)
    negatives = [s for s in rows if not s["reference"].strip()]
    positives = [s for s in rows if s["reference"].strip()]
    n_neg = min(len(negatives), args.limit // 2)
    chosen = {s["id"] for s in rng.sample(negatives, n_neg) + rng.sample(positives, args.limit - n_neg)}
    return [s for s in rows if s["id"] in chosen]


class Encoded(torch.utils.data.Dataset):
    """Catalog audio + template encoding, done in DataLoader workers."""

    def __init__(self, samples, predictions, template):
        self.samples, self.predictions, self.template = samples, predictions, template
        self.resolver = None

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        from open_audio_llm.data.catalog_resolver import LhotseCatalogAudioResolver
        from open_audio_llm.data.qwen3_asr import prepare_ts_clips

        if self.resolver is None:
            self.resolver = LhotseCatalogAudioResolver(CATALOG, ROOTS)
        sample = self.samples[index]
        slots = {s["name"]: s["ref"] for s in sample["record"]["audio_slots"]}
        enroll = load_ref(self.resolver, slots["enrollment"], ENROLL_SECONDS).astype(np.float32) / 32768
        mix = load_ref(self.resolver, slots["mixture"]).astype(np.float32) / 32768
        enroll, mix = prepare_ts_clips([enroll, mix], SR)
        raw_text = self.predictions[sample["id"]]["raw_text"]
        # Same row shape as training (online_dataset: native_messages + prepare_ts_clips), with the
        # model's own output as the assistant turn.
        row = {"messages": [{"role": "system", "content": sample["context"]},
                            {"role": "user", "content": "<audio>"},
                            {"role": "assistant", "content": raw_text}],
               "audios": [enroll], "mix_wav": mix}
        return index, self.template.encode(row)


def segments(input_ids, enroll_n_frames, tokenizer):
    """Token spans of the prompt. Audio pads are ``[enroll conv tokens, SEP, mixture conv tokens]``."""
    from open_audio_llm.tsasr.sep_token import conv_token_count

    pad_id = tokenizer.convert_tokens_to_ids("<|audio_pad|>")
    im_end = tokenizer.convert_tokens_to_ids("<|im_end|>")
    pads = (input_ids == pad_id).nonzero().flatten()
    first, n_pad = int(pads[0]), len(pads)
    assert int(pads[-1]) - first + 1 == n_pad, "audio pads are not contiguous"
    n_enroll = conv_token_count(enroll_n_frames)
    system_end = int((input_ids == im_end).nonzero()[0]) + 1  # system turn incl. <|im_start|>system and <|im_end|>
    return {"pad_first": first, "n_pad": n_pad, "n_enroll": n_enroll, "system_end": system_end,
            "enroll": slice(first, first + n_enroll), "sep": first + n_enroll,
            "mix": slice(first + n_enroll + 1, first + n_pad)}


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--run-dir", required=True, help="AmphionEval run dir (samples.jsonl, predictions.jsonl)")
    parser.add_argument("--out", required=True, help=".npz output; a .summary.json is written next to it")
    parser.add_argument("--limit", type=int, default=0, help="balanced draw of N samples; 0 = all")
    parser.add_argument("--ids-file", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    from swift.model import get_model_processor
    from swift.template import get_template

    import open_audio_llm.integrations.ms_swift.register_qwen3_asr  # noqa: F401  registers the model/template
    from open_audio_llm.tsasr.sep_token import audio_token_count
    from open_audio_llm.tsasr.ts_prompt import ts_system

    run = Path(args.run_dir)
    predictions = {p["id"]: p for p in map(json.loads, open(run / "predictions.jsonl")) if p["status"] == "success"}
    samples = select([json.loads(line) for line in open(run / "samples.jsonl")], predictions, args)
    print(f"{len(samples)} samples ({sum(not s['reference'].strip() for s in samples)} target-absent)", flush=True)

    model, processor = get_model_processor(args.checkpoint, model_type="amphion_asr_1.7b", torch_dtype=torch.bfloat16,
                                           attn_impl="eager", device_map="cuda:0")
    model.eval()
    thinker = model.thinker
    tower, text = thinker.audio_tower, thinker.model
    tower.config._attn_implementation = "sdpa"  # global audio attention, see module docstring
    assert all(layer.self_attn.config is tower.config for layer in tower.layers)
    assert text.config._attn_implementation == "eager"
    with safe_open(str(Path(args.checkpoint) / "model.safetensors"), framework="pt") as f:
        sep_ref = f.get_tensor("thinker.audio_tower.sep_token.weight")
    assert torch.equal(tower.sep_token.weight.detach().cpu(), sep_ref.to(tower.sep_token.weight.dtype)), "SEP not loaded"
    tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor
    template = get_template(processor, max_length=None, template_type="amphion_asr_1.7b")
    template.set_mode("train")  # labels mark the assistant turn

    language_id = tokenizer.convert_tokens_to_ids("language")
    candidates = []
    for word in LANGUAGE_CANDIDATES:
        ids = tokenizer.encode(word, add_special_tokens=False)
        assert len(ids) == 1, f"{word!r} is not one token: {ids}"
        candidates.append(ids[0])
    none_id = candidates[0]

    # Hooks: LLM residual (embedding = input of layer 0, then each layer's output, before final norm) and
    # audio-tower packed states (conv+PE = input of encoder layer 0, then each encoder layer's output).
    captured = {"llm": [None] * (len(text.layers) + 1), "audio": [None] * (len(tower.layers) + 1),
                "attn": [None] * len(text.layers), "decision": None}

    def save(store, index, pre):
        def hook(module, inputs, output=None):
            value = inputs[0] if pre else (output[0] if isinstance(output, tuple) else output)
            captured[store][index] = value
        return hook

    text.layers[0].register_forward_pre_hook(save("llm", 0, True))
    for i, layer in enumerate(text.layers):
        layer.register_forward_hook(save("llm", i + 1, False))
    def save_attention(index):
        def hook(module, inputs, output):
            captured["attn"][index] = output[1][0, :, captured["decision"]].float()  # (heads, T) decision row
        return hook

    for i, layer in enumerate(text.layers):
        layer.self_attn.register_forward_hook(save_attention(i))
    tower.layers[0].register_forward_pre_hook(save("audio", 0, True))
    for i, layer in enumerate(tower.layers):
        layer.register_forward_hook(save("audio", i + 1, False))

    n_llm, n_audio = len(captured["llm"]), len(captured["audio"])
    n_heads = text.config.num_attention_heads
    S = len(samples)
    out = {
        "residual": np.zeros((S, n_llm, text.config.hidden_size), np.float16),
        "audio_enroll_mean": np.zeros((S, n_audio, tower.config.d_model), np.float16),
        "audio_mix_mean": np.zeros((S, n_audio, tower.config.d_model), np.float16),
        "audio_sep": np.zeros((S, n_audio, tower.config.d_model), np.float16),
        "attn_mass_heads": np.zeros((S, n_llm - 1, n_heads, len(ATTN_BUCKETS)), np.float16),
        "attn_mass": np.zeros((S, n_llm - 1, len(ATTN_BUCKETS)), np.float32),
        "lens_logprob": np.zeros((S, n_llm, len(candidates)), np.float32),
        "lens_emitted_logprob": np.zeros((S, n_llm), np.float32),
        "lens_top1": np.zeros((S, n_llm), np.int64),
        "final_argmax": np.zeros(S, np.int64), "emitted_token": np.zeros(S, np.int64),
        "decision_pos": np.zeros(S, np.int64), "seq_len": np.zeros(S, np.int64),
        "n_audio_pad": np.zeros(S, np.int64), "n_enroll_tokens": np.zeros(S, np.int64),
        "n_mix_tokens": np.zeros(S, np.int64), "enroll_n_frames": np.zeros(S, np.int64),
        "n_mel_frames": np.zeros(S, np.int64), "retokenized_matches_vllm": np.zeros(S, bool),
    }
    count_errors, hs_check = [], None
    loader = torch.utils.data.DataLoader(Encoded(samples, predictions, template), batch_size=None,
                                         num_workers=args.workers, prefetch_factor=4 if args.workers else None)
    start = time.time()
    for done, (index, encoded) in enumerate(loader):
        sample, pred = samples[index], predictions[samples[index]["id"]]
        batch = template.data_collator([encoded])
        labels = batch.pop("labels")[0]
        batch = {k: v.to("cuda:0") if torch.is_tensor(v) else v for k, v in batch.items()}
        batch["input_features"] = batch["input_features"].to(torch.bfloat16)
        input_ids = batch["input_ids"][0].cpu()
        enroll_n = int(encoded["enroll_n_frames"])
        n_mel = int(encoded["input_features"].shape[-1])
        seg = segments(input_ids, enroll_n, tokenizer)
        decision = int((labels != -100).nonzero()[0])  # first assistant token = "language"
        assert int(input_ids[decision]) == language_id, tokenizer.decode(input_ids[decision])
        captured["decision"] = decision
        with torch.inference_mode():
            outputs = model(**batch, output_hidden_states=hs_check is None, use_cache=False)

        # Placeholder bookkeeping must agree with the template's enroll_n_frames and the tower output.
        expected = audio_token_count(n_mel, True, enroll_n)
        n_tower = int(captured["audio"][-1].shape[0])
        if not (seg["n_pad"] == expected == n_tower and enroll_n == int(ENROLL_SECONDS * SR) // 160):
            count_errors.append({"id": sample["id"], "n_pad": seg["n_pad"], "expected": expected,
                                 "tower": n_tower, "enroll_n_frames": enroll_n})
        if hs_check is None:  # document the HF hidden_states convention once against the hooks
            hs = outputs.hidden_states
            hs_check = {"n_hidden_states": len(hs),
                        "layers_0_to_27_equal_hooks": all(torch.equal(hs[i], captured["llm"][i]) for i in range(n_llm - 1)),
                        "last_equals_hook_pre_norm": bool(torch.equal(hs[-1], captured["llm"][-1])),
                        "last_equals_norm_of_hook": bool(torch.equal(hs[-1], text.norm(captured["llm"][-1])))}

        residual = torch.stack([h[0, decision] for h in captured["llm"]])  # (29, d)
        with torch.inference_mode():
            lens = torch.log_softmax(thinker.lm_head(text.norm(residual)).float(), -1)
        final_logits = outputs.logits[0, decision].float()
        emitted = int(input_ids[decision + 1])
        out["residual"][index] = residual.float().cpu().numpy()
        out["lens_logprob"][index] = lens[:, candidates].cpu().numpy()
        out["lens_emitted_logprob"][index] = lens[:, emitted].cpu().numpy()
        out["lens_top1"][index] = lens.argmax(-1).cpu().numpy()
        out["final_argmax"][index] = int(final_logits.argmax())
        if out["lens_top1"][index][-1] != out["final_argmax"][index]:
            print(f"warning: logit lens at the last layer disagrees with the model logits on {sample['id']}")

        audio_states = torch.stack(captured["audio"]).float()  # (25, N_tower, 1024)
        out["audio_enroll_mean"][index] = audio_states[:, :seg["n_enroll"]].mean(1).cpu().numpy()
        out["audio_sep"][index] = audio_states[:, seg["n_enroll"]].cpu().numpy()
        out["audio_mix_mean"][index] = audio_states[:, seg["n_enroll"] + 1:].mean(1).cpu().numpy()

        rows = torch.stack(captured["attn"])  # (28, heads, T)
        masses = torch.stack([rows[..., seg["enroll"]].sum(-1), rows[..., seg["sep"]],
                              rows[..., seg["mix"]].sum(-1), rows[..., :seg["system_end"]].sum(-1)], -1)
        masses = torch.cat([masses, (rows.sum(-1) - masses.sum(-1)).unsqueeze(-1)], -1)
        out["attn_mass_heads"][index] = masses.cpu().numpy()
        out["attn_mass"][index] = masses.mean(1).cpu().numpy()

        out["emitted_token"][index] = emitted
        out["decision_pos"][index] = decision
        out["seq_len"][index] = len(input_ids)
        out["n_audio_pad"][index] = seg["n_pad"]
        out["n_enroll_tokens"][index] = seg["n_enroll"]
        out["n_mix_tokens"][index] = seg["n_pad"] - seg["n_enroll"] - 1
        out["enroll_n_frames"][index] = enroll_n
        out["n_mel_frames"][index] = n_mel
        # vLLM's completion counts the generated tokens plus EOS; re-tokenizing raw_text should give the same ids.
        assistant = int((labels != -100).sum()) - 1  # minus the template suffix <|im_end|>
        out["retokenized_matches_vllm"][index] = assistant + 1 == pred.get("completion_tokens", -1)
        captured["llm"] = [None] * n_llm
        captured["audio"] = [None] * n_audio
        captured["attn"] = [None] * (n_llm - 1)
        if (done + 1) % 100 == 0:
            print(f"{done + 1}/{S} {(done + 1) / (time.time() - start):.2f} samples/s", flush=True)
    elapsed = time.time() - start

    meta = {
        "id": [s["id"] for s in samples],
        "source": [s["source"] for s in samples],
        "split": [s["source"].split(":", 1)[1].split("/", 1)[0] for s in samples],
        "reference": [s["reference"] for s in samples],
        "raw_text": [predictions[s["id"]]["raw_text"] for s in samples],
    }
    ref_empty = np.array([not s["reference"].strip() for s in samples])
    pred_empty = np.array([not predictions[s["id"]]["prediction"].strip() for s in samples])
    agree = out["final_argmax"] == out["emitted_token"]
    neg = ref_empty
    summary = {
        "samples": S, "target_absent": int(neg.sum()), "seconds": elapsed, "samples_per_s": S / elapsed,
        "first_token_agreement": float(agree.mean()),
        "first_token_agreement_target_present": float(agree[~neg].mean()) if (~neg).any() else None,
        "first_token_agreement_target_absent": float(agree[neg].mean()) if neg.any() else None,
        "absent_none_vs_not": {  # among target-absent samples: emitted (vLLM) vs final argmax, None or not
            f"vllm_{'none' if a else 'text'}__hf_{'none' if b else 'text'}":
                int(((out["emitted_token"][neg] == none_id) == a).__and__((out["final_argmax"][neg] == none_id) == b).sum())
            for a in (True, False) for b in (True, False)},
        "none_vs_not_agreement_all": float(((out["emitted_token"] == none_id) == (out["final_argmax"] == none_id)).mean()),
        "prediction_empty_iff_emitted_none": bool(((out["emitted_token"] == none_id) == pred_empty).all()),
        "lens_last_layer_equals_logits": float((out["lens_top1"][:, -1] == out["final_argmax"]).mean()),
        "placeholder_count_errors": count_errors,
        "retokenized_matches_vllm": float(out["retokenized_matches_vllm"].mean()),
        "system_prompt_is_ts_system": all(s["context"] == ts_system() for s in samples),
        "hidden_states_convention": hs_check,
        "candidate_tokens": dict(zip(LANGUAGE_CANDIDATES, candidates)), "language_token": language_id,
        "attn_buckets": ATTN_BUCKETS, "attention": {"llm": "eager", "audio_tower": "sdpa (global window)"},
        "checkpoint": args.checkpoint, "run_dir": args.run_dir, "limit": args.limit, "seed": args.seed,
        "ids_file": args.ids_file, "enroll_seconds": ENROLL_SECONDS,
    }
    summary["pass"] = summary["first_token_agreement"] >= MIN_AGREEMENT and not count_errors
    np.savez(args.out, reference_empty=ref_empty, prediction_empty=pred_empty,
             candidate_ids=np.array(candidates), attn_buckets=np.array(ATTN_BUCKETS),
             **{k: np.array(v) for k, v in meta.items()}, **out)
    Path(args.out).with_suffix(".summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "placeholder_count_errors"}, indent=2), flush=True)
    print(f"placeholder count errors: {len(count_errors)}")


if __name__ == "__main__":
    main()
