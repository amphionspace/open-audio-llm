"""Mel-domain oracle masks on 11.17's own input: Whisper log-mel (n_fft 400, hop 160, 128 bands, per-clip max - 8 floor,
(x + 4) / 4) of the mixture part, recomputed from the waveform; masks from the target / rest stems' mel power:
  melcheck: no mask (must equal the template's features and reproduce the plain decode);
  melibm:   keep bands where target power > rest power, others floored;
  melirm:   mixture power x target / (target + rest);
  melirm_tok: the same mask area-pooled to the audio-token rate (80 ms) and linearly upsampled (the resolution of the
              layer-12 mask heads of tasks 21-22).
The enrollment part is untouched. Greedy HF decoding as decode_adapter.py; also per-sample target-energy shares.
usage: mel_oracle_decode.py SUBSET_DIR OUT_JSONL CONDS SHARD NUM_SHARDS"""
import json, sys
from pathlib import Path
import numpy as np, soundfile as sf, torch
from swift.model import get_model_processor
from swift.template import get_template
import open_audio_llm.integrations.ms_swift.register_qwen3_asr  # noqa: F401
from open_audio_llm.data.qwen3_asr import prepare_ts_clips
sub, out_path, conds = Path(sys.argv[1]), sys.argv[2], sys.argv[3].split(","); shard, shards = int(sys.argv[4]), int(sys.argv[5])
CK = "/workspace/data/ae-state/model-cache/entries/model-afea989da22b77adc0ce9365"
model, processor = get_model_processor(CK, model_type="amphion_asr_1.7b", torch_dtype=torch.bfloat16, attn_impl="eager", device_map="cuda:0")
model.eval(); model.thinker.audio_tower.config._attn_implementation = "sdpa"
tok = processor.tokenizer if hasattr(processor, "tokenizer") else processor
fe = processor.feature_extractor
template = get_template(processor, max_length=None, template_type="amphion_asr_1.7b"); template.set_mode("transformers")
mel_filters = torch.from_numpy(fe.mel_filters).float()
win = torch.hann_window(fe.n_fft)
def power(w):
    s = torch.stft(torch.from_numpy(w).float(), fe.n_fft, fe.hop_length, window=win, return_complex=True)
    return mel_filters.T @ (s[..., :-1].abs() ** 2)  # (128, frames)
def logmel(p):
    l = torch.clamp(p, min=1e-10).log10(); l = torch.maximum(l, l.max() - 8.0); return (l + 4.0) / 4.0
rows = [json.loads(l) for l in open(sub / "manifest.jsonl", encoding="utf-8")]
stats_ok = json.load(open(sub / "oracle-stats.json"))
rows = [r for r in rows if r["id"] in stats_ok][shard::shards]
with open(out_path, "w", encoding="utf-8") as f:
    for r in rows:
        k = r["key"]
        e = sf.read(sub / "wav" / f"{k}.enroll.wav", dtype="float32")[0]; m = sf.read(sub / "wav" / f"{k}.mix.wav", dtype="float32")[0]
        T, O, N = (sf.read(sub / "stems" / f"{k}.{n}.wav", dtype="float32")[0] for n in ("target", "other", "noise"))
        e1, m1 = prepare_ts_clips([e, m], 16000)
        n = len(m1); fit = lambda x: np.pad(x[:n], (0, max(0, n - len(x))))
        T, R = fit(T), fit(O + N)
        row = {"messages": [{"role": "system", "content": r["context"]}, {"role": "user", "content": "<audio>"}], "audios": [e1], "mix_wav": m1}
        batch = template.data_collator([template.encode(row)])
        ef = int(batch["enroll_n_frames"][0]); total = int(batch["feature_attention_mask"][0].sum()); nmix = total - ef
        Pm, Pt, Pr = power(m1), power(T), power(R)
        Pm, Pt, Pr = Pm[:, :nmix], Pt[:, :nmix], Pr[:, :nmix]
        tmpl = batch["input_features"][0, :, ef:total].float()
        check = (logmel(Pm) - tmpl).abs().max().item()
        dom = (Pt > Pr).float(); share = float((Pt * (1 - dom)).sum() / Pt.sum().clamp_min(1e-10))
        irm = Pt / (Pt + Pr).clamp_min(1e-20)
        from open_audio_llm.tsasr.sep_token import conv_token_count
        ntok = conv_token_count(nmix)  # the ideal ratio mask at the audio-token rate (80 ms), upsampled back to mel frames
        irm_tok = torch.nn.functional.interpolate(torch.nn.functional.interpolate(irm[None], size=ntok, mode="area"), size=nmix, mode="linear", align_corners=False)[0]
        feats = {"melcheck": logmel(Pm), "melibm": logmel(Pm * dom), "melirm": logmel(Pm * irm), "melirm_tok": logmel(Pm * irm_tok)}
        for c in conds:
            b = {kk: (v.clone() if torch.is_tensor(v) else v) for kk, v in batch.items()}
            b["input_features"][0, :, ef:total] = feats[c].to(b["input_features"].dtype)
            b = {kk: (v.to("cuda:0") if torch.is_tensor(v) else v) for kk, v in b.items()}
            b["input_features"] = b["input_features"].to(torch.bfloat16); b.pop("labels", None)
            with torch.inference_mode():
                o = model.generate(**b, max_new_tokens=256, do_sample=False)
            seq = o if torch.is_tensor(o) else (o.sequences if hasattr(o, "sequences") else o[0])
            seq = seq if torch.is_tensor(seq) else seq.sequences
            gen = seq[0, b["input_ids"].shape[1]:] if seq.shape[1] > b["input_ids"].shape[1] else seq[0]
            raw = tok.decode(gen, skip_special_tokens=False).split("<|im_end|>")[0]
            text = raw.split("<asr_text>", 1)[1] if "<asr_text>" in raw else ""
            f.write(json.dumps({"id": r["id"], "cand": c, "text": text, "feat_check_max_abs": round(check, 5),
                                "mel_dominated_energy_share": round(share, 4)}, ensure_ascii=False) + "\n"); f.flush()
print("done", len(rows))
