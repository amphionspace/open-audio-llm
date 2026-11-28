"""Frozen 11.17 audio-tower features for [5 s enrollment | SEP | audio] (ms-swift template, global attention, as the
white-box ref_score.py): input of layer 0 (after the conv stack), outputs of layers 12 and 24, and the tower output
the LLM receives (after ln_post + proj, 2048-d). One fp16 file per layer and shard plus an index (key, cond, offset,
length, enroll token count). usage: extract_feats.py LIST_JSONL OUT_DIR SHARD NUM_SHARDS
LIST rows: {"key", "cond", "enroll": wav, "audio": wav}"""
import json, sys
from pathlib import Path
import numpy as np, soundfile as sf, torch
from swift.model import get_model_processor
from swift.template import get_template
import open_audio_llm.integrations.ms_swift.register_qwen3_asr  # noqa: F401
from open_audio_llm.data.qwen3_asr import prepare_ts_clips
CK = "/workspace/data/ae-state/model-cache/entries/model-afea989da22b77adc0ce9365"
LAYERS = {"l0": ("pre", 0), "l12": ("post", 11), "l24": ("post", 23)}
items = [json.loads(l) for l in open(sys.argv[1], encoding="utf-8")]
out, shard, shards = Path(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
items = items[shard::shards]
out.mkdir(parents=True, exist_ok=True)
model, processor = get_model_processor(CK, model_type="amphion_asr_1.7b", torch_dtype=torch.bfloat16, attn_impl="eager", device_map="cuda:0")
model.eval(); thinker = model.thinker; tower = thinker.audio_tower
tower.config._attn_implementation = "sdpa"
template = get_template(processor, max_length=None, template_type="amphion_asr_1.7b"); template.set_mode("train")
cap = {}
def hook(name, pre):
    def f(module, inputs, output=None):
        cap[name] = (inputs[0] if pre else (output[0] if isinstance(output, tuple) else output)).detach()
    return f
for name, (kind, i) in LAYERS.items():
    (tower.layers[i].register_forward_pre_hook(hook(name, True)) if kind == "pre"
     else tower.layers[i].register_forward_hook(hook(name, False)))
files = {n: open(out / f"{n}-{shard}.f16", "wb") for n in (*LAYERS, "out")}
index, offset = [], 0
for it in items:
    e = sf.read(it["enroll"], dtype="float32")[0]; a = sf.read(it["audio"], dtype="float32")[0]
    e1, a1 = prepare_ts_clips([e, a], 16000)
    row = {"messages": [{"role": "system", "content": "x"}, {"role": "user", "content": "<audio>"},
                        {"role": "assistant", "content": "x"}], "audios": [e1], "mix_wav": a1}
    batch = template.data_collator([template.encode(row)])
    with torch.inference_mode():
        feats = thinker.get_audio_features(batch["input_features"].to("cuda:0", torch.bfloat16),
                                           feature_attention_mask=batch["feature_attention_mask"].to("cuda:0"),
                                           enroll_n_frames=batch["enroll_n_frames"].to("cuda:0"))
    arrs = {n: cap[n].reshape(-1, cap[n].shape[-1]) for n in LAYERS}; arrs["out"] = feats.reshape(-1, feats.shape[-1])
    T = arrs["out"].shape[0]; assert all(v.shape[0] == T for v in arrs.values()), {k: v.shape for k, v in arrs.items()}
    for n, v in arrs.items():
        files[n].write(v.to(torch.float16).cpu().numpy().tobytes())
    index.append({"key": it["key"], "cond": it["cond"], "offset": offset, "length": T,
                  "enroll_frames": int(batch["enroll_n_frames"][0])})
    offset += T
for f in files.values(): f.close()
json.dump({"dims": {n: (2048 if n == "out" else 1024) for n in files}, "items": index}, open(out / f"index-{shard}.json", "w"))
print(f"shard {shard}: {len(index)} items, {offset} frames")
