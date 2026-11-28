"""Greedy 11.17 TS decoding (HF, ms-swift template, global audio attention, max 256 new tokens) of the v12 test subset
with a de-interference map inserted at one audio-tower layer, or (LAYER ft) a component fine-tune loaded, (output of layers[11] for l12, layers[23] for l24, or
the tower output for 'out'); ADAPTER 'none' decodes without one (HF reproduction check).
usage: decode_adapter.py LAYER ADAPTER_PT|none TEST_LIST_JSONL MANIFEST OUT_JSONL CONDS SHARD NUM_SHARDS"""
import json, sys
import numpy as np, soundfile as sf, torch
from swift.model import get_model_processor
from swift.template import get_template
import open_audio_llm.integrations.ms_swift.register_qwen3_asr  # noqa: F401
from open_audio_llm.data.qwen3_asr import prepare_ts_clips
sys.path.insert(0, __file__.rsplit("/", 1)[0])
from probe_map_model import Adapter
layer, ad_path, test_list, manifest, out_path, conds = sys.argv[1:7]
shard, shards = int(sys.argv[7]), int(sys.argv[8])
conds = conds.split(",")
CK = "/workspace/data/ae-state/model-cache/entries/model-afea989da22b77adc0ce9365"
model, processor = get_model_processor(CK, model_type="amphion_asr_1.7b", torch_dtype=torch.bfloat16, attn_impl="eager", device_map="cuda:0")
model.eval(); thinker = model.thinker; tower = thinker.audio_tower; tower.config._attn_implementation = "sdpa"
tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor
template = get_template(processor, max_length=None, template_type="amphion_asr_1.7b"); template.set_mode("transformers")
adapter = None
if layer == "ft":  # ADAPTER is a component fine-tune (ft_component.py): load its parameters into the model
    ck = torch.load(ad_path, map_location="cpu"); own = dict(model.named_parameters())
    for n, v in ck["state"].items(): own[n].data.copy_(v.to(own[n].dtype))
    print("loaded", len(ck["state"]), "fine-tuned tensors", flush=True)
elif ad_path != "none":
    ck = torch.load(ad_path, map_location="cpu"); adapter = Adapter(ck["dim"]); adapter.load_state_dict(ck["state"]); adapter = adapter.cuda().eval()
    def apply(h):
        return adapter(h.float()[None])[0].to(h.dtype)
    if layer == "out":
        orig = thinker.get_audio_features
        def patched(*a, **k): return apply(orig(*a, **k))
        thinker.get_audio_features = patched
    else:
        i = {"l12": 11, "l24": 23}[layer]
        def hook(module, inputs, output):
            if isinstance(output, tuple): return (apply(output[0]),) + tuple(output[1:])
            return apply(output)
        tower.layers[i].register_forward_hook(hook)
ctx = {json.loads(l)["key"]: json.loads(l)["context"] for l in open(manifest, encoding="utf-8")}
items = [json.loads(l) for l in open(test_list, encoding="utf-8")]
items = [it for it in items if it["cond"] in conds][shard::shards]
with open(out_path, "w", encoding="utf-8") as f:
    for it in items:
        e = sf.read(it["enroll"], dtype="float32")[0]; a = sf.read(it["audio"], dtype="float32")[0]
        e1, a1 = prepare_ts_clips([e, a], 16000)
        row = {"messages": [{"role": "system", "content": ctx[it["key"]]}, {"role": "user", "content": "<audio>"}],
               "audios": [e1], "mix_wav": a1}
        enc = template.encode(row); batch = template.data_collator([enc])
        batch = {k: (v.to("cuda:0") if torch.is_tensor(v) else v) for k, v in batch.items()}
        batch["input_features"] = batch["input_features"].to(torch.bfloat16)
        batch.pop("labels", None)
        with torch.inference_mode():
            out = model.generate(**batch, max_new_tokens=256, do_sample=False)
        seq = out if torch.is_tensor(out) else (out.sequences if hasattr(out, "sequences") else out[0])
        seq = seq if torch.is_tensor(seq) else seq.sequences
        gen = seq[0, batch["input_ids"].shape[1]:] if seq.shape[1] > batch["input_ids"].shape[1] else seq[0]
        raw = tokenizer.decode(gen, skip_special_tokens=False).split("<|im_end|>")[0]
        text = raw.split("<asr_text>", 1)[1] if "<asr_text>" in raw else ""
        f.write(json.dumps({"id": it["id"], "key": it["key"], "cond": it["cond"], "raw": raw, "text": text}, ensure_ascii=False) + "\n"); f.flush()
print("done", len(items))
