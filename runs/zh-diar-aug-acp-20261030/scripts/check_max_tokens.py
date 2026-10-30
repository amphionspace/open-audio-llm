"""Exact token length of every long timestamped record after 0.95x speed: overhead + audio tokens + target tokens,
with the decomposition checked against the real template on a few records first."""
import glob, gzip, json, sys, yaml
import numpy as np, torch
R2 = "/workspace/workspace/mingdong/repo/oal-diar-restart/runs/zh-diar-aug-acp-20261030"
sys.path[:0] = [f"{R2}/shared/code/src", f"{R2}/shared/code/audio-data-contract-src"]
import open_audio_llm.integrations.ms_swift.register_qwen3_asr  # noqa
from swift.model import get_processor
from swift.template import get_template
from qwen_asr.core.transformers_backend.modeling_qwen3_asr import _get_feat_extract_output_lengths
from open_audio_llm.data.qwen3_asr import native_messages
from open_audio_llm.data.sot import TIMESTAMP_FORMAT, scale_timestamps
from audio_data_contract import AudioRecord
MODEL = "/workspace/data/ae-runs/zh-emilia2-dialogue-6000/checkpoint-6000"
template = get_template(get_processor(MODEL, model_type="amphion_asr_1.7b"), template_type="amphion_asr_1.7b", max_length=100000)
template.set_mode("train")
tok = template.tokenizer
RATIO = 20 / 19
def audio_tokens(seconds):
    frames = torch.tensor([int(round(seconds * 16000)) // 160])
    return int(_get_feat_extract_output_lengths(frames))
def record_of(r, target):
    rec = AudioRecord.from_dict(r)
    from dataclasses import replace
    return replace(rec, target=target)
recipe = yaml.safe_load(open(f"{R2}/configs/train-data-h.yaml"))
roots = json.load(open(recipe["roots"]))
entries = {}
for f in glob.glob(f"{recipe['catalog']}/*.yaml") + glob.glob(f"{recipe['catalog']}/*.jsonl"):
    data = [json.loads(l) for l in open(f) if l.strip()] if f.endswith(".jsonl") else yaml.safe_load(open(f))
    for e in data if isinstance(data, list) else []:
        if isinstance(e, dict) and "dataset_id" in e: entries[(e["dataset_id"], e["version"])] = e
rows, seen = [], set()
for s in recipe["train"]:
    if s.get("max_duration", 0) < 250: continue
    e = entries[(s["dataset_id"], s["version"])]
    for name in e["splits"][s["split"]]["artifacts"].get("records", []):
        a = next(x for x in e["artifacts"] if x["name"] == name)
        path = f"{roots[a['root_alias']]}/{a['relative_path']}"
        if path in seen: continue
        seen.add(path)
        with gzip.open(path, "rt") as g:
            for line in g:
                r = json.loads(line)
                if (r.get("metadata") or {}).get("sot_output_format") != TIMESTAMP_FORMAT: continue
                if r["audio_slots"][0]["ref"]["duration"] < 250: continue
                rows.append((s["dataset_id"], r))
print("long records", len(rows), flush=True)
# Calibrate: real encode = overhead + audio tokens + messages' text tokens (system prompt + target).
def text_len(r, target):
    msgs = native_messages(record_of(r, target), "N/A")
    return msgs
checks = []
for ds, r in rows[:: max(1, len(rows) // 4)][:4]:
    d = r["audio_slots"][0]["ref"]["duration"]
    target = scale_timestamps(r["target"], RATIO)
    msgs = native_messages(record_of(r, target), "N/A")
    enc = len(template.encode({"messages": msgs, "audios": [np.zeros(int(round(d * RATIO * 16000)), dtype=np.float32)]})["input_ids"])
    checks.append(enc - audio_tokens(d * RATIO) - len(tok(target, add_special_tokens=False)["input_ids"]))
print("overhead per check", checks, flush=True)
overhead = max(checks)
worst = []
for ds, r in rows:
    d = r["audio_slots"][0]["ref"]["duration"]
    target = scale_timestamps(r["target"], RATIO)
    n = overhead + audio_tokens(d * RATIO) + len(tok(target, add_special_tokens=False)["input_ids"])
    worst.append((n, round(d, 1), ds, r["id"]))
worst.sort(reverse=True)
print("over 16384:", sum(w[0] > 16384 for w in worst), "over 16896:", sum(w[0] > 16896 for w in worst))
from collections import Counter
print("over 16384 by dataset:", Counter(w[2] for w in worst if w[0] > 16384))
for w in worst[:6]: print(w)
