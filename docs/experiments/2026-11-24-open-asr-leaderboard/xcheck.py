"""Score one native run twice: with the leaderboard repo's own eval_utils and with amphion-eval's copy.

Usage: xcheck.py <board-repo> <amphion-eval-src> <run-dir>
data_utils imports datasets / huggingface_hub at module level for loading only; they are stubbed.
"""
import json, sys, types
board, ae_src, run = sys.argv[1:4]
for name in ("datasets", "huggingface_hub"):
    stub = types.ModuleType(name)
    for attr in ("Audio", "IterableDataset", "load_dataset", "snapshot_download"):
        setattr(stub, attr, object)
    sys.modules[name] = stub
sys.path[:0] = [board, ae_src]
from normalizer import eval_utils  # the leaderboard's code
from amphion_eval.integrations.open_asr_leaderboard import scoring

rows = lambda p: [json.loads(l) for l in open(p) if l.strip()]
samples, predictions = rows(f"{run}/samples.jsonl"), rows(f"{run}/predictions.jsonl")
ours = scoring.score_run(samples, predictions)
datasets = scoring.dataset_rows(samples, predictions)
mismatch = 0
for name, entry in sorted(datasets.items()):
    lang = entry["language"]
    manifest = [{**r, "duration": None, "time": None} for r in entry["rows"]]
    from normalizer import data_utils
    if lang == "en":  # prepare_data filters on the normalized text
        manifest = [m for m in manifest if data_utils.is_target_text_in_range(data_utils.normalizer(m["text"]))]
    else:  # run_eval_ml filters on the raw reference
        manifest = [m for m in manifest if data_utils.is_target_text_in_range(m["text"])]
    theirs = eval_utils.score_manifest(manifest, language=lang, multilingual=lang != "en", num_workers=8)
    mine = ours["datasets"][name]
    same = (theirs["wer"], theirs["ins"], theirs["del"], theirs["sub"]) == (mine["error_rate"], mine["ins"], mine["del"], mine["sub"])
    mismatch += not same
    print(f"{name:32s} board {theirs['wer']:6.2f} ({theirs['ins']}/{theirs['del']}/{theirs['sub']})  ours {mine['error_rate']:6.2f} ({mine['ins']}/{mine['del']}/{mine['sub']})  {'OK' if same else 'MISMATCH'}")
print("mismatches:", mismatch)
