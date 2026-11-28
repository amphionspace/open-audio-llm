"""AmphionEval zh scoring tokens for the probe: train references (by key) and v12 test references (by key).
usage: tokens.py TRAIN_MANIFEST TEST_MANIFEST OUT_JSON   (amphion-eval python)"""
import json, sys
from amphion_eval.integrations.open_audio_llm import scoring
train = {json.loads(l)["key"]: scoring.tokens(json.loads(l)["reference"], "zh") for l in open(sys.argv[1], encoding="utf-8")}
test = {}
for l in open(sys.argv[2], encoding="utf-8"):
    r = json.loads(l); test[r["key"]] = {"tokens": scoring.tokens(r["reference"], "zh"), "n_interferers": r["n_interferers"], "id": r["id"]}
vocab = sorted({t for v in train.values() for t in v} | {t for v in test.values() for t in v["tokens"]})
json.dump({"vocab": vocab, "train": train, "test": test}, open(sys.argv[3], "w"), ensure_ascii=False)
print(len(vocab), "units;", len(train), "train;", len(test), "test")
