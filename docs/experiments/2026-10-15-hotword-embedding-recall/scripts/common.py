"""Shared data loading for the hotword-embedding recall analysis."""
import glob
import gzip
import json
import random

DATASETS = "/workspace/data/datasets"
ZH = ["aishell", "aishell2", "commonvoice_zh", "kespeech", "magicdata", "thchs30", "aidatatang", "aishell3", "talcs"]
EN = ["commonvoice_en", "librispeech_test_clean", "librispeech_test_other"]
POOL_SEED = 0


def load_supervisions(ds, split):
    files = sorted(glob.glob(f"{DATASETS}/{ds}_hotwords/versions/*/{split}_supervisions/*.jsonl.gz"))
    return [json.loads(line) for f in files for line in gzip.open(f, "rt")]


def hotwords_of(rows):
    return {w for r in rows for w in r["custom"]["hotwords"]}


def language_pool(datasets):
    pool = set()
    for ds in datasets:
        for split in ("train", "test"):
            pool |= hotwords_of(load_supervisions(ds, split))
    return pool


def build_pool(own, language_all, extra):
    """Own hotwords first, then `extra` random distractors from the language pool."""
    rest = sorted(language_all - set(own))
    random.Random(POOL_SEED).shuffle(rest)
    return sorted(own) + rest[:extra]
