"""Gold words written right with no list but wrong under pool50, by hotword type.

For each lost word, the utterance's pool50 hypothesis is checked for list words
it wrote more often than the reference has them, and their distractor kind.

    python lost.py <run dir> v3-20261020
"""

import collections
import glob
import gzip
import json
import sys

from amphion_eval.integrations.open_audio_llm import hotword_bench as hb
from rapidfuzz.distance import Levenshtein

ROOT = "/workspace/data/datasets/hotword_bench/versions"


def hits_and_alarms(record, candidates, text):
    chinese = hb._chinese(record["language"])
    gold = set(record["hotwords"])
    vocab = gold | set(candidates)
    ref, hyp = hb.tokens(record["target"], chinese), hb.tokens(text, chinese)
    ref_units = hb._units(ref, hb._spans(ref, vocab, chinese))
    hyp_units = hb._units(hyp, hb._spans(hyp, vocab, chinese))
    hits = collections.Counter()
    for tag, i1, i2, _, _ in Levenshtein.opcodes(ref_units, hyp_units):
        if tag == "equal":
            hits.update(
                u[1:]
                for u in ref_units[i1:i2]
                if u.startswith("\x00") and u[1:] in gold
            )
    others = set(candidates) - gold
    written = collections.Counter(w for _, _, w in hb._spans(hyp, others, chinese))
    spoken = collections.Counter(w for _, _, w in hb._spans(ref, others, chinese))
    return hits, {w for w, n in written.items() if n > spoken[w]}


def read_jsonl(path):
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def main(run, version):
    records = {}
    for path in glob.glob(f"{ROOT}/{version}/records/*.jsonl.gz"):
        for record in read_jsonl(path):
            records[record["id"].split(":", 1)[1]] = record
    predictions = collections.defaultdict(dict)
    for row in read_jsonl(f"{run}/predictions.jsonl"):
        predictions[row["sample_id"].split(":", 1)[1]][row["condition"]] = row[
            "prediction"
        ]
    out = collections.defaultdict(collections.Counter)
    for sample, by_condition in predictions.items():
        record = records[sample]
        if record["language"] != "zh" or not {"none", "pool50"} <= set(by_condition):
            continue
        types = record["metadata"].get("hotword_types", {})
        kinds = record["metadata"].get("distractor_kinds", {})
        before, _ = hits_and_alarms(record, [], by_condition["none"])
        after, alarms = hits_and_alarms(
            record, record["metadata"]["bias_lists"]["pool50"], by_condition["pool50"]
        )
        alarm_kinds = {kinds.get(w) for w in alarms}
        for word in set(record["hotwords"]):
            kind = types.get(word)
            if not kind or not before[word]:
                continue
            out[kind]["hit_none"] += 1
            if not after[word]:
                out[kind]["lost"] += 1
                out[kind][
                    "with_confusable" if "confusable" in alarm_kinds else "other"
                ] += 1
    for kind, c in sorted(out.items(), key=lambda item: -item[1]["hit_none"]):
        print(
            f"{kind}: hit_none {c['hit_none']}, lost {c['lost']} ({c['lost'] / c['hit_none']:.1%}), "
            f"with a confusable written {c['with_confusable'] / max(c['lost'], 1):.0%}"
        )


if __name__ == "__main__":
    main(*sys.argv[1:3])
