"""Rewrite the think lines that the round-2 recipe trains on into the clue + target form.

Round 2 (``runs/zh-spelling-think-r2-20261115``) reads 12 think files: 11 of
``context_correction_clean_zh@think-v2-20261113`` (cleaned real speech) and the one of
``context_correction_synth_zh@train-v1-20261114`` (synthetic). Each think-on line reads

    <pinyin> 这个音，说话人说是「<quote>」，写作「<chars>」。

and names one sound and the chars written for it, not the word of the transcript they go into, nor what not to
write. This script keeps every row (record ID, think on/off, reason, quote) and rewrites each line as

    线索：说话人说「<quote>」。目标：写「<target>」，不写「<alt>」。

- synthetic: ``target`` is the target word of the record (``labels.context_correction.segments``, role target:
  the whole name, e.g. 织音琴行) and ``alt`` its homophone spelling; one line per distinct target.
- real: ``target`` is the chars of the old line (the decisions do not mark the word they go into). ``alt`` is
  what the original transcript wrote, from the corrected windows' edits; without it the line ends after the
  target. The fixed sentence for records without an explanation is not in these files and stays as it is.

usage: make_think_v3.py   (writes OUT; reads the files above and the cleaning decisions)
"""
import gzip
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

DATA = Path("/workspace/data/datasets")
REAL = DATA / "context_correction_clean_zh/versions/think-v2-20261113"
CLEAN = DATA / "context_correction_clean_zh/versions/v1-20261110"  # decisions think-v2 was built from
SYNTH = DATA / "context_correction_synth_zh/versions/train-v1-20261114"
OUT = DATA / "context_correction_clean_zh/versions/think-v3-20261117"
REAL_SOURCES = ("aishell4_punc", "aishell4_tt_punc", "alimeeting_sdm_punc", "alimeeting_sdm_tt_punc", "emilia2_dialog",
                "emilia2_mix_utts", "emilia2_short", "magicdata_ramc_punc", "magicdata_ramc_tt_punc", "vitw",
                "wenetspeech_clean")
SAMPLES_PER_SOURCE = 20
OLD = re.compile(r"^(?P<pinyin>[a-zü ]+) 这个音，说话人说是「(?P<quote>.*)」，写作「(?P<chars>[^」]+)」。$")


def render(quote: str, target: str, alt: str | None) -> str:
    tail = f"，不写「{alt}」" if alt and alt != target else ""
    return f"线索：说话人说「{quote}」。目标：写「{target}」{tail}。"


def rows_of(path: Path):
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def parse(think: str) -> list[dict]:
    lines = [OLD.match(line) for line in think.split("\n")]
    assert all(lines), think
    return [m.groupdict() for m in lines]


def original_chars(source: str) -> dict[str, dict[str, set[str]]]:
    """unit_id -> {corrected char: chars the original transcript had there}, for corrected windows."""
    out = {}
    with open(CLEAN / source / "decisions.jsonl") as f:
        for line in f:
            row = json.loads(line)
            if row["decision"] != "corrected":
                continue
            back = defaultdict(set)
            for e in row["edits"]:
                back[e["to"]].add(e["from"])
            out[row["unit_id"]] = back
    return out


def real(source: str, summary: Counter, samples: list):
    back_of = original_chars(source)
    rows = rows_of(REAL / source / "think.jsonl.gz")
    for row in rows:
        if not row["think_on"]:
            continue
        back = back_of.get(row["unit_id"], {})
        lines = []
        for old in parse(row["think"]):
            chars = old["chars"]
            alts = [back.get(c, set()) for c in chars]
            alt = None
            if any(alts) and all(len(a) <= 1 for a in alts):  # one original char per corrected char, else unsure
                alt = "".join(next(iter(a)) if a else c for c, a in zip(chars, alts))
            summary[f"{source}:{'alt' if alt else 'no_alt'}"] += 1
            lines.append(render(old["quote"], chars, alt))
        new = "\n".join(lines)
        samples.append({"source": source, "record_id": row["record_id"], "old": row["think"], "new": new})
        row["think"] = new
    return rows


def synthetic(summary: Counter, samples: list):
    targets = {}
    for split in ("train", "train_ts"):
        for record in rows_of(SYNTH / "records" / f"{split}.jsonl.gz"):
            segments = record["labels"]["context_correction"]["segments"]
            pairs = []
            for s in segments:
                if s["role"] == "target" and (s["text"], s["alt"]) not in pairs:
                    pairs.append((s["text"], s["alt"]))
            targets[record["id"]] = pairs
    rows = rows_of(SYNTH / "think.jsonl.gz")
    for row in rows:
        if not row["think_on"]:
            continue
        quotes = {old["quote"] for old in parse(row["think"])}
        assert len(quotes) == 1, row  # the generator marks one evidence span per item
        quote = quotes.pop()
        pairs = targets[row["record_id"]]
        assert pairs, row
        new = "\n".join(render(quote, text, alt) for text, alt in pairs)
        summary[f"synth:{row['category']}:{len(pairs)}_targets"] += 1
        samples.append({"source": "synth", "record_id": row["record_id"], "old": row["think"], "new": new})
        row["think"] = new
    return rows


def write(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    summary, samples = Counter(), []
    files = {}
    for source in REAL_SOURCES:
        rows = real(source, summary, samples)
        files[source] = (OUT / source / "think.jsonl.gz", rows)
    files["context_correction_synth_zh"] = (OUT / "context_correction_synth_zh" / "think.jsonl.gz",
                                            synthetic(summary, samples))
    for name, (path, rows) in files.items():
        write(path, rows)
        summary[f"{name}:rows"] = len(rows)
        summary[f"{name}:think_on"] = sum(r["think_on"] for r in rows)
    (OUT / "summary.json").write_text(json.dumps(dict(sorted(summary.items())), ensure_ascii=False, indent=2))
    # For reading: every line that names what not to write, and a fixed draw of the rest of each source.
    rng = random.Random(20261117)
    by_source = defaultdict(list)
    for s in samples:
        by_source[s["source"]].append(s)
    with open(OUT / "samples.jsonl", "w") as f:
        for source, rows in by_source.items():
            real_alt = [s for s in rows if source != "synth" and "不写" in s["new"]]
            for s in real_alt + rng.sample(rows, min(SAMPLES_PER_SOURCE, len(rows))):
                f.write(json.dumps(s, ensure_ascii=False) + "\n")
    print(json.dumps(dict(sorted(summary.items())), ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
