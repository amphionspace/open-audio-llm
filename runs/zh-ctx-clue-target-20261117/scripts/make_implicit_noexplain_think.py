"""The no-explanation variant of the implicit data's think file, for the clue-target-implicit-noexplain arm.

Copies context_correction_synth_zh@implicit-v1-20261117/think.jsonl.gz (clue + target lines, written by amphiondata
think.py --style clue-target) to think-no-explanation.jsonl.gz in the same version directory, with the think of
every think-on row replaced by round 2's fixed sentence for records without an explanation
(``open_audio_llm.data.think.NO_EXPLANATION``). Same records, same think-on flags: the arm sees the implicit items
with their corrected transcripts, only without the reasoning.

usage: make_implicit_noexplain_think.py
"""
import gzip
import json
from pathlib import Path

VERSION = Path("/workspace/data/datasets/context_correction_synth_zh/versions/implicit-v1-20261117")
NO_EXPLANATION = "这段没有说明字怎么写。"  # open_audio_llm.data.think.NO_EXPLANATION


def main():
    with gzip.open(VERSION / "think.jsonl.gz", "rt", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    on = 0
    with gzip.open(VERSION / "think-no-explanation.jsonl.gz", "wt", encoding="utf-8") as f:
        for row in rows:
            if row["think_on"]:
                row["think"], on = NO_EXPLANATION, on + 1
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print({"rows": len(rows), "think_on": on})


if __name__ == "__main__":
    main()
