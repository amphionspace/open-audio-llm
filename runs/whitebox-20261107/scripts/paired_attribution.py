"""Task 12 follow-up: does pairing a target-present prompt with its target-absent twin cancel the threshold shift?

Twin = the same rendered mixture (same target speech, interferers, noise) with the enrollment swapped for the
nearest same-corpus look-alike speaker's enrollment (v2 tables, ``LookalikeTable``), cropped to the original
enrollment length; its reference is empty. Records come from the online-mix sources that have a look-alike
table (aishellmix / librimix ``train_1spk``) and render target-present. Each prompt is its own GRPO group, as in
training; the gradient parts are accumulated with ``grad_attribution`` under the kinds ``pair-present`` and
``pair-absent``, so the sum of the two (``analyze_paired``) is what one step on the pairs would do.

Usage (as grad_attribution.py): paired_attribution.py --out DIR [--prompts-per-type 64] [--seed 0] [--delta ...]
"""
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import grad_attribution as ga  # noqa: E402

PAIRED_SOURCES = ("aishellmix@", "librimix@")  # sources whose enrollments the v2 look-alike tables index
RENDERED = {}  # key -> prompt dict, filled by build_pairs


def build_pairs(args):
    from open_audio_llm.data.catalog_dataset import CatalogSwiftDataset, load_mono, read_data_config

    config = read_data_config(ga.DATA_CONFIG)
    dataset = CatalogSwiftDataset(config, encode=None, grpo=True, message_format="qwen3_asr")
    rng = random.Random(f"paired:{args.seed}")
    candidates = []
    for source, span in zip(dataset.sources, dataset.source_ranges):
        key = ga_source_key(source)
        if key.startswith(PAIRED_SOURCES) and source.get("online_mix"):
            candidates += list(span)
    rng.shuffle(candidates)
    chosen = []
    for index in candidates:
        if len(chosen) >= 2 * args.prompts_per_type:
            break
        record = dataset.records[index].record
        if record.slot("enrollment") is None:
            continue
        sample = dataset[(ga_epoch(), index, 0)]
        if not sample["solution"].split("<asr_text>", 1)[-1].strip():
            continue  # rendered target-absent: no present prompt to pair
        enroll_id = record.slot("enrollment").ref.cut_id
        twin = lookalike_enrollment(dataset, record.language, enroll_id, sample, load_mono, rng)
        if twin is None:
            continue
        n = len(chosen) // 2
        base = {"messages": sample["messages"], "mix_wav": sample["mix_wav"], "record_id": record.id,
                "mix_info": sample.get("online_mix_info", {})}
        RENDERED[2 * n] = {**base, "audios": sample["audios"], "solution": sample["solution"]}
        RENDERED[2 * n + 1] = {**base, "audios": [twin[0]], "solution": "language None<asr_text>",
                               "mix_info": {**base["mix_info"], "twin_enrollment": twin[1]}}
        chosen += [("pair-present", 2 * n), ("pair-absent", 2 * n + 1)]
    if len(chosen) < 2 * args.prompts_per_type:
        raise ValueError(f"only {len(chosen) // 2} pairs found")
    return dataset, None, chosen


def ga_source_key(source):
    from open_audio_llm.integrations.ms_swift.grpo_prompts import source_key

    return source_key(source)


def ga_epoch():
    return 10  # grpo.epoch of train-data-grpo-v25.yaml: fresh online-mix seeds, as the replay draws


def lookalike_enrollment(dataset, language, enroll_id, sample, load_mono, rng):
    """Nearest look-alike enrollment (another speaker), cut to the original enrollment's length."""
    import io

    import soundfile as sf

    for table, items in dataset.lookalike.get(language, []):
        found = [items[p] for p in table.pool_ids[table.candidates(enroll_id)][:4] if p in items]
        if not found:
            continue
        original, rate = sf.read(io.BytesIO(sample["audios"][0]), dtype="float32")
        audio = load_mono(found[rng.randrange(len(found))], rate)
        audio = np.pad(audio[:len(original)], (0, max(0, len(original) - len(audio))))
        return audio.astype(np.float32), getattr(found[0], "id", None)
    return None


def raw_prompt(dataset, plan, index):
    return RENDERED[index]


def main():
    ga.PROMPT_TYPES = {"pair-present": None, "pair-absent": None}
    ga.build_prompts = build_pairs
    ga.raw_prompt = raw_prompt
    ga.main()


if __name__ == "__main__":
    main()
