"""GRPO prompts drawn from a Catalog replay recipe in its SFT source proportions, plus mined prompts."""

from __future__ import annotations

import base64
import json
import random
from bisect import bisect_right
from itertools import islice
from pathlib import Path

from torch.utils.data import Dataset

from open_audio_llm.data.catalog_sampler import CatalogBatchSampler
from open_audio_llm.data.qwen3_asr import _wav_bytes


def source_key(source) -> str:
    return f"{source['dataset_id']}@{source['version']}:{source['split']}"


def read_mined(path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


class GRPOPromptPlan(Dataset):
    """A fixed list of ``(epoch, record, occurrence)`` draws.

    ``epoch`` selects fresh replay permutations and fresh online-mixing seeds, so the prompts are new
    mixtures of the SFT sources rather than the batches the SFT run already consumed. Sources named in
    ``mined_only`` are skipped by the replay draw: they enter only through ``mined`` plans, whose
    entries (written by ``open_audio_llm.scripts.grpo_mining`` against the same data config) take
    ``share`` of the list. A draw is reproduced exactly from its tuple, so a mined prompt is the very
    audio, prompt and augmentation the miner scored. GRPO's own sampler then shuffles and repeats the list.
    """

    def __init__(self, dataset, *, prompts: int, epoch: int, mined: list[dict] | None = None,
                 mined_only: list[str] | None = None, seed: int = 0):
        if dataset.config.get("replay") is None:
            raise ValueError("grpo prompts require a replay recipe")
        if type(prompts) is not int or prompts <= 0 or type(epoch) is not int or epoch < 0:
            raise ValueError("grpo.prompts must be a positive integer and grpo.epoch a non-negative integer")
        self.dataset = dataset
        ranges = {source_key(s): r for s, r in zip(dataset.sources, dataset.source_ranges)}
        self._source_starts = [r.start for r in dataset.source_ranges]
        self._source_keys = [source_key(s) for s in dataset.sources]
        skipped = [ranges[key] for key in mined_only or []]
        plans, total_share = [], 0.0
        # Record ids are checked when read in a worker: a Lance read here would make forked workers crash or hang.
        self.mined_ids = {}
        for item in mined or []:
            entries = read_mined(item["plan"])
            share = float(item["share"])
            if not entries or not 0 < share < 1:
                raise ValueError(f"mined plan {item['plan']} needs entries and a share in (0, 1)")
            for entry in entries:
                # The miner's data config must index records the same way as this one.
                _, index, _ = entry["index"]
                if index not in ranges.get(entry["source"], ()):
                    raise ValueError(f"{item['plan']}: record {index} is not in source {entry['source']}")
                self.mined_ids[index] = entry["record_id"]
            plans.append((entries, share))
            total_share += share
        if total_share >= 1:
            raise ValueError("mined shares must sum to less than 1")
        replay = prompts - sum(round(prompts * share) for _, share in plans)
        sampler = CatalogBatchSampler(dataset, batch_size=1)
        sampler.epoch = epoch
        draws = (t for t in sampler._mux(random.Random(f"{dataset.seed}:grpo:{epoch}"))
                 if not any(t[1] in r for r in skipped))
        self.plan = list(islice(draws, replay))
        if len(self.plan) < replay:
            raise ValueError(f"replay epoch yields {len(self.plan)} prompts, fewer than {replay}")
        # Where each prompt came from, for the rollout log: "replay" or the mined plan's task/plan name.
        self.origin = ["replay"] * len(self.plan)
        rng = random.Random(f"{dataset.seed}:grpo-mined:{seed}")
        for (entries, share), item in zip(plans, mined or []):
            # Small mined sets repeat; each repetition is the same prompt.
            count = round(prompts * share)
            self.plan += [tuple(rng.choice(entries)["index"]) for _ in range(count)]
            path = Path(item["plan"])
            self.origin += [f"{path.parent.parent.name}/{path.stem}"] * count

    def __len__(self) -> int:
        return len(self.plan)

    def __getitem__(self, index: int) -> dict:
        _, record, _ = self.plan[index]
        expected = self.mined_ids.get(record)
        if expected is not None and self.dataset.records[record].record.id != expected:
            raise ValueError(f"record {record} is not the mined record {expected}")
        sample = self.dataset[tuple(self.plan[index])]
        # ms-swift forwards only chat_template_kwargs from dataset columns to rollout and loss encoding.
        mix = sample.pop("mix_wav", None)
        if mix is not None:
            sample["chat_template_kwargs"] = {**(sample.get("chat_template_kwargs") or {}), "mix_wav": mix}
        sample.pop("ts_activity", None)
        # One JSON string per prompt (every sample carries it, as ms-swift's gathered log columns need), for
        # completions.jsonl / the W&B table: where the prompt came from and how its audio was built.
        source = self._source_keys[bisect_right(self._source_starts, record) - 1]
        target = str(sample.get("solution", "")).split("<asr_text>", 1)[-1]
        sample["grpo_meta"] = json.dumps({
            "origin": self.origin[index], "source": source, "record_id": self.dataset.records[record].record.id,
            "draw": list(self.plan[index]), "task": sample.get("task"), "target_empty": not target.strip(),
            **(sample.pop("online_mix_info", None) or {}),
        }, ensure_ascii=False)
        # Rollout requests type-check audios as strings: send decoded clips as base64 float WAV.
        sample["audios"] = [
            audio if isinstance(audio, str) else base64.b64encode(
                audio if isinstance(audio, bytes) else _wav_bytes(audio, self.dataset.sampling_rate)).decode()
            for audio in sample["audios"]
        ]
        return sample
