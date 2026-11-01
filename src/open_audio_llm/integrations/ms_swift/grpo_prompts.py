"""GRPO prompts drawn from a Catalog replay recipe in its SFT source proportions."""

from __future__ import annotations

import base64
import random
from itertools import islice

from torch.utils.data import Dataset

from open_audio_llm.data.catalog_sampler import CatalogBatchSampler
from open_audio_llm.data.qwen3_asr import _wav_bytes


class GRPOPromptPlan(Dataset):
    """A fixed list of ``(epoch, record, occurrence)`` draws from the replay mux.

    ``epoch`` selects fresh replay permutations and fresh online-mixing seeds, so the prompts are new
    mixtures of the SFT sources rather than the batches the SFT run already consumed. GRPO's own
    sampler then shuffles and repeats this list.
    """

    def __init__(self, dataset, *, prompts: int, epoch: int):
        if dataset.config.get("replay") is None:
            raise ValueError("grpo prompts require a replay recipe")
        if type(prompts) is not int or prompts <= 0 or type(epoch) is not int or epoch < 0:
            raise ValueError("grpo.prompts must be a positive integer and grpo.epoch a non-negative integer")
        sampler = CatalogBatchSampler(dataset, batch_size=1)
        sampler.epoch = epoch
        self.plan = list(islice(sampler._mux(random.Random(f"{dataset.seed}:grpo:{epoch}")), prompts))
        if len(self.plan) < prompts:
            raise ValueError(f"replay epoch yields {len(self.plan)} prompts, fewer than grpo.prompts={prompts}")
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.plan)

    def __getitem__(self, index: int) -> dict:
        sample = self.dataset[self.plan[index]]
        # ms-swift forwards only chat_template_kwargs from dataset columns to rollout and loss encoding.
        mix = sample.pop("mix_wav", None)
        if mix is not None:
            sample["chat_template_kwargs"] = {**(sample.get("chat_template_kwargs") or {}), "mix_wav": mix}
        sample.pop("ts_activity", None)
        # Rollout requests type-check audios as strings: send decoded clips as base64 float WAV.
        sample["audios"] = [
            audio if isinstance(audio, str) else base64.b64encode(
                audio if isinstance(audio, bytes) else _wav_bytes(audio, self.dataset.sampling_rate)).decode()
            for audio in sample["audios"]
        ]
        return sample
