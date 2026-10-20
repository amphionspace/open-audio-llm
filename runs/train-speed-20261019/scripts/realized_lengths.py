"""Encode the first samples of the real training plan and record their token lengths.

The keys come from the sampler's own mux stream (epoch 0), so online mixing, enrollment
choice and augmentation draw the same random numbers as in training. Each row holds the
sampler's sort key next to the realized total, audio-placeholder and supervised tokens.
"""
import argparse
import json
import random
from itertools import islice

import torch
from swift.model import get_model_processor
from swift.template import get_template

import open_audio_llm.integrations.ms_swift.register_qwen3_asr  # noqa: F401  registers amphion_asr_1.7b
from open_audio_llm.data.catalog_dataset import CatalogSwiftDataset, read_data_config
from open_audio_llm.data.catalog_sampler import CatalogBatchSampler

AUDIO_PAD = 151676


class Keys(torch.utils.data.Dataset):
    def __init__(self, dataset, keys):
        self.dataset, self.keys = dataset, keys

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, number):
        key = self.keys[number]
        row = {"number": number, "idx": key[1]}
        try:
            result = self.dataset[key]
        except Exception as error:  # Training drops nothing silently; record and report.
            row["error"] = f"{type(error).__name__}: {error}"[:300]
            return row
        ids = torch.as_tensor(result["input_ids"])
        labels = torch.as_tensor(result["labels"])
        row.update(tokens=int(ids.numel()), audio_tokens=int((ids == AUDIO_PAD).sum()),
                   supervised=int((labels != -100).sum()))
        return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_config", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--samples", type=int, required=True)
    parser.add_argument("--workers", type=int, default=48)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    _, processor = get_model_processor(args.model, model_type="amphion_asr_1.7b", load_model=False)
    template = get_template(processor, max_length=2048)
    template.set_mode("train")
    config = read_data_config(args.data_config)
    dataset = CatalogSwiftDataset(config, encode=template.encode, message_format="qwen3_asr")
    sampler = CatalogBatchSampler(dataset, batch_size=12, world_size=8)
    sampler.set_epoch(0)
    rng = random.Random(f"{dataset.seed}:{sampler.epoch}:mux")
    keys = list(islice(sampler._mux(rng), args.samples))
    owner = {}
    for number, indices in enumerate(dataset.source_ranges):
        for idx in {key[1] for key in keys} & set(range(indices.start, indices.stop)):
            owner[idx] = number
    loader = torch.utils.data.DataLoader(Keys(dataset, keys), batch_size=None,
                                         num_workers=args.workers, collate_fn=lambda row: row)
    with open(args.output, "w") as stream:
        for done, row in enumerate(loader, 1):
            source = dataset.sources[owner[row["idx"]]]
            row.update(source=owner[row["idx"]], dataset_id=source["dataset_id"], split=source["split"],
                       online_mix="online_mix" in source, cost=float(sampler.durations[row["idx"]]),
                       slots=int(sampler.slots[row["idx"]]))
            stream.write(json.dumps(row) + "\n")
            if done % 1000 == 0:
                print("encoded", done, flush=True)


if __name__ == "__main__":
    main()
