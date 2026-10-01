from collections import Counter
from settings import artifact
import json
from pathlib import Path
import sys
from open_audio_llm.data.catalog_dataset import CatalogSwiftDataset, read_data_config
from open_audio_llm.data.catalog_sampler import CatalogBatchSampler

root = artifact(sys.argv[1])
plan = json.loads((root / 'plan.json').read_text())
dataset = CatalogSwiftDataset(read_data_config(root / 'train-data.yaml'), message_format='qwen3_asr')
sampler = CatalogBatchSampler(dataset, batch_size=1, rank=0, world_size=2, shuffle=True)
counts = Counter(); quotas = Counter(); chosen = {}
for source, indexes, quota in zip(dataset.sources, dataset.source_ranges, sampler.quotas):
    counts[source['dataset_id']] += len(indexes); quotas[source['dataset_id']] += quota
    if source['dataset_id'] == plan['event_dataset']:
        assert len(indexes) == plan['event_records'] and quota == 3500
        for index in indexes:
            row = dataset.records[index]
            if row.duration not in chosen:
                sample = dataset[index]
                assert sample['task'] == 'speaker_attributed_asr'
                assert row.record.target in sample['solution'] and abs(sample['duration'] - row.duration) < 1 / 16000
                if plan['arm'] == 'treatment': assert row.record.metadata['clean']['pass'] is True
                chosen[row.duration] = {'id': row.record.id, 'audio_bytes': len(sample['audios'][0])}
assert sum(quotas.values()) == 10000 and set(chosen) == {30., 120., 300.}
result = {'passed': True, 'source_records': dict(counts), 'quotas_per_10000': dict(quotas), 'decoded': chosen}
(root / 'data-verification.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result), flush=True)
