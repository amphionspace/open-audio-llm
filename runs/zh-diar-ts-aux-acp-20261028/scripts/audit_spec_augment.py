"""Check that the SpecAugment policy the dataset attaches reaches the model input (tasks/06-augmentation-audit).

Encodes the same training records through the real training template twice, with spec_aug_prob forced to 1 and to 0,
and reports how many Mel bins/frames differ. Usage (snapshot code on PYTHONPATH):
    audit_spec_augment.py DATA_EFFECTIVE_YAML MODEL_DIR DATASET_ID VERSION SPLIT [n]
"""
import copy
import sys

import torch
import yaml

import open_audio_llm.integrations.ms_swift.register_qwen3_asr  # noqa: F401  registers the template
from open_audio_llm.data.catalog_dataset import CatalogSwiftDataset
from swift.model import get_processor
from swift.template import get_template


def main(config_path, model_dir, dataset_id, version, split, n=3):
    config = yaml.safe_load(open(config_path))
    config["train"] = [s for s in config["train"]
                       if (s["dataset_id"], s["version"], s["split"]) == (dataset_id, version, split)]
    assert config["train"], "source not in the config"
    template = get_template(get_processor(model_dir, model_type="amphion_asr_1.7b"),
                            template_type="amphion_asr_1.7b", max_length=16896)
    template.set_mode("train")
    features = {}
    for prob in (1.0, 0.0):
        variant = copy.deepcopy(config)
        variant["augmentation"]["spec_aug_prob"] = prob
        dataset = CatalogSwiftDataset(variant, training=True, encode=template.encode, message_format="qwen3_asr")
        features[prob] = [dataset[(0, i, None)]["input_features"] for i in range(int(n))]
    for i, (masked, plain) in enumerate(zip(features[1.0], features[0.0])):
        assert masked.shape == plain.shape
        diff = masked != plain
        print(f"record {i}: shape {tuple(plain.shape)}, differing values {int(diff.sum())}, "
              f"mel bins touched {int(diff.any(dim=-1).sum())}, frames touched {int(diff.any(dim=-2).sum())}, "
              f"masked values zero {bool((masked[diff] == 0).all()) if diff.any() else None}")


if __name__ == "__main__":
    main(*sys.argv[1:])
