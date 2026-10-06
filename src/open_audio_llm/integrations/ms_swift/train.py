"""Train directly from Catalog manifests without an offline ShareGPT export."""

from __future__ import annotations

import json
import sys
import time
import logging
import torch
from dataclasses import dataclass

from swift.arguments import RLHFArguments, SftArguments
from swift.pipelines.train.rlhf import SwiftRLHF
from swift.pipelines.train.sft import SwiftSft
from transformers import TrainerCallback

from open_audio_llm.data.catalog_dataset import CatalogSwiftDataset, read_data_config


class CatalogArgumentsMixin:
    def __post_init__(self):
        if not self.data_config:
            raise ValueError("--data_config is required")
        if (
            self.dataset
            or self.val_dataset
            or self.cached_dataset
            or self.cached_val_dataset
        ):
            raise ValueError(
                "Use train/validation in --data_config instead of dataset/cache arguments"
            )
        self._catalog_config = read_data_config(self.data_config)
        if any(source.get("enrollment") is not None for key in ("train", "validation")
               for source in self._catalog_config.get(key, [])):
            import os

            os.environ["OPEN_AUDIO_LLM_TARGET_SOT"] = "1"
        # SftArguments requires dataset identities, but the pipeline below owns
        # loading; these config references never go through the HF JSON loader.
        self.dataset = [self.data_config]
        self.val_dataset = (
            [self.data_config] if self._catalog_config.get("validation") else []
        )
        self.split_dataset_ratio = 0
        self.lazy_tokenize = True
        super().__post_init__()
        if (
            self.packing
            or self.padding_free
            or getattr(self, "group_by_length", False)
            or self.streaming
        ):
            raise ValueError(
                "Catalog training uses on-demand map-style samples; disable packing, padding_free, group_by_length and streaming"
            )


@dataclass
class CatalogSftArguments(CatalogArgumentsMixin, SftArguments):
    data_config: str | None = None
    continuous_training: bool = False
    reset_catalog_sampler: bool = False
    performance_logging: bool = True
    audio_encoder_parallel: bool = False
    audio_encoder_batching: bool = False
    retention_teacher: str | None = None
    retention_eval_script: str | None = None
    retention_eval_interval: int = 2000
    checkpoint_handoff: str | None = None
    checkpoint_selection: str = "final"
    storage_sync: str | None = None
    amphion_eval_check: str | None = None


@dataclass
class CatalogRLHFArguments(CatalogArgumentsMixin, RLHFArguments):
    data_config: str | None = None
    storage_sync: str | None = None


class DatasetEpochCallback(TrainerCallback):
    def __init__(self, dataset):
        self.dataset = dataset

    def on_epoch_begin(self, args, state, control, **kwargs):
        self.dataset.set_epoch(int(state.epoch or 0))


class CatalogTrainingMixin:
    def _prepare_dataset(self):
        started = time.perf_counter()
        grpo = getattr(self.args, "rlhf_type", None) == "grpo"
        if hasattr(self.args, "rlhf_type") and not grpo:
            raise ValueError("Catalog RLHF entry currently supports GRPO")
        config = self.args._catalog_config
        message_format = (
            "qwen3_asr"
            if getattr(self.args, "model_type", None) == "amphion_asr_1.7b"
            else "generic"
        )
        train = CatalogSwiftDataset(
            config, encode=None if grpo else self.template.encode, grpo=grpo,
            message_format=message_format,
            collect_metrics=getattr(self.args, "performance_logging", False),
        )
        validation = None
        if config.get("validation"):
            raw_eval = grpo or getattr(self.args, "predict_with_generate", False)
            validation = CatalogSwiftDataset(
                config,
                training=False,
                encode=None if raw_eval else self.template.encode,
                grpo=grpo,
                message_format=message_format,
            )
        logging.getLogger(__name__).warning(
            "Catalog metadata ready: train=%d validation=%d load_seconds=%.3f",
            len(train), len(validation) if validation is not None else 0,
            time.perf_counter() - started,
        )
        return train, validation

    def train(self, trainer):
        if getattr(self.args, 'audio_encoder_parallel', False) and getattr(trainer.model.config, 'target_sot_audio', False):
            raise ValueError('Conditional SOT requires audio_encoder_parallel=false')
        if getattr(self.args, 'audio_encoder_batching', False) and getattr(self.args, 'audio_encoder_parallel', False):
            raise ValueError('Choose either audio_encoder_batching or audio_encoder_parallel')
        original_create_optimizer = None
        if getattr(self.args, "rlhf_type", None) == "grpo":
            config = self.args._catalog_config
            if config.get("batching") or config.get("replay") is not None or any(
                any(
                    key in source
                    for key in ("weight", "samples", "reps", "shard_rotation")
                )
                for source in config["train"]
            ):
                raise ValueError(
                    "Catalog mux/duration batching is SFT-only; GRPO uses its generation-group sampler"
                )
            trainer.add_callback(DatasetEpochCallback(trainer.train_dataset))
            if trainer.args.deepspeed and trainer.args.optimizer in (None, "default"):
                # DeepSpeed removes empty LoRA parameter groups. Do this before
                # each scheduler creation, including DeepSpeed re-initialization.
                original_create_optimizer = trainer.create_optimizer

                def create_optimizer():
                    optimizer = original_create_optimizer()
                    optimizer.param_groups[:] = [
                        group for group in optimizer.param_groups if group["params"]
                    ]
                    return optimizer

                trainer.create_optimizer = create_optimizer
        else:
            from .catalog_loader import install_catalog_loader

            # TrainerFactory drops fields outside its TrainingArguments schema.
            trainer.args.continuous_training = self.args.continuous_training
            trainer.args.reset_catalog_sampler = self.args.reset_catalog_sampler
            if self.args._catalog_config.get("objective", {}).get("sample_mean"):
                from .retention import install_retention_objective

                install_retention_objective(trainer, self.args)
            install_catalog_loader(trainer, self._get_resume_checkpoint(trainer))
            if getattr(self.args, 'retention_eval_script', None):
                from .retention_eval import RetentionEvaluationCallback

                trainer.add_callback(RetentionEvaluationCallback(
                    self.args.retention_eval_script, self.args.retention_eval_interval,
                ))
        uploads = None
        if storage := getattr(self.args, 'storage_sync', None):
            if trainer.args.save_total_limit:
                raise ValueError('storage_sync removes uploaded checkpoints itself; unset save_total_limit')
            from .checkpoint_upload import CheckpointUploadCallback

            # Registered after callbacks that write into the checkpoint during on_save.
            uploads = CheckpointUploadCallback(json.loads(storage))
            trainer.add_callback(uploads)
        if handoff := getattr(self.args, 'checkpoint_handoff', None):
            from .checkpoint_handoff import CheckpointHandoffCallback

            # After the uploader so train-end handoff sees the confirmed remote copy.
            trainer.add_callback(CheckpointHandoffCallback(
                handoff, selection=self.args.checkpoint_selection,
                base_model=self.args.model, tuner_type=self.args.tuner_type,
                uploads=uploads,
                amphion_eval_check=getattr(self.args, 'amphion_eval_check', None),
            ))
        restore_audio = None
        if getattr(self.args, "audio_encoder_batching", False):
            from .audio_batching import enable_batched_audio

            # Keep the frozen retention teacher's numerical target unchanged.
            restore_audio = enable_batched_audio(trainer.model)
        elif getattr(self.args, "audio_encoder_parallel", False):
            from .audio_batching import enable_parallel_audio

            restore_audio = enable_parallel_audio(trainer.model)
        try:
            return super().train(trainer)
        finally:
            if original_create_optimizer is not None:
                trainer.create_optimizer = original_create_optimizer
            if restore_audio is not None:
                restore_audio()
            performance = getattr(trainer, "catalog_performance", None)
            if performance is not None:
                performance.close()


class CatalogSft(CatalogTrainingMixin, SwiftSft):
    args_class = CatalogSftArguments


class CatalogGRPO(CatalogTrainingMixin, SwiftRLHF):
    args_class = CatalogRLHFArguments


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in {"sft", "grpo"}:
        raise SystemExit(
            "Usage: python -m open_audio_llm.integrations.ms_swift.train {sft,grpo} --data_config CONFIG [swift arguments]"
        )
    mode = sys.argv[1]
    cls = CatalogSft if mode == "sft" else CatalogGRPO
    cls(sys.argv[2:]).main()
    # Only a successful rank tears down collectives. After a failure, peers may be
    # blocked in a collective and destroy_process_group would wait for them, so
    # exit with the error and let torchrun stop the remaining ranks.
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
