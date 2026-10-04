"""Connect Catalog sampling to ms-swift's SFT loop without double sharding."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import MethodType

import torch
from accelerate.data_loader import DataLoaderShard
from transformers import TrainerCallback

from open_audio_llm.data.catalog_sampler import CatalogBatchSampler


class CatalogSamplerCallback(TrainerCallback):
    def __init__(self, sampler):
        self.sampler = sampler

    def on_substep_end(self, args, state, control, **kwargs):
        self.sampler.mark_consumed()

    def on_step_end(self, args, state, control, **kwargs):
        self.sampler.mark_consumed()

    def on_save(self, args, state, control, **kwargs):
        if state.is_world_process_zero:
            path = (
                Path(args.output_dir)
                / f"checkpoint-{state.global_step}"
                / "catalog_sampler.json"
            )
            path.write_text(json.dumps(self.sampler.state_dict(), indent=2) + "\n")


def install_catalog_loader(trainer, resume_checkpoint=None):
    """The sampler owns DDP batch partitioning and checkpoint data position."""
    args = trainer.args
    continuous = getattr(args, "continuous_training", False)
    if continuous:
        if args.max_steps > 0:
            raise ValueError("continuous_training requires max_steps <= 0")
        if args.lr_scheduler_type not in {"constant", "constant_with_warmup", "inverse_sqrt"}:
            raise ValueError("continuous_training requires a schedule independent of total steps")
        if args.warmup_ratio:
            raise ValueError("continuous_training requires explicit warmup_steps, not warmup_ratio")
        # HF's finite-loop API needs a positive sentinel; no business stop is set.
        args.max_steps = sys.maxsize
    if args.max_steps <= 0:
        raise ValueError(
            "Catalog dynamic batching requires an explicit positive max_steps"
        )
    if trainer.template.sequence_parallel_size > 1:
        raise ValueError(
            "Catalog duration batching supports data parallel SFT, not sequence parallelism"
        )
    if args.deepspeed and args.deepspeed.get("tensor_parallel"):
        raise ValueError(
            "Catalog duration batching does not support tensor parallelism"
        )
    sampler = CatalogBatchSampler(
        trainer.train_dataset,
        batch_size=trainer._train_batch_size,
        rank=args.process_index,
        world_size=args.world_size,
        shuffle=args.train_dataloader_shuffle,
    )
    if (sampler.batch_merge * args.gradient_accumulation_steps) % sampler.merge_window:
        raise ValueError("batching.merge_window must fit complete gradient-accumulation updates")
    if resume_checkpoint and not getattr(args, "reset_catalog_sampler", False):
        path = Path(resume_checkpoint) / "catalog_sampler.json"
        if not path.is_file():
            raise ValueError(
                f"Sampler state missing: {path}; use the old model as initialization for a new run"
            )
        sampler.load_state_dict(json.loads(path.read_text()))
    if continuous and args.lr_scheduler_type == "constant" and getattr(args, "reset_catalog_sampler", False):
        original_load = trainer._load_optimizer_and_scheduler

        def load_optimizer_and_scheduler(self, checkpoint):
            original_load(checkpoint)
            if checkpoint:
                from transformers import get_constant_schedule

                # Keep AdamW moments and step, but replace the finished cosine LR.
                self.lr_scheduler = get_constant_schedule(
                    self.optimizer, last_epoch=self.lr_scheduler.last_epoch - 1
                )

        trainer._load_optimizer_and_scheduler = MethodType(load_optimizer_and_scheduler, trainer)
    if continuous and args.lr_scheduler_type == "inverse_sqrt":
        # transformers<=4.57 accepts scheduler_specific_kwargs in get_scheduler
        # but drops them in its inverse-sqrt branch. Install it directly so the
        # configured timescale is effective.
        if not hasattr(trainer, "create_scheduler"):
            raise ValueError("Continuous inverse-sqrt schedule requires Trainer.create_scheduler")
        def create_scheduler(self, num_training_steps, optimizer=None):
            if self.lr_scheduler is None:
                from transformers import get_inverse_sqrt_schedule

                selected = self.optimizer if optimizer is None else optimizer
                kwargs = dict(getattr(self.args, "lr_scheduler_kwargs", None) or {})
                self.lr_scheduler = get_inverse_sqrt_schedule(
                    selected,
                    num_warmup_steps=self.args.get_warmup_steps(num_training_steps),
                    **kwargs,
                )
                self._created_lr_scheduler = True
            return self.lr_scheduler
        trainer.create_scheduler = MethodType(create_scheduler, trainer)
        if getattr(args, "reset_catalog_sampler", False):
            original_load = trainer._load_optimizer_and_scheduler

            def load_optimizer_and_scheduler(self, checkpoint):
                original_load(checkpoint)
                if checkpoint:
                    from transformers import get_inverse_sqrt_schedule

                    kwargs = dict(getattr(self.args, "lr_scheduler_kwargs", None) or {})
                    self.lr_scheduler = get_inverse_sqrt_schedule(
                        self.optimizer,
                        num_warmup_steps=self.args.get_warmup_steps(sys.maxsize),
                        **kwargs,
                    )

            trainer._load_optimizer_and_scheduler = MethodType(
                load_optimizer_and_scheduler, trainer
            )
    # The saved consumed cursor replaces HF's global_step-based batch skipping.
    args.ignore_data_skip = True
    trainer.catalog_sampler = sampler
    trainer.add_callback(CatalogSamplerCallback(sampler))
    collator = trainer.data_collator
    if getattr(trainer.train_dataset, "collect_metrics", False):
        from .performance import PerformanceCollator, install_performance_logging

        collator = PerformanceCollator(collator)
        trainer.catalog_performance = install_performance_logging(trainer)

    def get_train_dataloader(self, skip_batches=0):
        if skip_batches:
            raise ValueError("Catalog sampler already restores the checkpoint cursor")
        params = {
            "batch_sampler": sampler,
            "collate_fn": collator,
            "num_workers": args.dataloader_num_workers,
            "pin_memory": args.dataloader_pin_memory,
            # Fresh workers after resume must not advance the model's CPU RNG.
            "generator": torch.Generator().manual_seed(self.train_dataset.seed),
        }
        if args.dataloader_num_workers:
            params.update(
                persistent_workers=args.dataloader_persistent_workers,
                prefetch_factor=args.dataloader_prefetch_factor,
            )
        return DataLoaderShard(
            self.train_dataset, device=self.accelerator.device, **params
        )

    initial_values = trainer.set_initial_training_values

    def set_initial_training_values(self, *positional, **kwargs):
        values = initial_values(*positional, **kwargs)
        # Epoch batch counts can change with shard rotation and bucketing.
        # Stop at max_steps, never at an extrapolated first-epoch count.
        return (sys.maxsize, *values[1:])

    trainer.get_train_dataloader = MethodType(get_train_dataloader, trainer)
    trainer.set_initial_training_values = MethodType(
        set_initial_training_values, trainer
    )
