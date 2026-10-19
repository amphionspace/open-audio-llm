"""Connect Catalog sampling to ms-swift's SFT loop without double sharding."""

from __future__ import annotations

import functools
import json
import os
import pickle
import sys
from pathlib import Path
from types import MethodType

import torch
from accelerate.data_loader import DataLoaderShard
from torch.utils.data import DataLoader
from swift.utils.utils import get_external_files
from transformers import TrainerCallback

from open_audio_llm.data.catalog_sampler import CatalogBatchSampler

# A batch normally arrives within seconds. A worker silent this long is hung; failing
# here names the cause instead of waiting for the 30-minute NCCL timeout on the other ranks.
DATALOADER_TIMEOUT_S = 600


class StopLoaderWorkersCallback(TrainerCallback):
    """Stop the training loader's persistent workers when training ends.

    With persistent workers and pinned memory, torch registers one atexit join per
    worker (5 s timeout) that runs before the iterator's own shutdown sends the stop
    signal, so each worker times out in turn: 20 workers per rank kept all 8 GPUs of a
    finished job idle for about 100 s.
    """

    def __init__(self):
        self.loader = None

    def on_train_end(self, args, state, control, **kwargs):
        iterator = getattr(self.loader, "_iterator", None)
        if iterator is not None:
            iterator._shutdown_workers()


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
    if sampler.packing == "sorted" and sampler.accumulation != args.gradient_accumulation_steps:
        raise ValueError("batching.accumulation must equal gradient_accumulation_steps")
    if (sampler.batch_merge * args.gradient_accumulation_steps) % sampler.merge_window:
        raise ValueError("batching.merge_window must fit complete gradient-accumulation updates")
    if resume_checkpoint and not getattr(args, "reset_catalog_sampler", False):
        path = Path(resume_checkpoint) / "catalog_sampler.json"
        if not path.is_file():
            raise ValueError(
                f"Sampler state missing: {path}; use the old model as initialization for a new run"
            )
        state = json.loads(path.read_text())
        if migration := getattr(args, "catalog_sampler_migration", None):
            migration = json.loads(migration) if isinstance(migration, str) else migration
            state = sampler.migrate_state_dict(
                state,
                world_size=migration["world_size"],
                previous_paths=migration.get("previous_paths", {}),
            )
        sampler.load_state_dict(state)
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
        original_load = trainer._load_optimizer_and_scheduler
        reset = getattr(args, "reset_catalog_sampler", False)

        def load_optimizer_and_scheduler(self, checkpoint):
            from transformers import get_inverse_sqrt_schedule

            kwargs = dict(getattr(self.args, "lr_scheduler_kwargs", None) or {})

            def configured():
                return get_inverse_sqrt_schedule(
                    self.optimizer,
                    num_warmup_steps=self.args.get_warmup_steps(sys.maxsize),
                    **kwargs,
                )

            # ms-swift>=4.5 builds the scheduler via the unbound HF
            # Trainer.create_scheduler, bypassing the patch above. HF calls this
            # loader on fresh and resumed runs, so install the configured schedule
            # here; the checkpoint then restores its step and the optimizer's LR.
            self.lr_scheduler = configured()
            original_load(checkpoint)
            if checkpoint and reset:
                self.lr_scheduler = configured()

        trainer._load_optimizer_and_scheduler = MethodType(
            load_optimizer_and_scheduler, trainer
        )
    # The saved consumed cursor replaces HF's global_step-based batch skipping.
    args.ignore_data_skip = True
    trainer.catalog_sampler = sampler
    trainer.add_callback(CatalogSamplerCallback(sampler))
    collator = trainer.data_collator
    if getattr(args, "packed_text", False):
        from .packed_text import PackedTextCollator

        collator = PackedTextCollator(collator)
    if getattr(trainer.train_dataset, "collect_metrics", False):
        from .performance import PerformanceCollator, install_performance_logging

        collator = PerformanceCollator(collator)
        trainer.catalog_performance = install_performance_logging(trainer)

    stop_workers = StopLoaderWorkersCallback()
    trainer.add_callback(stop_workers)

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
                timeout=DATALOADER_TIMEOUT_S,
            )
        stop_workers.loader = DataLoaderShard(
            self.train_dataset, device=self.accelerator.device, **params
        )
        return stop_workers.loader

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
    # Evaluation builds its loader after NCCL and the training pin-memory thread
    # exist. fork() would copy those locks into the workers. forkserver starts
    # workers from a clean process, so eval keeps dataloader_num_workers.
    trainer.get_eval_dataloader = MethodType(get_catalog_eval_dataloader, trainer)


def get_catalog_eval_dataloader(trainer, eval_dataset=None):
    """Shard evaluation like Trainer, but do not fork the CUDA process."""
    if eval_dataset is None and trainer.eval_dataset is None:
        raise ValueError("Trainer: evaluation requires an eval_dataset.")
    dataloader_key = eval_dataset if isinstance(eval_dataset, str) else "eval"
    cache = getattr(trainer, "_catalog_eval_dataloaders", None)
    if cache is not None and dataloader_key in cache:
        return cache[dataloader_key]
    resolved = (
        trainer.eval_dataset[eval_dataset]
        if isinstance(eval_dataset, str)
        else eval_dataset
        if eval_dataset is not None
        else trainer.eval_dataset
    )
    loader = trainer.accelerator.prepare(catalog_eval_dataloader(trainer, resolved))
    if cache is None:
        trainer._catalog_eval_dataloaders = {dataloader_key: loader}
    else:
        cache[dataloader_key] = loader
    return loader


def _method_self(fn):
    while isinstance(fn, functools.partial):
        fn = fn.func
    return fn.__self__ if isinstance(fn, MethodType) else None


def _detach_template(template):
    """Copy the template without the DeepSpeed engine.

    Eval workers are forkserver processes and must pickle the dataset and
    collator. Both hold this template. Its forward hooks keep the engine, and
    DeepSpeed's hook is a nested function that pickle cannot serialize.
    """
    saved = {
        name: getattr(template, name, None)
        for name in ("model", "dummy_model", "_handles", "_deepspeed_initialize")
    }
    template.model = None
    template.dummy_model = None
    template._handles = []
    template._deepspeed_initialize = None
    try:
        return pickle.loads(pickle.dumps(template))
    finally:
        for name, value in saved.items():
            setattr(template, name, value)


def _rebind(fn, template, detached):
    if isinstance(fn, functools.partial):
        if _method_self(fn) is not template:
            return fn
        rebound = MethodType(fn.func.__func__, detached)
        return functools.partial(rebound, *fn.args, **dict(fn.keywords or {}))
    if isinstance(fn, MethodType) and fn.__self__ is template:
        return MethodType(fn.__func__, detached)
    return fn


def _worker_view(eval_dataset, collator):
    template = _method_self(getattr(eval_dataset, "encode", None)) or _method_self(collator)
    if template is None:
        return eval_dataset, collator
    detached = _detach_template(template)
    if _method_self(getattr(eval_dataset, "encode", None)) is template:
        eval_dataset.encode = _rebind(eval_dataset.encode, template, detached)
    return eval_dataset, _rebind(collator, template, detached)


def _forkserver_context():
    """Forkserver context whose server has already imported the plugin files.

    The server reads its environment when it starts, so this must run before the
    first forkserver worker of the process.
    """
    from .plugin_preload import PLUGIN_FILES_ENV

    os.environ[PLUGIN_FILES_ENV] = os.pathsep.join(get_external_files())
    context = torch.multiprocessing.get_context("forkserver")
    context.set_forkserver_preload([f"{__package__}.plugin_preload"])
    return context


def catalog_eval_dataloader(trainer, eval_dataset):
    args = trainer.args
    eval_dataset, collator = _worker_view(eval_dataset, trainer.data_collator)
    if hasattr(trainer, "_get_collator_with_removed_columns"):
        collator = trainer._get_collator_with_removed_columns(
            collator, description="Evaluation"
        )
    # Eval's GPU step is a forward only. Train hides the audio read behind
    # backward; eval needs its own worker count and a deeper queue so the
    # next batches are ready when that forward finishes.
    num_workers = getattr(args, "dataloader_eval_num_workers", None)
    if num_workers is None:
        num_workers = args.dataloader_num_workers
    prefetch = getattr(args, "dataloader_eval_prefetch_factor", None)
    if prefetch is None:
        prefetch = args.dataloader_prefetch_factor or 4
    params = {
        "batch_size": args.eval_batch_size,
        "collate_fn": collator,
        "num_workers": num_workers,
        "pin_memory": args.dataloader_pin_memory,
        "drop_last": args.dataloader_drop_last,
        "sampler": trainer._get_eval_sampler(eval_dataset),
    }
    if num_workers:
        params.update(
            persistent_workers=True,
            prefetch_factor=prefetch,
            multiprocessing_context=_forkserver_context(),
            timeout=DATALOADER_TIMEOUT_S,
        )
    return DataLoader(eval_dataset, **params)
