"""Deliver an explicit saved checkpoint to external post-training consumers."""
from __future__ import annotations

import json
import os
from pathlib import Path

from transformers import TrainerCallback

# Bump when a field changes meaning or is removed; consumers reject unknown versions.
# v2: `checkpoint` may be removed locally once `checkpoint_remote` confirms the upload.
HANDOFF_SCHEMA_VERSION = 2


class CheckpointHandoffCallback(TrainerCallback):
    def __init__(self, path, *, base_model, tuner_type, selection="final", uploads=None):
        if selection not in {"final", "best"}:
            raise ValueError("Checkpoint selection must be final or best")
        self.path = Path(path).expanduser().resolve()
        self.selection = selection
        self.base_model = str(base_model)
        self.tuner_type = tuner_type
        self.uploads = uploads
        self.saved_checkpoint = None
        self.saved_step = None

    def on_save(self, args, state, control, **kwargs):
        if state.is_world_process_zero:
            self.saved_checkpoint = Path(args.output_dir).resolve() / f"checkpoint-{state.global_step}"
            self.saved_step = state.global_step

    def on_train_end(self, args, state, control, **kwargs):
        if not state.is_world_process_zero:
            return
        if self.selection == "best":
            checkpoint = Path(state.best_model_checkpoint).resolve() if state.best_model_checkpoint else None
        else:
            checkpoint = self.saved_checkpoint if self.saved_step == state.global_step else None
        remote = self.uploads.uploaded.get(checkpoint) if self.uploads and checkpoint else None
        if checkpoint is None or not (checkpoint.is_dir() or remote):
            raise RuntimeError("No saved checkpoint for the requested selection; save the final step or explicitly select best")
        payload = {
            "framework": "open-audio-llm", "schema_version": HANDOFF_SCHEMA_VERSION,
            "checkpoint": str(checkpoint), "checkpoint_remote": remote,
            "selection": self.selection,
            "training_run": str(Path(args.output_dir).resolve()),
            "global_step": state.global_step, "base_model": self.base_model,
            "tuner_type": self.tuner_type,
            "wandb_run_id": os.environ.get("WANDB_RUN_ID"),
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(".tmp")
        temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        temporary.replace(self.path)
