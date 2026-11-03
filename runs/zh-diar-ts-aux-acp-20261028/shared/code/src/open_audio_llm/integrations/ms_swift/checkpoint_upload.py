"""Upload saved checkpoints in the background and keep only unconfirmed or still-needed copies local."""
from __future__ import annotations

import json
import logging
import os
import queue
import shutil
import threading
from pathlib import Path

from transformers import TrainerCallback

from open_audio_llm.storage import StorageError, check, remote_path, upload_checkpoint

LOG = logging.getLogger(__name__)


class CheckpointUploadCallback(TrainerCallback):
    def __init__(self, storage):
        check(storage)
        self.storage = storage
        self.uploaded = {}  # resolved local checkpoint -> confirmed remote prefix
        self.records = []
        self.latest = None
        self.best = None
        self.queue = queue.Queue()
        self.worker = None
        self.record_file = None

    def on_save(self, args, state, control, **kwargs):
        if not state.is_world_process_zero:
            return
        # Absolute (not resolved) paths keep the remote layout under storage.local_root.
        checkpoint = Path(os.path.abspath(args.output_dir)) / f"checkpoint-{state.global_step}"
        self.record_file = checkpoint.parent / "storage-uploads.json"
        self.latest = checkpoint.resolve()
        self.best = Path(state.best_model_checkpoint).resolve() if state.best_model_checkpoint else None
        if self.worker is None:
            self.worker = threading.Thread(target=self._drain, name="checkpoint-upload", daemon=True)
            self.worker.start()
        self.queue.put(checkpoint)

    def _upload(self, checkpoint, resume_state):
        record = {"checkpoint": str(checkpoint), "remote": remote_path(checkpoint, self.storage),
                  "resume_state": resume_state}
        try:
            record["transfers"] = upload_checkpoint(checkpoint, self.storage, resume_state)
            self.uploaded[checkpoint.resolve()] = record["remote"]
        except StorageError as exc:  # A failed upload only keeps the local copy.
            record["problem"] = exc.problem
        except OSError as exc:
            record["problem"] = {"cause": "local-error", "detail": f"{type(exc).__name__}: {exc}"}
        if "problem" in record:
            LOG.error("Checkpoint upload failed; keeping %s locally: %s: %s", checkpoint,
                      record["problem"]["cause"], record["problem"]["detail"])
        self.records.append(record)
        temporary = self.record_file.with_suffix(".tmp")
        temporary.write_text(json.dumps(self.records, ensure_ascii=False, indent=2) + "\n")
        temporary.replace(self.record_file)

    def _remove_confirmed(self, keep):
        for checkpoint in self.uploaded:
            if checkpoint not in keep and checkpoint.is_dir():
                shutil.rmtree(checkpoint)

    def _drain(self):
        while (checkpoint := self.queue.get()) is not None:
            # Intermediate saves upload weights only; the latest and best stay local
            # for crash resume and load_best_model_at_end.
            self._upload(checkpoint, resume_state=False)
            self._remove_confirmed({self.latest, self.best})

    def on_train_end(self, args, state, control, **kwargs):
        if not state.is_world_process_zero or self.worker is None:
            return
        self.queue.put(None)
        self.worker.join()
        if self.latest.is_dir():
            self.uploaded.pop(self.latest, None)
            # The final checkpoint keeps its optimizer state so training can continue from storage.
            self._upload(Path(os.path.abspath(args.output_dir)) / self.latest.name, resume_state=True)
        self._remove_confirmed(set())
