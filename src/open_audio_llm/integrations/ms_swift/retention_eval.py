"""Pause DDP updates to evaluate saved candidates using the rank-zero GPU."""

from __future__ import annotations

import json
import logging
import os
import signal
import subprocess
import sys
from pathlib import Path

import torch
from transformers import TrainerCallback

LOG = logging.getLogger(__name__)


class RetentionEvaluationCallback(TrainerCallback):
    def __init__(self, script, interval):
        self.script = Path(script).expanduser().resolve()
        if not self.script.is_file() or interval <= 0:
            raise ValueError("Retention evaluation requires an existing Python script and positive interval")
        self.interval = interval

    def on_save(self, args, state, control, **kwargs):
        if state.global_step % self.interval and state.global_step not in {args.save_steps, state.max_steps}:
            return
        distributed = torch.distributed.is_initialized()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        if distributed:
            torch.distributed.barrier()
        error = [None]
        if state.is_world_process_zero:
            checkpoint = Path(args.output_dir) / f"checkpoint-{state.global_step}"
            output = Path(args.output_dir) / "retention-evaluations" / checkpoint.name
            command = [sys.executable, str(self.script), str(checkpoint), str(output)]
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = env.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0]
            for key in list(env):
                if key in {"RANK", "LOCAL_RANK", "WORLD_SIZE", "LOCAL_WORLD_SIZE", "GROUP_RANK", "ROLE_RANK", "ROLE_WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT"} or key.startswith("TORCHELASTIC_"):
                    env.pop(key)
            LOG.warning("Pausing DDP updates for retention evaluation: %s", checkpoint)
            try:
                output.mkdir(parents=True, exist_ok=True)
                with (output / "runner.log").open("w") as stream:
                    # Leave time to propagate failure before the DDP collective timeout.
                    with subprocess.Popen(command, env=env, stdout=stream, stderr=subprocess.STDOUT,
                                          start_new_session=True) as process:
                        try:
                            returncode = process.wait(timeout=max(1, args.ddp_timeout - 60))
                        except subprocess.TimeoutExpired:
                            # The runner can spawn a model worker; terminate that worker too.
                            os.killpg(process.pid, signal.SIGKILL)
                            process.wait()
                            raise
                (output / "runner-exit.json").write_text(json.dumps({"returncode": returncode}))
                if returncode:
                    raise RuntimeError(f"Evaluator exited {returncode}; see {output / 'runner.log'}")
            except Exception as exc:
                error[0] = repr(exc)
        if distributed:
            torch.distributed.broadcast_object_list(error, src=0)
        if error[0]:
            raise RuntimeError(f"Retention evaluation failed: {error[0]}")
