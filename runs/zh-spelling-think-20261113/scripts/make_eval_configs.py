"""Serve and evaluation configs of one checkpoint for the context-correction set, think off and think on.

Both servers load the think chat template (src/open_audio_llm/integrations/vllm/qwen3_asr_think.jinja); requests
without chat_template_kwargs render exactly the official Qwen3-ASR prompt, so the same servers answer both modes.

usage: make_eval_configs.py <name> <checkpoint dir> <asr gpu> <ts gpu> <asr port>
       (writes configs/eval-<name>/: serve-asr.yaml, serve-tsasr.yaml, after-training-context-correction{,-think}.yaml)
"""
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parents[1]
TEMPLATE = {"path": "../../shared/code-v1/src/open_audio_llm/integrations/vllm/qwen3_asr_think.jinja"}


def serve(name, checkpoint, gpu, port, ts):
    route = "tsasr" if ts else "asr"
    return {
        "version": 1,
        "runtime": {
            "python": {"path": "/workspace/workspace/mingdong/miniconda3/envs/vllm-serving/bin/python"},
            "cwd": {"path": "../../../.."},
            "pythonpath": [{"path": "../../shared/code-v1/src"}],
            "gpus": [gpu],
            "threads": {"omp": 4, "openblas": 1},
            "environment": {
                "VLLM_WORKER_MULTIPROC_METHOD": "spawn", "HF_HUB_OFFLINE": "1", "VLLM_ENGINE_READY_TIMEOUT_S": "1800",
                "PYTHONUNBUFFERED": "1", "AMPHION_TSASR_INSERT_SEP": "1" if ts else "0", "AMPHION_TSASR_ENROLL_SEC": "5",
                "AMPHION_TSASR_ENROLL_NOPAD": "0", "COT_TSASR_ENROLL_RANDOM": "0", "COT_AUDIO_CHUNKED_ATTN": "0"},
        },
        "task": {
            "kind": "serve", "backend": "vllm", "model": {"path": str(checkpoint)},
            "arguments": {
                "--host": "127.0.0.1", "--port": port + (1 if ts else 0), "--served-model-name": f"{name}-{route}",
                "--dtype": "bfloat16", "--tensor-parallel-size": 1, "--limit-mm-per-prompt": {"audio": 1},
                "--gpu-memory-utilization": 0.2, "--max-model-len": 8192, "--chat-template": TEMPLATE,
                "--hf-overrides": {"thinker_config": {"audio_config": {"n_window_infer": 1000000000}}}},
            "flags": ["--trust-remote-code", "--no-enable-prefix-caching"],
        },
        "recording": {"attempts_dir": {"path": f"../../tasks/03-eval/{name}/serve-{route}/attempts"}},
    }


def evaluation(name, port, think):
    return {
        "suite_id": "asr-regression-native@1",
        "model_name": f"{name}{'-think' if think else ''}",
        "evaluator": "ts-asr",
        "eval_set": "open-audio-llm/context-correction-zh@v1-20261018",
        "server_url": f"http://127.0.0.1:{port}", "served_model": f"{name}-asr",
        "tsasr_server_url": f"http://127.0.0.1:{port + 1}", "tsasr_served_model": f"{name}-tsasr",
        "worker_timeout": 600,
        "output_root": f"/workspace/data/ae-runs/spelling-think-20261113-{name}{'-think' if think else ''}",
        "parameters": {"split_group": "evaluation", "samples_per_source": 0, "batch_size": 32,
                       "max_new_tokens": 512, "seed": 42, "enroll_seconds": 5, **({"think": True} if think else {})},
        "wandb": {"entity": "1016097967-amphion", "project": "open-audio-llm"},
        "publish": {"enabled": False},
    }


def main():
    name, checkpoint, asr_gpu, ts_gpu, port = sys.argv[1], Path(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
    out = HERE / "configs" / f"eval-{name}"
    out.mkdir(parents=True, exist_ok=True)
    dump = lambda obj: yaml.safe_dump(obj, allow_unicode=True, sort_keys=False)
    (out / "serve-asr.yaml").write_text(dump(serve(name, checkpoint, asr_gpu, port, False)))
    (out / "serve-tsasr.yaml").write_text(dump(serve(name, checkpoint, ts_gpu, port, True)))
    (out / "after-training-context-correction.yaml").write_text(dump(evaluation(name, port, False)))
    (out / "after-training-context-correction-think.yaml").write_text(dump(evaluation(name, port, True)))
    print(out)


if __name__ == "__main__":
    main()
