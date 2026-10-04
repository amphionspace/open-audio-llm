"""Create a unified diarization-format training snapshot from the clean arm."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

import yaml

SINGLE_SPEAKER_DATASETS = {
    "aishell",
    "kespeech",
    "commonvoice_en_clean",
    "librispeech",
}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def single_speaker_source(source: dict, *, weight: float | None = None) -> dict:
    result = copy.deepcopy(source)
    result["single_speaker_format"] = True
    if weight is not None:
        result["weight"] = weight
    return result


def build_data_config(source: Path, output: Path) -> dict:
    config = yaml.safe_load(source.read_text())
    config["catalog"] = str(
        Path("runs/clean-events-ab-20260928/tasks/06-train-treatment/attempts/001/artifacts/catalog.jsonl").resolve()
    )
    config["roots"] = str(
        Path("runs/clean-events-ab-20260928/tasks/06-train-treatment/attempts/001/artifacts/roots.json").resolve()
    )
    config["metadata_cache"] = str(Path("runs/catalog-metadata-cache").resolve())

    train = []
    for source_item in config["train"]:
        dataset_id = source_item.get("dataset_id")
        if source_item.get("task") == "asr":
            if dataset_id not in SINGLE_SPEAKER_DATASETS:
                # WenetSpeech clean labels do not carry a speaker identity.
                continue
            train.append(single_speaker_source(source_item, weight=5.0))
        else:
            train.append(source_item)
    config["train"] = train
    # Every ordinary ASR source is either converted to a speaker-attributed
    # target or excluded when its supervision has no speaker identity. There
    # are therefore no plain-ASR examples for a teacher replay KL term.
    config.setdefault("objective", {})["replay_kl_weight"] = 0.0

    validation = []
    evaluation = []
    for source_item in config.get("validation", []):
        dataset_id = source_item.get("dataset_id")
        if source_item.get("task") == "asr":
            if dataset_id not in SINGLE_SPEAKER_DATASETS:
                continue
            validation.append(single_speaker_source(source_item))
            evaluation.append(copy.deepcopy(source_item))
        else:
            validation.append(source_item)
    config["validation"] = validation
    config["evaluation"] = evaluation
    config["format_policy"] = {
        "training_target": "speaker_attributed_asr",
        "single_speaker_sources": sorted(SINGLE_SPEAKER_DATASETS),
        "excluded_without_speaker_supervision": ["wenetspeech_clean"],
        "asr_regression_split": "evaluation",
    }
    output.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False))
    return config


def build_recipe(root: Path, data_config: Path, plan: dict) -> None:
    repo = Path(__file__).resolve().parents[1]
    initial = Path(
        "/222042021/mingdong/workspace/open-audio-llm/runs/"
        "qwen3-asr-sot-speaker-events-20260922/initial-checkpoint"
    )
    recipe = {
        "version": 1,
        "runtime": {
            "python": {"path": "/ai_sds_wuzz/MODELS/miniconda3/envs/amphionft/bin/python"},
            "cwd": {"path": str(repo)},
            "pythonpath": [
                {"path": str(repo / "src")},
                {"path": str(repo / "runs/qwen3-asr-sot-speaker-events-20260922/dependencies/audio-data-contract/src")},
            ],
            "gpus": [0, 1],
            "threads": {"omp": 2, "openblas": 1},
            "environment": {
                "TOKENIZERS_PARALLELISM": False,
                "HF_HUB_OFFLINE": "1",
                "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
                "PYTHONUNBUFFERED": "1",
                "PYTORCH_ALLOC_CONF": "expandable_segments:True",
            },
            "distributed": {"processes": 2, "port": 29783},
        },
        "data": {"config": {"path": str(data_config)}},
        "task": {
            "kind": "train",
            "module": "open_audio_llm.integrations.ms_swift.train",
            "positional": ["sft"],
            "arguments": {
                "--model_type": "amphion_asr_1.7b",
                "--tuner_type": "full",
                "--lora_rank": 64,
                "--lora_alpha": 128,
                "--target_modules": ["all-linear"],
                "--freeze_vit": False,
                "--freeze_aligner": False,
                "--freeze_llm": False,
                "--torch_dtype": "float32",
                "--attn_impl": "sdpa",
                "--max_length": 16384,
                "--per_device_train_batch_size": 8,
                "--per_device_eval_batch_size": 1,
                "--gradient_accumulation_steps": 4,
                "--learning_rate": 1.0e-5,
                "--vit_lr": 1.0e-5,
                "--aligner_lr": 2.0e-5,
                "--max_steps": 1000,
                "--lr_scheduler_type": "cosine",
                "--warmup_ratio": 0.0,
                "--warmup_steps": 500,
                "--gradient_checkpointing": True,
                "--vit_gradient_checkpointing": True,
                # Latest plugins attach the TS-only SEP even to diarization models.
                "--ddp_find_unused_parameters": True,
                "--dataloader_num_workers": 4,
                "--dataloader_persistent_workers": True,
                "--performance_logging": True,
                "--save_steps": 500,
                "--save_total_limit": 3,
                "--logging_steps": 5,
                "--report_to": "none",
                "--seed": 42,
                "--audio_encoder_parallel": False,
                "--audio_encoder_batching": True,
                "--use_logits_to_keep": False,
                "--prediction_loss_only": True,
                "--optim": "adamw_torch_fused",
                "--weight_decay": 0.01,
                "--max_grad_norm": 1.0,
                "--save_only_model": False,
                "--bf16": True,
                "--fp16": False,
                "--ddp_timeout": 300,
                "--eval_strategy": "no",
                "--load_args": False,
                "--add_version": False,
                "--retention_teacher": {"path": "/ai_sds_wuzz/MODELS/Qwen3-ASR-1.7B/Qwen/Qwen3-ASR-1___7B"},
                "--callbacks": ["sot_training_audit"],
                "--model": {"path": str(initial)},
                "--external_plugins": [
                    {"path": str(repo / "src/open_audio_llm/integrations/ms_swift/register_qwen3_asr.py")},
                    {"path": str(root / "scripts/training_audit.py")},
                ],
                "--output_dir": {"path": "{attempt}/artifacts/training"},
            },
        },
        "preflight": [
            {
                "kind": "prepare",
                "module": "open_audio_llm.data.catalog_cache",
                "arguments": {
                    "--data_config": {"path": str(data_config)},
                    "--workers": 4,
                    "--batch-size": 8,
                    "--world-size": 2,
                    "--preflight-report": {"path": "{attempt}/artifacts/data-preflight.json"},
                },
            }
        ],
        "recording": {"attempts_dir": {"path": "../tasks/train/attempts"}},
        "tracking": {
            "pythonpath": [{"path": str(repo / "src")}],
            "enabled": True,
            "python": {"path": str(repo / "runs/wandb-sync-env/bin/python")},
            "credentials_file": {"path": "~/.bashrc"},
            "entity": "1016097967-amphion",
            "project": "open-audio-llm",
            "name": "unified-diarization-asr-20261001",
            "interval": 5,
            "startup_timeout": 180,
            "finish_timeout": 90,
        },
    }
    (root / "configs/train-unified.yaml").write_text(
        yaml.safe_dump(recipe, allow_unicode=True, sort_keys=False)
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.output_root.resolve()
    (root / "shared/data").mkdir(parents=True, exist_ok=True)
    (root / "configs").mkdir(exist_ok=True)
    (root / "scripts").mkdir(exist_ok=True)
    source_config = args.source_config.resolve()
    data_config = root / "shared/data/unified-train.yaml"
    config = build_data_config(source_config, data_config)
    source_roots = Path(config["roots"])
    roots = json.loads(source_roots.read_text())
    treatment_artifacts = Path(
        "runs/clean-events-ab-20260928/tasks/06-train-treatment/attempts/001/artifacts"
    ).resolve()
    roots["events_treatment_ab_manifest"] = str(treatment_artifacts / "event-data")
    roots["events_treatment_ab_audio"] = str(
        Path("runs/clean-events-ab-20260928/tasks/03-synthesis/attempts/002/artifacts/generated").resolve()
    )
    unified_roots = root / "shared/data/roots.json"
    unified_roots.write_text(json.dumps(roots, ensure_ascii=False, indent=2) + "\n")
    config["roots"] = str(unified_roots.resolve())
    config["experiment_plan"] = str((root / "shared/plan.json").resolve())
    data_config.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False))

    source_plan = Path(
        "runs/clean-events-ab-20260928/tasks/06-train-treatment/attempts/001/artifacts/plan.json"
    )
    plan = json.loads(source_plan.read_text())
    for key in (
        "created_at", "launched_at", "launcher_pid", "verified_at", "verification",
        "handoff_completed_at",
    ):
        plan.pop(key, None)
    plan.update(
        {
            "name": root.name,
            "status": "prepared",
            "pilot_training": True,
            "training_steps": 1000,
            "format_policy": config["format_policy"],
            "source_data_config": str(source_config),
            "source_data_config_sha256": digest(source_config),
            "unified_data_config": str(data_config),
            "unified_data_config_sha256": digest(data_config),
            "model": "initial-checkpoint",
            "scope": "Unified diarization targets with speaker-labeled single-speaker ASR replay; bounded 1000-step pilot.",
            "retention_gate": "ASR regression uses the fixed evaluation split; single-speaker replay uses diarization targets.",
        }
    )
    (root / "shared/plan.json").write_text(json.dumps(plan, ensure_ascii=False, indent=2) + "\n")
    audit = Path(
        "runs/clean-events-ab-20260928/scripts/training_audit.py"
    )
    audit_output = root / "scripts/training_audit.py"
    if not audit_output.exists():
        audit_output.write_text(audit.read_text())
    build_recipe(root, data_config, plan)
    print(json.dumps({"root": str(root), "recipe": str(root / "configs/train-unified.yaml"),
                      "data": str(data_config), "train_sources": len(config["train"]),
                      "validation_sources": len(config["validation"]),
                      "evaluation_sources": len(config["evaluation"])}, ensure_ascii=False))


if __name__ == "__main__":
    main()
