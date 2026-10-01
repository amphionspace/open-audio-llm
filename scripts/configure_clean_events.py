"""Create reviewable pilot configs and navigation from preserved migration evidence."""

import argparse
import copy
import json
from datetime import datetime, timezone
from pathlib import Path

import yaml
from migrate_experiments import TASKS

from open_audio_llm.run_config import resolve_values

TRAIN_PARAMETERS = (
    "model_type",
    "tuner_type",
    "lora_rank",
    "lora_alpha",
    "target_modules",
    "freeze_vit",
    "freeze_aligner",
    "freeze_llm",
    "torch_dtype",
    "attn_impl",
    "max_length",
    "per_device_train_batch_size",
    "per_device_eval_batch_size",
    "gradient_accumulation_steps",
    "learning_rate",
    "vit_lr",
    "aligner_lr",
    "max_steps",
    "lr_scheduler_type",
    "warmup_ratio",
    "warmup_steps",
    "gradient_checkpointing",
    "vit_gradient_checkpointing",
    "ddp_find_unused_parameters",
    "dataloader_num_workers",
    "dataloader_persistent_workers",
    "performance_logging",
    "save_steps",
    "save_total_limit",
    "eval_steps",
    "logging_steps",
    "report_to",
    "seed",
    "audio_encoder_parallel",
    "audio_encoder_batching",
    "use_logits_to_keep",
    "prediction_loss_only",
    "optim",
    "weight_decay",
    "max_grad_norm",
    "bf16",
    "fp16",
    "save_only_model",
    "ddp_timeout",
    "eval_strategy",
    "load_args",
    "add_version",
    "retention_teacher",
    "callbacks",
)


def path(value):
    return {"path": str(value)}


def generate(root):
    repo = root.parent.parent
    mapping = json.loads((root / "tracking/migration-plan.json").read_text())["mapping"]
    files = {old: path("../" + new) for old, new in mapping.items()}
    reference = root.parent / "qwen3-asr-sot-speaker-events-20260922"
    model_python = "/ai_sds_wuzz/MODELS/miniconda3/envs/amphionft/bin/python"
    outputs = []
    configs = {}
    arm_data = {}
    for arm in ("control", "treatment"):
        source = root / mapping[arm]
        data = yaml.safe_load((source / "train-data.yaml").read_text())
        data["catalog"] = str(source / "catalog.jsonl")
        data["roots"] = str(root / f"configs/{arm}-roots.json")
        data["metadata_cache"] = str(root.parent / "catalog-metadata-cache")
        data["experiment_plan"] = str(source / "plan.json")
        roots = json.loads((source / "roots.json").read_text())
        for alias, raw in roots.items():
            old_path = Path(raw)
            if old_path.is_relative_to(root):
                parts = old_path.relative_to(root).parts
                if parts[0] in mapping:
                    roots[alias] = str(root / mapping[parts[0]] / Path(*parts[1:]))
        outputs.append(
            (root / f"configs/{arm}-roots.json", json.dumps(roots, indent=2) + "\n")
        )
        arm_data[arm] = data
        outputs.append(
            (
                root / f"configs/{arm}-data.yaml",
                yaml.safe_dump(data, allow_unicode=True, sort_keys=False),
            )
        )
    for index, (identifier, directory, title, condition) in enumerate(TASKS):
        python = model_python
        script = (
            "prepare.py",
            "identity.py",
            "synthesize.py",
            "prepare_training.py",
            None,
            None,
            "evaluate.py",
        )[index]
        runtime = {
            "python": path(python),
            "cwd": path("../../.."),
            "pythonpath": [
                path("../../../src"),
                path("../scripts"),
                path(str(reference / "dependencies/audio-data-contract/src")),
            ],
            "gpus": [0, 1] if index in {1, 4, 5, 6} else [],
            "threads": {"omp": 4, "openblas": 1},
            "environment": {
                "TOKENIZERS_PARALLELISM": False,
                "HF_HUB_OFFLINE": "1",
                "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
                "PYTHONUNBUFFERED": "1",
            },
        }
        if index == 2:
            runtime["python"] = path(
                "/222042021/mingdong/workspace/AmphionData/.venv/bin/python"
            )
        config = {
            "version": 1,
            "runtime": runtime,
            "task": {
                "kind": "eval" if index == 6 else "prepare",
                "script": path("../scripts/" + script) if script else None,
                "accepts_config": True,
                "arguments": {},
            },
            "experiment": {
                "root": path(".."),
                "runs": path("../.."),
                "output": path("{attempt}/artifacts"),
                "files": dict(files),
            },
            "parameters": {
                "source_data": path(
                    "/222042021/mingdong/data/sot-multispeaker/events-v1c-20260921"
                ),
                "reference_experiment": path(str(reference)),
            },
            "recording": {"attempts_dir": path("../tasks/" + directory + "/attempts")},
            "tracking": {
                "enabled": True,
                "python": path("../../wandb-sync-env/bin/python"),
                "pythonpath": [path("../../../src")],
                "credentials_file": path("~/.bashrc"),
                "entity": "1016097967-amphion",
                "project": "open-audio-llm",
                "name": root.name + "-" + identifier,
                "interval": 5,
                "metrics": [],
            },
        }
        output_names = [
            [
                "expanded-clean-pool.jsonl.gz",
                "source-turn-policy.json",
                "source-selection.json",
                "inventory.json",
                "prepare-progress.json",
                "roots.json",
            ],
            [
                "embeddings-0.npz",
                "embeddings-1.npz",
                "inputs-0.json",
                "inputs-1.json",
                "progress-0.json",
                "progress-1.json",
                "identity-audit.jsonl",
                "word-pool.jsonl.gz",
                "centroids.npz",
                "centroid-speakers.json",
                "clean-pool-summary.json",
                "roots.json",
                "identity-0.log",
                "identity-1.log",
            ],
            [
                "generated",
                "generation-plan.json",
                "generation-results.jsonl",
                "generation-failures.json",
                "generation-progress.json",
                "release-gate.json",
                "synthesis-summary.json",
                "audit-0",
                "audit-1",
                "audit-2",
            ],
            ["control", "treatment", "matched-data.json", "training-ready.json"],
            [],
            [],
            ["evaluation"],
        ][index]
        for name in output_names:
            config["experiment"]["files"][name] = path("{attempt}/artifacts/" + name)
        if index == 1:
            for worker in (0, 1):
                config["experiment"]["files"][f"identity-{worker}.log"] = path(
                    f"{{attempt}}/logs/identity-{worker}.log"
                )
        # New runs consume the newest completed producer's recorded output; migrated
        # artifacts are the explicit fallback for records predating this launcher.
        producers = {
            name: TASKS[source][1]
            for source in range(index)
            for name in [
                [
                    "expanded-clean-pool.jsonl.gz",
                    "source-turn-policy.json",
                    "inventory.json",
                    "roots.json",
                ],
                [
                    "word-pool.jsonl.gz",
                    "centroids.npz",
                    "centroid-speakers.json",
                    "clean-pool-summary.json",
                    "roots.json",
                ],
                ["generated", "release-gate.json", "synthesis-summary.json"],
                ["control", "treatment", "training-ready.json"],
                [],
                [],
                [],
            ][source]
        }
        config["experiment"]["producers"] = producers
        if index == 0:
            metric = "inventory.json"
        elif index == 1:
            config["experiment"]["copy_inputs"] = ["roots.json"]
            config["parameters"]["identity_model"] = path(
                "../../antspeaker-identity-20260924"
            )
            config["task"]["backend"] = "official_antspeaker_pytorch"
            metric = "clean-pool-summary.json"
        elif index == 2:
            config["parameters"]["tokenizer"] = path(
                "/ai_sds_wuzz/MODELS/Qwen3-ASR-1.7B/Qwen/Qwen3-ASR-1___7B"
            )
            config["task"]["kind"] = "synthesize"
            metric = "synthesis-summary.json"
            config["recording"]["result"] = path(
                "{attempt}/artifacts/release-gate.json"
            )
        elif index == 3:
            metric = "training-ready.json"
            config["recording"]["result"] = path(
                "{attempt}/artifacts/training-ready.json"
            )
            config["parameters"]["training_configs"] = {
                arm: path(f"{arm}-train.yaml") for arm in arm_data
            }
            config["parameters"]["verification_pythonpath"] = (
                str(repo / "src")
                + ":"
                + str(reference / "dependencies/audio-data-contract/src")
            )
        elif index in {4, 5}:
            arm = "control" if index == 4 else "treatment"
            historical = root / mapping[arm]
            args = json.loads((historical / "training/args.json").read_text())
            arguments = {
                "--" + key: args[key]
                for key in TRAIN_PARAMETERS
                if key in args and args[key] is not None
            }
            for key in ("--retention_teacher",):
                if arguments.get(key):
                    arguments[key] = path(arguments[key])
            arguments.update(
                {
                    "--model": path(str(reference / "initial-checkpoint")),
                    "--external_plugins": [
                        path(
                            str(
                                reference
                                / "source/src/open_audio_llm/integrations/ms_swift/register_qwen3_asr.py"
                            )
                        ),
                        path("../scripts/training_audit.py"),
                    ],
                    "--output_dir": path("{attempt}/artifacts/training"),
                    "--report_to": "none",
                }
            )
            config["task"] = {
                "kind": "train",
                "module": "open_audio_llm.integrations.ms_swift.train",
                "positional": ["sft"],
                "arguments": arguments,
            }
            config["data"] = {"config": path(arm + "-data.yaml")}
            config["experiment"]["training_arm"] = arm
            runtime["distributed"] = {
                "processes": 2,
                "port": 29781 if arm == "control" else 29782,
            }
            runtime["threads"] = {"omp": 2, "openblas": 1}
            runtime["environment"]["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
            runtime["pythonpath"][0] = path(str(reference / "source/src"))
            config["preflight"] = [
                {
                    "kind": "prepare",
                    "script": path("../scripts/gpu_preflight.py"),
                    "accepts_config": True,
                }
            ]
            metric = None
        else:
            metric = "evaluation/summary.json"
            config["task"].update(backend="vllm", resume_supported=True)
            config["experiment"]["producers"].update(
                control="05-train-control", treatment="06-train-treatment"
            )
            config["parameters"].update(
                reference_evaluation=path(
                    str(reference / "vllm-full-audio-checkpoint-21000-20260927")
                ),
                inference_python=path(model_python),
                scoring_python=path(
                    "/ai_sds_wuzz/MODELS/miniconda3/envs/sats-asr/bin/python"
                ),
                models={
                    arm: path("../" + mapping[arm] + "/training/checkpoint-1000")
                    for arm in arm_data
                },
            )
            config["parameters"]["models"]["initial"] = path(
                str(reference / "initial-checkpoint")
            )
            config["recording"]["result"] = path(
                "{attempt}/artifacts/evaluation/summary.json"
            )
            config["recording"]["quality_field"] = "supports_larger_followup"
            config["parameters"].update(
                encoder_window=13400,
                inference={
                    "gpu_memory_utilization": 0.8,
                    "max_num_batched_tokens": 16384,
                    "enforce_eager": True,
                    "tensor_parallel_size": 1,
                },
            )
        if metric:
            config["tracking"]["metrics"] = [
                {"file": path("{attempt}/artifacts/" + metric), "prefix": identifier}
            ]
        configs[identifier] = config
        filename = {
            "train-control": "control-train",
            "train-treatment": "treatment-train",
        }.get(identifier, identifier)
        outputs.append(
            (
                root / f"configs/{filename}.yaml",
                yaml.safe_dump(config, allow_unicode=True, sort_keys=False),
            )
        )
        imported = 2 if index in {2, 4} else 1
        for number in range(1, imported + 1):
            attempt = root / "tasks" / directory / "attempts" / f"{number:03d}"
            completed = not (
                (index == 2 and number == 1)
                or (index == 4 and number == 1)
                or index == 6
            )
            status = {
                "execution": "completed" if completed else "failed",
                "quality": "未核实",
                "imported": True,
                "started": "历史记录未提供统一开始时间",
                "ended": "见原始退出或日志记录",
                "exit_code": None,
                "attempt": str(attempt),
                "evidence": "artifacts/ 与 logs/；仅按实际产物和退出记录迁入",
            }
            if index in {4, 5} and completed:
                evidence = json.loads(
                    (attempt / "artifacts/training-exit.json").read_text()
                )
                status["exit_code"] = evidence["returncode"]
                status["ended"] = datetime.fromtimestamp(
                    evidence["time"], timezone.utc
                ).isoformat()
                status["tracking"] = json.loads(
                    (attempt / "artifacts/wandb-verification.json").read_text()
                )
                status["result"] = {"training_steps": 1000}
            if index == 4 and not completed:
                evidence = json.loads(
                    (attempt / "artifacts/legacy/training-exit.json").read_text()
                )
                status["exit_code"] = evidence["returncode"]
                status["ended"] = datetime.fromtimestamp(
                    evidence["time"], timezone.utc
                ).isoformat()
            if index == 2:
                status["quality"] = "通过" if completed else "未达标"
            if index in {0, 1, 3}:
                status["exit_code"] = None
                status["note"] = "有完成产物；历史退出码缺失，不能伪造为 0"
            outputs.append(
                (
                    attempt / "status.json",
                    json.dumps(status, ensure_ascii=False, indent=2) + "\n",
                )
            )
            historical_config = copy.deepcopy(config)
            historical_config["experiment"]["producers"] = {}
            historical_config["experiment"]["copy_inputs"] = []
            historical_config["experiment"]["files"] = dict(files)
            if index == 2 and number == 1:
                historical_config["experiment"]["files"]["generated"] = files[
                    "generated-before-supplement"
                ]
            if index == 4 and number == 1:
                historical_config["experiment"]["files"]["control"] = files[
                    "failed-control-attempt-20260929"
                ]
                saved = (
                    root
                    / mapping["failed-control-attempt-20260929"]
                    / "training/args.json"
                )
                old_args = json.loads(saved.read_text())
                historical_config["task"]["arguments"].update(
                    {
                        "--" + key: old_args[key]
                        for key in TRAIN_PARAMETERS
                        if key in old_args and old_args[key] is not None
                    }
                )
                historical_config["task"]["arguments"]["--output_dir"] = path(
                    str(saved.parent)
                )
            if index == 6:
                historical_config["recording"]["result"] = path(
                    str(root / mapping["evaluation"] / "summary.json")
                )
                historical_config["tracking"]["metrics"][0]["file"] = historical_config[
                    "recording"
                ]["result"]
            historical_config["recording"]["reconstructed"] = {
                "reason": "历史启动没有统一 YAML；按迁移映射和原始参数重建，不能视为原始运行快照",
                "evidence": str(attempt / "artifacts"),
            }
            historical_config["config_file"] = str(root / f"configs/{filename}.yaml")
            historical_config = resolve_values(
                historical_config, root / "configs", attempt
            )
            outputs.append(
                (
                    attempt / "effective.yaml",
                    yaml.safe_dump(
                        historical_config, allow_unicode=True, sort_keys=False
                    ),
                )
            )
            outputs.append(
                (
                    attempt / "README.md",
                    (
                        f"# {title}：执行 {number:03d}\n\n"
                        f"执行状态：{status['execution']}；质量状态：{status['quality']}。\n\n"
                        "这是迁入的历史执行，原始配置、日志和来源证明保持原文。\n\n"
                        "[产物](artifacts/)；[日志](logs/)；[状态](status.json)。\n\n"
                        "[重建配置](effective.yaml) 按原始参数和迁移映射生成，不视为当时保存的原始 YAML。\n"
                    ),
                )
            )
        task_readme = (
            f"# {title}\n\n输入：参见 [配置](../../configs/{filename}.yaml) 中的显式路径和依赖。\n\n"
            f"完成条件：{condition}。\n\n进展：历史执行已归档；执行状态与质量状态见各次记录。\n\n"
            f"运行：`open-audio-llm experiment run --config runs/{root.name}/experiment.yaml --task {identifier}`。\n\n"
            + "\n".join(
                f"- [执行 {number:03d}](attempts/{number:03d}/README.md)"
                for number in range(1, imported + 1)
            )
            + "\n"
        )
        outputs.append((root / "tasks" / directory / "README.md", task_readme))
    manifest = {
        "version": 1,
        "name": root.name,
        "purpose": "固定起点、相同训练配方，对比旧合成数据与自动清洗合成流程",
        "policy": {
            "evaluation_backend": "vllm",
            "identity_backend": "official_antspeaker_pytorch",
            "human_review_required": False,
            "steps_per_arm": 1000,
            "fixed_evaluation_samples": 338,
        },
        "tasks": [],
    }
    for i, (identifier, directory, title, _) in enumerate(TASKS):
        filename = {
            "train-control": "control-train",
            "train-treatment": "treatment-train",
        }.get(identifier, identifier)
        dependencies = [TASKS[i - 1][0]] if i else []
        if i == 6:
            dependencies = ["train-control", "train-treatment"]
        manifest["tasks"].append(
            {
                "id": identifier,
                "name": title,
                "config": f"configs/{filename}.yaml",
                "depends_on": dependencies,
            }
        )
    outputs.append(
        (
            root / "experiment.yaml",
            yaml.safe_dump(manifest, allow_unicode=True, sort_keys=False),
        )
    )
    rows = "\n".join(
        f"| [{title}](tasks/{directory}/README.md) | {'失败，待重试' if i == 6 else '已完成'} | "
        f"{'待核实' if i != 2 else '补充后自动门槛通过'} |"
        for i, (_, directory, title, _) in enumerate(TASKS)
    )
    outputs.append(
        (
            root / "README.md",
            "# 清洗事件数据对照实验\n\n"
            "两组均已完成 1000 步训练；固定集评测失败，尚无最终优劣结论。合成补充后达到 549 条、"
            "11.85 小时、645 个源说话人。声纹阈值为实验规则，未做校准；存疑项隔离，不再发起人工听审。\n\n"
            "| 任务 | 执行状态 | 质量状态 |\n|---|---|---|\n" + rows + "\n\n"
            "两组共享起始权重、训练设置和固定 338 条评测，只替换 35% 的事件数据。结果只能归因于"
            "来源质量与合成流程的整体变更。\n\n"
            "运行或预览：\n\n```bash\n"
            f"open-audio-llm experiment run --config runs/{root.name}/experiment.yaml --dry-run\n"
            f"open-audio-llm experiment run --config runs/{root.name}/experiment.yaml --task evaluate\n```\n\n"
            "[原始方案](shared/plan.json)、[原始概览](shared/history/README.md)、"
            "[迁移核验](tracking/migration-verification.json)。原始总控中的“未开始训练”与训练退出和 W&B 记录冲突，"
            "保留原文并标记待核实；不凭 PID 文件判断运行中。\n",
        )
    )
    outputs.append(
        (
            root / "tracking/README.md",
            (
                "# 实验状态与同步记录\n\n"
                "迁移核验通过，全部旧文件按大小和 inode 对照，未复制权重。\n\n"
                "[映射](migration-plan.json)、[迁移前清单](migration-before.json)、[核验](migration-verification.json)。\n\n"
                "历史状态与 W&B 核验在 [history/](history/)；进程日志在 [logs/](logs/)。"
                "其中旧“running”及“training_started=false”记录已过期或有冲突，不能视为当前状态。"
                "新执行由统一入口记录，W&B 远端核验文件在对应 attempts 目录。\n"
            ),
        )
    )
    return outputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--index", type=int)
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--records-only", action="store_true")
    options = parser.parse_args()
    outputs = generate(options.root.absolute())
    if options.records_only:
        outputs = [(p, c) for p, c in outputs if p.name == "effective.yaml"]
    if options.list:
        print(
            json.dumps(
                [
                    {"index": i, "path": str(p), "bytes": len(c)}
                    for i, (p, c) in enumerate(outputs)
                ]
            )
        )
        return
    if options.index is not None:
        outputs = [outputs[options.index]]
    if options.write:
        for filename, content in outputs:
            if filename.exists() and filename.read_text() != content:
                raise FileExistsError(filename)
        for filename, content in outputs:
            filename.parent.mkdir(parents=True, exist_ok=True)
            filename.write_text(content)
        print(json.dumps({"generated_files": len(outputs)}))
        return
    patch = "*** Begin Patch\n"
    for filename, content in outputs:
        patch += (
            f"*** Add File: {filename}\n"
            + "\n".join("+" + line for line in content.splitlines())
            + "\n"
        )
    print(json.dumps(patch + "*** End Patch"))


if __name__ == "__main__":
    main()
