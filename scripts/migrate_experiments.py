"""Move the clean-events pilot without copying weights; document other historical runs."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

TASKS = [
    (
        "source-expansion",
        "01-source-expansion",
        "扩充清洗来源",
        "来源、原文、版本和人工否定记录均可追溯",
    ),
    (
        "identity-screening",
        "02-identity-screening",
        "声纹筛查",
        "官方 AntSpeaker 自动核验；存疑项隔离",
    ),
    (
        "synthesis",
        "03-synthesis",
        "合成与补充",
        "至少 500 条、10 小时、300 个源说话人，覆盖 30/120/300 秒",
    ),
    (
        "prepare-training",
        "04-prepare-training",
        "准备对照训练数据",
        "两组样本、语言、时长、人数、场景匹配；数据核验通过",
    ),
    (
        "train-control",
        "05-train-control",
        "训练旧数据对照组",
        "1000 步训练完成，有退出码、模型和 W&B 证据",
    ),
    (
        "train-treatment",
        "06-train-treatment",
        "训练新数据实验组",
        "1000 步训练完成，有退出码、模型和 W&B 证据",
    ),
    (
        "evaluate",
        "07-evaluate",
        "固定集评测",
        "vLLM 完成 338 条、三组同条件评测，保留解析失败",
    ),
]


def destination(name):
    def target(task, attempt, area="artifacts", filename=None):
        return f"tasks/{TASKS[task - 1][1]}/attempts/{attempt:03d}/{area}/{filename or name}"

    if name == "README.md":
        return "shared/history/README.md"
    if name == "__pycache__":
        return "shared/history/__pycache__"
    if name.endswith(".py"):
        return "scripts/" + name
    if name in {"synthesis-source", "synthesis-source-supplement"}:
        return "shared/" + name
    if name == "failed-control-attempt-20260929":
        return target(5, 1, filename="legacy")
    if name in {"control", "treatment"}:
        return f"tasks/{TASKS[4 if name == 'control' else 5][1]}/attempts/{2 if name == 'control' else 1:03d}/artifacts"
    if name == "evaluation":
        return "tasks/07-evaluate/attempts/001/artifacts"
    if name == "generated-before-supplement":
        return target(3, 1, filename="generated")
    if name == "generated":
        return target(3, 2)
    if name.startswith("audit-"):
        return target(3, 2 if name == "audit-3" else 1)
    if name in {"roots.json", "plan.json", "shared-registration-verification.json"}:
        return "shared/" + name
    if name.startswith(
        ("track", "metrics-history", "wandb-", "status", "controller", "resume")
    ) or name in {"failed-resume.json", "failed-training-resume.json"}:
        return (
            "tracking/"
            + ("logs/" if name.endswith((".log", ".pid")) else "history/")
            + name
        )
    if name.startswith(("prepare_training", "matched-data", "training-ready")):
        return target(4, 1, "logs" if name.endswith(".log") else "artifacts")
    if name.startswith(("train_control", "train_treatment")):
        return target(
            5 if name.startswith("train_control") else 6,
            2 if name.startswith("train_control") else 1,
            "logs",
        )
    if name.startswith("evaluate"):
        return target(7, 1, "logs")
    if name.startswith(
        (
            "prepare",
            "inventory",
            "source-selection",
            "source-turn-policy",
            "expanded-clean-pool",
        )
    ):
        return target(1, 1, "logs" if name.endswith((".log", ".pid")) else "artifacts")
    if name.startswith(
        (
            "identity",
            "embeddings",
            "inputs",
            "progress-",
            "centroid",
            "clean-pool",
            "screen",
            "word-pool",
        )
    ):
        return target(2, 1, "logs" if name.endswith(".log") else "artifacts")
    if name.startswith(
        (
            "generate",
            "generation",
            "release-gate",
            "synthesis-summary",
            "supplement",
            "failed-initial-synthesis",
        )
    ):
        attempt = (
            1
            if name
            in {
                "generate.log",
                "generation-failures.json",
                "generation-plan.json",
                "generation-results.jsonl",
                "release-gate-before-supplement.json",
                "synthesis-summary-before-supplement.json",
                "failed-initial-synthesis.json",
            }
            else 2
        )
        return target(3, attempt, "logs" if name.endswith(".log") else "artifacts")
    raise ValueError(f"Unclassified experiment entry: {name}")


def inventory(root):
    records = []
    for item in sorted(root.rglob("*")):
        if item.is_symlink():
            records.append(
                {"path": str(item.relative_to(root)), "link": os.readlink(item)}
            )
        elif item.is_file():
            stat = item.stat()
            records.append(
                {
                    "path": str(item.relative_to(root)),
                    "bytes": stat.st_size,
                    "inode": stat.st_ino,
                    "device": stat.st_dev,
                }
            )
    return records


def remap(path, root, mapping):
    path = Path(os.path.abspath(path))
    if not path.is_relative_to(root):
        return path
    relative = path.relative_to(root)
    if relative.parts and relative.parts[0] in mapping:
        return root / mapping[relative.parts[0]] / Path(*relative.parts[1:])
    return path


def in_use(root):
    ancestors = set()
    pid = os.getpid()
    while pid > 1:
        ancestors.add(pid)
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        pid = int(fields[1])
    found = []
    for proc in Path("/proc").glob("[0-9]*"):
        if int(proc.name) in ancestors:
            continue
        try:
            cmd = (proc / "cmdline").read_bytes().replace(b"\0", b" ")
            paths = [proc / "cwd", *(proc / "fd").iterdir()]
            used = str(root).encode() in cmd or any(
                os.path.abspath(os.readlink(p)).startswith(str(root) + "/")
                for p in paths
            )
            if used:
                # Report only PID; command lines can contain credentials.
                found.append(int(proc.name))
        except (OSError, PermissionError):
            continue
    return found


def dump(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")


def migrate(root, apply=False):
    root = root.absolute()
    marker = root / "tracking/migration-verification.json"
    if marker.exists():
        raise ValueError(
            "Migration already completed; use its inventory to inspect the result"
        )
    mapping = {p.name: destination(p.name) for p in sorted(root.iterdir())}
    processes = in_use(root)
    before = inventory(root)
    plan = {
        "root": str(root),
        "mapping": mapping,
        "in_use_pids": processes,
        "files": len(before),
        "bytes": sum(r.get("bytes", 0) for r in before),
    }
    if not apply:
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return
    if processes:
        raise ValueError(
            f"Artifacts are in use by PIDs {processes}; migrate after those processes exit"
        )
    for old, new in mapping.items():
        if (root / new).exists() or (root / new).is_symlink():
            raise FileExistsError(new)
    dump(root / "tracking/migration-plan.json", plan)
    dump(root / "tracking/migration-before.json", before)
    for old, new in mapping.items():
        source, target = root / old, root / new
        target.parent.mkdir(parents=True, exist_ok=True)
        source.rename(target)
    repairs = []
    for row in before:
        if "link" not in row:
            continue
        old = root / row["path"]
        new = remap(old, root, mapping)
        old_target = Path(row["link"])
        old_target = old_target if old_target.is_absolute() else old.parent / old_target
        target = remap(old_target, root, mapping)
        # Relative links need adjustment even when their target is outside this experiment.
        text = (
            str(target)
            if Path(row["link"]).is_absolute()
            else os.path.relpath(target, new.parent)
        )
        if text != row["link"]:
            new.unlink()
            new.symlink_to(text)
            repairs.append({"path": str(new), "before": row["link"], "after": text})
    missing = []
    for row in before:
        new = remap(root / row["path"], root, mapping)
        if "link" in row:
            if not new.is_symlink():
                missing.append(row["path"])
        elif (
            not new.is_file()
            or new.stat().st_size != row["bytes"]
            or new.stat().st_ino != row["inode"]
        ):
            missing.append(row["path"])
    verification = {
        "passed": not missing,
        "files": len(before),
        "bytes": plan["bytes"],
        "missing_or_changed": missing,
        "symlink_repairs": repairs,
        "weights_copied": False,
        "in_use_pids": processes,
    }
    dump(marker, verification)
    if missing:
        raise RuntimeError(f"Migration verification failed: {missing}")
    # Preserve source bytes before updating executable copies.
    snapshot = root / "shared/legacy-scripts"
    snapshot.mkdir()
    for item in (root / "scripts").glob("*.py"):
        shutil.copy2(item, snapshot / item.name)
    print(
        json.dumps(
            {
                k: verification[k]
                for k in ("passed", "files", "bytes", "weights_copied")
            },
            ensure_ascii=False,
        )
    )


def overview(runs):
    excluded = {"catalog-metadata-cache", "wandb-sync-env", "wandb-sync"}
    labels = {
        "sot": "多说话人转写",
        "ts": "目标说话人识别",
        "hotwords": "热词识别",
        "identity": "说话人身份核验",
        "synthesis": "语音合成",
        "clean": "数据清洗",
        "release": "发布候选对照",
        "performance": "训练性能",
        "grpo": "GRPO 训练",
    }
    count = 0
    for root in sorted(runs.iterdir()):
        if (
            not root.is_dir()
            or root.name in excluded
            or root.name == "clean-events-ab-20260928"
        ):
            continue
        readme = root / "README.md"
        marker = "<!-- experiment-navigation -->"
        prior = readme.read_text() if readme.exists() else ""
        if marker in prior:
            continue
        purpose = next(
            (text for word, text in labels.items() if word in root.name),
            "实验与分析记录",
        )
        entries = sorted(
            p
            for p in root.iterdir()
            if p.name not in {"README.md", "__pycache__", "tasks"}
        )
        sections = []
        for label, predicate in (
            ("执行入口", lambda p: p.suffix in {".py", ".sh"}),
            ("配置与方案", lambda p: p.suffix in {".yaml", ".yml"} or "plan" in p.name),
            (
                "结果与证据",
                lambda p: (
                    "summary" in p.name
                    or "verification" in p.name
                    or "status" in p.name
                ),
            ),
            (
                "产物目录",
                lambda p: (
                    p.is_dir() and p.name not in {"source", "dependencies", "wandb"}
                ),
            ),
        ):
            paths = [p for p in entries if predicate(p)]
            sections.append(
                f"| {label} | "
                + (
                    "、".join(f"[{p.name}](../{p.name})" for p in paths)
                    or "未发现根层记录"
                )
                + " |"
            )
        task_dir = root / "tasks"
        task_dir.mkdir(exist_ok=True)
        (task_dir / "README.md").write_text(
            f"# {purpose}：任务导航\n\n输入与完成条件沿用原方案和配置；下表链接到保留原位的记录。\n\n"
            "| 工作 | 入口与证据 |\n|---|---|\n"
            + "\n".join(sections)
            + "\n\n进展与结论：历史状态未重新核实，不能根据目录或 PID 文件判定正在运行。"
            "有 summary、verification 时以其中原始结果为依据；没有结果时不视为通过。\n"
        )
        navigation = (
            f"# {purpose}：{root.name}\n\n{marker}\n\n"
            "这是历史实验，产物保留原位。输入、完成条件、进展和结果见"
            "[任务导航](tasks/README.md)及原始方案；本次仅补齐导航，未重跑或重判质量。\n\n"
            "执行状态：待核实；质量状态：沿用已有证据，缺少记录时保持未核实。\n\n"
        )
        readme.write_text(
            navigation + "## 原始概览\n\n" + prior
            if prior
            else navigation.rstrip() + "\n"
        )
        count += 1
    print(json.dumps({"historical_overviews_updated": count}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--historical-runs", type=Path)
    args = parser.parse_args()
    if args.root:
        migrate(args.root, args.apply)
    if args.historical_runs:
        overview(args.historical_runs)


if __name__ == "__main__":
    main()
