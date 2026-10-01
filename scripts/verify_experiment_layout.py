"""Verify preserved pilot files, links, training manifests and public dry-run configs."""

import argparse
import contextlib
import hashlib
import io
import json
import os
from pathlib import Path

import yaml
from migrate_experiments import remap

from open_audio_llm.cli import main as launch


def verify(root):
    before = json.loads((root / "tracking/migration-before.json").read_text())
    mapping = json.loads((root / "tracking/migration-plan.json").read_text())["mapping"]
    preserved = links = 0
    for row in before:
        old = root / row["path"]
        new = remap(old, root, mapping)
        if "link" in row:
            target = Path(row["link"])
            target = target if target.is_absolute() else old.parent / target
            expected = remap(target, root, mapping)
            assert new.is_symlink(), new
            assert Path(os.path.abspath(new.parent / os.readlink(new))) == expected, new
            links += 1
        else:
            if len(old.relative_to(root).parts) == 1 and old.suffix == ".py":
                new = root / "shared/legacy-scripts" / old.name
                assert new.stat().st_size == row["bytes"], new
            else:
                actual = new.stat()
                assert (actual.st_size, actual.st_ino, actual.st_dev) == (
                    row["bytes"],
                    row["inode"],
                    row["device"],
                ), new
            preserved += 1
    allowed = {
        "README.md",
        "experiment.yaml",
        "configs",
        "scripts",
        "tasks",
        "shared",
        "tracking",
    }
    assert {p.name for p in root.iterdir()} == allowed
    manifests = []
    ready = root / mapping["training-ready.json"]
    evidence = json.loads(ready.read_text())
    for arm in ("control", "treatment"):
        folder = root / mapping[arm]
        manifest = folder / "event-data/records.jsonl.gz"
        digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
        assert digest == evidence["arms"][arm]["manifest_sha256"], manifest
        data = yaml.safe_load((root / f"configs/{arm}-data.yaml").read_text())
        assert Path(data["catalog"]).is_file() and Path(data["roots"]).is_file()
        roots = json.loads(Path(data["roots"]).read_text())
        assert all(
            Path(value).exists()
            for key, value in roots.items()
            if key.startswith("events_")
        )
        manifests.append({"arm": arm, "sha256": digest})
    recipes = []
    repo = root.parent.parent
    for base in (repo / "examples/configs", root / "configs"):
        for path in sorted(base.rglob("*.yaml")):
            config = yaml.safe_load(path.read_text())
            if not isinstance(config, dict) or config.get("version") != 1:
                continue
            action = "deploy" if "compose" in config else config["task"]["kind"]
            with contextlib.redirect_stdout(io.StringIO()):
                assert launch([action, "--config", str(path), "--dry-run"]) == 0, path
            recipes.append(str(path.relative_to(repo)))
    attempt = root / "tasks/07-evaluate/attempts/001"
    with contextlib.redirect_stdout(io.StringIO()):
        assert (
            launch(
                [
                    "experiment",
                    "run",
                    "--config",
                    str(root / "experiment.yaml"),
                    "--task",
                    "evaluate",
                    "--resume",
                    str(attempt),
                    "--dry-run",
                ]
            )
            == 0
        )
    for record in root.glob("tasks/*/attempts/*"):
        settings = yaml.safe_load((record / "effective.yaml").read_text())
        assert settings["recording"]["reconstructed"], record
        assert (record / "status.json").is_file() and (record / "README.md").is_file()
    return {
        "passed": True,
        "preserved_files": preserved,
        "links": links,
        "bytes": sum(r.get("bytes", 0) for r in before),
        "manifests": manifests,
        "dry_run_configs": recipes,
        "resume_preview": True,
        "weights_copied": False,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    options = parser.parse_args()
    print(json.dumps(verify(options.root.absolute()), ensure_ascii=False, indent=2))
