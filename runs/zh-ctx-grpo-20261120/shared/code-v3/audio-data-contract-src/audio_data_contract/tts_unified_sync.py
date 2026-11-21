"""Register DATA-TTS-UNIFIED publications listed in its PUBLICATIONS.md overview.

Each listed base release becomes `<dataset_id>@tts-unified-<release_id>`. Each
listed annotation run is attached to the releases it covers as a
`tts-lance-annotation` artifact, under a train-split role named after its task.
Runs that are complete but not yet listed are left out, and runs removed from
the overview disappear on the next sync.

The generated tts_unified.yaml replaces the copy in a catalog directory only
after that directory, with the new file in place, still validates:

    audio-data-contract sync-tts-unified /workspace/data \\
        --catalog-dir /workspace/data/registry/catalog \\
        --views /workspace/data/registry/views \\
        --roots /workspace/data/registry/roots.json

With --notify, a failure sends one Feishu card (webhook as in storage-audit)
per distinct error, and the next successful run sends a recovery card.

ASR notes come from the 2026-10-09 text survey
(reports/tts-unified-asr-20261009/README.md).
"""

from __future__ import annotations

import hashlib
import io
import json
import sys
from pathlib import Path

from ruamel.yaml import YAML

from .catalog import validate_catalog
from .declarations import declaration_files, read_declarations
from .errors import ContractError
from .roots import load_roots
from .storage_audit import _card, notify_feishu
from .types import DatasetSpec
from .views import load_view_catalog

UNIFIED = "DATA-TTS-UNIFIED"
ROOT_ALIAS = "aidc_data"
FILENAME = "tts_unified.yaml"
OVERVIEW = "## 当前发布总览"
SOURCE_DIRS = {
    "aishell3": "AISHELL-3",
    "csemotions": "CSEMOTIONS",
    "emilia": "Emilia",
    "emilia2": "Emilia2",
    "emilia_yodas": "Emilia-YODAS",
    "galgame": "Galgame-VisualNovel-Reupload",
    "genshin_voice": "genshin-voice",
    "hifitts": "HiFiTTS",
    "hifitts2": "HiFiTTS2",
    "libriheavy": "libriheavy",
    "libritts_r": "LibriTTS-R",
    "ljspeech": "LJSpeech",
    "mls_sidon": "mls_sidon",
    "starrail_voice": "starrail-voice",
    "vctk": "VCTK",
    "wenetspeech4tts": "WenetSpeech4TTS",
    "wutheringwaves": "WutheringWaves-2.2",
    "zenless_voice": "zenless-voice",
}
# Datasets usable for ASR after the cleaning listed in the survey report.
ASR = {
    "aishell3": "人工转写；须排除 original_split=test（与 AISHELL-3 测试集相同）",
    "hifitts": "有声书原文；须排除 original_split=dev/test",
    "hifitts2": "有声书原文；须排除 dev_*/test_* 划分",
    "libriheavy": "书本原文（带大小写和标点），text_variants 另有 source_asr；超过 30 秒的约 44 万条需切分或过滤",
    "libritts_r": "须排除 dev/test（来自 LibriSpeech dev/test，会污染 LibriSpeech 评测）；音频经修复增强",
    "ljspeech": "单说话人朗读，数字未展开",
    "mls_sidon": "须排除 valid/test（与 MLS 评测集相同）；音频经 Sidon 修复",
    "vctk": "朗读文本；172 条无文本",
    "wenetspeech4tts": "来自 WenetSpeech 的弱标注，建议做置信度或双模型一致性过滤",
    "emilia": "机器转写，须做质量过滤；文本开头带空格",
    "emilia_yodas": "机器转写，须做质量过滤；文本开头带空格",
}

# Upstream defects left for tts-data-pipeline to fix; ASR rules only exclude them.
KNOWN_ISSUES = {
    "starrail_voice": [
        (
            "en/ja/ko 配音约 94% 配的是中文文本（en 中文 78,864 条、英文 4,786 条），疑为接入时各语言共用了中文台词；"
            "待 tts-data-pipeline 修复，本仓库不改，ASR 规则按 script_mismatch 排除"
        ),
    ],
}

# Registration history that the manifests themselves do not record.
NOTES = {
    "emilia2": {
        "manifest_correction": (
            "2026-10-12 发布方修正 manifest，只补 spec-10 字段（finished_at、code_sha256、speakers 等），"
            "Lance 表与快照版本 3 不变；原 manifest sha256 "
            "24779fb04467d1926d5ed7e90b310cffd1101059de6406c9eb6fbe424a1102c7（7,290,534 字节）"
            "记录在 manifest 的 manifest_correction 字段。登记于 2026-10-18 改为修正后的 sha256，"
            "此前按旧值无法打开本版本。"
        ),
    },
}


# Overview summaries (provenance.summary) for datasets declared only in this file.
SUMMARIES = {
    "csemotions": "中文情感语音（CSEMOTIONS），TTS 用",
    "emilia_yodas": "Emilia-YODAS 多语种 YouTube 语音",
    "galgame": "日语视觉小说配音，TTS 用",
    "genshin_voice": "《原神》多语配音，TTS 用",
    "hifitts": "英文有声书高保真朗读（Hi-Fi TTS）",
    "hifitts2": "英文有声书高保真朗读（Hi-Fi TTS 2）",
    "libriheavy": "英文有声书，带标点和大小写（Libriheavy）",
    "libritts_r": "英文有声书 TTS 语料，音频经修复（LibriTTS-R）",
    "ljspeech": "单人英文朗读（LJSpeech）",
    "mls_sidon": "8 种语言有声书，音频经 Sidon 修复（MLS）",
    "starrail_voice": "《崩坏：星穹铁道》多语配音，TTS 用",
    "vctk": "多口音英语朗读（VCTK）",
    "wutheringwaves": "《鸣潮》多语配音，TTS 用",
    "zenless_voice": "《绝区零》多语配音，TTS 用",
}


# Raw source copies in COS. Each release manifest input was found there with the
# same size on 2026-10-26; the LibriHeavy/ directory holds other files.
COS_SOURCES = {
    "aishell3": "cos://amphion-tts-data-1373259071/AISHELL-3/",
    "csemotions": "cos://amphion-tts-data-1373259071/CSEMOTIONS/",
    "emilia": "cos://amphion-tts-data-1373259071/Emilia/",
    "emilia2": "cos://amphion-tts-data-1373259071/Emilia2/",
    "emilia_yodas": "cos://amphion-tts-data-1373259071/Emilia-YODAS/",
    "galgame": "cos://amphion-tts-data-1373259071/Galgame-VisualNovel-Reupload/",
    "genshin_voice": "cos://amphion-tts-data-1373259071/genshin-voice/",
    "hifitts": "cos://amphion-tts-data-1373259071/HiFiTTS/",
    "libritts_r": "cos://amphion-tts-data-1373259071/LibriTTS-R/",
    "ljspeech": "cos://amphion-tts-data-1373259071/LJSpeech/",
    "starrail_voice": "cos://amphion-tts-data-1373259071/starrail-voice/",
    "vctk": "cos://amphion-tts-data-1373259071/VCTK/",
    "wenetspeech4tts": "cos://amphion-tts-data-1373259071/WenetSpeech4TTS/",
    "wutheringwaves": "cos://amphion-tts-data-1373259071/WutheringWaves-2.2/",
    "zenless_voice": "cos://amphion-tts-data-1373259071/zenless-voice/",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def published(unified: Path) -> tuple[list[tuple[str, str]], list[str]]:
    """(dataset_id, release_id) pairs and annotation run IDs in the overview table."""
    path = unified / "PUBLICATIONS.md"
    text = path.read_text(encoding="utf-8")
    _, heading, rest = text.partition(f"\n{OVERVIEW}\n")
    # The next section heading shows the table was not cut short mid-write.
    table, following, _ = rest.partition("\n## ")
    if not heading or not following:
        raise ContractError(f"{path}: no complete {OVERVIEW!r} section")
    releases: list[tuple[str, str]] = []
    annotations: list[str] = []
    for line in table.splitlines():
        if not line.startswith("|"):
            continue
        cells = [cell.strip().strip("`") for cell in line.strip().strip("|").split("|")]
        if cells[0] == "基础数据集":
            dataset_id, _, release_id = cells[1].partition("/")
            releases.append((dataset_id, release_id))
        elif cells[0] == "Annotation":
            annotations.append(cells[1])
    if not releases:
        raise ContractError(f"{path}: the overview lists no base releases")
    return releases, annotations


def release_entry(data_root: Path, dataset_id: str, release_id: str) -> dict:
    relative = f"{UNIFIED}/datasets/{dataset_id}/{release_id}/manifest.json"
    path = data_root / relative
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if (
        manifest.get("status") != "complete"
        or manifest.get("dataset_id") != dataset_id
        or manifest.get("release_id") != release_id
    ):
        raise ContractError(f"not a complete {dataset_id}/{release_id} release: {path}")
    languages = sorted(k for k in manifest["languages"] if k not in ("null", None))
    asr = ASR.get(dataset_id)
    provenance = {
        "source": UNIFIED,
        "contract": f"{UNIFIED}/CONTRACT.md",
        "contract_version": manifest["contract_version"],
        "release_id": release_id,
        # Older releases carry release_digest; emilia2's ingest-profile manifest
        # identifies itself by ingest_profile_id/input_manifest_sha256.
        **{
            key: manifest[key]
            for key in ("release_digest", "ingest_profile_id", "input_manifest_sha256")
            if key in manifest
        },
        **(
            {"source_directory": f"DATA-TTS/{SOURCE_DIRS[dataset_id]}"}
            if dataset_id in SOURCE_DIRS
            else {}
        ),
        "source_files": len(manifest["inputs"]),
        "excluded_source_files": [
            f["path"] for f in manifest.get("excluded_source_files", [])
        ],
        **{key: manifest[key] for key in ("finished_at",) if key in manifest},
        "description": "音频编码字节存放在 Lance 表内；原始划分在 metadata_json.original_split，统一输出全部为 train。ASR 用 tts_lance.iter_asr_samples（tts-asr-rules/v1）读取，会排除原始 dev/test 等。",
    }
    if dataset_id in SUMMARIES:
        provenance["summary"] = SUMMARIES[dataset_id]
    if dataset_id in COS_SOURCES:
        provenance["cos_locations"] = [{"uri": COS_SOURCES[dataset_id], "content": "source"}]
    if asr:
        provenance["asr_notes"] = asr
    if dataset_id in KNOWN_ISSUES:
        provenance["known_issues"] = KNOWN_ISSUES[dataset_id]
    provenance.update(NOTES.get(dataset_id, {}))
    return {
        "schema_version": "dataset-catalog/2.0",
        "dataset_id": dataset_id,
        "version": f"tts-unified-{release_id}",
        "languages": languages,
        "tasks": ["tts", "asr"] if asr else ["tts"],
        "artifacts": [
            {
                "name": "samples",
                "kind": "tts-lance-release",
                "root_alias": ROOT_ALIAS,
                "relative_path": relative,
                "expected_bytes": path.stat().st_size,
                "sha256": _sha256(path),
                "metadata": {
                    "lance_version": manifest["lance_version"],
                    "rows": manifest["rows"],
                    "schema_sha256": manifest["schema_sha256"],
                    "storage_version": manifest["storage_version"],
                },
            }
        ],
        "splits": {
            "train": {
                "artifacts": {"samples": ["samples"]},
                "statistics": {
                    "rows": manifest["rows"],
                    "duration_hours": round(manifest["duration_seconds"] / 3600, 1),
                },
            }
        },
        "provenance": provenance,
    }


def annotation_artifacts(data_root: Path, run_id: str):
    """Yield (dataset_id, release_id, task, artifact) for each table of one run."""
    matches = sorted((data_root / UNIFIED / "annotations").glob(f"*/{run_id}/manifest.json"))
    if len(matches) != 1:
        raise ContractError(f"annotation {run_id}: expected one manifest, found {len(matches)}")
    path = matches[0]
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("status") != "complete" or manifest.get("run_id") != run_id:
        raise ContractError(f"not a complete annotation {run_id}: {path}")
    size, digest = path.stat().st_size, _sha256(path)
    for output in manifest["outputs"]:
        table = output["table"]
        yield output["dataset_id"], output["release_id"], manifest["task"], {
            "name": run_id,
            "kind": "tts-lance-annotation",
            "root_alias": ROOT_ALIAS,
            "relative_path": path.relative_to(data_root).as_posix(),
            "expected_bytes": size,
            "sha256": digest,
            "metadata": {
                "task": manifest["task"],
                "schema_version": manifest["schema_version"],
                "profile": manifest["profile"]["name"],
                # Relative to the root alias, like relative_path.
                "table_path": f"{UNIFIED}/{table['table_path']}",
                **({"branch": table["branch"]} if table.get("branch") else {}),
                "lance_version": table["lance_version"],
                "rows": table["rows"],
                "coverage": output["coverage"],
            },
        }


def build_entries(data_root: Path) -> list[dict]:
    releases, annotations = published(data_root / UNIFIED)
    entries = {
        key: release_entry(data_root, *key) for key in sorted(set(releases))
    }
    attached = []
    for run_id in annotations:
        for dataset_id, release_id, task, artifact in annotation_artifacts(data_root, run_id):
            if (dataset_id, release_id) not in entries:
                raise ContractError(
                    f"annotation {run_id} covers {dataset_id}/{release_id}, "
                    "which the overview does not list"
                )
            attached.append((task, run_id, entries[(dataset_id, release_id)], artifact))
    for task, _, entry, artifact in sorted(attached, key=lambda item: item[:2]):
        entry["artifacts"].append(artifact)
        entry["splits"]["train"]["artifacts"].setdefault(task, []).append(artifact["name"])
    return list(entries.values())


def render(entries: list[dict]) -> str:
    yaml = YAML()
    yaml.indent(mapping=2, sequence=4, offset=2)
    yaml.width = 4096
    output = io.StringIO()
    yaml.dump(entries, output)
    return output.getvalue()


def _artifact_digests(rows) -> dict[str, str]:
    return {
        f"{row['dataset_id']}@{row['version']}:{artifact['name']}": artifact.get("sha256")
        for row in rows
        for artifact in row["artifacts"]
    }


def sync(
    data_root: str | Path,
    catalog_dir: str | Path,
    *,
    views: str | Path | None = None,
    roots: str | Path | None = None,
    dry_run: bool = False,
) -> dict:
    """Rewrite catalog_dir/tts_unified.yaml when the publications changed."""
    data_root, catalog_dir = Path(data_root), Path(catalog_dir)
    if roots is not None and Path(load_roots(roots).get(ROOT_ALIAS, "")) != data_root.resolve():
        raise ContractError(f"root alias {ROOT_ALIAS!r} in {roots} is not {data_root}")
    entries = build_entries(data_root)
    text = render(entries)
    target = catalog_dir / FILENAME
    old = target.read_text(encoding="utf-8") if target.exists() else ""
    before = _artifact_digests(row for _, row in read_declarations(target)) if old else {}
    after = _artifact_digests(entries)
    summary = {
        "target": str(target),
        "changed": text != old,
        "datasets": len(entries),
        "added": sorted(after.keys() - before.keys()),
        "removed": sorted(before.keys() - after.keys()),
        "updated": sorted(k for k in after.keys() & before.keys() if after[k] != before[k]),
    }
    if text == old or dry_run:
        return summary
    # Validate the directory as it will be after the swap, then swap atomically.
    specs = [DatasetSpec.from_dict(row) for row in entries]
    for source in declaration_files(catalog_dir):
        if source.name == FILENAME:
            continue
        for line_number, row in read_declarations(source):
            try:
                specs.append(DatasetSpec.from_dict(row))
            except ContractError as exc:
                raise ContractError(f"{source}:{line_number}: {exc}") from exc
    catalog = validate_catalog(specs)
    if views is not None:
        load_view_catalog(views, catalog)
    temporary = target.with_name(f".{FILENAME}.tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(target)
    return summary


def _notify(title: str, content: str, template: str) -> bool:
    try:
        notify_feishu(_card(title, [{"tag": "markdown", "content": content}], template))
    except (ContractError, OSError, ValueError) as exc:  # keep the sync outcome visible
        print(f"Feishu notification failed: {exc}", file=sys.stderr)
        return False
    return True


def _alert(state: Path | None, error: str) -> None:
    """Notify once per distinct error; state holds the last error notified."""
    if state is not None and state.exists() and state.read_text(encoding="utf-8") == error:
        return
    sent = _notify(
        "DATA-TTS-UNIFIED 自动登记失败",
        "registry 里的 tts_unified.yaml 没有改动，仍是上次成功同步的版本。\n"
        f"**原因**：{error[:1000]}\n同一个错误只提醒一次，恢复后会再通知。",
        "red",
    )
    if sent and state is not None:
        state.write_text(error, encoding="utf-8")


def _recovered(state: Path | None, summary: dict) -> None:
    if state is None or not state.exists():
        return
    counts = {key: len(summary[key]) for key in ("added", "removed", "updated")}
    if _notify(
        "DATA-TTS-UNIFIED 自动登记已恢复",
        "同步成功：新增 {added}、删除 {removed}、更新 {updated} 个产物。".format(**counts),
        "green",
    ):
        state.unlink()


def run(args) -> int:
    state = Path(args.state) if args.state else None
    try:
        summary = sync(
            args.data_root,
            args.catalog_dir,
            views=args.views,
            roots=args.roots,
            dry_run=args.dry_run,
        )
    except Exception as exc:
        if args.notify:
            _alert(state, f"{type(exc).__name__}: {exc}")
        raise
    if args.notify:
        _recovered(state, summary)
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0
