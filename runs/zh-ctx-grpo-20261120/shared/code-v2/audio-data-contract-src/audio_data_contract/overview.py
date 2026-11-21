"""Deterministic, human-readable overview of registered data."""

from __future__ import annotations

import json
import posixpath
import re
from collections import defaultdict
from dataclasses import dataclass
from html import escape
from pathlib import Path

from .catalog import load_catalog
from .declarations import declaration_files, read_declarations
from .errors import ContractError
from .types import DatasetSpec
from .views import load_view_catalog

TASK_NAMES = {
    "augmentation": "音频增强",
    "asr": "语音识别",
    "asr_hotwords": "热词增强语音识别",
    "ast": "语音翻译",
    "code_switch_asr": "语码转换语音识别",
    "continuous_speech_separation": "连续语音分离",
    "esc": "背景声场景描述",
    "overlap_speech": "重叠语音检测",
    "sec": "说话人情感分类",
    "sepc": "情感与说话风格描述",
    "ser": "语音情感识别",
    "speaker_attributed_asr": "带说话人标注的语音识别",
    "speaker_diarization": "说话人分段与归属",
    "speaker_identification": "说话人身份识别",
    "speaker_verification": "声纹验证",
    "ts_asr": "目标说话人语音识别",
    "tts": "语音合成",
}

SECTIONS = ("英文数据", "中文及粤语数据", "混合语言及多语数据", "其他语言数据")

QUALIFIERS = {
    "reported": "已登记",
    "nominal": "估算",
    "partial": "部分划分",
    "before_filter": "过滤前",
}

# Short bucket names for the overview's COS column; the versions page spells them out.
COS_BUCKETS = {
    "amphion-audio-backup-1373259071": "备份桶",
    "amphion-tts-data-1373259071": "TTS 桶",
}

# snapshot: a backup binding; files: a copy of the version's own files;
# source: the raw data the version was built from.
COS_KINDS = ("snapshot", "files", "source")


@dataclass(frozen=True)
class DurationSummary:
    """A duration that can be included in totals, or a reason it cannot."""

    hours: float | None
    kind: str
    basis: str

    @property
    def included(self) -> bool:
        return self.kind in {"reported", "nominal"}


def _cell(value: object) -> str:
    text = str(value).replace("\n", " ").replace("|", "\\|")
    return text or "—"


def _number(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _duration(spec: DatasetSpec) -> DurationSummary:
    legacy_total = _number(spec.provenance.get("legacy_total_hours"))
    if legacy_total is not None:
        return DurationSummary(legacy_total, "reported", "发布方或原登记总时长")

    root_splits = [split for split in spec.splits.values() if not split.group]
    reported: list[float] = []
    before_filter: list[float] = []
    missing = 0
    for split in root_splits:
        statistics = split.statistics
        value = _number(statistics.get("duration_hours", statistics.get("hours")))
        if value is not None:
            reported.append(value)
            continue
        previous = _number(statistics.get("hours_before_filter"))
        if previous is not None:
            before_filter.append(previous)
        else:
            missing += 1

    if root_splits and len(reported) == len(root_splits):
        return DurationSummary(sum(reported), "reported", "各顶层 split 时长之和")

    nominal_total = _number(spec.provenance.get("nominal_duration_hours"))
    if nominal_total is not None:
        return DurationSummary(nominal_total, "nominal", "名义时长（估算）")

    if reported:
        return DurationSummary(sum(reported), "partial", "仅部分顶层 split 登记了时长")
    if before_filter and not missing:
        return DurationSummary(
            sum(before_filter), "before_filter", "仅有过滤前时长，不代表当前版本"
        )
    return DurationSummary(None, "missing", "未登记")


def _format_hours(value: float) -> str:
    return f"{value:,.1f}"


def _task_name(task: str) -> str:
    return TASK_NAMES.get(task, task)


def _task_list(tasks: tuple[str, ...]) -> str:
    return "、".join(_task_name(task) for task in tasks)


def _quality_rows(specs: list[DatasetSpec]) -> list[tuple[str, str, str, str]]:
    rows: list[tuple[str, str, str, str]] = []
    for spec in specs:
        status = spec.provenance.get("quality_status")
        findings = spec.provenance.get("quality_findings", {})
        if status == "known_annotation_bounds_issues":
            train = int(findings.get("train_annotation_out_of_bounds_count", 0))
            dev = int(findings.get("dev_annotation_out_of_bounds_count", 0))
            test = int(findings.get("test_annotation_out_of_bounds_count", 0))
            overrun = float(findings.get("max_overrun_seconds", 0.0))
            rows.append(
                (
                    spec.key,
                    "有已知标注问题",
                    f"训练集 {train} 条标注越过音频边界；开发集 {dev} 条；测试集 {test} 条",
                    f"最大越界 {overrun:.3f} 秒，使用前应处理异常标注",
                )
            )

        review = spec.provenance.get("manual_review")
        split_statistics = [
            split.statistics for split in spec.splits.values()
        ]
        passed = sum(int(item.get("pass", 0)) for item in split_statistics)
        rejected = sum(int(item.get("reject", 0)) for item in split_statistics)
        if review or passed or rejected:
            total = passed + rejected
            rate = passed / total if total else 0.0
            reviewed = 0
            confirmed = 0
            if isinstance(review, dict):
                reviewed = int(review.get("content_review", 0)) + int(
                    review.get("replaced_review", 0)
                )
                confirmed = int(review.get("proposal_confirmed", 0))
            rows.append(
                (
                    spec.key,
                    "有自动筛选和人工抽检记录" if review else "有自动筛选记录",
                    f"自动筛选通过 {passed:,} 条、拒绝 {rejected:,} 条，通过率 {rate:.2%}",
                    (
                        f"人工抽检 {reviewed} 条，其中确认 {confirmed} 条；样本很小，不能代表整集准确率"
                        if review
                        else "未登记人工抽检；筛选通过不等于转写完全正确"
                    ),
                )
            )
    return rows


def _superseded(specs: list[DatasetSpec]) -> set[str]:
    """Keys of versions replaced by a registered consolidated version."""
    keys = {spec.key for spec in specs}
    result = set()
    for spec in specs:
        target = spec.provenance.get("superseded_by")
        if target is None:
            continue
        if target not in keys:
            raise ContractError(f"{spec.key}: superseded_by names an unknown version: {target}")
        result.add(spec.key)
    return result


def _dataset_families(specs: list[DatasetSpec]) -> dict[str, str]:
    """Group presentation identities using declared family and lineage only."""
    parents = {}
    for spec in specs:
        parent = spec.provenance.get("dataset_family")
        if not parent and spec.derived_from:
            parent = spec.derived_from.split("@", 1)[0]
        if parent and parent != spec.dataset_id:
            parents[spec.dataset_id] = parent
    result = {}
    for spec in specs:
        family = spec.dataset_id
        seen = set()
        while family in parents:
            if family in seen:
                raise ContractError(f"cyclic dataset family: {spec.dataset_id}")
            seen.add(family)
            family = parents[family]
        result[spec.dataset_id] = family
    return result


def _inventory(specs: list[DatasetSpec]):
    """Split current versions into dataset families, download plans and mixtures."""
    families = _dataset_families(specs)
    superseded = _superseded(specs)
    groups = defaultdict(list)
    planned = []
    mixtures = []
    for spec in specs:
        if spec.key in superseded:
            continue
        if spec.provenance.get("inventory_status") == "download_planned":
            planned.append(spec)
        elif spec.provenance.get("inventory_category") == "training_mixture":
            mixtures.append(spec)
        else:
            groups[families[spec.dataset_id]].append(spec)
    return families, groups, planned, mixtures


def _sections(groups: dict[str, list[DatasetSpec]]):
    sections = defaultdict(list)
    for family, members in sorted(groups.items()):
        languages = {language for spec in members for language in spec.languages}
        if languages == {"en"}:
            section = "英文数据"
        elif languages <= {"zh", "yue"}:
            section = "中文及粤语数据"
        elif len(languages) > 1 or "multi" in languages:
            section = "混合语言及多语数据"
        else:
            section = "其他语言数据"
        sections[section].append((family, members))
    return [(section, sections[section]) for section in SECTIONS if sections[section]]


def _languages_and_tasks(members: list[DatasetSpec]) -> tuple[str, str]:
    languages = "、".join(sorted({lang for spec in members for lang in spec.languages}))
    tasks = tuple(sorted({task for spec in members for task in spec.tasks}))
    return _cell(languages), _cell(_task_list(tasks))


def _sorted_specs(catalog_path: str | Path, views_path: str | Path) -> list[DatasetSpec]:
    catalog = load_catalog(catalog_path)
    load_view_catalog(views_path, catalog)
    return sorted(catalog, key=lambda spec: (spec.dataset_id, spec.version))


def _source_links(catalog_path: str | Path) -> dict[str, str]:
    links = {}
    for source in declaration_files(catalog_path):
        for number, data in read_declarations(source):
            links[f"{data['dataset_id']}@{data['version']}"] = (
                f"../../catalog/{source.name}#L{number}"
            )
    return links


def _cos_uri(uri: object, where: object) -> tuple[str, str]:
    if not isinstance(uri, str) or not uri.startswith("cos://") or "/" not in uri[6:]:
        raise ContractError(f"{where}: not a cos://bucket/path URI: {uri!r}")
    bucket, path = uri[6:].split("/", 1)
    return bucket, path


def _cos_locations(
    catalog_path: str | Path, specs: list[DatasetSpec]
) -> dict[str, list[tuple[str, str, str]]]:
    """(kind, bucket, path) per version, from backup records and declared copies."""
    locations = defaultdict(list)
    backups = Path(catalog_path) / "backups"
    for record_path in sorted(backups.glob("*.json")) if backups.is_dir() else []:
        record = json.loads(record_path.read_text(encoding="utf-8"))
        if "dataset_bindings" in record:
            for key, binding in record["dataset_bindings"].items():
                locations[key].append(
                    ("snapshot", *_cos_uri(binding["cos_uri"], record_path))
                )
        elif {"bucket", "prefix", "datasets"} <= record.keys():
            for key, mirror in record["datasets"].items():
                locations[key].append(
                    ("files", record["bucket"], f"{record['prefix']}{mirror['alias']}/")
                )
        else:
            raise ContractError(f"{record_path}: unknown COS backup record")
    for spec in specs:
        for entry in spec.provenance.get("cos_locations", []):
            if entry.get("content") not in ("files", "source"):
                raise ContractError(f"{spec.key}: cos_locations content must be files or source")
            locations[spec.key].append(
                (entry["content"], *_cos_uri(entry.get("uri"), spec.key))
            )
    return locations


def _cos_label(location: tuple[str, str, str], full: bool) -> str:
    kind, bucket, path = location
    if kind == "snapshot":
        name = path.rstrip("/").rsplit("/", 1)[-1]
        date = re.search(r"\d{8}", name)
        return f"快照 {date.group() if date else name}"
    shown = bucket if full else COS_BUCKETS.get(bucket, bucket)
    return f"`{shown}:{path}`" + ("（原始数据）" if kind == "source" else "")


def _cos_cell(
    specs: list[DatasetSpec],
    locations: dict[str, list[tuple[str, str, str]]],
    full: bool = False,
) -> str:
    found = sorted(
        {location for spec in specs for location in locations.get(spec.key, [])},
        key=lambda item: (COS_KINDS.index(item[0]), _cos_label(item, True)),
    )
    if not found:
        return "没有"
    if not full:
        # A directory already listed covers the paths below it.
        found = [
            location for location in found
            if not any(
                other[1] == location[1] and other[2] != location[2]
                and location[2].startswith(other[2]) and other[0] != "snapshot"
                for other in found
            )
        ]
    labels = [_cos_label(location, full) for location in found]
    if not full:
        # Many mirrored directories in one bucket collapse to their common
        # parent; data-versions.md lists each one.
        groups = defaultdict(list)
        for location in found:
            if location[0] == "files":
                groups[location[1]].append(location[2])
        labels = []
        for location in found:
            paths = groups.get(location[1], []) if location[0] == "files" else []
            if len(paths) < 3:
                labels.append(_cos_label(location, full))
            elif location[2] == paths[0]:
                common = posixpath.commonpath(paths)
                parent = ("files", location[1], f"{common}/" if common else "")
                labels.append(f"{_cos_label(parent, full)} 下 {len(paths)} 个目录")
    cell = "、".join(dict.fromkeys(labels))
    if any(spec.key not in locations for spec in specs):
        cell = "部分版本：" + cell
    return cell


def _predecessors(specs: list[DatasetSpec]) -> dict[str, list[DatasetSpec]]:
    """Versions merged into each consolidated version (superseded_by)."""
    merged = defaultdict(list)
    for spec in specs:
        target = spec.provenance.get("superseded_by")
        if target:
            merged[target].append(spec)
    return merged


def _cos_specs(
    members: list[DatasetSpec], merged: dict[str, list[DatasetSpec]]
) -> list[DatasetSpec]:
    """A family's versions plus the old versions merged into them."""
    return members + [old for spec in members for old in merged.get(spec.key, [])]


def _cos_legend(
    locations: dict[str, list[tuple[str, str, str]]], keys: set[str]
) -> list[str]:
    snapshots = sorted(
        {location for key in keys for location in locations.get(key, [])
         if location[0] == "snapshot"},
        key=lambda item: _cos_label(item, True),
    )
    lines = [
        f"- {_cos_label(location, True)}：`cos://{location[1]}/{location[2]}`"
        for location in snapshots
    ]
    buckets = "、".join(f"{short} = `{bucket}`" for bucket, short in COS_BUCKETS.items())
    return lines + [
        "- 快照内的文件按内容去重分批存放，要用 [cos_restore.py](../guides/cos-restore.md) 按版本取回；快照只表示该版本纳入了这次同步，文件是否已上传以 `locate` 结果为准。",
        f"- 其他地址是桶内目录，可直接 `ab pull`（{buckets}）。标“原始数据”的是登记版本所依据的原始文件，取回后还要按来源说明处理。",
        "- 没有：仓库里没有 COS 记录；“部分版本：”表示只有部分版本有。",
    ]


def _family_peak(specs: list[DatasetSpec]) -> tuple[float, str, str] | None:
    """The largest complete single-version or reference duration of a family."""
    candidates = []
    for spec in sorted(specs, key=lambda item: item.key):
        duration = _duration(spec)
        if duration.included:
            candidates.append((duration.hours, duration.kind, spec.key))
        reference = spec.provenance.get("reference_duration")
        if reference:
            candidates.append((reference["hours"], "reference", spec.key))
    if not candidates:
        return None
    return max(candidates, key=lambda item: (item[0], item[1] == "reported", item[2]))


def _peak_cell(specs: list[DatasetSpec]) -> str:
    peak = _family_peak(specs)
    if peak is None:
        return "—"
    hours, kind, _ = peak
    return ("" if kind == "reported" else "≈") + _format_hours(hours)


def _summary(specs: list[DatasetSpec]) -> str:
    for spec in specs:
        if spec.provenance.get("summary"):
            return _cell(spec.provenance["summary"])
    return "仅评测" if all(spec.provenance.get("consumer") for spec in specs) else "—"


def _anchor(name: str) -> str:
    """GitHub's anchor for a heading with this text."""
    return re.sub(r"[^\w\- ]", "", name.lower()).replace(" ", "-")


def _family_flags(specs: list[DatasetSpec]) -> list[str]:
    flags = []
    punctuation = {spec.provenance.get("has_punctuation") for spec in specs}
    if True in punctuation:
        flags.append("有含标点版本")
    if False in punctuation:
        flags.append("有无标点版本")
    if any(spec.derived_from for spec in specs):
        flags.append("含派生版本（与源数据可能重叠）")
    if all(spec.provenance.get("consumer") for spec in specs):
        flags.append("当前仅登记评测入口")
    return flags


def _version_rows(
    specs: list[DatasetSpec],
    links: dict[str, str],
    locations: dict[str, list[tuple[str, str, str]]],
) -> list[str]:
    rows = []
    for spec in specs:
        cos = _cos_cell([spec], locations, full=True)
        duration = _duration(spec)
        hours = "—" if duration.hours is None else _format_hours(duration.hours)
        scope = QUALIFIERS.get(duration.kind, "未登记")
        label = spec.provenance.get("duration_label")
        if label and duration.hours is not None:
            scope += f"：{_cell(label)}"
        rows.append(f"| [{spec.key}]({links[spec.key]}) | {hours} | {scope} | {cos} |")
    for spec in specs:
        reference = spec.provenance.get("reference_duration")
        if reference:
            rows.append(
                f"| [{spec.key}]({links[spec.key]}) | "
                f"{_format_hours(reference['hours'])} | [参考表]({reference['url']}) | "
                f"{_cos_cell([spec], locations, full=True)} |"
            )
    return rows


def _family_notes(specs: list[DatasetSpec]) -> list[str]:
    """Each distinct description once, after the versions that declare it."""
    keys_by_note = defaultdict(list)
    for spec in specs:
        description = spec.provenance.get("description")
        if description:
            keys_by_note[description.rstrip("。；")].append(f"`{spec.key}`")
    notes = [
        f"- {'、'.join(keys)}：{_cell(note)}" for note, keys in keys_by_note.items()
    ]
    return notes + [f"- {flag}" for flag in _family_flags(specs)]


def _chart_data(specs: list[DatasetSpec]) -> tuple[list[tuple], list[tuple]]:
    _, groups, _, _ = _inventory(specs)
    top = []
    counts = [0, 0, 0]
    for family, members in sorted(groups.items()):
        durations = [_duration(spec) for spec in members]
        has_reported = any(duration.kind == "reported" for duration in durations)
        has_reference = any(duration.hours is not None for duration in durations) or any(
            spec.provenance.get("reference_duration") for spec in members
        )
        counts[0 if has_reported else 1 if has_reference else 2] += 1
        peak = _family_peak(members)
        if peak:
            hours, kind, key = peak
            top.append((family, key, hours, kind))
    top.sort(key=lambda row: (-row[2], row[0]))
    coverage = [
        (
            "至少一个版本有完整时长登记",
            "不代表整个数据集所有版本均已覆盖",
            counts[0],
            "reported",
        ),
        (
            "仅参考、估算或部分时长",
            "含过滤前时长；没有完整登记版本",
            counts[1],
            "reference",
        ),
        ("尚未登记时长", "缺少时长的条目仍保留在总览", counts[2], "missing"),
    ]
    return top[:20], coverage


def _bar_chart(title: str, subtitle: str, rows: list[tuple], unit: str) -> str:
    """Render deterministic standalone SVG without runtime plotting dependencies."""
    colors = {
        "reported": "#2563eb",
        "reference": "#b45309",
        "nominal": "#7c3aed",
        "missing": "#64748b",
    }
    labels = {
        "reported": "已登记",
        "reference": "参考 / 部分",
        "nominal": "估算",
        "missing": "未登记",
    }
    height = 150 + 52 * len(rows)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="1240" height="{height}" viewBox="0 0 1240 {height}" role="img" aria-labelledby="title desc">',
        f'<title id="title">{escape(title)}</title>',
        f'<desc id="desc">{escape(subtitle)}</desc>',
        '<rect width="100%" height="100%" fill="white"/>',
        '<g font-family="Arial, Noto Sans CJK SC, Microsoft YaHei, sans-serif" fill="#172033">',
        f'<text x="24" y="34" font-size="23" font-weight="700">{escape(title)}</text>',
        f'<text x="24" y="60" font-size="13">{escape(subtitle)}</text>',
    ]
    for i, (kind, color) in enumerate(colors.items()):
        x = 24 + i * 170
        parts.append(f'<rect x="{x}" y="78" width="12" height="12" fill="{color}"/>')
        parts.append(f'<text x="{x + 19}" y="89" font-size="12">{labels[kind]}</text>')
    maximum = max((row[2] for row in rows), default=0) or 1
    for i, (name, scope, value, kind) in enumerate(rows):
        y = 122 + i * 52
        width = value / maximum * 440
        number = _format_hours(value) if unit == "小时" else str(value)
        parts.extend(
            [
                f'<text x="24" y="{y}" font-size="15" font-weight="600">{escape(name)}</text>',
                f'<text x="24" y="{y + 18}" font-size="11" fill="#475569">{escape(scope)}</text>',
                f'<rect x="640" y="{y - 12}" width="440" height="23" rx="3" fill="#f1f5f9"/>',
                f'<rect x="640" y="{y - 12}" width="{width:.2f}" height="23" rx="3" fill="{colors[kind]}"/>',
                f'<text x="1094" y="{y + 5}" font-size="14">{number} {unit}</text>',
            ]
        )
    if not rows:
        parts.append('<text x="24" y="124" font-size="15">暂无可展示的时长</text>')
    parts.extend(["</g>", "</svg>", ""])
    return "\n".join(parts)


def render_overview_charts(catalog_path: str | Path = "catalog") -> dict[str, str]:
    top, coverage = _chart_data(list(load_catalog(catalog_path)))
    return {
        "data-duration-top20.svg": _bar_chart(
            "数据集时长 Top 20",
            "每个数据集取最大单版本或参考值；线性比例，不跨版本相加，非当前完整规模排名。",
            top,
            "小时",
        ),
        "data-duration-coverage.svg": _bar_chart(
            "时长登记覆盖情况",
            f"共 {sum(row[2] for row in coverage)} 个数据集条目；不含下载来源声明及训练混合配方。",
            coverage,
            "个",
        ),
    }


def render_data_overview(
    catalog_path: str | Path = "catalog",
    views_path: str | Path = "views",
) -> str:
    """Render one row per data family, independently of consumer frameworks."""
    specs = _sorted_specs(catalog_path, views_path)
    families, groups, planned, mixtures = _inventory(specs)
    links = _source_links(catalog_path)
    locations = _cos_locations(catalog_path, specs)
    merged = _predecessors(specs)
    shown = {
        spec.key for members in groups.values() for spec in _cos_specs(members, merged)
    }
    lines = [
        "# 数据总览",
        "",
        f"当前登记 **{len(groups)} 个数据集**；另有 **{len(planned)} 个下载来源声明**和 **{len(mixtures)} 个训练混合配方**。登记不代表本机文件已齐备。",
        "",
        "时长取单个版本的最大值，不跨版本相加；≈ 为参考表或估算值，— 为暂无完整时长。各版本的时长、说明、COS 地址和质量证据见[版本明细](data-versions.md)。",
        "",
        "COS 列：",
        "",
        *_cos_legend(locations, shown),
        "",
        "## 数据规模与登记覆盖",
        "",
        "![数据集时长 Top 20](assets/data-duration-top20.svg)",
        "",
        "![时长登记覆盖情况](assets/data-duration-coverage.svg)",
        "",
        "Top 20 与下表取值相同；覆盖图按数据集计数，至少一个版本有完整时长即算已登记。",
        "",
    ]
    for section, entries in _sections(groups):
        lines.extend(
            [
                f"## {section}",
                "",
                "| 数据集 | 语言 | 任务 | 时长（小时） | 简介 | COS |",
                "|---|---|---|---:|---|---|",
            ]
        )
        for family, members in entries:
            languages, tasks = _languages_and_tasks(members)
            lines.append(
                f"| [{_cell(family)}](data-versions.md#{_anchor(family)}) | {languages} | "
                f"{tasks} | {_peak_cell(members)} | {_summary(members)} | "
                f"{_cos_cell(_cos_specs(members, merged), locations)} |"
            )
        lines.append("")
    lines.extend(
        [
            "## 下载来源声明",
            "",
            "仅登记来源、版本和预期位置，不代表已下载或可训练。",
            "",
            "| 数据集 | 版本 | 语言 | 支持任务 |",
            "|---|---|---|---|",
        ]
    )
    for spec in planned:
        lines.append(
            f"| [{_cell(families[spec.dataset_id])}]({links[spec.key]}) | {_cell(spec.version)} | {_cell('、'.join(spec.languages))} | {_cell(_task_list(spec.tasks))} |"
        )
    if mixtures:
        lines.extend(
            [
                "",
                "## 训练混合配方",
                "",
                "这些配方复用上面的数据，不增加源数据集数量或时长。",
                "",
            ]
        )
        for spec in mixtures:
            lines.append(f"- [{_cell(spec.key)}]({links[spec.key]})")
    lines.extend(
        [
            "",
            "## 数据声明与维护",
            "",
            "- [目录说明](../../catalog/README.md)：声明格式、字段和统计规则。",
            "- [组织规范](../reference/data-organization.md)：数据身份、版本、处理层与视图。",
            "- 新增数据时登记任务、语言、`provenance.summary`（一句话简介）和顶层 split 的 `statistics.duration_hours`；未知时长留空，不填 0。",
            "- 更新 catalog（含 `catalog/backups/`）或 views 后运行 `audio-data-contract generate-overview`；CI 用 `--check` 检查本页、版本明细和图表同步。",
            "",
        ]
    )
    return "\n".join(lines)


def render_data_versions(
    catalog_path: str | Path = "catalog",
    views_path: str | Path = "views",
) -> str:
    """Render every current version of each family, with its scope and notes."""
    specs = _sorted_specs(catalog_path, views_path)
    _, groups, _, _ = _inventory(specs)
    links = _source_links(catalog_path)
    locations = _cos_locations(catalog_path, specs)
    merged = _predecessors(specs)
    shown = {
        spec.key for members in groups.values() for spec in _cos_specs(members, merged)
    }
    lines = [
        "# 数据版本明细",
        "",
        "[数据总览](data-overview.md)中每个数据集的展开：各已登记版本的时长、口径、COS 地址和说明。",
        "",
        "- 时长单位为小时。不同版本、子集和标注片段可能重叠，不相加为总量。",
        "- 口径：已登记为各顶层划分时长之和或发布方总时长；估算为名义时长；部分划分为只有部分划分登记了时长；过滤前不能当作清洗后时长；参考表数字未核实当前文件与划分覆盖。",
        "- 已被合并版本取代（`superseded_by`）的旧版本不再列出，仍可按原 ID 和版本解析。",
        "- 任务是各版本能力的并集，具体版本和划分以链接内声明为准。标点、热词和文件哈希校验都不能单独证明做过内容清洗。",
        "",
        "COS 列：",
        "",
        *_cos_legend(locations, shown),
        "",
    ]
    for section, entries in _sections(groups):
        lines.extend([f"## {section}", ""])
        for family, members in entries:
            languages, tasks = _languages_and_tasks(members)
            lines.extend(
                [
                    f"### {family}",
                    "",
                    _summary(members),
                    "",
                    f"语言：{languages} · 任务：{tasks}",
                    "",
                    "| 版本 | 时长（小时） | 口径 | COS |",
                    "|---|---:|---|---|",
                    *_version_rows(members, links, locations),
                    "",
                ]
            )
            notes = _family_notes(members)
            old = [spec for member in members for spec in merged.get(member.key, [])]
            if old:
                notes.append(
                    "- 已并入的旧版本："
                    + "、".join(
                        f"`{spec.key}`（COS：{_cos_cell([spec], locations, full=True)}）"
                        for spec in old
                    )
                )
            if notes:
                lines.extend([*notes, ""])
    lines.extend(
        [
            "## 已记录的质量证据",
            "",
            "文件校验通过，不等于内容正确。以下仅列有明确记录的版本，缺少记录不代表质量差。",
            "",
            "| 数据版本 | 当前结论 | 证据 | 注意事项 |",
            "|---|---|---|---|",
        ]
    )
    for key, conclusion, evidence, caution in _quality_rows(specs):
        lines.append(
            f"| [{_cell(key)}]({links[key]}) | {_cell(conclusion)} | {_cell(evidence)} | {_cell(caution)} |"
        )
    lines.extend(
        [
            "",
            "CV EN 的 train/test 清单落地差异见 [CV EN 清洗说明](cv-en-cleaning.md)。",
            "",
        ]
    )
    return "\n".join(lines)


def update_data_overview(
    output: str | Path,
    *,
    catalog_path: str | Path = "catalog",
    views_path: str | Path = "views",
    check: bool = False,
) -> str:
    """Write the overview, or fail when a checked overview is stale."""

    destination = Path(output)
    artifacts = {
        destination: render_data_overview(catalog_path, views_path),
        destination.parent / "data-versions.md": render_data_versions(
            catalog_path, views_path
        ),
    }
    artifacts.update(
        {
            destination.parent / "assets" / name: content
            for name, content in render_overview_charts(catalog_path).items()
        }
    )
    if check:
        for path, expected in artifacts.items():
            try:
                current = path.read_text(encoding="utf-8")
            except OSError as exc:
                raise ContractError(
                    f"data overview artifact is missing: {path}"
                ) from exc
            if current != expected:
                raise ContractError(
                    f"data overview artifact is out of date: {path}; run "
                    "`audio-data-contract generate-overview`"
                )
        return "current"
    for path, content in artifacts.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    return "written"
