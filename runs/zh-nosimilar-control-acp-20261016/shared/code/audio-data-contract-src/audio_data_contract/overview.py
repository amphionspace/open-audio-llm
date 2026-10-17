"""Deterministic, human-readable overview of registered data."""

from __future__ import annotations

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
}


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


def _source_links(catalog_path: str | Path) -> dict[str, str]:
    links = {}
    for source in declaration_files(catalog_path):
        for number, data in read_declarations(source):
            links[f"{data['dataset_id']}@{data['version']}"] = (
                f"../../catalog/{source.name}#L{number}"
            )
    return links


def _family_hours(specs: list[DatasetSpec], links: dict[str, str]) -> str:
    entries = []
    for spec in specs:
        duration = _duration(spec)
        if duration.hours is None:
            continue
        label = spec.provenance.get("duration_label", spec.version)
        qualifier = {
            "reported": "已登记",
            "nominal": "估算",
            "partial": "部分划分",
            "before_filter": "过滤前",
        }[duration.kind]
        entries.append(
            f"{_format_hours(duration.hours)}（{qualifier}；"
            f"[{_cell(label)}]({links[spec.key]})）"
        )
    for spec in specs:
        reference = spec.provenance.get("reference_duration")
        if reference:
            entries.append(
                f"{_format_hours(reference['hours'])}（"
                f"[参考表]({reference['url']})；{_cell(spec.version)}）"
            )
    return "<br>".join(entries) or "未登记"


def _family_notes(specs: list[DatasetSpec]) -> str:
    notes = []
    for spec in specs:
        description = spec.provenance.get("description")
        if description and description.rstrip("。；") not in notes:
            notes.append(description.rstrip("。；"))
    punctuation = {spec.provenance.get("has_punctuation") for spec in specs}
    if True in punctuation:
        notes.append("有含标点版本")
    if False in punctuation:
        notes.append("有无标点版本")
    if any(spec.derived_from for spec in specs):
        notes.append("含派生版本（与源数据可能重叠）")
    if all(spec.provenance.get("consumer") for spec in specs):
        notes.append("当前仅登记评测入口")
    splits = sorted(
        {
            name
            for spec in specs
            for name, split in spec.splits.items()
            if not split.group
        }
    )
    if not notes:
        notes.append(
            "已登记划分：" + "、".join(splits) if splits else "尚未登记独立划分"
        )
    return _cell("；".join(notes))


def _chart_data(specs: list[DatasetSpec]) -> tuple[list[tuple], list[tuple]]:
    families = _dataset_families(specs)
    groups = defaultdict(list)
    for spec in specs:
        if (
            spec.provenance.get("inventory_status") == "download_planned"
            or spec.provenance.get("inventory_category") == "training_mixture"
        ):
            continue
        groups[families[spec.dataset_id]].append(spec)
    top = []
    counts = [0, 0, 0]
    for family, members in sorted(groups.items()):
        candidates = []
        has_reported = False
        has_reference = False
        for spec in sorted(members, key=lambda item: item.key):
            duration = _duration(spec)
            has_reported |= duration.kind == "reported"
            has_reference |= duration.hours is not None
            if duration.included:
                candidates.append((duration.hours, duration.kind, spec.key))
            reference = spec.provenance.get("reference_duration")
            if reference:
                has_reference = True
                candidates.append((reference["hours"], "reference", spec.key))
        counts[0 if has_reported else 1 if has_reference else 2] += 1
        if candidates:
            hours, kind, key = max(
                candidates, key=lambda item: (item[0], item[1] == "reported", item[2])
            )
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
    catalog = load_catalog(catalog_path)
    load_view_catalog(views_path, catalog)
    specs = sorted(catalog, key=lambda spec: (spec.dataset_id, spec.version))
    families = _dataset_families(specs)
    links = _source_links(catalog_path)
    groups = defaultdict(list)
    planned = []
    mixtures = []
    for spec in specs:
        if spec.provenance.get("inventory_status") == "download_planned":
            planned.append(spec)
        elif spec.provenance.get("inventory_category") == "training_mixture":
            mixtures.append(spec)
        else:
            groups[families[spec.dataset_id]].append(spec)

    lines = [
        "# 数据总览",
        "",
        "按数据集查看语言、支持任务、时长和特性。清洗、热词、加噪、评测入口和训练框架配置归入所属数据集，不另算一套源数据。",
        "",
        f"当前登记 **{len(groups)} 个数据集条目**；另有 **{len(planned)} 个下载来源声明**和 **{len(mixtures)} 个训练混合配方**。登记不代表本机文件已齐备。",
        "",
        "时长单位为小时。不同版本、子集和标注片段可能重叠，逐项列出，不相加为总量；没有时长的条目仍保留。参考表数字未核实当前文件与划分覆盖；“过滤前”不能当作清洗后时长。",
        "",
        "任务是该数据集各已登记版本的能力并集，具体版本和划分以链接内声明为准。标点、热词和文件哈希校验都不能单独证明做过内容清洗。",
        "",
    ]
    lines.extend(
        [
            "## 数据规模与登记覆盖",
            "",
            "![数据集时长 Top 20](assets/data-duration-top20.svg)",
            "",
            "每个数据集仅取最大的一项完整版本登记、参考或估算时长，并标出对应版本。部分划分和过滤前时长不参与排名；不同颜色区分登记值与参考 / 估算值。排名用于查看量级，不代表最新版本或整个数据集的完整时长。",
            "",
            "![时长登记覆盖情况](assets/data-duration-coverage.svg)",
            "",
            "覆盖图按数据集互斥分类：优先计入“至少一个版本有完整时长登记”，其次是“仅参考、估算或部分时长”，其余为“尚未登记”。完整登记只针对对应版本，不代表整个数据集所有子集和版本均已覆盖。",
            "",
        ]
    )
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
    for section in ["英文数据", "中文及粤语数据", "混合语言及多语数据", "其他语言数据"]:
        if not sections[section]:
            continue
        lines.extend(
            [
                f"## {section}",
                "",
                "| 数据集 | 语言 | 支持任务 | 时长（小时，注明范围） | 特性 / 备注 |",
                "|---|---|---|---|---|",
            ]
        )
        for family, members in sections[section]:
            primary = next(
                (spec for spec in members if spec.provenance.get("description")),
                next(
                    (
                        spec
                        for spec in members
                        if not spec.derived_from and not spec.provenance.get("consumer")
                    ),
                    members[0],
                ),
            )
            languages = "、".join(
                sorted({lang for spec in members for lang in spec.languages})
            )
            tasks = tuple(sorted({task for spec in members for task in spec.tasks}))
            lines.append(
                f"| [{_cell(family)}]({links[primary.key]}) | {_cell(languages)} | "
                f"{_cell(_task_list(tasks))} | {_family_hours(members, links)} | {_family_notes(members)} |"
            )
        lines.append("")
    lines.extend(
        [
            "## 下载来源声明",
            "",
            "以下条目来自下载计划，仅登记来源、版本和预期位置；这里不据此判断已下载或可训练。已有旧版的数据集也可能列有新版本下载计划。",
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
            "## 数据声明与维护",
            "",
            "- [目录说明](../../catalog/README.md)：声明格式、字段和统计规则。",
            "- [组织规范](../reference/data-organization.md)：数据身份、版本、处理层与视图。",
            "- 新增数据时登记任务、语言、特性和顶层 split 的 `statistics.duration_hours`；未知时长留空，不填 0。",
            "- 更新 catalog 或 views 后运行 `audio-data-contract generate-overview`；CI 用 `--check` 检查本页同步。",
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
    rendered = render_data_overview(catalog_path, views_path)
    artifacts = {destination: rendered}
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
