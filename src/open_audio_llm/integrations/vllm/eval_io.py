"""Unified output layout for evaluation runs (decode.py + vLLM eval).

Each ``EvalRunDir`` instance owns a single ``<base_dir>/<run_id>/`` directory
that hosts everything produced by one evaluation invocation. Per-test
artifacts live under ``per_test/<label>/``; the top-level holds a
``summary.json`` (machine-readable cross-dataset aggregation), a
``summary.xlsx`` (multi-sheet spreadsheet with paper-style ASR table +
per-task metric tables), a ``config.json`` snapshot of the CLI args +
resolved test plan, and a ``log-eval.log`` capturing the run log.

Both decoding paths (Amphion local and vLLM HTTP) share this layout so a
``diff`` between two ``summary.json`` files immediately shows where the
checkpoints disagree. The same module is used by
``src/decode.py`` and ``src/open_audio_llm/integrations/vllm/test_vllm_inference.py``.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence


# Directory layout constants — kept as module-level so ad-hoc tooling
# (e.g. doc generators) can also locate these files without instantiating
# an EvalRunDir.
PER_TEST_DIRNAME = "per_test"
SUMMARY_JSON_FILENAME = "summary.json"
SUMMARY_XLSX_FILENAME = "summary.xlsx"
# Legacy markdown summary file. New eval runs no longer produce it; the
# constant is kept so :class:`EvalRunDir` and tooling such as
# ``scripts/eval/fix_legacy_eval_outputs.py`` can still locate (and clean
# up) any stale ``summary.md`` left behind by earlier runs.
SUMMARY_MD_FILENAME = "summary.md"
CONFIG_JSON_FILENAME = "config.json"
LOG_FILENAME = "log-eval.log"
TRANSCRIPTS_FILENAME = "transcripts.jsonl"
WER_REPORT_FILENAME = "wer_report.txt"
METRICS_JSON_FILENAME = "metrics.json"

# Tasks whose reference is a single label (emotion class, sound event, ...)
# or a free-form caption — WER on these is meaningless. SER reports
# WA / UA / Macro-F1 instead (see write_summary_xlsx), ESC reports
# Tag-set Micro/Macro F1 + ROUGE-L (see _persist_esc_to_run /
# save_esc_results_to_run), and SEC remains a pure generation task with
# no auto metric (see _persist_sec_or_esc_to_run / save_sec_results_to_run).
# All three are excluded from the cross-test WER aggregation done by
# :func:`aggregate_overall`.
CLASSIFICATION_TASKS = frozenset({"ser", "sec", "esc"})


def _ts_str() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


class EvalRunDir:
    """A versioned evaluation output directory."""

    def __init__(self, run_dir: Path):
        self.run_dir = Path(run_dir)
        self.run_id = self.run_dir.name

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def create(
        cls,
        *,
        base_dir: Path,
        ckpt_label: str,
        run_name: Optional[str] = None,
    ) -> "EvalRunDir":
        """Materialise a fresh ``<base_dir>/<run_id>/`` directory.

        - ``run_id`` defaults to ``f"{ckpt_label}-{YYYYmmdd_HHMMSS}"``.
        - ``run_name`` (if given) overrides the entire run_id; useful for
          named campaigns (e.g. ``--run-name ts_neg_v1_full``).
        - On collision (rare) the new directory gets ``-2`` / ``-3`` etc.
          appended; existing runs are never overwritten in-place.
        """
        base_dir = Path(base_dir)
        base_dir.mkdir(parents=True, exist_ok=True)
        if run_name:
            base_run_id = run_name
        else:
            base_run_id = f"{_safe_ckpt_label(ckpt_label)}-{_ts_str()}"

        run_id = base_run_id
        suffix = 2
        while (base_dir / run_id).exists():
            run_id = f"{base_run_id}-{suffix}"
            suffix += 1
        run_dir = base_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=False)
        (run_dir / PER_TEST_DIRNAME).mkdir(parents=True, exist_ok=True)
        return cls(run_dir)

    # ------------------------------------------------------------------
    # Per-test paths
    # ------------------------------------------------------------------

    def per_test_dir(self, label: str) -> Path:
        d = self.run_dir / PER_TEST_DIRNAME / label
        d.mkdir(parents=True, exist_ok=True)
        return d

    # ------------------------------------------------------------------
    # Writers (each is small and side-effect-only by design)
    # ------------------------------------------------------------------

    def write_config(self, config_obj: Dict[str, Any]) -> Path:
        """Persist a JSON snapshot of CLI args + resolved test plan."""
        path = self.run_dir / CONFIG_JSON_FILENAME
        with path.open("w", encoding="utf-8") as f:
            json.dump(config_obj, f, ensure_ascii=False, indent=2,
                      default=str)
            f.write("\n")
        return path

    def write_test_metrics(self, label: str, metrics: Dict[str, Any]) -> Path:
        path = self.per_test_dir(label) / METRICS_JSON_FILENAME
        with path.open("w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2, default=str)
            f.write("\n")
        return path

    def write_test_transcripts(
        self, label: str, records: List[Dict[str, Any]],
    ) -> Path:
        """Write per-utterance JSONL records (one JSON obj per line)."""
        path = self.per_test_dir(label) / TRANSCRIPTS_FILENAME
        with path.open("w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False, default=str))
                f.write("\n")
        return path

    def write_test_wer_report(self, label: str, text: str) -> Path:
        path = self.per_test_dir(label) / WER_REPORT_FILENAME
        with path.open("w", encoding="utf-8") as f:
            f.write(text)
            if not text.endswith("\n"):
                f.write("\n")
        return path

    def write_summary(self, summary_obj: Dict[str, Any]) -> tuple[Path, Path]:
        """Write ``summary.json`` plus ``summary.xlsx`` side-by-side.

        Returns the paths to (json, xlsx). Earlier versions also wrote a
        ``summary.md`` here; the markdown report has been replaced by the
        spreadsheet (see :func:`write_summary_xlsx`). Any stale
        ``summary.md`` left over from a previous run in the same dir is
        deleted so the directory always reflects the current writer.
        """
        json_path = self.run_dir / SUMMARY_JSON_FILENAME
        xlsx_path = self.run_dir / SUMMARY_XLSX_FILENAME
        with json_path.open("w", encoding="utf-8") as f:
            json.dump(summary_obj, f, ensure_ascii=False, indent=2,
                      default=str)
            f.write("\n")
        write_summary_xlsx(summary_obj, xlsx_path)
        legacy_md = self.run_dir / SUMMARY_MD_FILENAME
        if legacy_md.exists():
            legacy_md.unlink()
        return json_path, xlsx_path

    # ------------------------------------------------------------------
    # Logger
    # ------------------------------------------------------------------

    def attach_file_logger(self, level: int = logging.INFO) -> Path:
        """Attach a FileHandler that writes everything to log-eval.log.

        Idempotent: re-attaching just adds another handler, but that's
        usually fine because the second instance only fires when the
        caller forgets to detach (rare). Uses the root logger so
        sub-modules' ``logging.info(...)`` calls also flow into the file.
        """
        path = self.run_dir / LOG_FILENAME
        handler = logging.FileHandler(path, mode="a", encoding="utf-8")
        handler.setLevel(level)
        formatter = logging.Formatter(
            "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
        )
        handler.setFormatter(formatter)
        root = logging.getLogger()
        root.setLevel(level)
        root.addHandler(handler)
        return path

    # ------------------------------------------------------------------
    # Legacy output aliases (backwards compatibility)
    # ------------------------------------------------------------------

    def write_test_legacy_aliases(
        self, base_dir: Path, label: str, suffix: str,
        task: Optional[str] = None,
    ) -> List[Path]:
        """Mirror per-test artifacts under the pre-run_id flat layout.

        Pre-PR, ``decode.py`` wrote each test's transcripts and WER report
        directly into ``<exp-dir>/eval/`` as
        ``{test_set_name}-{suffix}.jsonl`` / ``{test_set_name}-{suffix}_wer.txt``.
        Downstream tooling (e.g. WER aggregation scripts) globs for these
        files. To avoid breaking those callers we keep that flat layout as
        a thin alias on top of the new ``per_test/<label>/`` directory.

        ``task`` (optional) is the resolved task name from ``TestSpec``; when
        set to ``"ser"`` we additionally mirror the per-test report under
        ``{label}-{suffix}_ser.txt`` so SER pipelines that previously globbed
        the pre-PR ``save_ser_results`` filename keep finding their reports.

        Strategy: try a relative symlink first (fast, near-zero disk cost);
        fall back to copying when the filesystem doesn't allow symlinks
        (e.g. some NAS / Windows shares). Existing alias files are
        replaced so re-running an eval refreshes the link target.
        """
        base_dir = Path(base_dir)
        base_dir.mkdir(parents=True, exist_ok=True)
        per_test = self.per_test_dir(label)
        pairs = [
            (per_test / TRANSCRIPTS_FILENAME, base_dir / f"{label}-{suffix}.jsonl"),
            (per_test / WER_REPORT_FILENAME,  base_dir / f"{label}-{suffix}_wer.txt"),
            (per_test / METRICS_JSON_FILENAME, base_dir / f"{label}-{suffix}_metrics.json"),
        ]
        if task == "ser":
            pairs.append(
                (per_test / WER_REPORT_FILENAME, base_dir / f"{label}-{suffix}_ser.txt")
            )
        written: List[Path] = []
        for src, dst in pairs:
            if not src.exists():
                continue
            written.append(_make_alias(src, dst))
        return written

    def write_run_legacy_aliases(
        self, base_dir: Path, suffix: str,
    ) -> List[Path]:
        """Mirror run-level artifacts (config + log) under the flat layout.

        Pre-PR these were named ``decode_config-{suffix}.json`` and
        ``log-eval-{suffix}`` in ``<exp-dir>/eval/``. Same alias semantics
        as :meth:`write_test_legacy_aliases`.
        """
        base_dir = Path(base_dir)
        base_dir.mkdir(parents=True, exist_ok=True)
        pairs = [
            (self.run_dir / CONFIG_JSON_FILENAME, base_dir / f"decode_config-{suffix}.json"),
            (self.run_dir / LOG_FILENAME,         base_dir / f"log-eval-{suffix}"),
        ]
        written: List[Path] = []
        for src, dst in pairs:
            if not src.exists():
                continue
            written.append(_make_alias(src, dst))
        return written


def _make_alias(src: Path, dst: Path) -> Path:
    """Create a relative symlink ``dst`` -> ``src``; copy when symlink fails.

    Works idempotently: if ``dst`` already exists (file or symlink) it is
    removed before re-creating. Returns the path that was actually written.
    """
    import os
    import shutil
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    try:
        rel_target = os.path.relpath(src, dst.parent)
        dst.symlink_to(rel_target)
        return dst
    except (OSError, NotImplementedError):
        shutil.copy2(src, dst)
        return dst


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_ckpt_label(label: str) -> str:
    """Make sure the ckpt label is filesystem-safe.

    Keeps the original characters except for path separators / spaces, which
    are mapped to underscores so the resulting run_id stays a single dir
    component.
    """
    bad = "/\\ "
    out = "".join("_" if c in bad else c for c in (label or "run"))
    return out or "run"


def aggregate_overall(per_test_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Cross-test aggregation: ref_tokens-weighted overall WER + counts.

    Each entry in *per_test_results* is expected to mirror the per-test
    payload written by :meth:`EvalRunDir.write_test_metrics`, i.e. carry an
    ``overall`` field with ``ref_tokens`` / ``wer`` / ``substitutions`` /
    ``deletions`` / ``insertions`` (the dict produced by
    :func:`src.compute_wer.compute_wer`).

    Tasks listed in :data:`CLASSIFICATION_TASKS` (``ser`` / ``sec`` / ``esc``)
    are excluded from the WER pool — SER carries WA/UA/F1 in its own block
    and SEC/ESC have no auto metrics, so folding them into the WER would
    artificially dilute the corpus-level number with their zero-ref-token
    contributions.

    ``n_evaluated`` still counts every test (SER ⇒ ``num_samples``,
    SEC/ESC ⇒ ``n_total``, ASR ⇒ ``overall.num_sentences`` / ``n_evaluated``)
    so the Aggregate sheet in summary.xlsx reflects the full eval volume.
    """
    total_ref = 0
    total_sub = 0
    total_del = 0
    total_ins = 0
    n_tests_evaluated = 0
    n_tests_failed = 0
    n_tests_skipped = 0
    total_samples = 0
    for r in per_test_results:
        # ``skipped`` (added by vLLM whisper backend when a spec uses a
        # task whisper can't service) is semantically distinct from
        # ``error``: it means we *chose* not to run the spec, not that
        # we tried and the run blew up. Both still get excluded from
        # ref-token / WER aggregation so the corpus number stays clean,
        # but n_tests_skipped is reported separately so downstream
        # dashboards can show "8 ran, 4 skipped" instead of "4 failed".
        if r.get("skipped") is not None:
            n_tests_skipped += 1
            continue
        if r.get("error") is not None:
            n_tests_failed += 1
            continue
        n_tests_evaluated += 1
        spec = r.get("spec") or {}
        task = spec.get("task")

        if task == "ser":
            # SER: classification, no WER. Count samples for n_evaluated.
            total_samples += r.get("num_samples", 0) or 0
            continue
        if task in ("sec", "esc"):
            # SEC / ESC: generation, no auto metric. n_total comes from
            # _persist_sec_or_esc_to_run / save_sec_results_to_run.
            total_samples += r.get("n_total", 0) or 0
            continue

        overall = r.get("overall") or {}
        total_ref += overall.get("num_ref_tokens", 0) or overall.get("ref_tokens", 0) or 0
        total_sub += overall.get("substitutions", 0) or 0
        total_del += overall.get("deletions", 0) or 0
        total_ins += overall.get("insertions", 0) or 0
        total_samples += r.get("n_evaluated", 0) or overall.get("num_sentences", 0) or 0
    overall_wer = (
        100.0 * (total_sub + total_del + total_ins) / max(total_ref, 1)
        if total_ref > 0 else None
    )
    return {
        "n_tests_total": len(per_test_results),
        "n_tests_evaluated": n_tests_evaluated,
        "n_tests_failed": n_tests_failed,
        "n_tests_skipped": n_tests_skipped,
        "n_evaluated": total_samples,
        "ref_tokens": total_ref,
        "substitutions": total_sub,
        "deletions": total_del,
        "insertions": total_ins,
        "wer": overall_wer,   # already in %, matches compute_wer.compute_wer output
    }


# ---------------------------------------------------------------------------
# Paper-style ASR table layout
#
# 列布局对齐论文/对比表 "Model/Datasets" 横向多列, 每列要么是单 dataset 的
# WER, 要么是两个 dataset 的 "a | b" 双数 cell。这里硬编码 dataset 名 →
# 表头列名的映射, 让 ASR 区块顶部输出与图中表格一致的 "宽列单行" 样式。
#
# 渲染时按 ``entry["spec"]["dataset"]`` 严格匹配 dataset, 取 entry["overall"]["wer"]
# (按 ref_tokens 加权聚合后的整体 WER) 作为该 cell 的数字。
# ---------------------------------------------------------------------------
PAPER_STYLE_COLUMNS: List[tuple[str, List[str]]] = [
    ("LibriSpeech (clean | other)",
     ["librispeech_test_clean", "librispeech_test_other"]),
    ("GigaSpeech",                    ["gigaspeech"]),
    ("WenetSpeech (net | meeting)",
     ["wenetspeech_test_net", "wenetspeech_test_meeting"]),
    ("AIShell1",                      ["aishell"]),
    ("AIShell2",                      ["aishell2"]),
    ("KeSpeech",                      ["kespeech"]),
    ("CommonVoice (en | zh)",         ["commonvoice_en", "commonvoice_zh"]),
    # multilingual specs land in spec.dataset as "multilingual:<lang>:<name>"
    # (see eval_plan._safe_label); short alias accepted for
    # forward-compat / hand-written summaries.
    ("Fleurs (en | zh)",
     ["multilingual:en:fleurs_en", "multilingual:zh:fleurs_zh"]),
]


def _bucket_tests_by_task(
    tests: List[Dict[str, Any]],
) -> Dict[str, List[Dict[str, Any]]]:
    """Group spec entries by ``spec.task`` while skipping failed tests.

    Failed entries (``error`` set) are *not* placed in any task bucket — they
    surface in the dedicated "Failed tests" section at the end. This keeps
    each task's table free of incomplete rows.
    """
    buckets: Dict[str, List[Dict[str, Any]]] = {}
    for entry in tests:
        if entry.get("error") is not None:
            continue
        task = (entry.get("spec") or {}).get("task", "?")
        buckets.setdefault(task, []).append(entry)
    return buckets


def _model_display_name(summary: Dict[str, Any]) -> str:
    """Pick a short identifier for the 'Model/Datasets' cell.

    Priority: served_model_name > ckpt > run_id (so vLLM eval gets the
    served name and decode.py eval falls back to the checkpoint label).
    """
    model = summary.get("model") or {}
    for k in ("served_model_name", "ckpt"):
        v = model.get(k)
        if v:
            return str(v)
    return str(summary.get("run_id", "?"))


# ---------------------------------------------------------------------------
# XLSX writer
#
# The xlsx replaces the markdown summary that earlier versions of this module
# produced. Each task block becomes a dedicated worksheet so the spreadsheet
# is naturally filterable / sortable in Excel; a separate "Paper-style" sheet
# carries the wide single-row ASR table from the markdown era. ASR WER cells
# stay as plain numbers (already in percent units; format ``0.00`` so they
# still render as ``5.39``); SER / silence-aware metrics are stored as raw
# 0..1 floats with number_format ``0.00%`` so Excel multiplies/displays the
# percentage automatically.
# ---------------------------------------------------------------------------

# openpyxl rejects sheet names containing : * ? / \ [ ] (no plus, no space
# rules — both are fine) and caps the length at 31 chars. The labels below
# stay safely within that envelope.
_PAPER_STYLE_FLAT_HEADERS: List[str] = [
    "Model",
    "LibriSpeech_clean", "LibriSpeech_other",
    "GigaSpeech",
    "WenetSpeech_net", "WenetSpeech_meeting",
    "AIShell1",
    "AIShell2",
    "KeSpeech",
    "CommonVoice_en", "CommonVoice_zh",
    "Fleurs_en", "Fleurs_zh",
]

# Maps every flat column above to the dataset name in summary.json that
# fills that cell. Two-cell md columns (e.g. "LibriSpeech (clean | other)")
# expand into two consecutive xlsx columns; single-cell md columns map 1:1.
_PAPER_STYLE_FLAT_DATASETS: List[str] = [
    "librispeech_test_clean", "librispeech_test_other",
    "gigaspeech",
    "wenetspeech_test_net", "wenetspeech_test_meeting",
    "aishell",
    "aishell2",
    "kespeech",
    "commonvoice_en", "commonvoice_zh",
    "multilingual:en:fleurs_en", "multilingual:zh:fleurs_zh",
]


def _autosize_columns(ws: "Worksheet", header_lengths: Sequence[int]) -> None:  # noqa: F821
    """Set column widths to roughly fit the header text.

    A perfect autosize would walk every row, but headers usually dominate
    the width budget; eyeballing ``len(header) * 1.2`` keeps the file
    readable without iterating all data each time.
    """
    from openpyxl.utils import get_column_letter

    for idx, length in enumerate(header_lengths, start=1):
        col_letter = get_column_letter(idx)
        ws.column_dimensions[col_letter].width = max(10, int(length * 1.2) + 2)


def _ws_write_header(ws: "Worksheet", headers: Sequence[str]) -> None:  # noqa: F821
    """Write a header row, bold-format it, and freeze it."""
    from openpyxl.styles import Font

    bold = Font(bold=True)
    for col_idx, name in enumerate(headers, start=1):
        cell = ws.cell(row=1, column=col_idx, value=name)
        cell.font = bold
    ws.freeze_panes = "A2"
    _autosize_columns(ws, [len(h) for h in headers])


def _xlsx_meta_sheet(ws: "Worksheet", summary: Dict[str, Any]) -> None:  # noqa: F821
    _ws_write_header(ws, ["Field", "Value"])
    rows: List[tuple[str, Any]] = [
        ("run_id", summary.get("run_id", "")),
        ("source", summary.get("source", "")),
        ("generated_at", summary.get("generated_at", "")),
        ("test_plan_file", summary.get("test_plan_file", "")),
    ]
    for k, v in (summary.get("model") or {}).items():
        rows.append((f"model.{k}", v))
    for r_idx, (k, v) in enumerate(rows, start=2):
        ws.cell(row=r_idx, column=1, value=k)
        ws.cell(row=r_idx, column=2, value=v)
    _autosize_columns(
        ws,
        [
            max(len("Field"), max((len(str(k)) for k, _ in rows), default=5)),
            max(
                len("Value"),
                max((len(str(v)) for _, v in rows), default=10),
            ),
        ],
    )


def _xlsx_paper_style_sheet(
    ws: "Worksheet",  # noqa: F821
    summary: Dict[str, Any],
    asr_entries: List[Dict[str, Any]],
) -> None:
    """One bold row per model with WER cells in percent units."""
    _ws_write_header(ws, _PAPER_STYLE_FLAT_HEADERS)

    by_dataset: Dict[str, Dict[str, Any]] = {}
    for entry in asr_entries:
        ds = (entry.get("spec") or {}).get("dataset")
        if ds:
            by_dataset[ds] = entry

    ws.cell(row=2, column=1, value=_model_display_name(summary))
    for col_offset, dataset in enumerate(_PAPER_STYLE_FLAT_DATASETS):
        entry = by_dataset.get(dataset)
        wer = (entry.get("overall") or {}).get("wer") if entry else None
        cell = ws.cell(row=2, column=col_offset + 2, value=wer)
        if wer is not None:
            cell.number_format = "0.00"


def _xlsx_dataset_overall_sheet(
    ws: "Worksheet", entries: List[Dict[str, Any]],  # noqa: F821
    *,
    extra_cols: Optional[List[tuple]] = None,
) -> None:
    """One row per dataset with overall (ref_tokens-weighted) WER.

    Used for ASR / ASR + Hotwords / TS-ASR sheets — the user-facing view
    is ``dataset × metric`` so a checkpoint comparison can be built by
    diffing two xlsx files row-by-row. Per-language detail (KeSpeech 方言桶,
    TS-ASR en/zh split) and silence-aware metrics (FA / Miss) live in the
    per-test artifacts under ``per_test/<label>/wer_report.txt``;
    folding them in here would force half-empty rows that defeat the
    "one row per dataset" promise.

    ``extra_cols`` is a list of ``(header, fn(entry) -> value)`` or
    ``(header, fn(entry) -> value, number_format)`` tuples inserted
    between the Dataset column and the n / WER columns. Used by the
    "ASR + Hotwords (Random/Retrieve)" sheets to surface K, KER, SACC,
    Recall@K, etc. with the appropriate Excel number format. Default
    None keeps the legacy 3-column layout for everyone else.
    """
    extras = [
        (t[0], t[1], t[2] if len(t) > 2 else None) for t in (extra_cols or [])
    ]
    headers = ["Dataset"] + [h for h, _, _ in extras] + ["n", "WER %"]
    _ws_write_header(ws, headers)
    for r_idx, entry in enumerate(entries, start=2):
        dataset = (entry.get("spec") or {}).get("dataset", "?")
        ov = entry.get("overall") or {}
        n = ov.get("num_sentences",
                   ov.get("n_evaluated",
                          entry.get("n_evaluated", entry.get("n_total", 0))))
        wer = ov.get("wer")
        ws.cell(row=r_idx, column=1, value=dataset)
        col = 2
        for _, fn, fmt in extras:
            try:
                value = fn(entry)
            except Exception:
                value = None
            cell = ws.cell(row=r_idx, column=col, value=value)
            if fmt and value is not None:
                cell.number_format = fmt
            col += 1
        ws.cell(row=r_idx, column=col, value=n)
        cell = ws.cell(row=r_idx, column=col + 1, value=wer)
        if wer is not None:
            cell.number_format = "0.00"


def _xlsx_tsasr_sheet(
    ws: "Worksheet", entries: List[Dict[str, Any]],  # noqa: F821
) -> None:
    """TS-ASR sheet: one row per dataset with WER + silence-aware metrics.

    Unlike the generic ``_xlsx_dataset_overall_sheet`` (Dataset / n / WER %)
    this sheet surfaces the ``silence_metrics`` block produced by
    ``compute_silence_metrics`` so virtual-speaker presence/absence behaviour
    is visible at a glance for TS-ASR runs. Columns:

      Dataset | n_pos | n_neg | WER % | FA % | Miss % | Silence Match %

    - ``n_pos`` / ``n_neg`` come from ``silence_metrics.n_pos_ref`` /
      ``n_neg_ref`` (number of supervisions with non-empty / empty ref).
    - ``WER %`` is in 0..100 percent units (matches the ASR sheet so a
      diff between two xlsx files stays apples-to-apples; format ``0.00``).
    - ``FA % / Miss % / Silence Match %`` are stored as raw 0..1 floats
      with ``0.00%`` number_format so Excel renders e.g. ``20.43%``.
    - libri2mix / libri3mix-style datasets have ``n_neg_ref == 0`` so
      ``false_alarm_rate`` / ``exact_silence_match`` are None and the
      corresponding cells stay blank (Excel default) — Miss can still
      carry a value when positive samples elicited empty hypotheses.
    - Old summary.json files without a ``silence_metrics`` block degrade
      cleanly: Dataset / WER % still populate, the 4 new cells stay blank.
    """
    headers = [
        "Dataset", "n_pos", "n_neg", "WER %",
        "FA %", "Miss %", "Silence Match %",
    ]
    _ws_write_header(ws, headers)
    for r_idx, entry in enumerate(entries, start=2):
        dataset = (entry.get("spec") or {}).get("dataset", "?")
        sm = entry.get("silence_metrics") or {}
        wer = (entry.get("overall") or {}).get("wer")
        ws.cell(row=r_idx, column=1, value=dataset)
        ws.cell(row=r_idx, column=2, value=sm.get("n_pos_ref"))
        ws.cell(row=r_idx, column=3, value=sm.get("n_neg_ref"))
        cell = ws.cell(row=r_idx, column=4, value=wer)
        if wer is not None:
            cell.number_format = "0.00"
        for col_offset, key in enumerate(
            ("false_alarm_rate", "miss_rate", "exact_silence_match"),
            start=5,
        ):
            value = sm.get(key)
            cell = ws.cell(row=r_idx, column=col_offset, value=value)
            if value is not None:
                cell.number_format = "0.00%"


def _xlsx_ser_sheet(
    ws: "Worksheet", entries: List[Dict[str, Any]],  # noqa: F821
) -> None:
    _ws_write_header(ws, ["Dataset", "n", "WA", "UA", "Macro-F1"])
    for r_idx, entry in enumerate(entries, start=2):
        dataset = (entry.get("spec") or {}).get("dataset", "?")
        ws.cell(row=r_idx, column=1, value=dataset)
        ws.cell(row=r_idx, column=2, value=entry.get("num_samples", 0))
        for col_offset, key in enumerate(("wa", "ua", "macro_f1"), start=3):
            cell = ws.cell(row=r_idx, column=col_offset, value=entry.get(key))
            if cell.value is not None:
                cell.number_format = "0.00%"


def _xlsx_secesc_sheet(
    ws: "Worksheet", entries: List[Dict[str, Any]],  # noqa: F821
) -> None:
    _ws_write_header(ws, ["Dataset", "Task", "n"])
    for r_idx, entry in enumerate(entries, start=2):
        spec = entry.get("spec") or {}
        ws.cell(row=r_idx, column=1, value=spec.get("dataset", "?"))
        ws.cell(row=r_idx, column=2, value=spec.get("task", "?"))
        ws.cell(row=r_idx, column=3, value=entry.get("n_total", 0))


def _xlsx_esc_sheet(
    ws: "Worksheet", entries: List[Dict[str, Any]],  # noqa: F821
) -> None:
    """ESC: Tag-set Micro/Macro F1 + ROUGE-L F1 (paired with TP/FP/FN counts).

    Mirrors the SER sheet layout but with ESC's specific metric columns
    (see :func:`compute_wer.compute_esc_metrics`). Percentage cells get
    ``0.00%`` formatting so Excel renders 0.2895 as ``28.95%``.
    """
    headers = [
        "Dataset", "n",
        "Tag micro F1", "Tag micro P", "Tag micro R",
        "Tag macro F1", "Tag macro P", "Tag macro R",
        "ROUGE-L F1", "ROUGE-L P", "ROUGE-L R",
        "TP", "FP", "FN", "Vocab",
    ]
    _ws_write_header(ws, headers)
    pct_keys = (
        "tag_micro_f1", "tag_micro_p", "tag_micro_r",
        "tag_macro_f1", "tag_macro_p", "tag_macro_r",
        "rouge_l_f1", "rouge_l_p", "rouge_l_r",
    )
    for r_idx, entry in enumerate(entries, start=2):
        dataset = (entry.get("spec") or {}).get("dataset", "?")
        ws.cell(row=r_idx, column=1, value=dataset)
        ws.cell(row=r_idx, column=2, value=entry.get("n_evaluated",
                                                     entry.get("n_total", 0)))
        for col_offset, key in enumerate(pct_keys, start=3):
            cell = ws.cell(row=r_idx, column=col_offset, value=entry.get(key))
            if cell.value is not None:
                cell.number_format = "0.00%"
        for col_offset, key in enumerate(("tp", "fp", "fn"), start=12):
            ws.cell(row=r_idx, column=col_offset, value=entry.get(key, 0))
        ws.cell(row=r_idx, column=15, value=len(entry.get("labels") or []))


def _xlsx_aggregate_sheet(
    ws: "Worksheet", agg: Dict[str, Any],  # noqa: F821
) -> None:
    _ws_write_header(ws, ["Metric", "Value"])
    rows: List[tuple[str, Any, Optional[str]]] = [
        ("n_evaluated",        agg.get("n_evaluated"),        None),
        ("n_tests_evaluated",  agg.get("n_tests_evaluated"),  None),
        ("n_tests_failed",     agg.get("n_tests_failed"),     None),
        ("WER % (ref_tokens-weighted)", agg.get("wer"),       "0.00"),
    ]
    for r_idx, (k, v, fmt) in enumerate(rows, start=2):
        ws.cell(row=r_idx, column=1, value=k)
        cell = ws.cell(row=r_idx, column=2, value=v)
        if fmt and v is not None:
            cell.number_format = fmt


def _xlsx_performance_sheet(
    ws: "Worksheet", tests: List[Dict[str, Any]],  # noqa: F821
) -> None:
    _ws_write_header(ws, ["Test", "audio_s", "RTF", "wall_s", "speedup"])
    row = 2
    for entry in tests:
        if entry.get("error"):
            continue
        perf = entry.get("perf") or {}
        if not perf:
            continue
        spec = entry.get("spec") or {}
        ws.cell(
            row=row, column=1,
            value=spec.get("label", spec.get("dataset", "?")),
        )
        ws.cell(row=row, column=2, value=perf.get("total_duration_s"))
        ws.cell(row=row, column=3, value=perf.get("rtf"))
        ws.cell(row=row, column=4, value=perf.get("wall_clock_s"))
        ws.cell(row=row, column=5, value=perf.get("speedup"))
        # Floats as-is; Excel default General is fine for ad-hoc inspection.
        row += 1


def _xlsx_failed_sheet(
    ws: "Worksheet", tests: List[Dict[str, Any]],  # noqa: F821
) -> None:
    _ws_write_header(ws, ["Test", "Error"])
    row = 2
    for entry in tests:
        if entry.get("error") is None:
            continue
        spec = entry.get("spec") or {}
        ws.cell(
            row=row, column=1,
            value=spec.get("label", spec.get("dataset", "?")),
        )
        ws.cell(row=row, column=2, value=str(entry.get("error")))
        row += 1


_HW_EXTENDED_METRIC_COLS: List[tuple] = [
    # (header, fn(entry) -> value, excel number_format).
    # Values are already in % (compute_hotword_metrics multiplies by 100),
    # so "0.00" displays e.g. 22.63 verbatim — same convention as WER %.
    ("KER %",   lambda e: ((e.get("overall") or {}).get("ker")),   "0.00"),
    ("SACC %",  lambda e: ((e.get("overall") or {}).get("sacc")),  "0.00"),
    ("B-WER %", lambda e: ((e.get("overall") or {}).get("b_wer")), "0.00"),
    ("U-WER %", lambda e: ((e.get("overall") or {}).get("u_wer")), "0.00"),
    ("PRR %",   lambda e: ((e.get("overall") or {}).get("prr")),   "0.00"),
    ("PF1 %",   lambda e: ((e.get("overall") or {}).get("pf1")),   "0.00"),
]


def _split_hotword_entries_by_mode(
    entries: List[Dict[str, Any]],
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Split asr_hotwords entries into (random_or_none, retrieve) sheets
    based on ``spec.overrides.hotword_mode``. Plans without the field
    fall into the random/none bucket so legacy yaml files keep landing
    in the same physical sheet (renamed to "ASR + Hotwords (Random)")."""
    retrieve_entries: List[Dict[str, Any]] = []
    random_entries: List[Dict[str, Any]] = []
    for e in entries:
        mode = (
            ((e.get("spec") or {}).get("overrides") or {}).get("hotword_mode")
        )
        if mode in ("retrieve", "tower_retrieve"):
            retrieve_entries.append(e)
        else:
            random_entries.append(e)
    return random_entries, retrieve_entries


def write_summary_xlsx(summary: Dict[str, Any], path: Path) -> Path:
    """Write a multi-sheet xlsx mirroring the markdown summary.

    Sheets (created in this order, but only the ones with data are kept;
    an empty workbook would otherwise default to a phantom "Sheet"):
      - ``Meta``                          — run_id / source / model.* / plan file
      - ``Paper-style``                   — wide single-row ASR table
      - ``ASR``                           — per-language WER detail
      - ``ASR + Hotwords (Random)``       — random-padding asr_hotwords
                                            specs (legacy "ASR + Hotwords"
                                            renamed for symmetry with the
                                            retrieve sibling sheet);
                                            columns: Dataset / K / KER %
                                            / SACC % / B-WER % / U-WER %
                                            / PRR % / PF1 % / n / WER %.
      - ``ASR + Hotwords (Retrieve)``     — two-stage retrieve specs;
                                            columns add Top-K / Recall@K %
                                            / PrRR % up front.
      - ``TS-ASR``                        — Dataset / n_pos / n_neg / WER %
                                            / FA % / Miss % / Silence Match %
                                            (silence-aware metrics from
                                            compute_silence_metrics)
      - ``SER``                           — WA / UA / Macro-F1 (0..1, formatted as %)
      - ``ESC``                           — Tag-set Micro/Macro F1 + ROUGE-L F1 + TP/FP/FN
      - ``SEC``                           — sample counts only (no auto metric)
      - ``Aggregate``                     — cross-task ref_tokens-weighted overall
      - ``Performance``                   — per-test RTF / wall / speedup
      - ``Failed tests``                  — any spec carrying ``error``
    """
    from openpyxl import Workbook

    by_task = _bucket_tests_by_task(summary.get("tests", []))
    asr_entries        = by_task.get("asr", [])
    hotword_entries    = by_task.get("asr_hotwords", [])
    tsasr_entries      = by_task.get("ts_asr", [])
    ser_entries        = by_task.get("ser", [])
    esc_entries        = by_task.get("esc", [])
    sec_entries        = by_task.get("sec", [])
    failed_entries     = [
        e for e in summary.get("tests", []) if e.get("error") is not None
    ]
    agg                = summary.get("aggregate") or {}

    random_hw_entries, retrieve_hw_entries = _split_hotword_entries_by_mode(
        hotword_entries,
    )

    wb = Workbook()
    # Workbook() ships with a default "Sheet" worksheet — repurpose it as
    # Meta so we never end up with an unused tab in the output file.
    meta_ws = wb.active
    meta_ws.title = "Meta"
    _xlsx_meta_sheet(meta_ws, summary)

    if asr_entries:
        _xlsx_paper_style_sheet(wb.create_sheet("Paper-style"), summary, asr_entries)
        _xlsx_dataset_overall_sheet(wb.create_sheet("ASR"), asr_entries)
    if random_hw_entries:
        def _random_pool_label(e: Dict[str, Any]) -> Optional[str]:
            import os as _os
            ov = (e.get("spec") or {}).get("overrides") or {}
            pool = (ov.get("random_hotword_pool_file") or "").strip()
            if pool:
                return _os.path.splitext(_os.path.basename(pool))[0]
            return None

        random_extras = [
            (
                "Label",
                _random_pool_label,
            ),
            (
                "K",
                lambda e: ((e.get("spec") or {}).get("overrides") or {})
                    .get("hotwords_pad_to"),
            ),
        ] + _HW_EXTENDED_METRIC_COLS
        _xlsx_dataset_overall_sheet(
            wb.create_sheet("ASR + Hotwords (Random)"),
            random_hw_entries,
            extra_cols=random_extras,
        )
    if retrieve_hw_entries:
        def _pool_label(e: Dict[str, Any]) -> Optional[str]:
            import re as _re
            ov = (e.get("spec") or {}).get("overrides") or {}
            tsv = ov.get("tower_biasing_tsv_file") or ""
            pool = ov.get("tower_hotword_pool_file") or ""
            if tsv:
                m = _re.search(r"biasing[_-](\d+)", tsv)
                return f"biasing_{m.group(1)}" if m else tsv.split("/")[-1]
            if pool:
                return "all"
            return None

        retrieve_extras = [
            (
                "Pool",
                _pool_label,
            ),
            (
                "Top-K",
                lambda e: ((e.get("spec") or {}).get("overrides") or {})
                    .get("retrieve_top_k"),
            ),
            ("Recall@K %", lambda e: ((e.get("overall") or {}).get("recall_at_k")), "0.00"),
            ("PrRR %",     lambda e: ((e.get("overall") or {}).get("prrr")),        "0.00"),
        ] + _HW_EXTENDED_METRIC_COLS
        _xlsx_dataset_overall_sheet(
            wb.create_sheet("ASR + Hotwords (Retrieve)"),
            retrieve_hw_entries,
            extra_cols=retrieve_extras,
        )
    if tsasr_entries:
        _xlsx_tsasr_sheet(wb.create_sheet("TS-ASR"), tsasr_entries)
    if ser_entries:
        _xlsx_ser_sheet(wb.create_sheet("SER"), ser_entries)
    if esc_entries:
        _xlsx_esc_sheet(wb.create_sheet("ESC"), esc_entries)
    if sec_entries:
        _xlsx_secesc_sheet(wb.create_sheet("SEC"), sec_entries)
    if agg.get("wer") is not None:
        _xlsx_aggregate_sheet(wb.create_sheet("Aggregate"), agg)
    if any((e.get("perf") and not e.get("error")) for e in summary.get("tests", [])):
        _xlsx_performance_sheet(
            wb.create_sheet("Performance"), summary.get("tests", []),
        )
    if failed_entries:
        _xlsx_failed_sheet(wb.create_sheet("Failed tests"), failed_entries)

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    wb.save(str(path))
    return path
