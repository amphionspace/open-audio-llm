"""YAML test-plan loader shared by decode.py and the vLLM eval script.

A test plan is a small declarative spec that lists the (dataset, task, ...)
combinations to evaluate in a single run. Driven from a YAML file::

    run_name: amphion_4b_full          # optional; default = <ckpt>-<timestamp>
    defaults:                          # all fields optional, applied to every test
      max_new_tokens: 200
      repetition_penalty: 1.0
      max_hyp_ratio: 10.0
      num_samples: null
    tests:
      - dataset: ts_hw_test            # task auto-derived to "ts_asr"
      - dataset: librispeech           # task auto-derived to "asr"
      - dataset: aishell
      - dataset: magicdata
        task: asr_hotwords             # explicit override
        num_hotwords: 10               # task-specific knob

Both ``src/decode.py`` and ``src/open_audio_llm/integrations/vllm/test_vllm_inference.py``
import :func:`load_test_plan` and iterate ``plan.tests`` so the two paths
share the exact same definition of "what to evaluate".
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from open_audio_llm.integrations.vllm.dataset_registry import (
    _resolve_test_cuts_from_dataset_name,
    get_default_task,
)


# Keys legal at the test-plan top level. Anything else is a typo.
_TOP_LEVEL_KEYS = {"run_name", "defaults", "tests"}
# Keys legal inside a single test entry. ``overrides`` collects everything
# that's not in this set (so per-task knobs like num_hotwords just work).
_TEST_RESERVED_KEYS = {"dataset", "task", "label"}


@dataclass
class TestSpec:
    """One (dataset, task) evaluation request derived from a plan entry."""

    dataset: str                # registry key or "<lang>:<name>" multilingual form
    task: str                   # always non-empty after loader
    label: str                  # used as per_test/<label>/ directory name
    overrides: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "dataset": self.dataset,
            "task": self.task,
            "label": self.label,
            "overrides": dict(self.overrides),
        }


@dataclass
class TestPlan:
    """A list of TestSpecs plus optional run_name.

    The loader is the single source of truth for default-task derivation,
    overrides merging, label de-duplication, and dataset existence checks.
    Downstream consumers should treat the resulting plan as fully resolved.
    """

    run_name: Optional[str]
    tests: List[TestSpec]
    plan_file: Optional[str] = None      # path to the YAML, for provenance

    def to_dict(self) -> dict:
        return {
            "run_name": self.run_name,
            "plan_file": self.plan_file,
            "tests": [t.to_dict() for t in self.tests],
        }


# ---------------------------------------------------------------------------
# Public loader
# ---------------------------------------------------------------------------

def load_test_plan(
    *,
    plan_file: Optional[str] = None,
    test_dataset: Optional[str] = None,
    cli_task: Optional[str] = None,
    cli_overrides: Optional[Dict[str, Any]] = None,
    args: Optional[argparse.Namespace] = None,
    validate_existence: bool = True,
) -> TestPlan:
    """Resolve a TestPlan from CLI args and/or a YAML plan file.

    Exactly one of *plan_file* / *test_dataset* must be provided; passing both
    or neither raises :class:`ValueError`. *cli_task* / *cli_overrides* only
    apply to the single-test CLI mode (in plan-file mode all per-test knobs
    must come from the YAML).

    When *validate_existence* is True (default) every TestSpec is round-tripped
    through :func:`_resolve_test_cuts_from_dataset_name` so missing manifests /
    typo'd dataset names are caught up-front, before the model is loaded.
    """
    if bool(plan_file) == bool(test_dataset):
        raise ValueError(
            "load_test_plan requires exactly one of --test-plan-file or "
            "--test-dataset (got both or neither)."
        )

    if plan_file is not None:
        if cli_task is not None:
            raise ValueError(
                "--task cannot be combined with --test-plan-file; specify "
                "task per entry inside the YAML."
            )
        plan = _load_from_yaml(plan_file)
    else:
        plan = _build_single_spec_plan(
            dataset=test_dataset,
            task=cli_task,
            overrides=cli_overrides or {},
        )

    if validate_existence and args is not None:
        _validate_specs_exist(plan, args)

    return plan


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_from_yaml(plan_file: str) -> TestPlan:
    path = Path(plan_file)
    if not path.is_file():
        raise FileNotFoundError(f"Test plan file not found: {plan_file}")
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    if not isinstance(raw, dict):
        raise ValueError(
            f"Test plan {plan_file}: top-level YAML must be a mapping, got "
            f"{type(raw).__name__}"
        )

    unknown = set(raw) - _TOP_LEVEL_KEYS
    if unknown:
        raise ValueError(
            f"Test plan {plan_file}: unknown top-level keys: {sorted(unknown)}"
            f" (allowed: {sorted(_TOP_LEVEL_KEYS)})"
        )

    raw_tests = raw.get("tests")
    if not isinstance(raw_tests, list) or not raw_tests:
        raise ValueError(
            f"Test plan {plan_file}: 'tests' must be a non-empty list."
        )

    defaults = raw.get("defaults") or {}
    if not isinstance(defaults, dict):
        raise ValueError(
            f"Test plan {plan_file}: 'defaults' must be a mapping if present."
        )

    specs: List[TestSpec] = []
    used_labels: Dict[str, int] = {}
    for i, entry in enumerate(raw_tests):
        if not isinstance(entry, dict):
            raise ValueError(
                f"Test plan {plan_file}: tests[{i}] must be a mapping."
            )
        dataset = entry.get("dataset")
        if not dataset or not isinstance(dataset, str):
            raise ValueError(
                f"Test plan {plan_file}: tests[{i}] missing required "
                f"string 'dataset'."
            )
        task = entry.get("task") or get_default_task(dataset)
        if not task:
            raise ValueError(
                f"Test plan {plan_file}: tests[{i}] dataset '{dataset}' has "
                "no default task; please specify 'task: <name>' explicitly."
            )

        # Anything in the entry beyond reserved keys becomes an override; keys
        # already present in defaults are merged with entry-level winning.
        overrides = dict(defaults)
        for k, v in entry.items():
            if k in _TEST_RESERVED_KEYS:
                continue
            overrides[k] = v

        explicit_label = entry.get("label")
        base_label = explicit_label or f"{_safe_label(dataset)}__{task}"
        label = _dedupe_label(base_label, used_labels)
        specs.append(TestSpec(
            dataset=dataset, task=task, label=label, overrides=overrides,
        ))

    run_name = raw.get("run_name")
    if run_name is not None and not isinstance(run_name, str):
        raise ValueError(
            f"Test plan {plan_file}: 'run_name' must be a string if present."
        )

    _validate_hotword_overrides(specs, plan_file=plan_file)
    specs = _expand_hotwords_sweep(specs, used_labels, plan_file=plan_file)
    specs = _expand_retrieve_top_k_sweep(
        specs, used_labels, plan_file=plan_file,
    )
    return TestPlan(run_name=run_name, tests=specs, plan_file=str(path))


def _build_single_spec_plan(
    *, dataset: str, task: Optional[str], overrides: Dict[str, Any],
) -> TestPlan:
    if not dataset:
        raise ValueError("--test-dataset must be a non-empty string.")
    resolved_task = task or get_default_task(dataset)
    if not resolved_task:
        raise ValueError(
            f"Dataset '{dataset}' has no default task; pass --task explicitly."
        )
    label = f"{_safe_label(dataset)}__{resolved_task}"
    return TestPlan(
        run_name=None,
        tests=[TestSpec(
            dataset=dataset, task=resolved_task, label=label,
            overrides=dict(overrides),
        )],
        plan_file=None,
    )


def _safe_label(dataset: str) -> str:
    """Sanitise a dataset name (which may contain ``:`` for multilingual) so
    it is safe to use as a directory component on every filesystem.
    """
    return dataset.replace(":", "__")


def _dedupe_label(base: str, used: Dict[str, int]) -> str:
    """Ensure label uniqueness by appending ``__2`` / ``__3`` on collision."""
    if base not in used:
        used[base] = 1
        return base
    used[base] += 1
    candidate = f"{base}__{used[base]}"
    # Walk forward in the rare case of pathological collisions.
    while candidate in used:
        used[base] += 1
        candidate = f"{base}__{used[base]}"
    used[candidate] = 1
    return candidate


def _expand_hotwords_sweep(
    specs: List[TestSpec],
    used_labels: Dict[str, int],
    *,
    plan_file: Optional[str] = None,
) -> List[TestSpec]:
    """Expand specs carrying ``hotwords_pad_to_sweep`` into one sub-spec per K.

    Each sub-spec inherits the parent's dataset + task + remaining overrides
    but pins ``hotwords_pad_to`` to a single value from the sweep list and
    suffixes its label with ``__hwK{K}``.

    Special case ``K == 0``: prompt-side ``Hotwords:`` line is fully
    suppressed (``no_hotwords=True``) so the sub-spec acts as a "no
    hotwords" baseline. ``hotwords_pad_to=0`` is still recorded on the
    overrides so downstream tooling (e.g. summary.xlsx K column) can show
    "0" alongside the sweep siblings instead of an empty cell.

    Combining a sweep with an explicit ``hotwords_pad_to`` is rejected as
    ambiguous; non-int / negative values in the sweep list are caught here
    so the error fires at plan-load time rather than mid-run.
    """
    where = f"Test plan {plan_file}: " if plan_file else ""
    out: List[TestSpec] = []
    for spec in specs:
        if "hotwords_pad_to_sweep" not in spec.overrides:
            out.append(spec)
            continue

        sweep = spec.overrides["hotwords_pad_to_sweep"]
        if not isinstance(sweep, list) or not sweep:
            raise ValueError(
                f"{where}spec '{spec.label}': 'hotwords_pad_to_sweep' must be "
                f"a non-empty list, got {sweep!r}."
            )
        if "hotwords_pad_to" in spec.overrides:
            raise ValueError(
                f"{where}spec '{spec.label}': set either 'hotwords_pad_to' "
                f"or 'hotwords_pad_to_sweep', not both."
            )

        seen_k: set = set()
        ks: List[int] = []
        for v in sweep:
            if not isinstance(v, int) or isinstance(v, bool) or v < 0:
                raise ValueError(
                    f"{where}spec '{spec.label}': "
                    f"'hotwords_pad_to_sweep' values must be non-negative "
                    f"ints (0 = no-hotwords baseline), got {v!r}."
                )
            if v in seen_k:
                continue
            seen_k.add(v)
            ks.append(v)

        base_overrides = {
            k: v for k, v in spec.overrides.items()
            if k != "hotwords_pad_to_sweep"
        }
        for k in ks:
            sub_overrides = dict(base_overrides)
            sub_overrides["hotwords_pad_to"] = k
            if k == 0:
                # K=0 is the explicit no-hotwords baseline; suppress the
                # Hotwords: prompt line entirely. _pad_hotwords is also a
                # no-op when pad_to=0, so manifest-supplied real hotwords
                # never reach the model — exactly what a baseline needs.
                sub_overrides["no_hotwords"] = True
            sub_label = _dedupe_label(f"{spec.label}__hwK{k}", used_labels)
            out.append(TestSpec(
                dataset=spec.dataset,
                task=spec.task,
                label=sub_label,
                overrides=sub_overrides,
            ))
    return out


_VALID_HOTWORD_MODES = {"none", "random", "retrieve", "tower_retrieve"}


def _validate_hotword_overrides(
    specs: List[TestSpec],
    *,
    plan_file: Optional[str] = None,
) -> None:
    """Cross-check hotword_mode / pad / retrieve fields are mutually consistent.

    Caught at plan-load time so users see typos before the model is loaded:

    - ``hotword_mode`` (if present) must be one of {none, random, retrieve,
      tower_retrieve}.
    - ``hotword_mode`` in {retrieve, tower_retrieve} cannot coexist with
      ``hotwords_pad_to`` > 0 / ``hotwords_pad_to_sweep`` (random-padding
      vs retrieve are two different algorithms; mixing them double-injects
      hotwords into the prompt).
    - ``retrieve_top_k`` / ``retrieve_top_k_sweep`` may only appear when
      ``hotword_mode`` is retrieve or tower_retrieve (otherwise they're
      silently ignored, which is confusing to debug).
    - ``retrieve_top_k_sweep`` and ``retrieve_top_k`` are mutually
      exclusive on the same spec; pick one.
    """
    where = f"Test plan {plan_file}: " if plan_file else ""
    for spec in specs:
        ov = spec.overrides
        mode = ov.get("hotword_mode")
        if mode is not None and mode not in _VALID_HOTWORD_MODES:
            raise ValueError(
                f"{where}spec '{spec.label}': hotword_mode must be one of "
                f"{sorted(_VALID_HOTWORD_MODES)}, got {mode!r}."
            )
        is_retrieve = mode in ("retrieve", "tower_retrieve")
        has_pad_sweep = "hotwords_pad_to_sweep" in ov
        has_pad_value = (
            "hotwords_pad_to" in ov
            and ov.get("hotwords_pad_to") not in (None, 0)
        )
        has_topk_sweep = "retrieve_top_k_sweep" in ov
        has_topk_value = "retrieve_top_k" in ov

        if is_retrieve and (has_pad_sweep or has_pad_value):
            raise ValueError(
                f"{where}spec '{spec.label}': hotword_mode={mode!r} cannot "
                f"coexist with hotwords_pad_to / hotwords_pad_to_sweep "
                f"(retrieve picks the hotword set itself; remove the "
                f"random-padding fields)."
            )
        if (has_topk_sweep or has_topk_value) and not is_retrieve:
            raise ValueError(
                f"{where}spec '{spec.label}': retrieve_top_k* requires "
                f"hotword_mode=retrieve or tower_retrieve; got "
                f"hotword_mode={mode!r}."
            )
        if has_topk_sweep and has_topk_value:
            raise ValueError(
                f"{where}spec '{spec.label}': set either retrieve_top_k or "
                f"retrieve_top_k_sweep, not both."
            )


def _expand_retrieve_top_k_sweep(
    specs: List[TestSpec],
    used_labels: Dict[str, int],
    *,
    plan_file: Optional[str] = None,
) -> List[TestSpec]:
    """Expand specs carrying ``retrieve_top_k_sweep`` into one sub-spec per K.

    Mirrors :func:`_expand_hotwords_sweep`: each sub-spec inherits dataset /
    task / remaining overrides, pins ``retrieve_top_k`` to a single value
    from the sweep list, and suffixes its label with ``__retK{K}``. Used
    by plans like ``configs/eval_plans/hotwords_retrieve.yaml`` to cover
    e.g. ``[5, 10, 15, 20]`` with one config.

    ``K == 0`` is rejected here because retrieve mode without a top-k
    target is meaningless (the no-hotwords baseline lives in the random
    sweep at ``hotwords_pad_to_sweep: [0]``).
    """
    where = f"Test plan {plan_file}: " if plan_file else ""
    out: List[TestSpec] = []
    for spec in specs:
        if "retrieve_top_k_sweep" not in spec.overrides:
            out.append(spec)
            continue

        sweep = spec.overrides["retrieve_top_k_sweep"]
        if not isinstance(sweep, list) or not sweep:
            raise ValueError(
                f"{where}spec '{spec.label}': 'retrieve_top_k_sweep' must "
                f"be a non-empty list, got {sweep!r}."
            )

        seen_k: set = set()
        ks: List[int] = []
        for v in sweep:
            if not isinstance(v, int) or isinstance(v, bool) or v <= 0:
                raise ValueError(
                    f"{where}spec '{spec.label}': "
                    f"'retrieve_top_k_sweep' values must be positive ints "
                    f"(K=0 is the no-hotwords baseline; use "
                    f"hotwords_pad_to_sweep:[0] for that), got {v!r}."
                )
            if v in seen_k:
                continue
            seen_k.add(v)
            ks.append(v)

        base_overrides = {
            k: v for k, v in spec.overrides.items()
            if k != "retrieve_top_k_sweep"
        }
        for k in ks:
            sub_overrides = dict(base_overrides)
            sub_overrides["retrieve_top_k"] = k
            sub_label = _dedupe_label(f"{spec.label}__retK{k}", used_labels)
            out.append(TestSpec(
                dataset=spec.dataset,
                task=spec.task,
                label=sub_label,
                overrides=sub_overrides,
            ))
    return out


def _validate_specs_exist(plan: TestPlan, args: argparse.Namespace) -> None:
    """Resolve every spec via the registry to surface manifest errors early."""
    for spec in plan.tests:
        try:
            _resolve_test_cuts_from_dataset_name(spec.dataset, args)
        except Exception as e:
            raise ValueError(
                f"Test plan validation failed for spec '{spec.label}' "
                f"(dataset='{spec.dataset}', task='{spec.task}'): {e}"
            ) from e
        logging.debug(
            "test_plan: validated dataset '%s' for spec '%s'",
            spec.dataset, spec.label,
        )
