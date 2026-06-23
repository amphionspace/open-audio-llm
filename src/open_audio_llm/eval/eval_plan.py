"""Small eval-plan representation for migrated tasks."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path


@dataclass
class EvalDataset:
    name: str
    index_path: str
    task: str = "asr"
    split: str = "test"
    options: dict = field(default_factory=dict)


@dataclass
class EvalPlan:
    datasets: list[EvalDataset]

    @classmethod
    def from_json(cls, path: str | Path) -> "EvalPlan":
        with Path(path).open("r", encoding="utf-8") as f:
            raw = json.load(f)
        return cls(datasets=[EvalDataset(**item) for item in raw.get("datasets", [])])
