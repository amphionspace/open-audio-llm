"""Logical dataset views layered over physical catalog entries."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from .catalog import DatasetCatalog
from .declarations import declaration_files, read_declarations
from .errors import ContractError
from .types import DatasetSpec, DatasetViewSpec


class DatasetViewCatalog:
    def __init__(self, views: Iterable[DatasetViewSpec]):
        self._by_key: dict[str, DatasetViewSpec] = {}
        self._versions: dict[str, list[DatasetViewSpec]] = {}
        for view in views:
            if view.key in self._by_key:
                raise ContractError(f"duplicate dataset view key: {view.key}")
            self._by_key[view.key] = view
            self._versions.setdefault(view.view_id, []).append(view)
        for versions in self._versions.values():
            versions.sort(key=lambda item: item.version)

    def __len__(self) -> int:
        return len(self._by_key)

    def __iter__(self):
        return iter(self._by_key.values())

    def get(self, view_id: str, version: str | None = None) -> DatasetViewSpec:
        versions = self._versions.get(view_id)
        if not versions:
            raise ContractError(f"unknown dataset view: {view_id!r}")
        if version is None:
            if len(versions) != 1:
                available = [item.version for item in versions]
                raise ContractError(
                    f"dataset view {view_id!r} requires a version; available: {available}"
                )
            return versions[0]
        try:
            return self._by_key[f"{view_id}@{version}"]
        except KeyError as exc:
            raise ContractError(
                f"unknown dataset view version: {view_id}@{version}"
            ) from exc


def validate_view_catalog(
    views: Iterable[DatasetViewSpec], datasets: DatasetCatalog
) -> DatasetViewCatalog:
    catalog = DatasetViewCatalog(views)
    for view in catalog:
        datasets.get(view.source.dataset_id, view.source.version)
        result = datasets.get(view.result.dataset_id, view.result.version)
        allowed_parents = {view.source.dataset_id}
        if view.result.dataset_id == view.source.dataset_id:
            allowed_parents.add(None)
        if view.lineage_status == "exact" and result.derived_from not in allowed_parents:
            raise ContractError(
                f"view {view.key} result derives from {result.derived_from!r}, "
                f"not {view.source.dataset_id!r}"
            )
    return catalog


def load_view_catalog(path: str | Path, datasets: DatasetCatalog) -> DatasetViewCatalog:
    views: list[DatasetViewSpec] = []
    for source in declaration_files(path):
        for line_number, row in read_declarations(source):
            try:
                views.append(DatasetViewSpec.from_dict(row))
            except ContractError as exc:
                raise ContractError(f"{source}:{line_number}: {exc}") from exc
    return validate_view_catalog(views, datasets)


def resolve_view(
    views: DatasetViewCatalog,
    datasets: DatasetCatalog,
    view_id: str,
    version: str | None = None,
) -> DatasetSpec:
    view = views.get(view_id, version)
    return datasets.get(view.result.dataset_id, view.result.version)
