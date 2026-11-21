"""Small typed registries for model components."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Generic, TypeVar

T = TypeVar("T")


class ComponentRegistry(Generic[T]):
    """Name -> factory registry with explicit errors for unsupported plugins."""

    def __init__(self, name: str):
        self.name = name
        self._items: dict[str, Callable[..., T]] = {}

    def register(self, key: str, factory: Callable[..., T] | None = None):
        def _decorator(fn: Callable[..., T]) -> Callable[..., T]:
            if key in self._items:
                raise ValueError(f"{self.name} component already registered: {key}")
            self._items[key] = fn
            return fn

        return _decorator(factory) if factory is not None else _decorator

    def build(self, key: str, *args: Any, **kwargs: Any) -> T:
        try:
            factory = self._items[key]
        except KeyError as exc:
            raise ValueError(
                f"Unsupported {self.name}: {key!r}. "
                f"Available: {sorted(self._items)}"
            ) from exc
        return factory(*args, **kwargs)

    def keys(self) -> list[str]:
        return sorted(self._items)


audio_tower_registry: ComponentRegistry[Any] = ComponentRegistry("audio_tower")
connector_registry: ComponentRegistry[Any] = ComponentRegistry("connector")
