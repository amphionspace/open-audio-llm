"""Special-token helpers."""

from __future__ import annotations


def token_id_is_set(token_id: int | None) -> bool:
    return token_id is not None and token_id >= 0
