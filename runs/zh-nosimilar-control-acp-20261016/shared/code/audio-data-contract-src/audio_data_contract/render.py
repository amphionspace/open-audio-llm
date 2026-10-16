"""Deterministic rendering from stable audio facts to multimodal messages."""

from __future__ import annotations

import random
from collections.abc import Mapping
from typing import Any

from .errors import ContractError
from .types import (
    AudioContent,
    AudioExample,
    AudioRecord,
    AudioRef,
    Message,
    PromptAudio,
    PromptTemplate,
    PromptText,
    TextContent,
)


def render_example(
    record: AudioRecord,
    template: PromptTemplate,
    *,
    seed: int = 0,
    context: Mapping[str, Any] | None = None,
) -> AudioExample:
    rng = random.Random(f"{template.template_id}:{template.version}:{record.id}:{seed}")
    variant_index = rng.randrange(len(template.user_variants))
    variant = template.user_variants[variant_index]
    values: dict[str, Any] = {
        "target": record.target,
        "language": record.language,
        "hotwords": ",".join(record.hotwords) if record.hotwords else "N/A",
    }
    if context:
        values.update(context)
    content = []
    for block in variant:
        if isinstance(block, PromptText):
            try:
                content.append(TextContent(block.text.format_map(values)))
            except KeyError as exc:
                raise ContractError(
                    f"template {template.template_id!r} requires missing context {exc.args[0]!r}"
                ) from exc
        elif isinstance(block, PromptAudio):
            slot = record.slot(block.slot)
            content.append(
                AudioContent(ref=slot.ref, slot=slot.name, purpose=slot.purpose)
            )
        else:
            raise ContractError(f"unsupported prompt block: {type(block).__name__}")
    metadata = dict(record.metadata)
    metadata["prompt_template"] = {
        "id": template.template_id,
        "version": template.version,
        "seed": seed,
        "variant": variant_index,
    }
    return AudioExample(
        id=record.id,
        task=record.task,
        messages=(
            Message(role="user", content=tuple(content)),
            Message(role="assistant", content=(TextContent(record.target),)),
        ),
        labels=dict(record.labels),
        metadata=metadata,
    )


def extract_audio_refs(example: AudioExample) -> list[AudioRef]:
    return [
        item.ref
        for message in example.messages
        for item in message.content
        if isinstance(item, AudioContent)
    ]
