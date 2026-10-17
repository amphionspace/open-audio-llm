"""Q-Former connector extension point."""

from __future__ import annotations

from .mlp_downsample import MLPDownsampleConnector
from open_audio_llm.registry import connector_registry


@connector_registry.register("qformer")
class QFormerConnector(MLPDownsampleConnector):
    """Temporary compatible connector until a true Q-Former is introduced."""
