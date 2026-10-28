"""ms-swift integration package."""

from .patches import apply_patches

apply_patches()

__all__ = ["apply_patches"]
