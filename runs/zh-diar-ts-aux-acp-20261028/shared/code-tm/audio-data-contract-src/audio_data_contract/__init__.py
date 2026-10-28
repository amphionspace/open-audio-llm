"""Portable contracts for shared audio datasets."""

from .catalog import (
    DatasetCatalog,
    load_catalog,
    resolve_artifact,
    resolve_split,
    validate_catalog,
    verify_artifact_file,
)
from .records import (
    load_examples,
    load_records,
    read_artifact,
    write_examples,
    write_records,
)
from .render import extract_audio_refs, render_example
from .roots import load_roots
from .state import (
    DatasetState,
    DownloadState,
    inspect_download,
    load_state,
    write_state_atomic,
)
from .types import (
    ArtifactRef,
    AudioContent,
    AudioExample,
    AudioRecord,
    AudioRef,
    AudioSlot,
    DatasetRef,
    DatasetSpec,
    DatasetViewSpec,
    Message,
    PromptAudio,
    PromptTemplate,
    PromptText,
    Split,
    TextContent,
    TransformStep,
)
from .views import (
    DatasetViewCatalog,
    load_view_catalog,
    resolve_view,
    validate_view_catalog,
)

__all__ = [
    "ArtifactRef",
    "AudioContent",
    "AudioExample",
    "AudioRecord",
    "AudioRef",
    "AudioSlot",
    "DatasetCatalog",
    "DatasetRef",
    "DatasetSpec",
    "DatasetState",
    "DatasetViewCatalog",
    "DatasetViewSpec",
    "DownloadState",
    "Message",
    "PromptAudio",
    "PromptTemplate",
    "PromptText",
    "Split",
    "TextContent",
    "TransformStep",
    "extract_audio_refs",
    "inspect_download",
    "load_catalog",
    "load_examples",
    "load_records",
    "load_roots",
    "load_state",
    "load_view_catalog",
    "read_artifact",
    "render_example",
    "resolve_artifact",
    "resolve_split",
    "resolve_view",
    "validate_catalog",
    "validate_view_catalog",
    "verify_artifact_file",
    "write_examples",
    "write_records",
    "write_state_atomic",
]

__version__ = "0.2.1"
