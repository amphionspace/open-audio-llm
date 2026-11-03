"""Read DATA-TTS-UNIFIED Lance releases as AudioRecords with embedded audio bytes.

These tables follow the TTS data contract (27 base columns, audio stored in the
table), not the audio-record-lance mapping in lance.py. The catalog pins the
release manifest by sha256 and its Lance snapshot; reads never use latest.

    release = open_release(catalog, "aishell3", "tts-unified-v0.1", "samples", roots)
    shard, shards = worker_shard(rank, world_size)
    for record, audio in iter_samples(release, shard=shard, num_shards=shards):
        waveform, sr = soundfile.read(io.BytesIO(audio))
"""

from __future__ import annotations

import hashlib
import json
import random
import re
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from .catalog import resolve_artifact
from .errors import ContractError
from .types import AudioRecord, AudioRef, AudioSlot

ARTIFACT_KIND = "tts-lance-release"
SPLIT = "train"
_SCALARS = [
    "sample_id",
    "text",
    "text_kind",
    "language",
    "speaker_id",
    "speaker_scope",
    "duration_seconds",
    "sample_rate",
    "channels",
    "metadata_json",
]


@dataclass(frozen=True)
class TtsRelease:
    dataset_id: str
    version: str
    manifest_path: Path
    manifest: dict[str, Any]
    dataset: Any


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def open_release(catalog, dataset_id, version, artifact_name, roots) -> TtsRelease:
    """Open a registered release at exactly the manifest and snapshot it pins."""
    try:
        import lance
    except ImportError as exc:
        raise ContractError("install audio-data-contract[lance] to read Lance") from exc

    ref = catalog.get(dataset_id, version).artifact(artifact_name)
    if ref.kind != ARTIFACT_KIND:
        raise ContractError(f"artifact {artifact_name!r} is not a {ARTIFACT_KIND}")
    path = resolve_artifact(catalog, dataset_id, version, artifact_name, roots)
    if _sha256(path) != ref.sha256:
        raise ContractError(f"release manifest sha256 differs from catalog: {path}")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("status") != "complete":
        raise ContractError(f"release manifest is not complete: {path}")
    for key in ("lance_version", "rows"):
        if manifest.get(key) != ref.metadata[key]:
            raise ContractError(f"release manifest {key} differs from catalog: {path}")
    try:
        dataset = lance.dataset(
            str(path.parent / manifest["table_path"]), version=manifest["lance_version"]
        )
        rows = dataset.count_rows()
    except Exception as exc:
        raise ContractError(f"cannot open pinned Lance snapshot for {path}: {exc}") from exc
    if rows != manifest["rows"]:
        raise ContractError(f"Lance snapshot row count differs from manifest: {path}")
    return TtsRelease(dataset_id, version, path, manifest, dataset)


def _record(release: TtsRelease, row: dict[str, Any]) -> AudioRecord:
    metadata = {
        key: row[key]
        for key in (
            "text_kind",
            "speaker_id",
            "speaker_scope",
            "duration_seconds",
            "sample_rate",
            "channels",
        )
        if row[key] is not None
    }
    original_split = json.loads(row["metadata_json"]).get("original_split")
    if original_split is not None:
        metadata["original_split"] = original_split
    ref = AudioRef(release.dataset_id, release.version, SPLIT, row["sample_id"])
    return AudioRecord(
        id=row["sample_id"],
        task="asr",
        audio_slots=(AudioSlot("audio", ref),),
        target=row["text"],
        language=row["language"] or "N/A",
        metadata=metadata,
    )


def iter_samples(
    release: TtsRelease,
    *,
    filter: str | None = None,
    with_audio: bool = True,
    shard: int = 0,
    num_shards: int = 1,
    seed: int = 0,
    epoch: int = 0,
    batch_size: int = 1024,
):
    """Yield (AudioRecord, audio bytes or None) for rows with text in this shard.

    Shards are whole fragments, reshuffled per (seed, epoch) like lance_stream;
    `filter` is an extra Lance SQL predicate, e.g. "language = 'zh'".
    """
    if not 0 <= shard < num_shards:
        raise ContractError("shard must be in [0, num_shards)")
    expression = "text IS NOT NULL" + (f" AND ({filter})" if filter else "")
    columns = _SCALARS + (["audio"] if with_audio else [])
    fragments = sorted(release.dataset.get_fragments(), key=lambda f: f.fragment_id)
    random.Random(f"{seed}:{epoch}").shuffle(fragments)
    for fragment in fragments[shard::num_shards]:
        for batch in fragment.to_batches(
            columns=columns, filter=expression, batch_size=batch_size
        ):
            for row in batch.to_pylist():
                audio = row["audio"]["bytes"] if with_audio else None
                yield _record(release, row), audio


def read_audio(release: TtsRelease, sample_ids) -> dict[str, bytes]:
    """Encoded audio bytes for the given sample IDs, via the sample_id index."""
    ids = list(dict.fromkeys(sample_ids))
    if not ids:
        return {}
    quoted = ", ".join("'" + value.replace("'", "''") + "'" for value in ids)
    table = release.dataset.to_table(
        columns=["sample_id", "audio"], filter=f"sample_id IN ({quoted})"
    )
    found = {
        row["sample_id"]: row["audio"]["bytes"] for row in table.to_pylist()
    }
    missing = [value for value in ids if value not in found]
    if missing:
        raise ContractError(f"unknown sample IDs in {release.dataset_id}: {missing[:5]}")
    return found


# ASR read rules over the TTS text; nothing is written back to the release.
ASR_RULES_VERSION = "tts-asr-rules/v1"
_FORMAT_TAGS = re.compile(r"</?(?:i|b|u)>|<color=[^>]*>|</color>")
_EMPTY_BRACKETS = re.compile(r"\[\s*\]")
_MARKUP = re.compile(r"[{}<>\[\]]")
_LEXICAL = re.compile(r"[^\W_]")
_KANA = re.compile(r"[぀-ヿ]")
_HANGUL = re.compile(r"[가-힯]")
_HAN = re.compile(r"[一-鿿]")
_CJK = re.compile(r"[぀-ヿ가-힯一-鿿]")
# Japanese lines this long with no kana are Chinese text under a ja label.
_JA_MIN_HAN_WITHOUT_KANA = 6


def _script_mismatch(language, text):
    if language == "zh":
        return bool(_KANA.search(text) or _HANGUL.search(text))
    if language == "ja":
        return bool(_HANGUL.search(text)) or (
            not _KANA.search(text)
            and len(_HAN.findall(text)) >= _JA_MIN_HAN_WITHOUT_KANA
        )
    if language == "ko":
        return not _HANGUL.search(text) and bool(_CJK.search(text))
    return language != "N/A" and bool(_CJK.search(text))


def apply_asr_rules(record: AudioRecord, *, max_duration: float | None = 30.0):
    """(cleaned record, None) or (None, reason) under ASR_RULES_VERSION.

    Excludes original dev/test/valid rows, unresolvable markup (game
    placeholders, ruby, audiobook [Illustration: ...]), text without letters,
    language/script mismatches and audio longer than max_duration seconds.
    Language tags keep only the primary subtag (zh-CN -> zh, en-US -> en).
    """
    metadata = dict(record.metadata)
    split = metadata.get("original_split")
    if split is not None and not str(split).startswith("train"):
        return None, "eval_split"
    duration = metadata.get("duration_seconds")
    if max_duration is not None and duration is not None and duration > max_duration:
        return None, "too_long"
    text = _FORMAT_TAGS.sub("", record.target)
    text = " ".join(_EMPTY_BRACKETS.sub(" ", text).split()).lstrip("#").lstrip()
    if _MARKUP.search(text):
        return None, "markup"
    if not _LEXICAL.search(text):
        return None, "no_lexical"
    language = record.language.split("-")[0].lower() if record.language != "N/A" else "N/A"
    if _script_mismatch(language, text):
        return None, "script_mismatch"
    if language != record.language:
        metadata["source_language"] = record.language
    metadata["asr_rules"] = ASR_RULES_VERSION
    return replace(record, target=text, language=language, metadata=metadata), None


def iter_asr_samples(release: TtsRelease, *, max_duration: float | None = 30.0, **kwargs):
    """iter_samples() restricted to rows kept by apply_asr_rules()."""
    for record, audio in iter_samples(release, **kwargs):
        cleaned, _ = apply_asr_rules(record, max_duration=max_duration)
        if cleaned is not None:
            yield cleaned, audio
