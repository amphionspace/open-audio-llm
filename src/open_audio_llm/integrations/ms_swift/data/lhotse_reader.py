"""Lhotse format I/O utilities.

Supports two lhotse input formats (auto-detected):
  - Supervisions + Recordings: separate *_supervisions_* and *_recordings_* files
  - Cuts: single *_cuts_* file with embedded supervisions and recording per cut
"""

import glob
import gzip
import os

try:
    import ujson as json
except ImportError:  # pragma: no cover - optional speed-up
    import json


def infer_recordings_path(supervisions_path):
    """Infer the recordings file path from the supervisions path.

    Lhotse convention: {prefix}_supervisions_{split}[_extras].jsonl[.gz]
    corresponds to  {prefix}_recordings_{split}[...].jsonl[.gz]
    in the same directory.
    """
    dirname = os.path.dirname(supervisions_path)
    basename = os.path.basename(supervisions_path)

    parts = basename.split('_supervisions_', 1)
    if len(parts) != 2:
        raise ValueError(
            f'Cannot infer recordings path: supervisions filename does not contain '
            f'"_supervisions_": {basename}')

    prefix = parts[0]
    suffix = parts[1]
    split = suffix.split('_')[0].split('.')[0]

    narrow = os.path.join(dirname, f'{prefix}_recordings_{split}*')
    candidates = sorted(glob.glob(narrow))

    if not candidates:
        broad = os.path.join(dirname, f'{prefix}_recordings_*')
        candidates = sorted(glob.glob(broad))
        pattern = broad
    else:
        pattern = narrow

    if len(candidates) == 1:
        return candidates[0]
    elif len(candidates) == 0:
        raise FileNotFoundError(
            f'No recordings file matching "{pattern}" found. '
            f'Please specify recordings path explicitly.')
    else:
        raise ValueError(
            f'Multiple recordings files match "{pattern}": {candidates}. '
            f'Please specify recordings path explicitly.')


def load_jsonl_gz(path):
    """Load a gzipped or plain JSONL file into a dict keyed by ``id``."""
    opener = gzip.open if path.endswith('.gz') else open
    data = {}
    with opener(path, 'rt', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            data[item['id']] = item
    return data


def detect_lhotse_format(path):
    """Peek at the first record to determine if the file is cuts or supervisions."""
    opener = gzip.open if path.endswith('.gz') else open
    with opener(path, 'rt', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            if 'supervisions' in item or item.get('type') in ('MonoCut', 'MixedCut'):
                return 'cuts'
            return 'supervisions'
    return 'supervisions'


def load_cuts_gz(path):
    """Load a lhotse cuts file, returning (supervisions_dict, recordings_dict)."""
    opener = gzip.open if path.endswith('.gz') else open
    supervisions = {}
    recordings = {}
    with opener(path, 'rt', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            cut = json.loads(line)
            rec = cut.get('recording')
            if rec:
                recordings[rec['id']] = rec
            for sup in cut.get('supervisions', []):
                supervisions[sup['id']] = sup
    return supervisions, recordings


def load_supervisions_auto(path):
    """Load supervisions from either a supervisions or cuts file (auto-detect)."""
    fmt = detect_lhotse_format(path)
    if fmt == 'cuts':
        sups, _ = load_cuts_gz(path)
        return sups
    return load_jsonl_gz(path)
