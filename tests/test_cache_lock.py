import fcntl
import multiprocessing
import time

import pytest

from open_audio_llm.data import catalog_cache
from open_audio_llm.data.catalog_cache import lock_exclusive


def test_lock_waits_through_eagain_from_fuse(tmp_path, monkeypatch):
    # quarkfs reports a contended blocking flock as EAGAIN instead of waiting.
    calls = []
    real_flock = fcntl.flock

    def flock(handle, operation):
        calls.append(operation)
        if len(calls) < 3:
            raise BlockingIOError(11, "Resource temporarily unavailable")
        return real_flock(handle, operation)

    monkeypatch.setattr(catalog_cache.fcntl, "flock", flock)
    monkeypatch.setattr(catalog_cache, "LOCK_RETRY_SECONDS", 0)
    with (tmp_path / "build.lock").open("a") as lock:
        lock_exclusive(lock)
    assert calls == [fcntl.LOCK_EX | fcntl.LOCK_NB] * 3


def hold(path, ready, seconds):
    with open(path, "a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        ready.set()
        time.sleep(seconds)


def test_lock_is_exclusive_until_holder_releases(tmp_path):
    path = tmp_path / "build.lock"
    ready = multiprocessing.Event()
    holder = multiprocessing.Process(target=hold, args=(path, ready, 1.0))
    holder.start()
    assert ready.wait(10)
    started = time.monotonic()
    with path.open("a") as lock:
        lock_exclusive(lock)
        waited = time.monotonic() - started
    holder.join()
    assert waited > 0.5


def test_lock_open_retries_eexist_from_concurrent_creation(tmp_path, monkeypatch):
    # quarkfs can fail a plain O_CREAT open with EEXIST when two hosts create it at once.
    attempts = []

    def flaky_open(path, mode):
        attempts.append(path)
        if len(attempts) < 3:
            raise FileExistsError(17, "File exists", str(path))
        return open(path, mode)

    monkeypatch.setattr(catalog_cache, "open", flaky_open, raising=False)
    monkeypatch.setattr(catalog_cache, "LOCK_RETRY_SECONDS", 0)
    with catalog_cache.exclusive_lock(tmp_path / "build.lock"):
        assert (tmp_path / "build.lock").exists()
    assert len(attempts) == 3


def test_source_index_publish_keeps_first_complete_build(tmp_path):
    index = catalog_cache.SourceRecordIndex.__new__(catalog_cache.SourceRecordIndex)
    index.path = tmp_path / "key"
    for name in ("a.tmp", "b.tmp"):
        staging = tmp_path / name
        staging.mkdir()
        (staging / "complete.json").write_text(name)
        index._publish(staging)
        assert not staging.exists()
    assert (index.path / "complete.json").read_text() == "a.tmp"


def test_source_index_publish_rejects_incomplete_directory(tmp_path):
    index = catalog_cache.SourceRecordIndex.__new__(catalog_cache.SourceRecordIndex)
    index.path = tmp_path / "key"
    (index.path / "partial").mkdir(parents=True)
    staging = tmp_path / "a.tmp"
    staging.mkdir()
    (staging / "complete.json").write_text("{}")
    with pytest.raises(RuntimeError, match="Incomplete index directory"):
        index._publish(staging)
    assert not staging.exists() and (index.path / "partial").exists()
