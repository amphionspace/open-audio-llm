import fcntl
import multiprocessing
import time

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
