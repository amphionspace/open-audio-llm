import gzip
import json
import sqlite3

from open_audio_llm.data.catalog_cache import AudioIndex


def test_audio_index_publishes_rows_and_keeps_the_last_duplicate(tmp_path):
    source = tmp_path / "audio-index.jsonl.gz"
    rows = [
        {"cut_id": "a", "duration": 1.5, "relative_path": "a.wav"},
        {"cut_id": "b", "duration": 2.0, "relative_path": "b.wav"},
        {"cut_id": "a", "duration": 3.25, "relative_path": "a-later.wav"},
    ]
    with gzip.open(source, "wt", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row) + "\n")

    cache = tmp_path / "cache"
    index = AudioIndex(source, cache)
    assert index.duration("a") == 3.25
    assert index.duration("b") == 2.0
    assert index["a"]["relative_path"] == "a-later.wav"
    published = list(cache.glob("audio-*.sqlite"))
    assert len(published) == 1
    assert not list(cache.glob("*.publishing"))
    with sqlite3.connect(published[0]) as db:
        page_size = db.execute("PRAGMA page_size").fetchone()[0]
        count = db.execute("SELECT COUNT(*) FROM audio").fetchone()[0]
    assert page_size == 65536
    assert count == 2

    again = AudioIndex(source, cache)
    assert again.path == index.path
    assert again["b"]["duration"] == 2.0
