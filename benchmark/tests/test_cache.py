"""Unit tests for benchmark/cache.py."""

import json
import tempfile
from pathlib import Path

import pytest

from cache import DiskCache


class _FakeRunner:
    name = "fake"
    cache_config = "model-a:int8"


class _FakeRunnerNoConfig:
    name = "fake-no-config"


def test_cache_miss_returns_none():
    with tempfile.TemporaryDirectory() as tmp:
        cache = DiskCache(tmp)
        assert cache.get(_FakeRunner(), "nonexistent.wav") is None


def test_cache_round_trip():
    with tempfile.TemporaryDirectory() as tmp:
        cache = DiskCache(tmp)
        runner = _FakeRunner()
        wav = "clip.wav"
        cache.set(runner, wav, "hello world", 1.234)

        entry = cache.get(runner, wav)
        assert entry is not None
        assert entry["hypothesis"] == "hello world"
        assert entry["proc_time"] == 1.234
        assert entry["runner"] == "fake"
        assert entry["config"] == "model-a:int8"


def test_cache_disabled_returns_none():
    with tempfile.TemporaryDirectory() as tmp:
        cache = DiskCache(tmp, enabled=False)
        runner = _FakeRunner()
        cache.set(runner, "clip.wav", "hello", 0.5)
        assert cache.get(runner, "clip.wav") is None


def test_cache_key_depends_on_runner_config():
    with tempfile.TemporaryDirectory() as tmp:
        cache = DiskCache(tmp)
        wav = "clip.wav"

        class RunnerA:
            name = "runner"
            cache_config = "a"

        class RunnerB:
            name = "runner"
            cache_config = "b"

        cache.set(RunnerA(), wav, "from a", 1.0)
        cache.set(RunnerB(), wav, "from b", 2.0)

        assert cache.get(RunnerA(), wav)["hypothesis"] == "from a"
        assert cache.get(RunnerB(), wav)["hypothesis"] == "from b"


def test_cache_key_without_config_uses_name_only():
    with tempfile.TemporaryDirectory() as tmp:
        cache = DiskCache(tmp)
        runner = _FakeRunnerNoConfig()
        cache.set(runner, "clip.wav", "hello", 0.1)
        assert cache.get(runner, "clip.wav")["hypothesis"] == "hello"


def test_corrupt_cache_entry_is_miss():
    with tempfile.TemporaryDirectory() as tmp:
        cache = DiskCache(tmp)
        # Force a cache file with invalid JSON.
        key = cache._key(_FakeRunner(), "clip.wav")
        cache_path = cache._path(key)
        cache_path.write_text("not json")

        assert cache.get(_FakeRunner(), "clip.wav") is None


def test_clear_cache_removes_entries():
    with tempfile.TemporaryDirectory() as tmp:
        cache = DiskCache(tmp)
        cache.set(_FakeRunner(), "a.wav", "one", 1.0)
        cache.set(_FakeRunner(), "b.wav", "two", 2.0)
        assert cache.clear() == 2
        assert cache.get(_FakeRunner(), "a.wav") is None
        assert cache.get(_FakeRunner(), "b.wav") is None


def test_cache_writes_are_atomic():
    with tempfile.TemporaryDirectory() as tmp:
        cache = DiskCache(tmp)
        cache.set(_FakeRunner(), "clip.wav", "hello", 0.1)
        key = cache._key(_FakeRunner(), "clip.wav")
        # Only the final .json file should exist, no leftover .tmp files.
        files = list(Path(tmp).glob("*"))
        assert len(files) == 1
        assert files[0].name == f"{key}.json"


def test_set_cleans_up_temp_on_write_failure(tmp_path):
    cache = DiskCache(tmp_path)
    # Make the cache directory read-only so os.replace / mkstemp will fail.
    cache.cache_dir.chmod(0o555)
    try:
        with pytest.raises(Exception):
            cache.set(_FakeRunner(), str(tmp_path / "clip.wav"), "hyp", 1.0)
    finally:
        cache.cache_dir.chmod(0o755)
    assert list(cache.cache_dir.glob("*.tmp")) == []


def test_gigastt_runner_cache_config_is_stable():
    from runners.gigastt import GigasttRunner

    runner = GigasttRunner(model_dir="/models", use_int8=True, port=9877)
    # The resolved binary path is discovered lazily and must NOT affect the
    # cache key, otherwise the second run would see a miss.
    runner._binary = "/usr/local/bin/gigastt"
    assert runner.cache_config == "/models:True:v2.2.0"
