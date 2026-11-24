"""Unit tests for benchmark/benchmark.py orchestration."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

import benchmark


def test_clear_cache_flag_removes_entries_and_exits(tmp_path):
    cache_dir = tmp_path / "cache"
    cache = benchmark.DiskCache(cache_dir)
    runner = SimpleNamespace(name="fake")
    cache.set(runner, "clip.wav", "x", 0.5)
    report = cache_dir / "results.json"
    report.write_text('{"samples": 1}')

    with pytest.raises(SystemExit) as exc:
        with patch.object(sys, "argv", ["benchmark.py", "--cache-dir", str(cache_dir), "--clear-cache"]):
            benchmark._main()

    assert exc.value.code == 0
    assert cache.get(runner, "clip.wav") is None
    assert list(cache_dir.iterdir()) == [report]
    assert report.read_text() == '{"samples": 1}'


def _fake_runner(name: str, available: bool = True):
    runner = MagicMock()
    runner.name = name
    runner.is_available.return_value = available
    runner.__enter__ = MagicMock(return_value=runner)
    runner.__exit__ = MagicMock(return_value=False)
    return runner


def _fake_result(name: str = "fake") -> dict:
    return {
        "name": name,
        "samples": 0,
        "failures": 0,
        "cached_hits": 0,
        "wer": 0.0,
        "ci_low": 0.0,
        "ci_high": 0.0,
        "total_errors": 0,
        "total_ref_words": 0,
        "naive_wer": 0.0,
        "naive_ci_low": 0.0,
        "naive_ci_high": 0.0,
        "naive_total_errors": 0,
        "naive_total_ref_words": 0,
        "naive_delta": 0.0,
        "total_audio_sec": 0.0,
        "total_proc_sec": 0.0,
        "rtf": 0.0,
        "details": [],
        "histograms": {},
    }


def test_mode_batch_does_not_select_stream_runner(tmp_path):
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    batch = _fake_runner("gigastt")
    stream = _fake_runner("gigastt-stream")
    manifest = [{"filename": "a.wav", "reference": "hello"}]
    fake_result = {**_fake_result(), "samples": 1, "total_ref_words": 1}

    with patch.object(benchmark, "ALL_RUNNERS", [batch, stream]):
        with patch.object(benchmark, "load_manifest", return_value={"samples": manifest, "skipped_empty_refs": 0}):
            with patch.object(benchmark, "run_benchmark", return_value=fake_result) as mock_run:
                with patch.object(sys, "argv", [
                    "benchmark.py",
                    "--cache-dir", str(cache_dir),
                    "--mode", "batch",
                    "--output", str(tmp_path / "results.json"),
                    "--max-samples", "1",
                ]):
                    benchmark._main()
    names = [c.args[0].name for c in mock_run.call_args_list]
    assert names == ["gigastt"]


def test_mode_stream_selects_only_stream_runner(tmp_path):
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    batch = _fake_runner("gigastt")
    stream = _fake_runner("gigastt-stream")
    manifest = [{"filename": "a.wav", "reference": "hello"}]
    fake_result = {**_fake_result(), "samples": 1, "total_ref_words": 1}

    with patch.object(benchmark, "ALL_RUNNERS", [batch, stream]):
        with patch.object(benchmark, "load_manifest", return_value={"samples": manifest, "skipped_empty_refs": 0}):
            with patch.object(benchmark, "run_benchmark", return_value=fake_result) as mock_run:
                with patch.object(sys, "argv", [
                    "benchmark.py",
                    "--cache-dir", str(cache_dir),
                    "--mode", "stream",
                    "--output", str(tmp_path / "results.json"),
                    "--max-samples", "1",
                ]):
                    benchmark._main()
    names = [c.args[0].name for c in mock_run.call_args_list]
    assert names == ["gigastt-stream"]


def test_mode_both_selects_batch_and_stream_and_writes_delta(tmp_path):
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    batch = _fake_runner("gigastt")
    stream = _fake_runner("gigastt-stream")
    whisper = _fake_runner("whisper_cpp")
    manifest = [{"filename": "a.wav", "reference": "hello"}]
    out = tmp_path / "results.json"

    def _run(runner, *_args, **_kwargs):
        if runner.name == "gigastt":
            return {
                **_fake_result("gigastt"),
                "details": [{"file": "a.wav", "ref_words": 10, "errors": 1, "failed": False}],
            }
        if runner.name == "gigastt-stream":
            return {
                **_fake_result("gigastt-stream"),
                "details": [{"file": "a.wav", "ref_words": 10, "errors": 2, "failed": False}],
            }
        return _fake_result(runner.name)

    with patch.object(benchmark, "ALL_RUNNERS", [batch, stream, whisper]):
        with patch.object(benchmark, "load_manifest", return_value={"samples": manifest, "skipped_empty_refs": 0}):
            with patch.object(benchmark, "run_benchmark", side_effect=_run) as mock_run:
                with patch.object(sys, "argv", [
                    "benchmark.py",
                    "--cache-dir", str(cache_dir),
                    "--mode", "both",
                    "--runners", "gigastt",
                    "--output", str(out),
                    "--max-samples", "1",
                ]):
                    benchmark._main()

    names = [c.args[0].name for c in mock_run.call_args_list]
    assert names == ["gigastt", "gigastt-stream"]
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["mode"] == "both"
    assert payload["stream_vs_batch"]["paired"] == 1
    assert payload["stream_vs_batch"]["delta_pp"] == pytest.approx(10.0)


def test_runner_selection_filters_by_name(tmp_path):
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()

    fake_available = _fake_runner("fake_available")
    fake_unavailable = _fake_runner("fake_unavailable", available=False)

    manifest = [{"filename": "a.wav", "reference": "hello"}]
    fake_result = {**_fake_result(), "samples": 1, "total_ref_words": 1}

    with patch.object(benchmark, "ALL_RUNNERS", [fake_available, fake_unavailable]):
        with patch.object(benchmark, "load_manifest", return_value={"samples": manifest, "skipped_empty_refs": 0}):
            with patch.object(benchmark, "run_benchmark", return_value=fake_result) as mock_run:
                with patch.object(sys, "argv", [
                    "benchmark.py",
                    "--cache-dir", str(cache_dir),
                    "--runners", "fake_available",
                    "--output", str(tmp_path / "results.json"),
                    "--max-samples", "1",
                ]):
                    benchmark._main()

    assert fake_available.is_available.called
    assert not fake_unavailable.is_available.called
    mock_run.assert_called_once()


def test_run_benchmark_counts_failure():
    runner = MagicMock()
    runner.name = "fake"
    runner.transcribe.side_effect = RuntimeError("boom")

    manifest = [{"filename": "a.wav", "reference": "hello world"}]
    result = benchmark.run_benchmark(runner, manifest, max_samples=1)

    assert result["failures"] == 1
    assert result["wer"] == 100.0
    assert result["details"][0]["failed"] is True


def test_run_benchmark_uses_cache():
    class _FakeCache:
        def __init__(self, entries):
            self._entries = entries

        def get(self, runner, wav_path):
            return self._entries.get(wav_path)

        def set(self, *args, **kwargs):
            pass

    runner = MagicMock()
    runner.name = "fake"
    cache = _FakeCache({"a.wav": {"hypothesis": "hello", "proc_time": 0.1}})
    manifest = [{"filename": "a.wav", "reference": "hello"}]
    result = benchmark.run_benchmark(runner, manifest, cache=cache)

    runner.transcribe.assert_not_called()
    assert result["cached_hits"] == 1
    assert result["details"][0]["cached"] is True
    assert result["details"][0]["hypothesis"] == "hello"


def test_run_benchmark_reports_naive_wer():
    # gigastt-style hypothesis (digits) vs spoken-number reference: the ITN pass
    # forgives the digit/word convention difference, the verbatim pass does not.
    # run_benchmark must surface both, with naive_delta = wer - naive_wer.
    class _FakeCache:
        def get(self, runner, wav_path):
            return {"hypothesis": "5%", "proc_time": 0.1}

        def set(self, *args, **kwargs):
            pass

    runner = MagicMock()
    runner.name = "fake"
    manifest = [{"filename": "a.wav", "reference": "пять процентов"}]
    result = benchmark.run_benchmark(runner, manifest, cache=_FakeCache())

    assert result["wer"] == 0.0
    assert result["naive_wer"] > 0.0
    assert result["naive_delta"] == round(result["wer"] - result["naive_wer"], 2)
    assert result["naive_total_errors"] >= 1
    assert result["details"][0]["naive_wer"] > 0.0
    assert result["details"][0]["naive_errors"] >= 1


def test_no_cache_flag_disables_cache(tmp_path):
    cache_dir = tmp_path / "cache"

    fake_runner = _fake_runner("fake")
    fake_result = _fake_result()

    with patch.object(benchmark, "ALL_RUNNERS", [fake_runner]):
        with patch.object(benchmark, "load_manifest", return_value={"samples": [], "skipped_empty_refs": 0}):
            with patch.object(benchmark, "run_benchmark", return_value=fake_result):
                with patch.object(sys, "argv", [
                    "benchmark.py",
                    "--cache-dir", str(cache_dir),
                    "--runners", "fake",
                    "--no-cache",
                ]):
                    benchmark._main()

    assert not cache_dir.exists() or not list(cache_dir.iterdir())


def test_profile_flag_writes_prof_file(tmp_path, monkeypatch):
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    prof_path = tmp_path / benchmark.PROFILE_PATH

    fake_runner = _fake_runner("fake")
    fake_result = _fake_result()

    monkeypatch.chdir(tmp_path)
    with patch.object(benchmark, "ALL_RUNNERS", [fake_runner]):
        with patch.object(benchmark, "load_manifest", return_value={"samples": [], "skipped_empty_refs": 0}):
            with patch.object(benchmark, "run_benchmark", return_value=fake_result):
                with patch.object(sys, "argv", [
                    "benchmark.py",
                    "--cache-dir", str(cache_dir),
                    "--runners", "fake",
                    "--profile",
                ]):
                    benchmark.main()

    assert prof_path.exists()
    assert prof_path.stat().st_size > 0


def test_profile_flag_does_not_mutate_argv(tmp_path, monkeypatch):
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    argv = ["benchmark.py", "--cache-dir", str(cache_dir), "--profile"]

    monkeypatch.chdir(tmp_path)
    with patch.object(benchmark, "ALL_RUNNERS", []):
        with patch.object(benchmark, "load_manifest", return_value={"samples": [], "skipped_empty_refs": 0}):
            with patch.object(benchmark, "run_benchmark", return_value=_fake_result()):
                with patch.object(sys, "argv", argv):
                    with pytest.raises(SystemExit):
                        benchmark.main()
                    assert sys.argv == argv
