"""Unit tests for runner cache_config stability."""

from pathlib import Path

import pytest


def test_gigastt_cache_config_excludes_binary_path():
    from runners.gigastt import GIGASTT_CACHE_SCHEMA_VERSION, GigasttRunner

    runner = GigasttRunner(model_dir="/models", use_int8=True, port=9877)
    runner._binary = "/usr/local/bin/gigastt"
    assert runner.cache_config == f"/models:True:{GIGASTT_CACHE_SCHEMA_VERSION}"


def test_faster_whisper_cache_config_includes_params():
    from runners.faster_whisper import FasterWhisperRunner

    runner = FasterWhisperRunner(model_size="base", device="cpu", compute_type="int8")
    assert runner.cache_config == "base:cpu:int8"


def test_whisper_cpp_cache_config_includes_model_and_tag(tmp_path):
    from runners.whisper_cpp import WhisperCppRunner

    runner = WhisperCppRunner(
        model_name="ggml-small.bin",
        download_dir=str(tmp_path),
        source_tag="v1.0.0",
    )
    assert runner.cache_config == "ggml-small.bin:v1.0.0"


def test_vosk_cache_config_is_model_name(tmp_path):
    from runners.vosk import VoskRunner

    runner = VoskRunner(model_name="vosk-model-ru-0.42", download_dir=str(tmp_path))
    assert runner.cache_config == "vosk-model-ru-0.42"


def test_vosk_054_cache_config_is_model_name(tmp_path):
    from runners.vosk_054 import Vosk054Runner

    runner = Vosk054Runner(model_name="vosk-model-ru-0.54", download_dir=str(tmp_path))
    assert runner.cache_config == "vosk-model-ru-0.54"


def test_t_one_cache_config_reflects_env(monkeypatch):
    from runners.t_one import TOneRunner

    monkeypatch.setenv("BENCHMARK_TONE_DECODER", "beam_search")
    monkeypatch.setenv("BENCHMARK_TONE_KENLM", "/path/to/kenlm.bin")
    runner = TOneRunner(device="cpu")
    assert runner.cache_config == "cpu:beam_search:/path/to/kenlm.bin"
