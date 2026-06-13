#!/usr/bin/env python3
"""Measure model footprint: file size, peak RSS, cold-start time."""

import argparse
import json
import os
import platform
import time
from pathlib import Path

import psutil


GIGASTT_MODEL_DIR = Path.home() / ".gigastt" / "models"
VOSK_CACHE = Path.home() / ".cache" / "vosk"
WHISPER_CACHE = Path.home() / ".cache" / "whisper.cpp"


def dir_size(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def model_size_for_runner(runner) -> dict:
    name = runner.name
    if name.startswith("gigastt"):
        return {
            "int8_bytes": dir_size(GIGASTT_MODEL_DIR / "v3_e2e_rnnt_encoder_int8.onnx"),
            "fp32_bytes": dir_size(GIGASTT_MODEL_DIR),
        }
    if name.startswith("vosk"):
        model_name = getattr(runner, "model_name", "vosk-model-ru-0.42")
        return {"total_bytes": dir_size(VOSK_CACHE / model_name)}
    if name == "whisper.cpp":
        return {"total_bytes": dir_size(WHISPER_CACHE / getattr(runner, "model_name", "ggml-large-v3.bin"))}
    if name.startswith("faster-whisper"):
        # faster-whisper stores under ~/.cache/huggingface/hub by model size
        return {"cache_dir": str(Path.home() / ".cache" / "huggingface" / "hub")}
    return {}


def measure_cold_start(runner_cls, **kwargs) -> dict:
    """Smoke measurement: start the runner and record RSS/time."""
    proc = psutil.Process(os.getpid())
    before_rss = proc.memory_info().rss
    started = time.perf_counter()
    runner = runner_cls(**kwargs)
    available = runner.is_available()
    ready = time.perf_counter()
    after_rss = proc.memory_info().rss
    return {
        "available": available,
        "cold_start_sec": round(ready - started, 3),
        "rss_delta_bytes": after_rss - before_rss,
    }


def main():
    parser = argparse.ArgumentParser(description="Footprint benchmark")
    parser.add_argument("--output", default="results_footprint.json")
    args = parser.parse_args()

    from runners import (
        FasterWhisperRunner,
        FasterWhisperTurboRunner,
        GigasttRunner,
        VoskRunner,
        Vosk054Runner,
        WhisperCppRunner,
    )

    all_runners = [
        ("gigastt", GigasttRunner, {}),
        ("faster-whisper", FasterWhisperRunner, {}),
        ("faster-whisper-turbo", FasterWhisperTurboRunner, {}),
        ("whisper.cpp", WhisperCppRunner, {}),
        ("vosk", VoskRunner, {}),
        ("vosk-0.54", Vosk054Runner, {}),
    ]

    results = []
    for label, cls, kwargs in all_runners:
        runner = cls(**kwargs)
        footprint = {
            "name": label,
            "model_size": model_size_for_runner(runner),
            "cold_start": measure_cold_start(cls, **kwargs),
        }
        results.append(footprint)

    output = {
        "host": {"cpu": platform.processor() or platform.machine(), "ram_bytes": psutil.virtual_memory().total},
        "runners": results,
    }
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
