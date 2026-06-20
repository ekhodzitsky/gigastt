#!/usr/bin/env python3
"""Measure model footprint per engine: file size, peak RSS, and cold-start time.

Footprint is one of gigastt's real advantages (INT8 ~225 MB vs Whisper ~3 GB,
Vosk 1.3-1.8 GB). For each available runner this records:
  - on-disk model size (best-effort, engine-specific),
  - peak resident memory during the first transcription (cold start) and after a
    couple more (steady state),
  - cold-start wall time (runner ready -> first transcription done).

Output: ``results_footprint.json``. Smoke-friendly: only a handful of samples are
used. ``psutil`` is used when available; otherwise it falls back to ``resource``.
"""

import argparse
import importlib
import json
import os
import time
from pathlib import Path

import common


def _peak_rss_mb(pid: int | None = None) -> float | None:
    try:
        import psutil

        proc = psutil.Process(pid) if pid else psutil.Process()
        return round(proc.memory_info().rss / (1024 * 1024), 1)
    except Exception:
        if pid is not None:
            return None  # resource can only report this process, not a subprocess
        try:
            import resource

            maxrss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            # ru_maxrss is bytes on macOS, KiB on Linux.
            scale = 1024 * 1024 if os.uname().sysname == "Darwin" else 1024
            return round(maxrss / scale, 1)
        except Exception:
            return None


def _model_size_mb(runner) -> float | None:
    """Best-effort on-disk model size for a runner, if locatable."""
    name = runner.name
    home = Path.home()
    if name.startswith("gigastt"):
        # rnnt is the default head since v2.3; fall back to e2e_rnnt.
        models = home / ".gigastt" / "models"
        for fn in ("v3_rnnt_encoder_int8.onnx", "v3_e2e_rnnt_encoder_int8.onnx"):
            enc = models / fn
            if enc.exists():
                return round(enc.stat().st_size / (1024 * 1024), 1)
    if name.startswith("vosk"):
        model_dir = getattr(runner, "download_dir", None)
        model_name = getattr(runner, "model_name", None)
        if model_dir and model_name:
            d = Path(model_dir) / model_name
            if d.exists():
                total = sum(f.stat().st_size for f in d.rglob("*") if f.is_file())
                return round(total / (1024 * 1024), 1)
    return None


def measure_runner(runner, samples: list[dict]) -> dict:
    start = time.perf_counter()
    runner.transcribe(samples[0]["filename"])  # cold start = model load + inference
    cold_start_s = time.perf_counter() - start

    sub = getattr(runner, "_proc", None)
    pid = sub.pid if sub is not None else None
    peak_cold = _peak_rss_mb(pid)

    for s in samples[1:3]:
        try:
            runner.transcribe(s["filename"])
        except Exception:
            pass
    peak_steady = _peak_rss_mb(pid)

    return {
        "name": runner.name,
        "model_size_mb": _model_size_mb(runner),
        "cold_start_sec": round(cold_start_s, 2),
        "peak_rss_mb_cold": peak_cold,
        "peak_rss_mb_steady": peak_steady,
    }


def _load_runner_classes() -> list:
    candidates = [
        "GigasttRunner",
        "GigasttCoreMLRunner",
        "FasterWhisperRunner",
        "FasterWhisperTurboRunner",
        "WhisperCppRunner",
        "VoskRunner",
        "Vosk054Runner",
        "TOneRunner",
    ]
    classes = []
    for name in candidates:
        try:
            classes.append(getattr(importlib.import_module("runners"), name))
        except Exception as e:
            print(f"[benchmark_footprint] Skipping {name}: {e}")
    return classes


def main():
    parser = argparse.ArgumentParser(description="Model footprint benchmark")
    parser.add_argument("--dataset", default=os.environ.get("GIGASTT_BENCHMARK_DATASET", "golos_crowd"))
    parser.add_argument("--max-samples", type=int, default=3, help="Samples for cold/steady measurement")
    parser.add_argument("--output", default="results_footprint.json")
    parser.add_argument("--runners", default="all", help="Comma-separated runner names")
    args = parser.parse_args()

    runner_classes = _load_runner_classes()
    manifest_data = common.load_manifest(max_samples=args.max_samples, dataset=args.dataset)
    samples = manifest_data["samples"]
    if not samples:
        print("No samples available; cannot measure footprint.")
        return

    all_runners = [cls() for cls in runner_classes]
    requested = set(args.runners.split(",")) if args.runners != "all" else {"all"}
    active = [r for r in all_runners if "all" in requested or r.name in requested]

    results = []
    for runner in active:
        if not runner.is_available():
            print(f"Skipping {runner.name} (not available)")
            continue
        print(f"\n=== {runner.name} ===")
        try:
            if hasattr(runner, "__exit__"):
                with runner:
                    results.append(measure_runner(runner, samples))
            else:
                results.append(measure_runner(runner, samples))
        except Exception as e:
            print(f"[{runner.name}] footprint measurement failed: {e}")

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump({"dataset": args.dataset, "runners": results}, f, ensure_ascii=False, indent=2)
    print(f"\nResults written to {args.output}")


if __name__ == "__main__":
    main()
