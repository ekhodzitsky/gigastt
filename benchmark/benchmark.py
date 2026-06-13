#!/usr/bin/env python3
"""Cross-ASR benchmark: gigastt vs whisper.cpp vs faster-whisper vs Vosk.

Usage:
    python benchmark.py --max-samples 100 --output results.json

Environment:
    GIGASTT_BENCHMARK_MAX_SAMPLES  default limit (0 = unlimited)
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional

from common import (
    audio_duration,
    bootstrap_ci,
    collect_repro_metadata,
    compute_wer,
    load_manifest,
)
from runners import (
    FasterWhisperRunner,
    FasterWhisperTurboRunner,
    GigasttCoreMLRunner,
    GigasttRunner,
    Vosk054Runner,
    VoskRunner,
    WhisperCppRunner,
)


def run_benchmark(runner, manifest: list[dict], max_samples: Optional[int] = None) -> dict:
    """Run a single ASR runner over the manifest."""
    if max_samples:
        manifest = manifest[:max_samples]

    total_ref_words = 0
    total_errors = 0
    total_audio_sec = 0.0
    total_proc_sec = 0.0
    failures = 0
    details = []
    per_sample = []

    print(f"\n=== {runner.name} ===")
    for idx, sample in enumerate(manifest):
        wav_path = sample["filename"]
        ref = sample["reference"]
        dur = sample.get("duration") or audio_duration(wav_path)

        try:
            hyp, proc_time = runner.transcribe(wav_path)
            success = True
        except Exception as e:
            print(f"  [{idx + 1}/{len(manifest)}] ERROR: {e}")
            hyp = ""
            proc_time = 0.0
            success = False
            failures += 1

        wer, errors, ref_count = compute_wer(ref, hyp)

        total_ref_words += ref_count
        total_errors += errors
        per_sample.append((ref_count, errors))

        if success:
            total_audio_sec += dur
            total_proc_sec += proc_time

        details.append({
            "file": wav_path,
            "reference": ref,
            "hypothesis": hyp,
            "wer": round(wer, 2),
            "errors": errors,
            "ref_words": ref_count,
            "audio_sec": round(dur, 2),
            "proc_sec": round(proc_time, 2),
            "failed": not success,
        })

        if (idx + 1) % 10 == 0 or idx + 1 == len(manifest):
            rtf = proc_time / dur if dur > 0 and success else 0.0
            print(f"  [{idx + 1}/{len(manifest)}] WER={wer:.1f}%  RTF={rtf:.2f}x  {Path(wav_path).name}")

    overall_wer = (total_errors / total_ref_words * 100.0) if total_ref_words > 0 else 0.0
    overall_rtf = total_proc_sec / total_audio_sec if total_audio_sec > 0 else 0.0
    ci_low, ci_high = bootstrap_ci(per_sample, iterations=1000)

    return {
        "name": runner.name,
        "samples": len(details),
        "failures": failures,
        "wer": round(overall_wer, 2),
        "ci_low": round(ci_low, 2),
        "ci_high": round(ci_high, 2),
        "total_errors": total_errors,
        "total_ref_words": total_ref_words,
        "total_audio_sec": round(total_audio_sec, 2),
        "total_proc_sec": round(total_proc_sec, 2),
        "rtf": round(overall_rtf, 3),
        "details": details,
    }


def print_results_table(results: list[dict]):
    print("\n" + "=" * 90)
    print(
        f"{'Engine':<20} {'Samples':>8} {'Failures':>9} {'WER %':>8} "
        f"{'95% CI':>16} {'RTF':>8} {'Errors':>10} {'Words':>10}"
    )
    print("-" * 90)
    for r in results:
        ci = f"[{r['ci_low']:.1f}, {r['ci_high']:.1f}]"
        print(
            f"{r['name']:<20} "
            f"{r['samples']:>8} "
            f"{r['failures']:>9} "
            f"{r['wer']:>8.2f} "
            f"{ci:>16} "
            f"{r['rtf']:>8.3f} "
            f"{r['total_errors']:>10} "
            f"{r['total_ref_words']:>10}"
        )
    print("=" * 90)


def main():
    parser = argparse.ArgumentParser(description="Cross-ASR benchmark")
    parser.add_argument("--max-samples", type=int, default=int(os.environ.get("GIGASTT_BENCHMARK_MAX_SAMPLES", "100")),
                        help="Maximum samples to process (0 = unlimited)")
    parser.add_argument("--output", type=str, default="results.json", help="Output JSON path")
    parser.add_argument("--runners", type=str, default="all",
                        help="Comma-separated list: gigastt,whisper_cpp,faster_whisper,vosk (or 'all')")
    parser.add_argument("--dataset", type=str, default=os.environ.get("GIGASTT_BENCHMARK_DATASET", "golos_crowd"),
                        help="Dataset manifest name (e.g. golos_crowd, golos_farfield)")
    args = parser.parse_args()

    max_samples = args.max_samples if args.max_samples > 0 else None
    manifest_data = load_manifest(max_samples=max_samples, dataset=args.dataset)
    manifest = manifest_data["samples"]
    skipped_empty_refs = manifest_data["skipped_empty_refs"]
    print(
        f"Loaded {len(manifest)} samples from dataset '{args.dataset}' "
        f"({skipped_empty_refs} skipped with empty reference)"
    )

    requested = set(args.runners.split(",")) if args.runners != "all" else {"all"}
    all_runners = [
        GigasttRunner(),
        GigasttCoreMLRunner(),
        WhisperCppRunner(),
        FasterWhisperRunner(),
        FasterWhisperTurboRunner(),
        VoskRunner(),
        Vosk054Runner(),
    ]

    active_runners = []
    for r in all_runners:
        normalized = r.name.replace(".", "_").replace("-", "_")
        if "all" in requested or normalized in requested or r.name in requested:
            if r.is_available():
                active_runners.append(r)
            else:
                print(f"Skipping {r.name} (not available)")

    if not active_runners:
        print("No runners available. Install dependencies:")
        print("  pip install -r requirements.txt")
        print("  # For whisper.cpp: auto-downloaded on first run")
        sys.exit(1)

    results = []
    for runner in active_runners:
        # Use explicit context manager lifecycle for runners that support it
        if hasattr(runner, "__enter__"):
            with runner:
                result = run_benchmark(runner, manifest, max_samples=None)  # already truncated
        else:
            result = run_benchmark(runner, manifest, max_samples=None)
        results.append(result)

    print_results_table(results)

    # Write JSON
    total_failures = sum(r["failures"] for r in results)
    output = {
        "dataset": args.dataset,
        "manifest_samples": len(manifest) + skipped_empty_refs,
        "skipped_empty_refs": skipped_empty_refs,
        "total_failures": total_failures,
        "runners": results,
        "metadata": collect_repro_metadata(active_runners, dataset_name=args.dataset),
    }
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\nResults written to {args.output}")


if __name__ == "__main__":
    main()
