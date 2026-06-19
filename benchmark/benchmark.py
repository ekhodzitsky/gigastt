#!/usr/bin/env python3
"""Cross-ASR benchmark: gigastt vs whisper.cpp vs faster-whisper vs Vosk.

Usage:
    python benchmark.py --max-samples 100 --output results.json

Environment:
    GIGASTT_BENCHMARK_MAX_SAMPLES  default limit (0 = unlimited)
"""

import argparse
import contextlib
import json
import os
import sys
from pathlib import Path
from typing import Optional

from cache import DiskCache
from common import (
    audio_duration,
    bootstrap_ci,
    collect_repro_metadata,
    compute_histograms,
    compute_wer,
    compute_wer_naive,
    load_manifest,
)
from runners import (
    FasterWhisperRunner,
    FasterWhisperTurboRunner,
    GigasttRunner,
    TOneRunner,
    VoskRunner,
    Vosk054Runner,
    WhisperCppRunner,
)


PROFILE_PATH = "benchmark.prof"
PROGRESS_INTERVAL = 10

ALL_RUNNERS = [
    GigasttRunner,
    WhisperCppRunner,
    FasterWhisperRunner,
    FasterWhisperTurboRunner,
    VoskRunner,
    Vosk054Runner,
    TOneRunner,
]


def run_benchmark(
    runner,
    manifest: list[dict],
    max_samples: Optional[int] = None,
    cache: Optional[DiskCache] = None,
) -> dict:
    """Run a single ASR runner over the manifest."""
    if max_samples:
        manifest = manifest[:max_samples]

    total_ref_words = 0
    total_errors = 0
    total_naive_errors = 0
    total_naive_ref_words = 0
    total_audio_sec = 0.0
    total_proc_sec = 0.0
    failures = 0
    cached_hits = 0
    details = []
    per_sample = []
    naive_per_sample = []

    print(f"\n=== {runner.name} ===")
    for idx, sample in enumerate(manifest):
        wav_path = sample["filename"]
        ref = sample["reference"]
        dur = sample.get("duration") or audio_duration(wav_path)

        cached = None
        if cache is not None:
            cached = cache.get(runner, wav_path)

        if cached is not None:
            hyp = cached["hypothesis"]
            proc_time = cached["proc_time"]
            success = True
            cached_hits += 1
            source = "cache"
        else:
            try:
                hyp, proc_time = runner.transcribe(wav_path)
                success = True
                if cache is not None:
                    cache.set(runner, wav_path, hyp, proc_time)
                source = "transcribe"
            except Exception as e:
                print(f"  [{idx + 1}/{len(manifest)}] ERROR: {e}")
                hyp = ""
                proc_time = 0.0
                success = False
                failures += 1
                source = "error"

        wer, errors, ref_count = compute_wer(ref, hyp)
        naive_wer, naive_errors, naive_ref_count = compute_wer_naive(ref, hyp)

        total_ref_words += ref_count
        total_errors += errors
        per_sample.append((ref_count, errors))
        total_naive_ref_words += naive_ref_count
        total_naive_errors += naive_errors
        naive_per_sample.append((naive_ref_count, naive_errors))

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
            "naive_wer": round(naive_wer, 2),
            "naive_errors": naive_errors,
            "naive_ref_words": naive_ref_count,
            "audio_sec": round(dur, 2),
            "proc_sec": round(proc_time, 2),
            "failed": not success,
            "cached": source == "cache",
        })

        if (idx + 1) % PROGRESS_INTERVAL == 0 or idx + 1 == len(manifest):
            rtf = proc_time / dur if dur > 0 and success else 0.0
            marker = " [C]" if source == "cache" else ""
            print(
                f"  [{idx + 1}/{len(manifest)}] WER={wer:.1f}%  RTF={rtf:.2f}x  "
                f"{Path(wav_path).name}{marker}"
            )

    overall_wer = (total_errors / total_ref_words * 100.0) if total_ref_words > 0 else 0.0
    overall_naive_wer = (
        (total_naive_errors / total_naive_ref_words * 100.0)
        if total_naive_ref_words > 0
        else 0.0
    )
    overall_rtf = total_proc_sec / total_audio_sec if total_audio_sec > 0 else 0.0
    ci_low, ci_high = bootstrap_ci(per_sample, iterations=1000)
    naive_ci_low, naive_ci_high = bootstrap_ci(naive_per_sample, iterations=1000)

    return {
        "name": runner.name,
        "samples": len(details),
        "failures": failures,
        "cached_hits": cached_hits,
        "wer": round(overall_wer, 2),
        "ci_low": round(ci_low, 2),
        "ci_high": round(ci_high, 2),
        "total_errors": total_errors,
        "total_ref_words": total_ref_words,
        # Verbatim ("naive") WER reported alongside the normalized WER. The gap
        # (naive_delta = wer - naive_wer, usually negative) is the share of the
        # apparent error that is writing convention — number style, punctuation,
        # transliteration — rather than acoustic recognition error.
        "naive_wer": round(overall_naive_wer, 2),
        "naive_ci_low": round(naive_ci_low, 2),
        "naive_ci_high": round(naive_ci_high, 2),
        "naive_total_errors": total_naive_errors,
        "naive_total_ref_words": total_naive_ref_words,
        "naive_delta": round(overall_wer - overall_naive_wer, 2),
        "total_audio_sec": round(total_audio_sec, 2),
        "total_proc_sec": round(total_proc_sec, 2),
        "rtf": round(overall_rtf, 3),
        "details": details,
        "histograms": compute_histograms(details),
    }


def print_results_table(results: list[dict]):
    print("\n" + "=" * 104)
    print(
        f"{'Engine':<20} {'Samples':>8} {'Failures':>9} {'WER %':>8} "
        f"{'95% CI':>16} {'naive %':>8} {'Δ pp':>7} {'RTF':>8} {'Errors':>10}"
    )
    print("-" * 104)
    for r in results:
        ci = f"[{r['ci_low']:.1f}, {r['ci_high']:.1f}]"
        print(
            f"{r['name']:<20} "
            f"{r['samples']:>8} "
            f"{r['failures']:>9} "
            f"{r['wer']:>8.2f} "
            f"{ci:>16} "
            f"{r.get('naive_wer', 0.0):>8.2f} "
            f"{r.get('naive_delta', 0.0):>+7.2f} "
            f"{r['rtf']:>8.3f} "
            f"{r['total_errors']:>10}"
        )
    print("=" * 104)
    print(
        "  WER % = normalized (ITN) · naive % = verbatim · "
        "Δ pp = WER − naive (negative ⇒ normalization, not acoustics, closed the gap)"
    )


def print_histograms(results: list[dict]):
    for r in results:
        hists = r.get("histograms")
        if not hists:
            continue
        print(f"\n--- Histograms: {r['name']} ---")
        for dim_name, buckets in hists.items():
            print(f"\n{dim_name}:")
            print(
                f"  {'Bucket':<16} {'Samples':>8} {'Words':>8} "
                f"{'Errors':>8} {'WER %':>8}"
            )
            for b in buckets:
                print(
                    f"  {b['bucket']:<16} "
                    f"{b['samples']:>8} {b['ref_words']:>8} "
                    f"{b['errors']:>8} {b['wer']:>8.2f}"
                )


def _parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cross-ASR benchmark")
    parser.add_argument("--max-samples", type=int, default=int(os.environ.get("GIGASTT_BENCHMARK_MAX_SAMPLES", "100")),
                        help="Maximum samples to process (0 = unlimited)")
    parser.add_argument("--output", type=str, default="results.json", help="Output JSON path")
    parser.add_argument("--runners", type=str, default="all",
                        help="Comma-separated list: gigastt,whisper_cpp,faster_whisper,vosk (or 'all')")
    parser.add_argument("--dataset", type=str, default=os.environ.get("GIGASTT_BENCHMARK_DATASET", "golos_crowd"),
                        help="Dataset manifest name (e.g. golos_crowd, golos_farfield)")
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=os.environ.get("GIGASTT_BENCHMARK_CACHE_DIR", "~/.gigastt/benchmark_cache"),
        help="Directory for cached transcription results",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable transcription result cache",
    )
    parser.add_argument(
        "--clear-cache",
        action="store_true",
        help="Clear the transcription cache and exit",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help=f"Run cProfile and dump stats to {PROFILE_PATH}",
    )
    return parser.parse_args(argv)


def _main(args: Optional[argparse.Namespace] = None):
    if args is None:
        args = _parse_args()

    cache = DiskCache(args.cache_dir, enabled=not args.no_cache)
    if args.clear_cache:
        removed = cache.clear()
        print(f"Cleared {removed} cached entries from {cache.cache_dir}")
        sys.exit(0)

    max_samples = args.max_samples if args.max_samples > 0 else None
    manifest_data = load_manifest(max_samples=max_samples, dataset=args.dataset)
    manifest = manifest_data["samples"]
    skipped_empty_refs = manifest_data["skipped_empty_refs"]
    print(
        f"Loaded {len(manifest)} samples from dataset '{args.dataset}' "
        f"({skipped_empty_refs} skipped with empty reference)"
    )
    if cache.enabled:
        print(f"Cache enabled: {cache.cache_dir}")

    requested = set(args.runners.split(",")) if args.runners != "all" else {"all"}

    active_runners = []
    for runner_or_cls in ALL_RUNNERS:
        r = runner_or_cls() if isinstance(runner_or_cls, type) else runner_or_cls
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
        cm = runner if hasattr(runner, "__enter__") else contextlib.nullcontext(runner)
        with cm:
            result = run_benchmark(runner, manifest, max_samples=None, cache=cache)
        results.append(result)

    print_results_table(results)
    print_histograms(results)
    if cache.enabled:
        total_cached = sum(r.get("cached_hits", 0) for r in results)
        print(f"Total cache hits: {total_cached}")

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


def main():
    args = _parse_args()
    if args.profile:
        import cProfile
        prof = cProfile.Profile()
        prof.enable()
        try:
            _main(args)
        finally:
            prof.disable()
            prof.dump_stats(PROFILE_PATH)
            print(f"\nProfile written to {PROFILE_PATH}")
            print(f"View with: python -m pstats {PROFILE_PATH}")
    else:
        _main(args)


if __name__ == "__main__":
    main()
