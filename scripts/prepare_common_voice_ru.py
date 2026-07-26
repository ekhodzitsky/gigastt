#!/usr/bin/env python3
"""Prepare a fixed Common Voice Russian test slice for benchmarking.

Dataset provenance:
  - Name: Mozilla Common Voice (Russian / ru)
  - Split: test
  - License: CC0-1.0 (public domain dedication)
  - Attribution: Mozilla Common Voice contributors
  - Project: https://commonvoice.mozilla.org/ru

**Hub note (2025-10+).** Official ``mozilla-foundation/common_voice_*`` datasets
on Hugging Face are empty; Common Voice is distributed via
[Mozilla Data Collective](https://datacollective.mozillafoundation.org/).
This script defaults to a community mirror that still hosts Corpus **21.0**
Russian on the Hub:

  - https://huggingface.co/datasets/artyomboyko/common_voice_21_0_ru

Override ``--dataset-id`` / ``--config`` when you have a preferred source.
Audio is written with ``Audio(decode=False)`` (raw container bytes) so
``torchcodec`` is not required; gigastt/symphonia decode MP3/WAV at eval time.

Deterministic slice: ``random.seed(seed)``.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Prepare Common Voice Russian benchmark slice")
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("~/.gigastt/benchmarks/common_voice_ru").expanduser(),
        help="Directory to write audio files to",
    )
    p.add_argument(
        "--manifest-path",
        type=Path,
        default=Path(__file__).parent.parent / "benchmark/manifests/common_voice_ru.json",
        help="Path to write the manifest JSON file",
    )
    p.add_argument("--slice-size", type=int, default=1000, help="Samples in the manifest")
    p.add_argument("--seed", type=int, default=42, help="Random seed for deterministic selection")
    p.add_argument(
        "--dataset-id",
        type=str,
        default="artyomboyko/common_voice_21_0_ru",
        help="Hugging Face dataset id (default: CV 21.0 RU community mirror)",
    )
    p.add_argument(
        "--config",
        type=str,
        default="",
        help="Dataset config/locale (empty = single-config dataset, e.g. the default mirror)",
    )
    p.add_argument(
        "--split",
        type=str,
        default="test",
        help="Dataset split",
    )
    p.add_argument(
        "--max-scan",
        type=int,
        default=0,
        help="Stop after this many usable rows (0 = full split). Useful for smoke tests.",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    from datasets import Audio, load_dataset

    print(f"Loading {args.dataset_id} {args.config or '(default config)'} {args.split} ...")
    load_kw: dict = {"path": args.dataset_id, "split": args.split, "streaming": True}
    # datasets API: load_dataset(path, name=config, ...)
    if args.config:
        ds = load_dataset(args.dataset_id, args.config, split=args.split, streaming=True)
    else:
        ds = load_dataset(args.dataset_id, split=args.split, streaming=True)
    ds = ds.cast_column("audio", Audio(decode=False))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    written: list[tuple[str, str]] = []
    for i, row in enumerate(ds):
        ref = (row.get("sentence") or row.get("transcription") or "").strip()
        if not ref:
            continue
        audio = row["audio"]
        data = audio.get("bytes") if isinstance(audio, dict) else None
        if not data:
            continue
        path_hint = (audio.get("path") if isinstance(audio, dict) else None) or f"{i}.mp3"
        name = Path(path_hint).name
        if not name.lower().endswith((".mp3", ".wav", ".flac", ".ogg", ".m4a")):
            name = f"{i:05d}.mp3"
        # avoid clobber if names collide across rows
        out = args.output_dir / name
        if out.exists() and out.stat().st_size != len(data):
            name = f"{i:05d}_{name}"
            out = args.output_dir / name
        out.write_bytes(data)
        written.append((name, ref))
        if len(written) % 200 == 0:
            print(f"  wrote {len(written)}")
        if args.max_scan and len(written) >= args.max_scan:
            break

    total = len(written)
    print(f"Total with audio + reference: {total}")
    if total == 0:
        print("No usable samples found.", file=sys.stderr)
        return 1

    n = min(args.slice_size, total)
    rng = random.Random(args.seed)
    selected = written if n == total else rng.sample(written, n)
    samples = [{"filename": f, "reference": r} for f, r in selected]

    manifest = {
        "dataset": "common_voice_ru",
        "audio_root": "~/.gigastt/benchmarks/common_voice_ru",
        "slice_seed": args.seed,
        "slice_size": n,
        "total_available": total,
        "language": "ru",
        "license": "CC0-1.0",
        "source": f"https://huggingface.co/datasets/{args.dataset_id}",
        "upstream": "Mozilla Common Voice (community HF mirror; official Hub datasets empty since Oct 2025)",
        "attribution": "Mozilla Common Voice contributors",
        "fleurs_config": args.config or None,
        "samples": samples,
    }
    # drop null
    if manifest["fleurs_config"] is None:
        del manifest["fleurs_config"]

    args.manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(args.manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(f"Selected {n}/{total}; manifest: {args.manifest_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
