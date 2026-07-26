#!/usr/bin/env python3
"""Prepare a ToneWebinars slice for held-out conversational WER benchmarking.

Dataset:
  - https://huggingface.co/datasets/Vikhrmodels/ToneWebinars
  - Russian (and some English) webinar / lecture speech, timed segments
  - License: Apache-2.0 (check HF card)

Streams HF with ``Audio(decode=False)`` (no torchcodec). Default uses the
``validation`` split and a seeded sample of the first ``--max-scan`` usable
Russian rows (full validation is ~21k / ~52 GB — avoid a full download).
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Prepare ToneWebinars benchmark slice")
    p.add_argument("--split", default="validation", help="HF split (validation/train)")
    p.add_argument("--slice-size", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("~/.gigastt/benchmarks/tone_webinars").expanduser(),
    )
    p.add_argument(
        "--manifest-path",
        type=Path,
        default=Path(__file__).parent.parent / "benchmark/manifests/tone_webinars.json",
    )
    p.add_argument(
        "--max-scan",
        type=int,
        default=2500,
        help="Max usable rows to pull before sampling (0 = unlimited; costly).",
    )
    p.add_argument(
        "--allow-english",
        action="store_true",
        help="Keep mostly-Latin rows (default: require Cyrillic majority).",
    )
    return p.parse_args()


def is_mostly_russian(text: str) -> bool:
    cyr = 0
    lat = 0
    for ch in text:
        o = ord(ch)
        if 0x0400 <= o <= 0x04FF:
            cyr += 1
        elif ch.isascii() and ch.isalpha():
            lat += 1
    return cyr > 0 and cyr >= lat


def ext_for_bytes(data: bytes) -> str:
    if data[:4] == b"RIFF" and data[8:12] == b"WAVE":
        return ".wav"
    if data[:3] == b"ID3" or data[:2] == b"\xff\xfb":
        return ".mp3"
    if data[:4] == b"fLaC":
        return ".flac"
    if data[:4] == b"OggS":
        return ".ogg"
    return ".wav"


def main() -> int:
    args = parse_args()
    from datasets import Audio, load_dataset

    print(f"Loading Vikhrmodels/ToneWebinars split={args.split} (streaming) ...")
    ds = load_dataset("Vikhrmodels/ToneWebinars", split=args.split, streaming=True)
    ds = ds.cast_column("audio", Audio(decode=False))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    written: list[tuple[str, str]] = []
    skipped_lang = 0
    skipped_empty = 0

    for i, row in enumerate(ds):
        ref = (row.get("text") or "").strip()
        if not ref:
            skipped_empty += 1
            continue
        if not args.allow_english and not is_mostly_russian(ref):
            skipped_lang += 1
            continue
        audio = row["audio"]
        data = audio.get("bytes") if isinstance(audio, dict) else None
        if not data:
            continue
        ext = ext_for_bytes(data)
        name = f"{i:06d}{ext}"
        out = args.output_dir / name
        out.write_bytes(data)
        written.append((name, ref))
        if len(written) % 50 == 0:
            print(f"  wrote {len(written)} (scanned {i + 1}, skip_lang={skipped_lang})")
        if args.max_scan and len(written) >= args.max_scan:
            break

    total = len(written)
    print(
        f"Usable Russian rows written: {total} "
        f"(skip_empty={skipped_empty}, skip_lang={skipped_lang})"
    )
    if total == 0:
        print("No usable samples.", file=sys.stderr)
        return 1

    n = min(args.slice_size, total)
    rng = random.Random(args.seed)
    selected = written if n == total else rng.sample(written, n)
    samples = [{"filename": f, "reference": r} for f, r in selected]

    manifest = {
        "dataset": "tone_webinars",
        "audio_root": "~/.gigastt/benchmarks/tone_webinars",
        "slice_seed": args.seed,
        "slice_size": n,
        "max_scan": args.max_scan,
        "total_available": total,
        "language": "ru",
        "split": args.split,
        "license": "Apache-2.0",
        "source": "https://huggingface.co/datasets/Vikhrmodels/ToneWebinars",
        "attribution": (
            "Vikhrmodels/ToneWebinars (from ZeroAgency/shkolkovo-bobr.video-webinars-audio)"
        ),
        "domain": "conversational / webinar lecture",
        "samples": samples,
    }
    args.manifest_path.parent.mkdir(parents=True, exist_ok=True)
    args.manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Selected {n}/{total}; manifest: {args.manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
