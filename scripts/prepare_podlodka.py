#!/usr/bin/env python3
"""Prepare a Podlodka Speech slice for held-out WER benchmarking.

Dataset:
  - https://huggingface.co/datasets/bond005/podlodka_speech
  - Conversational / podcast Russian (episode segments)
  - License: check the HF dataset card before redistributing

Writes audio bytes with ``Audio(decode=False)`` (no torchcodec) and a
standard benchmark manifest (seeded sample for large splits).
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Prepare Podlodka Speech benchmark slice")
    p.add_argument("--split", default="test", help="HF split (test/train/…)")
    p.add_argument("--slice-size", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("~/.gigastt/benchmarks/podlodka").expanduser(),
    )
    p.add_argument(
        "--manifest-path",
        type=Path,
        default=Path(__file__).parent.parent / "benchmark/manifests/podlodka.json",
    )
    p.add_argument(
        "--max-scan",
        type=int,
        default=0,
        help="Stop after this many usable rows (0 = full split).",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    from datasets import Audio, load_dataset

    print(f"Loading bond005/podlodka_speech split={args.split} ...")
    ds = load_dataset("bond005/podlodka_speech", split=args.split, streaming=True)
    ds = ds.cast_column("audio", Audio(decode=False))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    written: list[tuple[str, str]] = []
    for i, row in enumerate(ds):
        ref = (row.get("transcription") or row.get("text") or "").strip()
        if not ref:
            continue
        audio = row["audio"]
        data = audio.get("bytes") if isinstance(audio, dict) else None
        if not data:
            continue
        path_hint = (audio.get("path") if isinstance(audio, dict) else None) or f"{i}.wav"
        name = Path(path_hint).name
        if not name.lower().endswith((".mp3", ".wav", ".flac", ".ogg", ".m4a", ".opus")):
            name = f"{i:05d}.wav"
        out = args.output_dir / name
        if out.exists() and out.stat().st_size != len(data):
            name = f"{i:05d}_{name}"
            out = args.output_dir / name
        out.write_bytes(data)
        written.append((name, ref))
        if len(written) % 100 == 0:
            print(f"  wrote {len(written)}")
        if args.max_scan and len(written) >= args.max_scan:
            break

    total = len(written)
    print(f"Total with audio+ref: {total}")
    if total == 0:
        print("No usable samples.", file=sys.stderr)
        return 1

    n = min(args.slice_size, total)
    rng = random.Random(args.seed)
    selected = written if n == total else rng.sample(written, n)
    samples = [{"filename": f, "reference": r} for f, r in selected]

    manifest = {
        "dataset": "podlodka",
        "audio_root": "~/.gigastt/benchmarks/podlodka",
        "slice_seed": args.seed,
        "slice_size": n,
        "total_available": total,
        "language": "ru",
        "split": args.split,
        "license": "see HF card bond005/podlodka_speech",
        "source": "https://huggingface.co/datasets/bond005/podlodka_speech",
        "attribution": "bond005/podlodka_speech (Podlodka podcast ASR corpus)",
        "domain": "conversational / podcast",
        "samples": samples,
    }
    args.manifest_path.parent.mkdir(parents=True, exist_ok=True)
    args.manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Selected {n}/{total}; manifest: {args.manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
