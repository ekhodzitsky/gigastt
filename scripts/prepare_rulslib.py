#!/usr/bin/env python3
"""Prepare a Russian LibriSpeech (RuLS / OpenSLR SLR96) test slice.

Downloads ``ruls_data.tar.gz`` from OpenSLR (~9.1 GB) unless already present,
extracts, and builds a seeded manifest of up to ``--slice-size`` utterances.

Source: https://www.openslr.org/96/
License: Public Domain (USA) / LibriVox-derived.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

DEFAULT_URL = "https://openslr.trmal.net/resources/96/ruls_data.tar.gz"
MIRRORS = [
    DEFAULT_URL,
    "https://openslr.elda.org/resources/96/ruls_data.tar.gz",
    "https://openslr.magicdatatech.com/resources/96/ruls_data.tar.gz",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Prepare RuLS (OpenSLR 96) benchmark slice")
    p.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("~/.gigastt/benchmarks/ruls_raw").expanduser(),
        help="Where to store the tar + extracted tree",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("~/.gigastt/benchmarks/ruls").expanduser(),
        help="Directory of copied/symlinked audio for the slice",
    )
    p.add_argument(
        "--manifest-path",
        type=Path,
        default=Path(__file__).parent.parent / "benchmark/manifests/ruls.json",
    )
    p.add_argument("--slice-size", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--prefer-split",
        default="test",
        help="Substring in path to prefer (test/dev/train); falls back to all",
    )
    p.add_argument("--skip-download", action="store_true")
    return p.parse_args()


def download_tar(dest: Path) -> Path:
    dest.mkdir(parents=True, exist_ok=True)
    tar_path = dest / "ruls_data.tar.gz"
    if tar_path.exists() and tar_path.stat().st_size > 1_000_000:
        print(f"Using existing {tar_path} ({tar_path.stat().st_size // 10**6} MB)")
        return tar_path
    for url in MIRRORS:
        print(f"Downloading {url} ...")
        try:
            subprocess.run(
                ["curl", "-fL", "--retry", "3", "-o", str(tar_path), url],
                check=True,
            )
            if tar_path.stat().st_size > 1_000_000:
                return tar_path
        except Exception as e:
            print(f"  failed: {e}")
    raise SystemExit("Could not download RuLS tar from any mirror")


def extract(tar_path: Path, dest: Path) -> Path:
    mark = dest / ".extracted"
    if mark.exists():
        print(f"Already extracted under {dest}")
        return dest
    dest.mkdir(parents=True, exist_ok=True)
    print(f"Extracting {tar_path} → {dest} (slow) ...")
    with tarfile.open(tar_path, "r:gz") as tf:
        tf.extractall(dest)
    mark.write_text("ok\n", encoding="utf-8")
    return dest


def find_transcripts(root: Path) -> list[tuple[Path, str]]:
    """Collect (wav_path, reference) pairs from common RuLS layouts."""
    pairs: list[tuple[Path, str]] = []
    # Common patterns: *.trans.txt next to wavs, or transcripts.tsv, or utt2text
    for trans in root.rglob("*.trans.txt"):
        base = trans.parent
        for line in trans.read_text(encoding="utf-8", errors="replace").splitlines():
            line = line.strip()
            if not line:
                continue
            parts = line.split(maxsplit=1)
            if len(parts) < 2:
                continue
            utt_id, text = parts[0], parts[1].strip()
            for ext in (".wav", ".flac", ".mp3", ".opus"):
                cand = base / f"{utt_id}{ext}"
                if cand.exists():
                    pairs.append((cand, text))
                    break
    if pairs:
        return pairs

    for tsv in root.rglob("*.tsv"):
        for i, line in enumerate(tsv.read_text(encoding="utf-8", errors="replace").splitlines()):
            if i == 0 and ("path" in line.lower() or "file" in line.lower()):
                continue
            cols = line.split("\t")
            if len(cols) < 2:
                continue
            # try path + text in first two cols either order
            for a, b in ((cols[0], cols[1]), (cols[1], cols[0])):
                p = root / a if not Path(a).is_absolute() else Path(a)
                if not p.exists():
                    p = tsv.parent / a
                if p.exists() and p.suffix.lower() in {".wav", ".flac", ".mp3", ".opus"}:
                    pairs.append((p, b.strip()))
                    break
    if pairs:
        return pairs

    # Last resort: pair same-stem .txt next to audio
    for audio in root.rglob("*"):
        if audio.suffix.lower() not in {".wav", ".flac", ".mp3", ".opus"}:
            continue
        txt = audio.with_suffix(".txt")
        if txt.exists():
            ref = txt.read_text(encoding="utf-8", errors="replace").strip()
            if ref:
                pairs.append((audio, ref))
    return pairs


def main() -> int:
    args = parse_args()
    if not args.skip_download:
        tar_path = download_tar(args.cache_dir)
        root = extract(tar_path, args.cache_dir / "extracted")
    else:
        root = args.cache_dir / "extracted"
        if not root.exists():
            root = args.cache_dir

    print(f"Scanning transcripts under {root} ...")
    pairs = find_transcripts(root)
    print(f"Found {len(pairs)} utterance pairs")
    if not pairs:
        # print tree hint
        kids = list(root.iterdir())[:20] if root.exists() else []
        print("No pairs found. Top-level:", [p.name for p in kids], file=sys.stderr)
        return 1

    prefer = [p for p in pairs if args.prefer_split.lower() in str(p[0]).lower()]
    pool = prefer if prefer else pairs
    print(f"Pool after prefer_split={args.prefer_split!r}: {len(pool)} (of {len(pairs)})")

    n = min(args.slice_size, len(pool))
    rng = random.Random(args.seed)
    selected = pool if n == len(pool) else rng.sample(pool, n)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    samples = []
    for src, ref in selected:
        name = src.name
        dest = args.output_dir / name
        if not dest.exists():
            try:
                dest.symlink_to(src.resolve())
            except OSError:
                shutil.copy2(src, dest)
        samples.append({"filename": name, "reference": ref})

    manifest = {
        "dataset": "ruls",
        "audio_root": "~/.gigastt/benchmarks/ruls",
        "slice_seed": args.seed,
        "slice_size": n,
        "total_available": len(pool),
        "language": "ru",
        "prefer_split": args.prefer_split,
        "license": "Public Domain (USA) / LibriVox-derived",
        "source": "https://www.openslr.org/96/",
        "attribution": "Russian LibriSpeech (RuLS), OpenSLR SLR96",
        "domain": "audiobook read",
        "samples": samples,
    }
    args.manifest_path.parent.mkdir(parents=True, exist_ok=True)
    args.manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {n} samples; manifest: {args.manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
