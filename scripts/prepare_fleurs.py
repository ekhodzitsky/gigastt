#!/usr/bin/env python3
"""Prepare a FLEURS test slice for benchmarking a language, for any FLEURS config.

FLEURS (Few-shot Learning Evaluation of Universal Representations of Speech) is a
102-language read-speech ASR benchmark. Used here to measure the GigaAM
Multilingual (``ml_ctc`` / ``ml_ctc_large``) heads on their non-Russian
Cyrillic/Turkic languages — Kazakh (``kk_kz``), Kyrgyz (``ky_kg``), Uzbek
(``uz_uz``).

Dataset provenance:
  - Name: FLEURS (google/fleurs)
  - Source: https://huggingface.co/datasets/google/fleurs
  - Paper: Conneau et al., "FLEURS: Few-shot Learning Evaluation of Universal
    Representations of Speech", 2022 (arXiv:2205.12446)
  - License: CC BY 4.0

FLEURS stores 16 kHz mono WAV audio. We read it with ``Audio(decode=False)`` to
get the raw WAV bytes (avoids the datasets audio-decoder / torchcodec dependency)
and write them straight to disk — gigastt decodes WAV via symphonia, so no
resampling/conversion is needed. The reference is the lowercase ``transcription``
field. Deterministic slice via ``random.seed(seed)``.
"""

import argparse
import json
import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Prepare a FLEURS benchmark slice")
    p.add_argument("--config", type=str, required=True, help="FLEURS config (kk_kz, ky_kg, uz_uz, ...)")
    p.add_argument("--slice-size", type=int, default=1000, help="Max samples in the manifest")
    p.add_argument("--seed", type=int, default=42, help="Random seed for deterministic selection")
    p.add_argument("--split", type=str, default="test", help="Dataset split")
    p.add_argument(
        "--field",
        default="transcription",
        choices=["transcription", "raw_transcription"],
        help="Reference field. `transcription` is lowercase/normalized (WER); "
        "`raw_transcription` keeps punctuation + casing (punctuation benchmark) and "
        "names the manifest `fleurs_<lang>_punct`.",
    )
    p.add_argument("--output-dir", type=Path, default=None, help="WAV output dir (default derived from lang)")
    p.add_argument("--manifest-path", type=Path, default=None, help="Manifest JSON path (default derived from lang)")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    lang = args.config.split("_")[0]  # kk_kz -> kk
    name = f"fleurs_{lang}" + ("_punct" if args.field == "raw_transcription" else "")
    output_dir = (args.output_dir or Path(f"~/.gigastt/benchmarks/{name}")).expanduser()
    manifest_path = args.manifest_path or (REPO_ROOT / f"benchmark/manifests/{name}.json")

    from datasets import Audio, load_dataset

    print(f"Loading google/fleurs ({args.config}) {args.split} split ...")
    ds = load_dataset("google/fleurs", args.config, split=args.split, streaming=True)
    ds = ds.cast_column("audio", Audio(decode=False))

    # Single streaming pass: FLEURS test splits are small; write every WAV, then
    # pick the deterministic slice for the manifest.
    output_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for row in ds:
        ref = (row.get(args.field) or "").strip()
        if not ref:
            continue
        audio = row["audio"]
        data = audio.get("bytes")
        if not data:
            continue
        wav_name = Path(audio.get("path") or f"{row.get('id')}.wav").name
        (output_dir / wav_name).write_bytes(data)
        written.append((wav_name, ref))
        if len(written) % 100 == 0:
            print(f"  wrote {len(written)} wavs")

    total_available = len(written)
    print(f"Total test utterances with audio + reference: {total_available}")
    if total_available == 0:
        print("No usable samples found.", file=sys.stderr)
        return 1

    slice_size = min(args.slice_size, total_available)
    rng = random.Random(args.seed)
    selected = written if slice_size == total_available else rng.sample(written, slice_size)
    samples = [{"filename": wav, "reference": ref} for wav, ref in selected]

    manifest = {
        "dataset": name,
        "audio_root": f"~/.gigastt/benchmarks/{name}",
        "slice_seed": args.seed,
        "slice_size": slice_size,
        "total_available": total_available,
        "language": lang,
        "fleurs_config": args.config,
        "license": "CC BY 4.0",
        "source": "https://huggingface.co/datasets/google/fleurs",
        "attribution": "FLEURS (Conneau et al., 2022)",
        "samples": samples,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(f"Selected {slice_size}/{total_available}; manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
