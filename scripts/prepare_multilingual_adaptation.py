#!/usr/bin/env python3
"""Prepare a small, deterministic FLEURS adaptation pilot with telephone augmentation.

Uses official train/validation splits only. Read speech with simulated codecs is
not a real-call corpus. The public call and YouTube evaluation sets are excluded.
Requires pyarrow and ffmpeg; audio stays in the supplied local output directory.
"""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import pyarrow.parquet as pq

from wer_unicode import normalize

REVISION = "168de341b3db6859a9bac1c50a2ef5e3b47647e0"
HASHES = {"train": "7f5d4a6c58db465dadfef9718dc4a61f1137f13196c3634cf803acd6eb859457",
          "validation": "35f8de4d59af07b236a9174e64093f64cfa74e5a47bf488f909e4138a8d24315"}


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train-count", type=int, default=128)
    parser.add_argument("--validation-count", type=int, default=32)
    args = parser.parse_args()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    seen = {}
    for split, limit in [("train", args.train_count), ("validation", args.validation_count)]:
        if limit < 1:
            raise ValueError("Sample counts must be positive")
        source = args.cache / f"fleurs_kk_{split}.parquet"
        if digest(source) != HASHES[split]:
            raise ValueError(f"Checksum mismatch: {source}")
        audio_dir = args.output / split
        audio_dir.mkdir(exist_ok=True)
        selected = []
        index = 0
        ids = set()
        texts = set()
        hashes = set()
        for batch in pq.ParquetFile(source).iter_batches(batch_size=16):
            for row in batch.to_pylist():
                row_index = index
                index += 1
                seconds = row["num_samples"] / 16000
                if not 1 <= seconds <= 20:
                    continue
                ref = " ".join(normalize(row["transcription"]))
                if not ref:
                    continue
                stem = f"fleurs_kk_{row['id']}_{row_index}"
                clean = audio_dir / (stem + ".wav")
                clean.write_bytes(row["audio"]["bytes"])
                hashes.add(digest(clean))
                ids.add(row["id"])
                texts.add(ref)
                telephone = audio_dir / (stem + "_telephone.wav")
                subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(clean), "-ac", "1",
                                "-af", "highpass=f=300,lowpass=f=3400", "-ar", "8000",
                                "-c:a", "pcm_mulaw", str(telephone)], check=True)
                for condition, path in [("original", clean), ("simulated_telephone", telephone)]:
                    selected.append({"id": stem, "dataset": "fleurs_kk_adaptation", "split": split,
                                     "condition": condition, "path": str(path), "sha256": digest(path),
                                     "reference": ref, "duration_seconds": seconds, "row_index": row_index,
                                     "sentence_id": row["id"]})
                if len(selected) == limit * 2:
                    break
            if len(selected) == limit * 2:
                break
        if len(selected) != limit * 2:
            raise ValueError(f"Not enough eligible {split} rows")
        seen[split] = (ids, texts, hashes)
        if split == "validation":
            for label, train, val in zip(["sentence IDs", "normalized texts", "audio hashes"], seen["train"], seen[split]):
                if train & val:
                    raise ValueError(f"Train/validation overlap in {label}")
        manifest = {"source": "https://huggingface.co/datasets/google/fleurs", "revision": REVISION,
                    "license": "CC BY 4.0", "attribution": "Conneau et al., FLEURS (2022)",
                    "config": "kk_kz", "split": split, "source_sha256": HASHES[split],
                    "selection": f"First {limit} nonempty-reference rows with duration 1..20s in source order",
                    "limitations": "Human read speech; simulated telephone degradation, not real calls. Official dataset split; no individual speaker IDs in these rows.",
                    "samples": selected}
        (args.output / f"{split}_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
        print(f"Prepared {split}: {len(selected)} clean/telephone pairs entries", flush=True)


if __name__ == "__main__":
    main()
