#!/usr/bin/env python3
"""Prepare pinned public Kazakh mixed-speech and Uzbek read-speech controls.

Requires pyarrow (benchmark environment). Downloads ~665 MB in total.
Audio stays in --cache; compact manifests are written to --output.
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import random
import subprocess
import urllib.request

import pyarrow.parquet as pq

KAZAKH_REVISION = "6575c5cd9e9e481edcb4d9928fda2b2f60d3e5ad"
FLEURS_SHA256 = "4772b17310b5cd0494d9f33a19140288ea13bad1c44fd8833aca4069b1e8ec07"
FLEURS_URL = "https://huggingface.co/datasets/google/fleurs/resolve/refs%2Fconvert%2Fparquet/uz_uz/test/0000.parquet"


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--uzbek-samples", type=int, default=100)
    args = parser.parse_args()
    args.cache = args.cache.resolve()
    args.cache.mkdir(parents=True, exist_ok=True)
    args.output.mkdir(parents=True, exist_ok=True)
    repository = args.cache / "kaz-codeswitch"
    if not repository.exists():
        subprocess.run(["git", "clone", "https://github.com/Tim2190/Kaz-ASR-codeswitch-benchmark", str(repository)], check=True)
    revision = subprocess.check_output(["git", "-C", str(repository), "rev-parse", "HEAD"], text=True).strip()
    if revision != KAZAKH_REVISION:
        raise ValueError(f"Check out {KAZAKH_REVISION} in {repository} before proceeding")
    samples = []
    for row in csv.DictReader((repository / "metadata.csv").open()):
        path = repository / "audio" / (row["audio_id"] + ".wav")
        samples.append({
            "id": row["audio_id"], "dataset": "kazakh_russian_natural",
            "path": str(path), "sha256": digest(path),
            "reference": row["transcript_verbatim"],
            "reference_normalized": row["transcript_normalized_written"],
            "source": row["audio_source_link"],
            "flags": {key: row[key] for key in row if key.startswith("has_")},
        })
    kazakh = {"source": "https://github.com/Tim2190/Kaz-ASR-codeswitch-benchmark",
              "revision": revision, "license": "Audio CC BY per upstream attribution; annotations MIT", "samples": samples}
    (args.output / "kazakh_manifest.json").write_text(json.dumps(kazakh, ensure_ascii=False, indent=2) + "\n")
    parquet = args.cache / "fleurs_uz_test.parquet"
    if not parquet.exists():
        urllib.request.urlretrieve(FLEURS_URL, parquet)
    if digest(parquet) != FLEURS_SHA256:
        raise ValueError("FLEURS parquet checksum mismatch")
    count = pq.ParquetFile(parquet).metadata.num_rows
    if not 1 <= args.uzbek_samples <= count:
        raise ValueError(f"Choose 1..{count} Uzbek samples")
    selected = set(random.Random(42).sample(range(count), args.uzbek_samples))
    audio_dir = args.cache / "fleurs_uz"
    audio_dir.mkdir(exist_ok=True)
    samples = []
    index = 0
    for batch in pq.ParquetFile(parquet).iter_batches(batch_size=32):
        for row in batch.to_pylist():
            if index in selected:
                path = audio_dir / Path(row["audio"]["path"]).name
                path.write_bytes(row["audio"]["bytes"])
                samples.append({"id": f"fleurs_uz_{row['id']}_{index}", "dataset": "uzbek_read",
                                "path": str(path), "sha256": digest(path), "reference": row["transcription"],
                                "row_index": index, "gender": row["gender"]})
            index += 1
    uzbek = {"source": "https://huggingface.co/datasets/google/fleurs", "license": "CC BY 4.0",
             "attribution": "Conneau et al., FLEURS (2022)", "download_url": FLEURS_URL,
             "parquet_sha256": FLEURS_SHA256, "split": "test", "config": "uz_uz",
             "selection": "random.Random(42).sample(range(862), n), in source row order",
             "samples": samples}
    (args.output / "uzbek_manifest.json").write_text(json.dumps(uzbek, ensure_ascii=False, indent=2) + "\n")
    print(f"Prepared {len(kazakh['samples'])} Kazakh mixed and {len(samples)} Uzbek read clips")


if __name__ == "__main__":
    main()
