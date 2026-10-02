#!/usr/bin/env python3
"""Select exactly 1,000 original FLEURS recordings per language, without training.

Use complete pinned test/validation parquet files in --cache. Prefer all test
rows, then deterministic validation top-ups. Audio remains in the local cache.
Kazakh prior adaptation validation sentence IDs and audio hashes are excluded.
Requires pyarrow; uses the existing Unicode scorer for reference validation.
"""

import argparse
import hashlib
import json
from pathlib import Path
import random

import pyarrow.parquet as pq

from wer_unicode import normalize


REVISION = "168de341b3db6859a9bac1c50a2ef5e3b47647e0"
SOURCES = {
    "ru": ("ru_ru", "1e5bbee150c05a219c6c318e95d765886d34b4d179e8ab06dbecdb2791ac9a5d", "21ff4b967df7be93985f8b2e51bebfd633ccfc7844cbc2b2e204e33e04bc9150"),
    "en": ("en_us", "6428a4d04d3aac29e16b45e039bb1470a8bd7aa334cf92f7984c9c520d1f234d", "7c3eeb11a9597bd52cdc1b0d637e85389fe094cfd8763913e7bf4fdf7a853959"),
    "kk": ("kk_kz", "7f1f136a96e8210998f67b66652fbe5dc36da36943d89c174e654557a14fc60a", "35f8de4d59af07b236a9174e64093f64cfa74e5a47bf488f909e4138a8d24315"),
    "ky": ("ky_kg", "c40b27cc7640b6b5fa6f736ca795934b21b4ca9b7054ee2c5ad3bc17a35ef398", "d7794ac934c2456f4644e4e8eb55160e606c6b5abdaf6ab257b6fd89b4a1522c"),
    "uz": ("uz_uz", "4772b17310b5cd0494d9f33a19140288ea13bad1c44fd8833aca4069b1e8ec07", "3af9f89d373f70214bdec3a4053390028944bfbb6b88632fff6f490a4958d51e"),
}


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def prepare(args, language):
    config, test_hash, validation_hash = SOURCES[language]
    excluded_ids, excluded_hashes = set(), set()
    exclusion = None
    if language == "kk":
        if not args.exclude_kazakh_validation or not args.exclude_kazakh_validation.is_file():
            raise ValueError("Kazakh requires the previous adaptation validation manifest for explicit exclusions")
        rows = json.loads(args.exclude_kazakh_validation.read_text())["samples"]
        excluded_ids = {row["sentence_id"] for row in rows}
        excluded_hashes = {row["sha256"] for row in rows}
        exclusion = {"manifest_sha256": digest(args.exclude_kazakh_validation),
                     "sentence_ids": sorted(excluded_ids), "audio_sha256": sorted(excluded_hashes),
                     "policy": "Exclude every matching sentence ID or audio hash before sampling, across both test and validation"}
    pools = {}
    sources = []
    exclusion_counts = {}
    audio_dir = args.cache / "audio" / language
    audio_dir.mkdir(parents=True, exist_ok=True)
    for split, expected in [("test", test_hash), ("validation", validation_hash)]:
        parquet = args.cache / f"fleurs_{config}_{split}.parquet"
        if digest(parquet) != expected:
            raise ValueError(f"Source checksum mismatch: {parquet}")
        reader = pq.ParquetFile(parquet)
        pool = []
        counts = {"prior_adaptation_sentence_or_audio": 0, "empty_reference_or_nonpositive_duration": 0}
        index = 0
        for batch in reader.iter_batches(batch_size=16):
            for row in batch.to_pylist():
                row_index = index
                index += 1
                raw = row["audio"]["bytes"]
                sha = hashlib.sha256(raw).hexdigest()
                if row["id"] in excluded_ids or sha in excluded_hashes:
                    counts["prior_adaptation_sentence_or_audio"] += 1
                    continue
                if not normalize(row["transcription"]) or row["num_samples"] <= 0:
                    counts["empty_reference_or_nonpositive_duration"] += 1
                    continue
                identity = f"fleurs_{config}_{split}_{row_index:04d}"
                path = audio_dir / (identity + ".wav")
                path.write_bytes(raw)
                pool.append({"id": identity, "dataset": f"fleurs_{config}_{split}",
                             "language": language, "config": config, "split": split,
                             "row_index": row_index, "sentence_id": row["id"],
                             "path": str(path), "sha256": sha, "reference": row["transcription"],
                             "duration_s": row["num_samples"] / 16000, "num_samples": row["num_samples"],
                             "source_sample_rate": 16000, "gender": row["gender"]})
        pools[split] = pool
        sources.append({"split": split, "rows": reader.metadata.num_rows, "sha256": expected,
                        "size_bytes": parquet.stat().st_size,
                        "url": f"https://huggingface.co/datasets/google/fleurs/resolve/{REVISION}/{config}/{split}/0000.parquet"})
        exclusion_counts[split] = counts
    validation = pools["validation"].copy()
    random.Random(42).shuffle(validation)
    selected, hashes = [], set()
    duplicates = 0
    for sample in pools["test"] + validation:
        if sample["sha256"] in hashes:
            duplicates += 1
            continue
        hashes.add(sample["sha256"])
        selected.append(sample)
        if len(selected) == 1000:
            break
    if len(selected) != 1000:
        raise ValueError(f"{language}: only {len(selected)} eligible unique recordings; never pad or use train")
    # Publish in source order after the seeded selection, keeping runs reproducible.
    selected.sort(key=lambda row: (row["split"] != "test", row["row_index"]))
    metadata = {"source": "https://huggingface.co/datasets/google/fleurs", "revision": REVISION,
                "license": "CC BY 4.0", "attribution": "Conneau et al., FLEURS (2022)",
                "language": language, "config": config, "count": 1000,
                "selection": "All eligible test rows first, then random.Random(42).shuffle(eligible_validation_rows) top-up; reject repeated audio-file SHA256; final manifest sorted by split and source row index",
                "seed": 42, "sources": sources, "source_exclusions": exclusion_counts,
                "duplicate_audio_candidates_skipped": duplicates, "kazakh_prior_adaptation_exclusion": exclusion,
                "selected_split_counts": {split: sum(row["split"] == split for row in selected) for split in pools},
                "distinct_audio_hashes": len({row["sha256"] for row in selected}),
                "distinct_sentence_ids": len({row["sentence_id"] for row in selected}),
                "distinct_normalized_texts": len({" ".join(normalize(row["reference"])) for row in selected}),
                "digit_bearing_references": sum(any(char.isdigit() for char in row["reference"]) for row in selected),
                "duration_over_30_seconds": sum(row["duration_s"] > 30 for row in selected),
                "audio_seconds": sum(row["duration_s"] for row in selected),
                "limitations": "Original read speech, not real telephone calls. Test and validation are reported separately. Multiple recordings of the same sentence are legitimate distinct audio, not independent text prompts. No training split or model training is used. Prior evaluation exposure and model pretraining overlap remain possible.",
                "samples": selected}
    output = args.output / f"{language}_manifest.json"
    temporary = output.with_suffix(".partial")
    temporary.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n")
    summary = {key: value for key, value in metadata.items() if key != "samples"}
    summary["manifest_sha256"] = digest(temporary)
    (args.output / f"{language}_provenance.json").write_text(json.dumps(summary, indent=2) + "\n")
    temporary.replace(output)
    print(json.dumps({"language": language, "manifest": str(output), "splits": metadata["selected_split_counts"],
                      "audio_seconds": metadata["audio_seconds"], "exclusions": exclusion_counts}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--language", choices=list(SOURCES), nargs="+", default=list(SOURCES))
    parser.add_argument("--exclude-kazakh-validation", type=Path)
    args = parser.parse_args()
    args.cache = args.cache.expanduser().resolve()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    for language in args.language:
        prepare(args, language)


if __name__ == "__main__":
    main()
