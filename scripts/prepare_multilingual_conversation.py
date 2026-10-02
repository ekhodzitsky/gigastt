#!/usr/bin/env python3
"""Prepare public Uzbek Telegram speech and the LDC Kazakh telephone sample.

Audio is kept locally. LDC's sample is not an openly licensed full corpus;
do not redistribute its audio or transcript as part of benchmark artifacts.
Requires pyarrow and soundfile in the benchmark environment, plus ffmpeg.
"""

import argparse
import json
from pathlib import Path
import random
import re
import subprocess
import urllib.request

import pyarrow as pa
import soundfile as sf

from prepare_multilingual_public import digest

REVISION = "c6e1b4f66d45e9d7644003d1564837786295850a"
HASHES = ["c83379b77b1ba1cfe58458e7305a11628a831fc8d95d357293f6e16839ecba83",
          "986885f18da1f39d93a2b6fc965a2c55714c7b2d21f13d25311490325943afdc"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cache = args.cache.resolve()
    cache.mkdir(parents=True, exist_ok=True)
    args.output.mkdir(parents=True, exist_ok=True)
    selected = set(random.Random(42).sample(range(745), 100))
    audio_dir = cache / "uz_telegram"
    audio_dir.mkdir(exist_ok=True)
    samples = []
    index = 0
    for shard, checksum in enumerate(HASHES):
        path = cache / f"uz_telegram_{shard}.arrow"
        if not path.exists():
            url = f"https://huggingface.co/datasets/BoburAmirov/asr_evaluate_set/resolve/{REVISION}/data-0000{shard}-of-00002.arrow"
            urllib.request.urlretrieve(url, path)
        if digest(path) != checksum:
            raise ValueError(f"Arrow checksum mismatch: {path}")
        with pa.memory_map(str(path), "r") as mapped:
            for batch in pa.ipc.open_stream(mapped):
                for offset in range(batch.num_rows):
                    if index in selected:
                        row = batch.slice(offset, 1).to_pylist()[0]
                        audio = row["audio"]
                        output = audio_dir / Path(row["name"]).name
                        sf.write(output, audio["array"], audio["sampling_rate"], subtype="PCM_16")
                        samples.append({"id": f"telegram_{index}", "dataset": "uzbek_telegram_natural",
                                        "path": str(output), "sha256": digest(output),
                                        "reference": row["transcript"], "row_index": index,
                                        "source_name": row["name"], "source_sample_rate": audio["sampling_rate"]})
                    index += 1
    if index != 745 or len(samples) != 100:
        raise ValueError("Unexpected Telegram dataset length")
    manifest = {"source": "https://huggingface.co/datasets/BoburAmirov/asr_evaluate_set",
                "revision": REVISION, "license": "Apache-2.0 per publisher",
                "selection": "random.Random(42).sample(range(745), 100), in source order",
                "notes": "Public Telegram voice messages, manual references per publisher; train is the sole published split; not telephone calls",
                "arrow_sha256": HASHES, "samples": samples}
    (args.output / "telegram_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    for extension in ("sph", "txt"):
        path = cache / f"ldc_kazakh.{extension}"
        if not path.exists():
            urllib.request.urlretrieve(f"https://catalog.ldc.upenn.edu/desc/addenda/LDC2018S13.{extension}", path)
    wav = cache / "ldc_kazakh.wav"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(cache / "ldc_kazakh.sph"), str(wav)], check=True)
    text = (cache / "ldc_kazakh.txt").read_text()
    reference = " ".join(re.sub(r"\[[\d.]+\]|<[^>]+>", " ", text).split())
    manifest = {"source": "https://catalog.ldc.upenn.edu/LDC2018S13",
                "license": "Public catalog demonstration sample; full corpus requires LDC agreement. No redistribution of sample audio or transcript.",
                "reference_preparation": "Remove timestamp lines and angle-bracket nonlexical event tags; keep all spoken words",
                "sph_sha256": digest(cache / "ldc_kazakh.sph"), "transcript_sha256": digest(cache / "ldc_kazakh.txt"),
                "samples": [{"id": "ldc_kazakh_sample", "dataset": "kazakh_real_telephone_single_sample",
                             "path": str(wav), "sha256": digest(wav), "reference": reference}]}
    (cache / "telephone_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print("Prepared 100 Uzbek Telegram clips and one real Kazakh telephone sample")


if __name__ == "__main__":
    main()
