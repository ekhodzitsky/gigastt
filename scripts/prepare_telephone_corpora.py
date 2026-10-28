#!/usr/bin/env python3
"""Prepare the public MATERIAL Kazakh telephone demonstration locally.

The full LDC corpus requires an agreement. Keep the downloaded audio, human
transcript, and manifests containing reference text outside the repository.
Only provenance, hashes, and aggregate measurements should be published.
Requires soundfile and numpy from the existing benchmark environment.
"""

import argparse
import hashlib
import json
from pathlib import Path
import re
import urllib.request

import soundfile as sf


SOURCE = "https://catalog.ldc.upenn.edu/LDC2025S03"
BASE = "https://catalog.ldc.upenn.edu/desc/addenda/"
CUTOFF = 56.140
HASHES = {
    "LDC2025S03.wav": "ebee68a1297efcb7b99d40b2b4aaa91b0e48461ff4fe913a105c9cc6d9c423a9",
    "LDC2025S03.transcription.txt": "ec4961caab734ff15009a09539b41d07da605ad2f02838d5b75adcafac0ae3e1",
}


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def clean(text):
    return " ".join(re.sub(r"<[^>]+>", " ", text).split())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()
    cache = args.cache.resolve()
    cache.mkdir(parents=True, exist_ok=True)
    for filename in ("LDC2025S03.wav", "LDC2025S03.transcription.txt"):
        path = cache / filename
        if not path.exists():
            temporary = path.with_suffix(path.suffix + ".partial")
            with urllib.request.urlopen(BASE + filename, timeout=120) as response:
                temporary.write_bytes(response.read())
            temporary.replace(path)
        if digest(path) != HASHES[filename]:
            raise ValueError(f"Source checksum mismatch: {path}")
    wav = cache / "LDC2025S03.wav"
    transcript = cache / "LDC2025S03.transcription.txt"
    audio, rate = sf.read(wav, dtype="int16", always_2d=True)
    if rate != 8000 or audio.shape != (960000, 2):
        raise ValueError(f"Unexpected audio shape: {rate}, {audio.shape}")
    annotations = {"inLine": [], "outLine": []}
    for line in transcript.read_text().splitlines():
        timestamp, channel, reference = line.split("\t")
        annotations[channel].append((float(timestamp), clean(reference)))
    samples = []
    segments = []
    for channel_index, channel in enumerate(("inLine", "outLine")):
        path = cache / f"material_kk_{channel}.wav"
        sf.write(path, audio[:round(CUTOFF * rate), channel_index], rate, subtype="PCM_16")
        rows = annotations[channel]
        reference = " ".join(text for timestamp, text in rows if timestamp < CUTOFF)
        samples.append({"id": f"material_kk_{channel}", "dataset": "kazakh_material_real_telephone",
                        "path": str(path), "sha256": digest(path), "reference": reference,
                        "call_id": "ldc2025s03_public_sample", "channel": channel,
                        "duration_seconds": CUTOFF, "source_sample_rate": rate})
        for index, ((start, text), (end, _)) in enumerate(zip(rows, rows[1:])):
            # Exclude incomplete final annotated turns and non-lexical intervals.
            if not re.search(r"\w", text) or end > CUTOFF:
                continue
            path = cache / f"material_kk_{channel}_{index:02d}.wav"
            sf.write(path, audio[round(start * rate):round(end * rate), channel_index], rate, subtype="PCM_16")
            segments.append({"id": f"material_kk_{channel}_{index:02d}",
                             "dataset": "kazakh_material_real_telephone_oracle_segments",
                             "path": str(path), "sha256": digest(path), "reference": text,
                             "call_id": "ldc2025s03_public_sample", "channel": channel,
                             "start_seconds": start, "end_seconds": end,
                             "duration_seconds": end - start, "source_sample_rate": rate})
    metadata = {
        "source": SOURCE, "audio_url": BASE + wav.name, "transcript_url": BASE + transcript.name,
        "license": "Public LDC demonstration; full corpus requires an agreement. No sample redistribution.",
        "audio_sha256": digest(wav), "transcript_sha256": digest(transcript),
        "audio_original_seconds": 120, "audio_original_channels": 2,
        "audio_original_codec": "PCM A-law, 8000 Hz",
        "reference_preparation": "Strip angle-bracket events; preserve words and filled pauses. Cut both channels at 56.140 s, before the last exchange; published transcript only covers approximately the first minute of a 120 s recording.",
        "channel_mapping": "inLine=channel 0, outLine=channel 1; checked against speech/silence intervals at 0-1, 2.4-5.5 and 6-12 seconds, without ASR-based selection.",
        "independence": "Distinct transcript from LDC2018S13 public sample; MATERIAL contains Babel material, so speakers/recordings cannot be certified independent of the full Babel corpus.",
        "unique_calls": 1, "not_confirmed_code_switching": True,
        "oracle_segments_warning": "Human reference timestamps give privileged segmentation; report separately from full-channel/VAD inference, never as extra independent calls.",
    }
    for name, rows in (("telephone_expansion_manifest.json", samples), ("telephone_segments_manifest.json", segments)):
        (cache / name).write_text(json.dumps({**metadata, "samples": rows}, ensure_ascii=False, indent=2) + "\n")
    summary = {**metadata, "channels_evaluated": len(samples), "channel_seconds": sum(x["duration_seconds"] for x in samples),
               "oracle_segments": len(segments), "oracle_segment_seconds": sum(x["duration_seconds"] for x in segments),
               "samples": [{key: value for key, value in row.items() if key not in ("reference", "path")} for row in samples]}
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Prepared one call: {len(samples)} channels and {len(segments)} diagnostic segments")


if __name__ == "__main__":
    main()
