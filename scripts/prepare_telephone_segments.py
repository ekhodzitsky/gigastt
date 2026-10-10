#!/usr/bin/env python3
"""Create diagnostic speech clips using LDC reference timestamps (oracle cuts).

These cuts are not an automatic segmentation algorithm. Keep the resulting
LDC audio, manifest and reference text local; do not redistribute them.
"""

import argparse
import hashlib
import json
from pathlib import Path
import re
import wave


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True, help="Contains ldc_kazakh.wav and .txt")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    transcript = args.cache / "ldc_kazakh.txt"
    parts = re.split(r"\[(\d+\.\d+)\]", transcript.read_text())
    with wave.open(str(args.cache / "ldc_kazakh.wav")) as source:
        rate = source.getframerate()
        params = source.getparams()
        if (source.getnchannels(), source.getsampwidth(), rate) != (1, 2, 8000):
            raise ValueError("Expected mono PCM16 at 8 kHz")
        pcm = source.readframes(source.getnframes())
        duration = source.getnframes() / rate
    samples = []
    for i in range(1, len(parts), 2):
        start = float(parts[i])
        end = float(parts[i + 2]) if i + 2 < len(parts) else duration
        ref = " ".join(re.sub(r"<[^>]*>", " ", parts[i + 1]).split())
        if not ref:
            continue
        if not 0 <= start < end <= duration:
            raise ValueError("Reference timestamp outside audio")
        path = args.output / f"{len(samples):02d}.wav"
        with wave.open(str(path), "wb") as target:
            target.setparams(params)
            target.writeframes(pcm[round(start * rate) * 2:round(end * rate) * 2])
        samples.append({"id": path.stem, "path": str(path),
                        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                        "reference": ref, "dataset": "kazakh_phone_oracle_segments",
                        "start": start, "end": end})
    manifest = {"source": "https://catalog.ldc.upenn.edu/LDC2018S13",
                "license": "Public demonstration; no redistribution",
                "segmentation": "Oracle reference timestamps, no padding; diagnostic only, not automatic production segmentation",
                "samples": samples}
    (args.output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(f"Prepared {len(samples)} oracle segments")


if __name__ == "__main__":
    main()
