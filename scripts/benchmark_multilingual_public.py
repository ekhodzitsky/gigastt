#!/usr/bin/env python3
"""Measure a running gigastt server on a provenance-bearing audio manifest.

Requires jiwer in the benchmark Python environment and ffmpeg on PATH.
The manifest has samples with id, path, reference, dataset, and optional
reference_normalized fields. Paths may be absolute or relative to the manifest.
Both original audio and an explicitly simulated 300–3400 Hz / 8 kHz G.711
mu-law condition are measured. No recognition results are silently dropped.
"""

import argparse
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
import time
import urllib.request

import jiwer

from wer_unicode import normalize


def sha256(path):
    with open(path, "rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def score(reference, hypothesis):
    ref = " ".join(normalize(reference))
    hyp = " ".join(normalize(hypothesis))
    if not ref:
        raise ValueError("Empty normalized reference")
    words = jiwer.process_words(ref, hyp)
    chars = jiwer.process_characters(ref, hyp)
    return {
        "reference_words": words.hits + words.substitutions + words.deletions,
        "word_errors": words.substitutions + words.deletions + words.insertions,
        "substitutions": words.substitutions,
        "deletions": words.deletions,
        "insertions": words.insertions,
        "reference_chars": chars.hits + chars.substitutions + chars.deletions,
        "char_errors": chars.substitutions + chars.deletions + chars.insertions,
    }


def summarize(details):
    summaries = []
    for dataset, condition in sorted({(x["dataset"], x["condition"]) for x in details}):
        rows = [x for x in details if (x["dataset"], x["condition"]) == (dataset, condition)]
        for layer in ("verbatim", "normalized"):
            scores = [x["scores"][layer] for x in rows]
            duration = sum(x["duration_s"] for x in rows)
            elapsed = sum(x["elapsed_s"] for x in rows)
            summaries.append({
                "dataset": dataset, "condition": condition, "reference_layer": layer,
                "n": len(rows), "audio_s": duration, "elapsed_s": elapsed,
                "rtf": elapsed / duration,
                "wer_pct": 100 * sum(s["word_errors"] for s in scores) / sum(s["reference_words"] for s in scores),
                "cer_pct": 100 * sum(s["char_errors"] for s in scores) / sum(s["reference_chars"] for s in scores),
                "empty_hypotheses": sum(not x["hypothesis"].strip() for x in rows),
            })
    return summaries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--url", default="http://127.0.0.1:19876")
    parser.add_argument("--variant", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--conditions", nargs="+", default=["original", "simulated_telephone", "simulated_telephone_external_16k"],
                        choices=["original", "original_external_16k", "simulated_telephone", "simulated_telephone_external_16k"])
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    samples = manifest["samples"]
    if not samples or len({s["id"] for s in samples}) != len(samples):
        raise ValueError("Manifest must contain nonempty, unique sample IDs")
    with urllib.request.urlopen(args.url + "/health", timeout=30) as response:
        health = json.load(response)
    if health.get("variant") != args.variant or health.get("punctuation") or health.get("itn"):
        raise ValueError(f"Unexpected server configuration: {health}")
    output = {
        "health": health, "manifest": manifest, "platform": platform.platform(),
        "python": sys.version, "scoring": "scripts/wer_unicode.py normalize; jiwer; micro WER/CER including spaces; no ITN",
        "details": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def transcribe(path):
        request = urllib.request.Request(args.url + "/v1/transcribe", data=path.read_bytes(),
                                         headers={"Content-Type": "application/octet-stream"}, method="POST")
        start = time.perf_counter()
        with urllib.request.urlopen(request, timeout=300) as response:
            result = json.load(response)
        return result, time.perf_counter() - start

    with tempfile.TemporaryDirectory(prefix="gigastt-public-audio-") as temporary:
        for index, sample in enumerate(samples):
            path = Path(sample["path"]).expanduser()
            if not path.is_absolute():
                path = args.manifest.resolve().parent / path
            if sha256(path) != sample["sha256"]:
                raise ValueError(f"Audio checksum mismatch: {path}")
            if index == 0:
                transcribe(path)  # Unmeasured warm-up; steady-state HTTP timing below.
            inputs = {"original": path}
            if any(c.startswith("simulated_") for c in args.conditions):
                degraded = Path(temporary) / "narrowband.wav"
                subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(path), "-ac", "1",
                                "-af", "highpass=f=300,lowpass=f=3400", "-ar", "8000",
                                "-c:a", "pcm_mulaw", str(degraded)], check=True)
                inputs["simulated_telephone"] = degraded
                decoded = Path(temporary) / "narrowband_external_16k.wav"
                subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(degraded),
                                "-ar", "16000", "-c:a", "pcm_s16le", str(decoded)], check=True)
                inputs["simulated_telephone_external_16k"] = decoded
            if "original_external_16k" in args.conditions:
                decoded = Path(temporary) / "original_external_16k.wav"
                subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(path), "-ac", "1",
                                "-ar", "16000", "-c:a", "pcm_s16le", str(decoded)], check=True)
                inputs["original_external_16k"] = decoded
            for condition in args.conditions:
                audio = inputs[condition]
                result, elapsed = transcribe(audio)
                hypothesis = result["text"]
                refs = {"verbatim": sample["reference"],
                        "normalized": sample.get("reference_normalized", sample["reference"])}
                output["details"].append({
                    **sample, "condition": condition, "input_sha256": sha256(audio),
                    "hypothesis": hypothesis, "elapsed_s": elapsed, "duration_s": result["duration"],
                    "scores": {layer: score(ref, hypothesis) for layer, ref in refs.items()},
                })
                args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n")
            print(f"{args.variant}: {index + 1}/{len(samples)} {sample['id']}", flush=True)
    output["summary"] = summarize(output["details"])
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(output["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
