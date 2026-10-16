#!/usr/bin/env python3
"""Evaluate predeclared fixed 10/20-second chunks without oracle boundaries.

Inputs and private outputs must stay outside the repository for LDC demos.
Only hashes, counts and aggregate scores are saved to --public-output.
Requires soundfile and the existing benchmark Python dependencies.
"""

import argparse
import io
import json
from pathlib import Path
import time
import urllib.request

import soundfile as sf

from benchmark_multilingual_public import score, sha256


def dump(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")


def recording_scores(rows):
    groups = {}
    for row in rows:
        key = (row["recording_id"], row["window_seconds"])
        group = groups.setdefault(key, {"recording_id": key[0], "window_seconds": key[1],
                                        "channels": 0, "word_errors": 0, "reference_words": 0})
        group["channels"] += 1
        group["word_errors"] += row["scores"]["word_errors"]
        group["reference_words"] += row["scores"]["reference_words"]
    for group in groups.values():
        group["wer_percent"] = 100 * group["word_errors"] / group["reference_words"]
    return list(groups.values())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, nargs="+", required=True)
    parser.add_argument("--variant", choices=["ml_ctc", "ml_ctc_large"], required=True)
    parser.add_argument("--url", default="http://127.0.0.1:19880")
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--private-output", type=Path, required=True)
    parser.add_argument("--public-output", type=Path, required=True)
    args = parser.parse_args()
    with urllib.request.urlopen(args.url + "/health", timeout=30) as response:
        health = json.load(response)
    if health.get("variant") != args.variant or health.get("punctuation") or health.get("itn"):
        raise ValueError(f"Unexpected server settings: {health}")
    rows = []
    for manifest in args.manifest:
        data = json.loads(manifest.read_text())
        rows.extend(data["samples"])
    if not rows or len({row["id"] for row in rows}) != len(rows):
        raise ValueError("Expected nonempty unique sample IDs")
    details = []
    metadata = {
        "model_variant": args.variant, "binary_sha256": sha256(args.binary),
        "model_sha256": sha256(args.model), "health": health,
        "windows_seconds": [10, 20], "boundary_policy": "Nonoverlapping fixed-duration chunks from sample zero; final chunk may be shorter. No VAD, reference timestamps, overlap or hypothesis stitching.",
        "scoring": "Concatenate all chunk hypotheses in chronological order, then score against the complete channel reference with shared Unicode normalization and jiwer.",
        "limitations": "Two public recordings, three channels total; fixed boundaries can split words. Already inspected development diagnostics, not independent final holdout. No speed comparison because other jobs run concurrently.",
        "manifest_sha256": [sha256(path) for path in args.manifest],
    }
    for row in rows:
        path = Path(row["path"])
        if sha256(path) != row["sha256"]:
            raise ValueError(f"Audio checksum mismatch: {path}")
        audio, rate = sf.read(path, dtype="int16", always_2d=True)
        if rate != 8000 or audio.shape[1] != 1:
            raise ValueError("Expected mono 8 kHz telephone input")
        for window in (10, 20):
            hypotheses = []
            chunks = []
            started = time.monotonic()
            for start in range(0, len(audio), window * rate):
                pcm = audio[start:start + window * rate]
                stream = io.BytesIO()
                sf.write(stream, pcm, rate, format="WAV", subtype="PCM_16")
                request = urllib.request.Request(args.url + "/v1/transcribe", data=stream.getvalue(),
                                                 headers={"Content-Type": "application/octet-stream"}, method="POST")
                with urllib.request.urlopen(request, timeout=300) as response:
                    result = json.load(response)
                hypotheses.append(result["text"])
                chunks.append({"start_seconds": start / rate, "duration_seconds": len(pcm) / rate,
                               "hypothesis": result["text"]})
            text = " ".join(hypotheses)
            scores = score(row["reference"], text)
            details.append({"id": row["id"], "dataset": row["dataset"], "source_audio_sha256": row["sha256"],
                            "recording_id": row.get("call_id", row["id"]),
                            "window_seconds": window, "duration_seconds": len(audio) / rate,
                            "chunks": len(chunks), "empty_chunks": sum(not h.strip() for h in hypotheses),
                            "elapsed_seconds": time.monotonic() - started,
                            "wer_percent": 100 * scores["word_errors"] / scores["reference_words"],
                            "scores": scores, "reference": row["reference"], "hypothesis": text,
                            "chunk_details": chunks})
            dump(args.private_output, {"metadata": metadata, "details": details})
            public_rows = [{k: v for k, v in value.items() if k not in {"reference", "hypothesis", "chunk_details"}} for value in details]
            dump(args.public_output, {"metadata": metadata, "completed": len(details) == len(rows) * 2,
                                     "details": public_rows, "recordings": recording_scores(public_rows)})
            print(row["id"], window, round(details[-1]["wer_percent"], 4), flush=True)


if __name__ == "__main__":
    main()
