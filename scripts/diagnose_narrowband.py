#!/usr/bin/env python3
"""Ablate waveform precision/window/log floor on a decoded PCM/feature dump.

Create INPUT.pcm.f32 and INPUT.mel.f32 using the inspect_audio Rust example.
Needs numpy, onnxruntime and onnx-asr (only its packaged reference filterbank).
This is a diagnostic NumPy frontend, not a replacement production benchmark.
"""

import argparse
import importlib.resources
import itertools
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-prefix", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--start", type=float, default=0)
    parser.add_argument("--end", type=float)
    parser.add_argument("--omit-text", action="store_true")
    args = parser.parse_args()
    options = ort.SessionOptions()
    options.intra_op_num_threads = 6
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(args.model_dir / "multilingual_ctc.int8.onnx"),
                                  options, providers=["CPUExecutionProvider"])
    vocab = [line.rsplit(" ", 1)[0] for line in
             (args.model_dir / "multilingual_vocab.txt").read_text().splitlines()]
    with importlib.resources.as_file(importlib.resources.files("onnx_asr.preprocessors.data") / "fbanks.npz") as path:
        with np.load(path) as data:
            filters = data["gigaam_v3"].copy()
            window = data["gigaam_v3_window"].copy()
    pcm = np.fromfile(str(args.input_prefix) + ".pcm.f32", dtype="<f4")
    pcm = pcm[int(args.start * 16000):None if args.end is None else int(args.end * 16000)]
    rows = []

    def decode(features, **condition):
        logits = session.run(None, {"features": features[None].astype("float32"),
                                   "feature_lengths": np.array([features.shape[1]], dtype="int64")})[0]
        ids = np.argmax(logits, axis=-1).ravel()
        text = "".join(vocab[i] for i, _ in itertools.groupby(ids) if i != len(vocab) - 1).replace("▁", " ").strip()
        row = {**condition, "words": len(text.split()), "high_band_mean": float(features[50:].mean())}
        if not args.omit_text:
            row["text"] = text
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)

    if not args.start and args.end is None:
        rust_mel = np.fromfile(str(args.input_prefix) + ".mel.f32", dtype="<f4").reshape(64, -1)
        for floor in [1e-10, 1e-9, 1e-8, 1e-7]:
            decode(np.maximum(rust_mel, np.log(floor)), frontend="Rust dump", floor=floor)
    for bits in [None, 16, 15, 14, 12]:
        waveform = pcm if bits is None else np.clip(np.round(pcm * 2 ** (bits - 1)), -2 ** (bits - 1), 2 ** (bits - 1) - 1) / 2 ** (bits - 1)
        for name, weights in [("symmetric_f32", np.hanning(320).astype("float32")),
                              ("author_periodic_bf16", window)]:
            frames = np.lib.stride_tricks.sliding_window_view(waveform, 320)[::160] * weights
            power = np.abs(np.fft.rfft(frames)).astype("float32") ** 2
            mel = np.log(np.clip(power @ filters, 1e-9, 1e9)).T
            decode(mel, frontend="NumPy, author bf16 mel filters", bits=bits, window=name, floor=1e-9)
    args.output.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
