#!/usr/bin/env python3
"""Reconstruct the qualified GigaAM v4 frontend from its pinned safetensors."""

import argparse
import hashlib
import json
import struct
from pathlib import Path

CHECKPOINT_SHA256 = "2c38690c38edbd9dfc2190f1a2dacdcfbea2a692078457fc145a6034c031840a"
PREFIX = "model.preprocessor.featurizer.0."


def extract(checkpoint: Path) -> bytes:
    with checkpoint.open("rb") as handle:
        if hashlib.file_digest(handle, "sha256").hexdigest() != CHECKPOINT_SHA256:
            raise ValueError("checkpoint does not match the qualified v4 revision")
        handle.seek(0)
        header_size = struct.unpack("<Q", handle.read(8))[0]
        header = json.loads(handle.read(header_size))
        data_start = 8 + header_size

        def tensor(name, shape):
            entry = header[PREFIX + name]
            if entry["dtype"] != "F32" or entry["shape"] != shape:
                raise ValueError("unexpected qualified frontend tensor shape or dtype")
            start, end = entry["data_offsets"]
            handle.seek(data_start + start)
            result = handle.read(end - start)
            count = 1
            for dim in shape:
                count *= dim
            if len(result) != count * 4:
                raise ValueError("invalid qualified frontend tensor length")
            return result

        window = tensor("spectrogram.window", [320])
        filterbank = tensor("mel_scale.fb", [161, 64])
        # Preserve the exact little-endian float32 bits; transpose only axes.
        mel_major = b"".join(
            filterbank[(freq * 64 + mel) * 4 : (freq * 64 + mel + 1) * 4]
            for mel in range(64)
            for freq in range(161)
        )
        return window + mel_major


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    result = extract(args.checkpoint)
    if args.output.exists():
        if args.output.read_bytes() != result:
            raise ValueError("refusing to replace a different existing output")
    else:
        args.output.write_bytes(result)
    print(hashlib.sha256(result).hexdigest())


if __name__ == "__main__":
    main()
