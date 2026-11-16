#!/usr/bin/env python3
"""Install the original E2E RNNT FP32 pack separately from default INT8 models.

Python standard library only. Copies local files with --source-dir, otherwise
downloads the pinned upstream revision. Every payload is size/hash verified.
"""
import argparse
import hashlib
from pathlib import Path
import tempfile
import urllib.request


REVISION = '322c3b29492673eb7d0b434bfa9dfb8653e34d02'
BASE_URL = f'https://huggingface.co/istupakov/gigaam-v3-onnx/resolve/{REVISION}'
FILES = {
    'v3_e2e_rnnt_encoder.onnx': (885084534, 'cd60b3764a832e8560ae6d3ad0b10adc1a42ffae412b9476f25620aae4f4a508'),
    'v3_e2e_rnnt_decoder.onnx': (4599910, '7b0a16d67fd2cb37061decc93c69e364a9ab27afee3c57495d55b1c974cf7231'),
    'v3_e2e_rnnt_joint.onnx': (2712896, '602ff7017a93311aad34df1437c8d7f49911353c13d6eae7a6ee7b041339465c'),
    'v3_e2e_rnnt_vocab.txt': (13354, '39abae20e692998290c574e606f11a9edef2902a1995463fcff63d1490cf22b7'),
}
MANIFEST = '''architecture = "e2e_rnnt"
precision = "fp32"

[files]
encoder = "v3_e2e_rnnt_encoder.onnx"
decoder = "v3_e2e_rnnt_decoder.onnx"
joint = "v3_e2e_rnnt_joint.onnx"
vocab = "v3_e2e_rnnt_vocab.txt"
'''


def copy_verified(source, destination, size, expected):
    digest = hashlib.sha256()
    copied = 0
    with destination.open('xb') as output:
        while chunk := source.read(1024 * 1024):
            copied += len(chunk)
            if copied > size:
                raise ValueError(f'Unexpected size: {destination.name}')
            digest.update(chunk)
            output.write(chunk)
    if copied != size:
        raise ValueError(f'Unexpected size: {destination.name}')
    if digest.hexdigest() != expected:
        raise ValueError(f'SHA-256 mismatch: {destination.name}')


def install(destination, source_dir=None):
    """Stage verified regular files, then publish a separate model directory."""
    destination = Path(destination).expanduser().absolute()
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f'Destination must not exist: {destination}')
    if source_dir is not None:
        source_dir = Path(source_dir).expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.gigastt-fp32-', dir=destination.parent) as temp:
        staging = Path(temp) / 'pack'
        staging.mkdir()
        for name, (size, expected) in FILES.items():
            if source_dir is None:
                source = urllib.request.urlopen(f'{BASE_URL}/{name}', timeout=120)
            else:
                source = (source_dir / name).open('rb')
            with source:
                copy_verified(source, staging / name, size, expected)
        (staging / 'manifest.toml').write_text(MANIFEST, encoding='utf-8')
        # A failed download never exposes a partial pack. Existing nonempty
        # directories cannot be replaced by a directory rename.
        if destination.exists() or destination.is_symlink():
            raise FileExistsError(f'Destination appeared during installation: {destination}')
        staging.rename(destination)
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--destination', required=True, type=Path,
                        help='New, separate model directory (must not exist)')
    parser.add_argument('--source-dir', type=Path,
                        help='Copy pinned original assets locally instead of downloading')
    args = parser.parse_args()
    try:
        destination = install(args.destination, args.source_dir)
    except (OSError, ValueError) as error:
        parser.exit(1, f'Installation failed: {error}\n')
    print(f'Installed FP32 E2E pack: {destination}')
    print('Use --model-dir with this directory; the default build runs on CPU.')


if __name__ == '__main__':
    main()
