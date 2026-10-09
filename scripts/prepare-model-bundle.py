#!/usr/bin/env python3
"""Rebuild the reviewed INT8 candidate from hash-pinned FP32 packaging sources.

Never call the runtime download command to produce a new model: it fetches the
previously published INT8 bundle. Baseline and candidate live in separate dirs.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import urllib.request

MANIFEST = Path(__file__).resolve().parents[1] / 'benchmark/model-release.json'


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def verify(path, expected):
    if not re.fullmatch('[0-9a-f]{64}', expected) or digest(path) != expected:
        raise ValueError(f'checksum mismatch: {path}')


def fetch(url, path, expected):
    if path.is_file():
        verify(path, expected)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix('.partial')
    try:
        with urllib.request.urlopen(url, timeout=120) as source, partial.open('wb') as out:
            shutil.copyfileobj(source, out, 1024 * 1024)
        verify(partial, expected)
        partial.replace(path)
    finally:
        partial.unlink(missing_ok=True)


def prepare(binary, root):
    manifest = json.loads(MANIFEST.read_text())
    for head, recipe in manifest['heads'].items():
        candidate, baseline = root / 'candidate' / head, root / 'baseline' / head
        for directory in (candidate, baseline):
            directory.mkdir(parents=True, exist_ok=True)
        source_name, encoder = f'v3_{head}_encoder.onnx', f'v3_{head}_encoder_int8.onnx'
        for name, checksum in recipe['source_files'].items():
            fetch(f"{manifest['source_repo']}/{name}", candidate / name, checksum)
            if name != source_name:
                if (baseline / name).exists():
                    verify(baseline / name, checksum)
                else:
                    shutil.copyfile(candidate / name, baseline / name)
        fetch(f"{manifest['baseline_release']}/{encoder}", baseline / encoder,
              recipe['baseline_encoder_sha256'])
        subprocess.run([str(binary.resolve()), 'quantize', '--model-dir', str(candidate.resolve()),
                        '--model-variant', head, '--force'], check=True, timeout=600,
                       env={k: v for k, v in os.environ.items() if not k.startswith('GIGASTT_')})
        verify(candidate / encoder, recipe['candidate_encoder_sha256'])
    dist = root / 'dist'
    dist.mkdir(exist_ok=False)
    sums = []
    for head, recipe in manifest['heads'].items():
        encoder = f'v3_{head}_encoder_int8.onnx'
        for name, checksum in {**{n: h for n, h in recipe['source_files'].items()
                                 if n != f'v3_{head}_encoder.onnx'},
                               encoder: recipe['candidate_encoder_sha256']}.items():
            shutil.copyfile(root / 'candidate' / head / name, dist / name)
            verify(dist / name, checksum)
            sums.append(f'{checksum}  {name}\n')
    (dist / 'SHA256SUMS.txt').write_text(''.join(sorted(sums)))
    return manifest['tag']


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', required=True, type=Path)
    parser.add_argument('--root', required=True, type=Path)
    args = parser.parse_args()
    print(prepare(args.binary, args.root))
