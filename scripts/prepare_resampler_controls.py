#!/usr/bin/env python3
"""Prepare three fixed laboratory resampler controls from the precision manifest.

Requires numpy, ffmpeg and the resample_narrowband Rust example. No runtime
configuration is changed. Use only the same simulated Kazakh evaluation inputs.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import wave

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--resampler', type=Path, default=Path('target/debug/examples/resample_narrowband'))
    args = parser.parse_args()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    for sample in json.loads(args.manifest.read_text()):
        if sample['language'] != 'kazakh' or sample['condition'] != 'simulated_telephone':
            continue
        source = Path(sample['path'])
        if hashlib.sha256(source.read_bytes()).hexdigest() != sample['audio_sha256']:
            raise ValueError(f'Input checksum mismatch: {source}')
        raw = args.output / (sample['id'] + '.f32')
        subprocess.run(['ffmpeg', '-v', 'error', '-y', '-i', str(source), '-ac', '1',
                        '-ar', '8000', '-f', 'f32le', str(raw)], check=True)
        for name, length, cutoff in [('current', 256, .95), ('short32', 32, .97), ('short64', 64, .97)]:
            floats = args.output / (sample['id'] + '_' + name + '.f32')
            wav = floats.with_suffix('.wav')
            subprocess.run([str(args.resampler.resolve()), str(raw), str(floats), str(length), str(cutoff)], check=True)
            values = np.fromfile(floats, dtype='<f4') * 32768
            pcm = np.clip(np.copysign(np.floor(np.abs(values) + .5), values), -32768, 32767).astype('<i2')
            with wave.open(str(wav), 'wb') as output:
                output.setnchannels(1)
                output.setsampwidth(2)
                output.setframerate(16000)
                output.writeframes(pcm.tobytes())
            rows.append(dict(id=sample['id'] + '_' + name, path=str(wav), reference=sample['reference'],
                             sha256=hashlib.sha256(wav.read_bytes()).hexdigest(), dataset='kazakh_filter_' + name))
    if not rows:
        raise ValueError('No matching Kazakh simulated telephone samples')
    (args.output / 'manifest.json').write_text(json.dumps(dict(
        source='Fixed Kazakh sample set; diagnostic only; 32/64 taps at .97 cutoff versus current256/.95, all PCM16',
        samples=rows), ensure_ascii=False, indent=2) + '\n')


if __name__ == '__main__':
    main()
