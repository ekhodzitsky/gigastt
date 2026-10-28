#!/usr/bin/env python3
"""Fixed native-product controls for small multilingual deletion failures.

No training or model changes. The cases are failure-enriched diagnostics,
not an independent language-quality benchmark.
"""
import argparse
import concurrent.futures
import hashlib
import json
from pathlib import Path
import subprocess
import urllib.request

import numpy as np
import soundfile as sf

from benchmark_multilingual_1000 import score


def digest(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def save(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    tmp.replace(path)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cases', type=Path, required=True)
    p.add_argument('--work', type=Path, required=True)
    p.add_argument('--inspect-audio', type=Path, required=True)
    p.add_argument('--url', default='http://127.0.0.1:19911')
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    cases = json.loads(args.cases.read_text())['rows']
    args.work.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(args.url + '/health', timeout=30) as f:
        health = json.load(f)
    assert health['variant'] == 'ml_ctc' and health['punctuation'] is False and health['itn'] is False
    entries = []
    prepared = []

    def dump(path, prefix):
        subprocess.run([str(args.inspect_audio.resolve()), str(path), str(prefix)], check=True, capture_output=True)
        return np.fromfile(str(prefix) + '.pcm.f32', dtype='<f4')

    for i, row in enumerate(cases):
        assert digest(row['path']) == row['audio_sha256']
        key = row['id'] + '_' + row['condition']
        directory = args.work / key
        directory.mkdir(exist_ok=True)
        native = dump(row['path'], directory / 'native')
        variants = [('source_file', Path(row['path'])), ('decoded_float', native)]
        if row['condition'] != 'original':
            samples, rate = sf.read(row['path'], dtype='float32')
            assert rate == 8000 and samples.ndim == 1
            expected = len(samples) * 2
            assert expected - len(native) == 4, 'Length assumption changed'
            padded = directory / 'source_with_flush.wav'
            sf.write(padded, np.pad(samples, (0, 512)), 8000, subtype='FLOAT')
            flushed = dump(padded, directory / 'flushed')
            assert np.array_equal(flushed[:len(native)], native), 'Flush changed existing samples'
            assert len(flushed) >= 256 + expected
            variants.extend([
                ('length_only_pad4', np.pad(native, (0, 4))),
                ('delay_only_shift256', np.pad(native[256:], (0, 256))),
                ('flush_trim256', flushed[256:256 + expected]),
            ])
        for variant, value in variants:
            if isinstance(value, Path):
                path = value
                feature_prefix = directory / 'native'
            else:
                path = directory / (variant + '.wav')
                sf.write(path, value, 16000, subtype='FLOAT')
                roundtrip, rate = sf.read(path, dtype='float32')
                assert rate == 16000 and np.array_equal(value, roundtrip)
                feature_prefix = directory / variant
                decoded = dump(path, feature_prefix)
                assert np.array_equal(value, decoded), 'Native float WAV decode changed PCM'
            prepared.append({**row, 'variant': variant, 'input_path': str(path), 'input_sha256': digest(path),
                             'feature_path': str(feature_prefix) + '.mel.f32',
                             'feature_sha256': digest(str(feature_prefix) + '.mel.f32'),
                             'frames': json.loads(Path(str(feature_prefix) + '.json').read_text())['mel_shape'][1]})
        if (i + 1) % 10 == 0:
            print('Prepared', i + 1, '/', len(cases), flush=True)
    assert len(prepared) == 444
    metadata = {'cases_sha256': digest(args.cases), 'script_sha256': digest(__file__),
                'inspect_audio_sha256': digest(args.inspect_audio), 'health': health,
                'protocol': 'source_file and decoded_float for all 111 inputs; length_only_pad4, delay_only_shift256 and true flush_trim256 for all 74 telephone inputs. Delay 256 is the rubato API value, fixed before outcomes. Exact flush prefix and float WAV roundtrip verified.'}
    save(args.work / 'prepared.json', {'metadata': metadata, 'rows': prepared})

    def infer(row):
        request = urllib.request.Request(args.url + '/v1/transcribe', data=Path(row['input_path']).read_bytes(),
                                         headers={'Content-Type': 'application/octet-stream'}, method='POST')
        with urllib.request.urlopen(request, timeout=300) as f:
            result = json.load(f)
        text = result['text']
        return {**row, 'hypothesis': text, 'scores': score(row['reference'], text),
                'exact_baseline_match': text.strip() == row['baseline_hypothesis'].strip()}

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        for row in pool.map(infer, prepared):
            entries.append(row)
            save(args.output, {'metadata': metadata, 'completed': len(entries) == len(prepared), 'n': len(entries), 'rows': entries})
            if len(entries) % 20 == 0:
                print('Recognized', len(entries), '/', len(prepared), flush=True)
    parity = [r for r in entries if r['variant'] in ('source_file', 'decoded_float')]
    assert all(r['exact_baseline_match'] for r in parity), 'Native parity failed; do not interpret interventions'
    print('Complete: native baseline parity', len(parity), '/', len(parity), flush=True)


if __name__ == '__main__':
    main()
