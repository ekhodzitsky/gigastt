#!/usr/bin/env python3
"""Verify a standalone selective-precision pack against frozen native screening."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import time
import urllib.error
import urllib.request


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def get(url, data=None):
    request = urllib.request.Request(url, data=data, headers={'Content-Type': 'application/octet-stream'})
    with urllib.request.urlopen(request, timeout=300) as response:
        return json.load(response)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', type=Path, required=True)
    parser.add_argument('--pack', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    cache = Path.home() / '.cache'
    prepared = cache / 'gigastt-small-diagnosis/inputs/prepared.json'
    native = cache / 'gigastt-small-quantization/s8_channel_reduced_conv_fp32/screen/private_results.json'
    samples = [r for r in json.loads(prepared.read_text())['rows'] if r['variant'] == 'decoded_float']
    expected = {(r['id'], r['condition']): r['hypothesis'] for r in json.loads(native.read_text())['rows']}
    assert len(samples) == len(expected) == 111
    assert not (args.pack / 'multilingual_ctc.int8.onnx').exists(), 'Original encoder companion must be absent'
    assert digest(args.pack / 'multilingual_small_selective.int8.onnx') == '18fa5fab6ece0123d7813f7abc8da129530f6d1a7da17c85c4b3948f27b8e3e0'
    assert digest(args.pack / 'multilingual_vocab.txt') == '4d130287892e1099fedfb3f93c4b4cf8a263151158801680b28977d1be4133f4'
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        port = sock.getsockname()[1]
    url = f'http://127.0.0.1:{port}'
    command = [str(args.binary.resolve()), '--offline', 'serve', '--model-dir', str(args.pack.resolve()),
               '--model-variant', 'ml_ctc', '--execution-provider', 'cpu', '--pool-size', '2',
               '--encoder-intra-threads', '3', '--punctuation', 'off', '--itn', 'off', '--port', str(port)]
    env = {k:v for k,v in os.environ.items() if not k.startswith('GIGASTT_')}
    env['RUST_LOG'] = 'warn'
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.with_suffix('.log').open('w') as log:
        process = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT)
        try:
            deadline = time.monotonic() + 180
            while True:
                if process.poll() is not None:
                    raise RuntimeError('Standalone startup failed')
                try:
                    health = get(url + '/health')
                    models = get(url + '/v1/models')
                    break
                except (urllib.error.URLError, TimeoutError):
                    if time.monotonic() >= deadline:
                        raise TimeoutError('Standalone readiness timed out')
                    time.sleep(.2)
            assert health['variant'] == 'ml_ctc' and not health['punctuation'] and not health['itn']
            assert models['pool_size'] == 2 and models['execution_provider'] == 'cpu'
            def request(row):
                assert digest(row['path']) == row['audio_sha256']
                actual = get(url + '/v1/transcribe', Path(row['path']).read_bytes())['text'].strip()
                reference = expected[(row['id'], row['condition'])].strip()
                return {'id': row['id'], 'condition': row['condition'], 'audio_sha256': row['audio_sha256'],
                        'hypothesis_sha256': hashlib.sha256(actual.encode()).hexdigest(),
                        'expected_hypothesis_sha256': hashlib.sha256(reference.encode()).hexdigest(),
                        'match': actual == reference}
            with ThreadPoolExecutor(max_workers=2) as pool:
                rows = list(pool.map(request, samples))
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
    result = {'completed': True, 'binary_sha256': digest(args.binary), 'script_sha256': digest(__file__),
              'pack_files': {p.name:digest(p) for p in sorted(args.pack.iterdir()) if p.is_file()},
              'prepared_sha256': digest(prepared), 'native_private_sha256': digest(native),
              'command': command, 'health': health, 'models': models, 'n': len(rows),
              'exact_matches': sum(r['match'] for r in rows), 'original_companion_present': False, 'rows': rows}
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    assert result['exact_matches'] == 111
    print('Standalone startup and111/111 native transcript parity PASS')


if __name__ == '__main__':
    main()
