#!/usr/bin/env python3
"""Run an isolated candidate pack through the frozen native HTTP server.

The full benchmark delegates scoring/resume to benchmark_multilingual_1000.py.
Three condition workers each own one server, two requests and three intra-op
threads per pool slot. Contended full-run timings are not resource benchmarks.
"""
import argparse
import concurrent.futures
import contextlib
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request

from benchmark_multilingual_1000 import atomic_json, score


BINARY_SHA = 'bb30ca980374af9eb6e9fae9d239c3ce34a85e69e202bc8f8bac6c1e7ae36566'
SELECTED = 's8_channel_reduced_conv_fp32'
REPORT = Path('benchmark/results/multilingual_small_quantization_20261001')
WORK = Path.home() / '.cache/gigastt-small-quantization'


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write_identical(path, content):
    if path.exists():
        if path.read_bytes() != content:
            raise ValueError('Existing pack differs: ' + str(path))
    else:
        path.write_bytes(content)


def prepare_pack(binary):
    if digest(binary) != BINARY_SHA:
        raise ValueError('Frozen server binary mismatch')
    model = json.loads((REPORT / (SELECTED + '.json')).read_text())
    if digest(model['path']) != model['sha256']:
        raise ValueError('Candidate weight identity mismatch')
    pack = WORK / 'selected_product_pack'
    pack.mkdir(parents=True, exist_ok=True)
    encoder = pack / 'lab_multilingual_ctc.int8.onnx'
    if not encoder.exists():
        encoder.symlink_to(Path(model['path']).resolve())
    if digest(encoder) != model['sha256']:
        raise ValueError('Existing pack encoder mismatch')
    # The frozen CLI's presence check predates manifest-aware engine loading.
    # Provide the real pinned original under its standard name; the manifest
    # still selects only the laboratory encoder for inference.
    companion = pack / 'multilingual_ctc.int8.onnx'
    if not companion.exists():
        companion.symlink_to(Path.home() / '.gigastt/models/multilingual_ctc.int8.onnx')
    if digest(companion) != model['original_sha256']:
        raise ValueError('Pinned bootstrap companion mismatch')
    vocab = pack / 'multilingual_vocab.txt'
    write_identical(vocab, (Path.home() / '.gigastt/models/multilingual_vocab.txt').read_bytes())
    manifest = pack / 'manifest.toml'
    write_identical(manifest, b'architecture = "ml_ctc"\n[files]\nencoder = "lab_multilingual_ctc.int8.onnx"\nencoder_int8 = "lab_multilingual_ctc.int8.onnx"\nvocab = "multilingual_vocab.txt"\n')
    info = {'candidate': SELECTED, 'model_dir': str(pack), 'binary': str(binary),
            'binary_sha256': BINARY_SHA, 'encoder_sha256': digest(encoder),
            'vocab_sha256': digest(vocab), 'manifest_sha256': digest(manifest),
            'bootstrap_presence_companion_sha256': digest(companion),
            'bootstrap_presence_companion_loaded': False,
            'custom_manifest_policy': 'Supported custom basename; explicit external SHA verification; no production checksum code changed.',
            'pool_size': 2, 'encoder_intra_threads': 3, 'http_workers_per_server': 2,
            'script_sha256': digest(__file__)}
    atomic_json(REPORT / 'product_pack.json', info)
    return pack, info


def server_command(binary, pack, port):
    return [str(binary), 'serve', '--offline', '--host', '127.0.0.1', '--port', str(port),
            '--model-dir', str(pack), '--model-variant', 'ml_ctc', '--execution-provider', 'cpu', '--pool-size', '2',
            '--encoder-intra-threads', '3', '--punctuation', 'off', '--itn', 'off']


def get_json(url):
    with urllib.request.urlopen(url, timeout=5) as response:
        return json.load(response)


@contextlib.contextmanager
def server(binary, pack, port, label):
    # Fail before launching if the port belongs to an unrelated process.
    with socket.socket() as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        probe.bind(('127.0.0.1', port))
    env = {k: v for k, v in os.environ.items() if not k.startswith('GIGASTT_')}
    command = server_command(binary, pack, port)
    url = f'http://127.0.0.1:{port}'
    with (WORK / (label + '_server.log')).open('a') as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env)
        try:
            deadline = time.monotonic() + 180
            while True:
                if process.poll() is not None:
                    raise RuntimeError('Server startup failed; inspect ' + label + '_server.log')
                try:
                    health = get_json(url + '/health')
                    models = get_json(url + '/v1/models')
                    break
                except (urllib.error.URLError, TimeoutError):
                    if time.monotonic() >= deadline:
                        raise TimeoutError('Server health deadline exceeded')
                    time.sleep(0.5)
            if health.get('variant') != 'ml_ctc' or health.get('punctuation') is not False or health.get('itn') is not False:
                raise ValueError('Incorrect server configuration')
            if models.get('pool_size') != 2 or models.get('execution_provider') != 'cpu':
                raise ValueError('Expected pool2 CPU execution: ' + json.dumps(models))
            atomic_json(WORK / (label + '_server.json'), {'pid': process.pid, 'command': command,
                        'health': health, 'model_info': models, 'binary_sha256': digest(binary)})
            yield url
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()


def parity(binary, pack, info):
    prepared = Path.home() / '.cache/gigastt-small-diagnosis/inputs/prepared.json'
    rows = [r for r in json.loads(prepared.read_text())['rows'] if r['variant'] == 'decoded_float']
    native_path = WORK / SELECTED / 'screen/private_results.json'
    native = {(r['id'], r['condition']): r for r in json.loads(native_path.read_text())['rows']}
    if len(rows) != 111 or len(native) != 111:
        raise ValueError('Expected exactly111 fixed diagnostic inputs')
    output = []
    with server(binary, pack, 19921, 'parity') as url:
        def request(row):
            # Feed original uploaded source bytes; the feature probe used the
            # exact product frontend arrays already verified for these files.
            if digest(row['path']) != row['audio_sha256']:
                raise ValueError('Source audio changed')
            body = Path(row['path']).read_bytes()
            req = urllib.request.Request(url + '/v1/transcribe', data=body,
                                         headers={'Content-Type': 'application/octet-stream'}, method='POST')
            with urllib.request.urlopen(req, timeout=300) as response:
                actual = json.load(response)['text']
            expected = native[(row['id'], row['condition'])]['hypothesis']
            return {'id': row['id'], 'condition': row['condition'], 'audio_sha256': row['audio_sha256'],
                    'feature_sha256': row['feature_sha256'], 'exact_text_match': actual.strip() == expected.strip(),
                    'word_scores_match': score(row['reference'], actual) == score(row['reference'], expected)}
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            for result in pool.map(request, rows):
                output.append(result)
                if len(output) % 20 == 0:
                    print('HTTP parity', len(output), '/111', flush=True)
    result = {'metadata': info, 'prepared_sha256': digest(prepared), 'native_private_sha256': digest(native_path),
              'n': len(output), 'exact_matches': sum(r['exact_text_match'] for r in output),
              'score_matches': sum(r['word_scores_match'] for r in output), 'rows': output}
    result['passed'] = result['exact_matches'] == result['score_matches'] == 111
    atomic_json(REPORT / 'product_http_parity.json', result)
    if not result['passed']:
        raise ValueError('HTTP/native parity failed; full evaluation forbidden')
    print('HTTP/native parity111/111 PASS', flush=True)


def full(binary, pack, info):
    checked = json.loads((REPORT / 'product_http_parity.json').read_text())
    if not checked['passed'] or checked['metadata']['encoder_sha256'] != info['encoder_sha256'] or checked['metadata']['binary_sha256'] != BINARY_SHA:
        raise ValueError('Matching native HTTP parity is required')
    destination = REPORT / 'full'
    destination.mkdir(exist_ok=True)

    def condition_run(condition, port):
        with server(binary, pack, port, 'full_' + condition) as url:
            for language in ['ru', 'en', 'kk', 'ky', 'uz']:
                manifest = (Path('benchmark/results/multilingual_1000_20261001') / (language + '_manifest.json')
                            if condition == 'original' else
                            Path('benchmark/results/multilingual_telephony_20261001') / (language + '_' + condition + '_manifest.json'))
                label = language + '_' + condition
                command = [sys.executable, 'scripts/benchmark_multilingual_1000.py', '--manifest', str(manifest),
                           '--output', str(destination / (label + '.json')), '--binary', str(binary),
                           '--variant', 'ml_ctc', '--url', url, '--workers', '2', '--timeout', '300', '--fsync']
                for path in [pack / 'lab_multilingual_ctc.int8.onnx', pack / 'multilingual_vocab.txt', pack / 'manifest.toml']:
                    command.extend(['--model', str(path)])
                with (WORK / (label + '_full.log')).open('a') as log:
                    run = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
                if run.returncode not in (0, 1) or not (destination / (label + '.json')).is_file():
                    raise RuntimeError('Runner did not complete: ' + label)
                data = json.loads((destination / (label + '.json')).read_text())
                if not data['completed'] or data['n'] != 1000:
                    raise ValueError('Incomplete group: ' + label)
                print('Full completed', label, 'n=1000 failures=' + str(data['failed_requests']), flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        futures = [executor.submit(condition_run, condition, port) for condition, port in
                   [('original', 19921), ('alaw', 19922), ('mulaw', 19923)]]
        for future in futures:
            future.result()
    print('Full15000 native requests completed', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['prepare', 'parity', 'full', 'parity-and-full'], required=True)
    parser.add_argument('--binary', type=Path, default=Path.home() / '.cache/gigastt-multilingual-1000/gigastt-candidate')
    args = parser.parse_args()
    pack, info = prepare_pack(args.binary)
    if args.mode in ('parity', 'parity-and-full'):
        parity(args.binary, pack, info)
    if args.mode in ('full', 'parity-and-full'):
        full(args.binary, pack, info)


if __name__ == '__main__':
    main()
