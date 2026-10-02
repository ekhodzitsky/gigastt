#!/usr/bin/env python3
"""Measure an acquired conversational corpus on one frozen standard large model."""
import argparse
import json
import os
from pathlib import Path
import socket
import subprocess
import time
import urllib.error
import urllib.request
from benchmark_multilingual_1000 import atomic_json, run
from benchmark_multilingual_public import sha256

BINARY = Path.home() / '.cache/gigastt-multilingual-1000/gigastt-candidate'
BINARY_SHA = 'bb30ca980374af9eb6e9fae9d239c3ce34a85e69e202bc8f8bac6c1e7ae36566'
MODELS = Path.home() / '.gigastt/models'
MODEL_SHA = 'b2ad9c38fc04197ba758105d33f7404fd13d977958722e0f49e3f3e22521f1c6'
VOCAB_SHA = '4d130287892e1099fedfb3f93c4b4cf8a263151158801680b28977d1be4133f4'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--port', type=int, default=20021)
    args = parser.parse_args()
    encoder = MODELS / 'multilingual_large_ctc.int8.onnx'
    vocab = MODELS / 'multilingual_vocab.txt'
    if (sha256(BINARY), sha256(encoder), sha256(vocab)) != (BINARY_SHA, MODEL_SHA, VOCAB_SHA):
        raise ValueError('Frozen binary/model identity mismatch')
    samples = json.loads(args.manifest.read_text())['samples']
    if len({r['language'] for r in samples}) != 1:
        raise ValueError('One language corpus is required per run')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    command = [str(BINARY), '--offline', 'serve', '--model-dir', str(MODELS),
               '--model-variant', 'ml_ctc_large', '--execution-provider', 'cpu',
               '--pool-size', '2', '--encoder-intra-threads', '3', '--punctuation', 'off',
               '--itn', 'off', '--port', str(args.port)]
    protocol = {'manifest_sha256': sha256(args.manifest), 'n': len(samples),
                'binary_sha256': BINARY_SHA, 'encoder_sha256': MODEL_SHA, 'vocab_sha256': VOCAB_SHA,
                'command': command, 'conditions': 'Original decoded audio only; no VAD, ITN, punctuation or preprocessing interventions.',
                'scoring': 'Existing Unicode micro WER/CER; one fixed primary reference; no minimum over reference alternatives.',
                'failures': 'All failed requests and empty hypotheses retained; no retries or replacement recordings.',
                'script_sha256': sha256(Path(__file__))}
    protocol_path = args.output.with_suffix('.protocol.json')
    if protocol_path.exists() and json.loads(protocol_path.read_text()) != protocol:
        raise ValueError('Existing protocol differs')
    atomic_json(protocol_path, protocol)
    with socket.socket() as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        probe.bind(('127.0.0.1', args.port))
    url = f'http://127.0.0.1:{args.port}'
    env = {k:v for k,v in os.environ.items() if not k.startswith('GIGASTT_')}
    env['RUST_LOG'] = 'warn'
    with args.output.with_suffix('.server.log').open('a') as log:
        process = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT)
        try:
            deadline = time.monotonic() + 180
            while True:
                if process.poll() is not None:
                    raise RuntimeError('Server exited before readiness')
                try:
                    with urllib.request.urlopen(url + '/health', timeout=5) as r:
                        health = json.load(r)
                    with urllib.request.urlopen(url + '/v1/models', timeout=5) as r:
                        model_info = json.load(r)
                    break
                except (urllib.error.URLError, TimeoutError):
                    if time.monotonic() >= deadline:
                        raise TimeoutError('Server readiness timed out')
                    time.sleep(.2)
            if model_info['pool_size'] != 2 or model_info['execution_provider'] != 'cpu':
                raise ValueError('Wrong runtime configuration')
            status = run(argparse.Namespace(manifest=args.manifest, output=args.output, binary=BINARY,
                         variant='ml_ctc_large', url=url, workers=2, timeout=300, fsync=True,
                         model=[encoder, vocab]), expected_per_language=len(samples))
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
    raise SystemExit(status)


if __name__ == '__main__':
    main()
