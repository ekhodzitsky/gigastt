#!/usr/bin/env python3
"""Run both ASR heads once, then compare punctuation on their frozen outputs.

Uses prebuilt release examples; never downloads models or runs GitHub Actions.
Reruns refuse to overwrite evidence. See docs/punctuation-heads-evaluation.md.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import struct
import subprocess


def load_module(filename):
    spec = importlib.util.spec_from_file_location(filename.stem, filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def duration(path):
    data = path.read_bytes()
    offset, rate, size = 12, None, None
    while offset + 8 <= len(data):
        tag, length = struct.unpack_from('<4sI', data, offset)
        if tag == b'fmt ':
            rate = struct.unpack_from('<I', data, offset + 16)[0]
        if tag == b'data':
            size = length
        offset += 8 + length + length % 2
    if not rate or not size:
        raise ValueError(f'missing WAV format/data: {path}')
    return size / rate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['decode-probe', 'restore-probe', 'models', 'big', 'audio-root', 'output']:
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--cpus', default='0-3')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    previous = load_module(root / 'scripts/compare-punctuation.py')
    manifest = root / 'benchmark/manifests/fleurs_ru_punct.json'
    args.output.mkdir(parents=True, exist_ok=False)
    cases = []
    for row in json.loads(manifest.read_text())['samples']:
        audio = args.audio_root / row['filename']
        cases.append(dict(id=row['filename'], reference=row['reference'],
                          audio=str(audio.resolve()), duration=duration(audio)))
    inputs = args.output / 'cases.json'
    inputs.write_text(json.dumps(cases, ensure_ascii=False, indent=2))
    environment = dict(os.environ, RAYON_NUM_THREADS='1')
    protocol = {'manifest_sha256': hashlib.sha256(manifest.read_bytes()).hexdigest(),
                'cpus': args.cpus, 'rayon_threads': 1, 'commands': [],
                'executables': {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                                for p in [args.decode_probe, args.restore_probe]},
                'audio_sha256': {r['id']: hashlib.sha256(Path(r['audio']).read_bytes()).hexdigest() for r in cases}}

    def run(command):
        protocol['commands'].append(command)
        (args.output / 'protocol.json').write_text(json.dumps(protocol, indent=2))
        subprocess.run(['taskset', '-c', args.cpus, *command], env=environment, check=True)

    for head in ['rnnt', 'e2e_rnnt']:
        output = args.output / (head + '.jsonl')
        run([str(args.decode_probe), head, str(args.models), str(inputs), str(output)])
        rows = [json.loads(s) for s in output.read_text().splitlines()][1:-1]
        if len(rows) != len(cases):
            raise ValueError('incomplete ASR output')
        for mode in (['product'] if head == 'rnnt' else ['direct', 'clean']):
            texts = [{'id': r['id'], 'input': r['itn_text'] if head == 'rnnt' else
                      previous.asr_text(r['text']) if mode == 'clean' else r['text']} for r in rows]
            inputs_punct = args.output / f'{head}-{mode}-inputs.json'
            inputs_punct.write_text(json.dumps(texts, ensure_ascii=False, indent=2))
            for label, backend, model in [('onnx', 'onnx', args.models / 'punct'),
                                          ('small', 'native', args.models / 'punct'),
                                          ('big', 'native', args.big)]:
                if head == 'e2e_rnnt' and mode == 'direct' and label == 'onnx':
                    continue
                result = args.output / f'{head}-{mode}-{label}.json'
                print('restore', result.name, flush=True)
                run([str(args.restore_probe), 'quality', backend, str(model),
                     str(inputs_punct), str(result)])


if __name__ == '__main__':
    main()
