#!/usr/bin/env python3
"""Measure both model heads on the frozen Golos slice, retaining all transcripts."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import wave

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'benchmark'))
from common import compute_wer, compute_wer_naive


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', required=True, type=Path)
    parser.add_argument('--models', required=True, type=Path, help='Root with baseline/ and candidate/ packs')
    parser.add_argument('--audio', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    manifest_path = ROOT / 'benchmark/manifests/golos_crowd_1k.json'
    manifest = json.loads(manifest_path.read_text())
    samples = [s for s in manifest['samples'] if s['reference'].strip()]
    output, binary = args.output.resolve(), args.binary.resolve()
    output.mkdir(parents=True, exist_ok=False)
    inputs = output / 'input'
    inputs.mkdir()
    report = dict(schema=1, wer_unit='fraction', binary_sha256=digest(binary),
                  manifest_sha256=digest(manifest_path), platform=platform.platform(),
                  measurement_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                  audio_sha256={}, audio_seconds=0, samples=len(samples), nominal_samples=len(manifest['samples']),
                  skipped_empty_references=len(manifest['samples']) - len(samples),
                  encoder_threads=6, pool_size=1, models={}, results={})
    for sample in samples:
        source = args.audio / sample['filename']
        report['audio_sha256'][sample['filename']] = digest(source)
        try:
            os.link(source, inputs / source.name)
        except OSError:
            shutil.copyfile(source, inputs / source.name)
        with wave.open(str(source)) as audio:
            report['audio_seconds'] += audio.getnframes() / audio.getframerate()
    environment = {k: v for k, v in os.environ.items() if not k.startswith('GIGASTT_')}
    environment['RUST_LOG'] = 'warn'
    for head in ('rnnt', 'e2e_rnnt'):
        report['models'][head], report['results'][head] = {}, {}
        encoder = f'v3_{head}_encoder_int8.onnx'
        names = [encoder, f'v3_{head}_decoder.onnx', f'v3_{head}_joint.onnx',
                 'v3_vocab.txt' if head == 'rnnt' else 'v3_e2e_rnnt_vocab.txt']
        for arm in ('baseline', 'candidate'):
            pack = args.models.resolve() / arm / head
            report['models'][head][arm] = dict(files={name: digest(pack / name) for name in names},
                                               encoder_bytes=(pack / encoder).stat().st_size)
            stem = output / f'{head}-{arm}'
            stem.mkdir()
            command = ['/usr/bin/time', '-f', '%e %M', '-o', str(stem.with_suffix('.time')),
                       str(binary), 'transcribe-batch', str(inputs), str(stem), '--model-dir', str(pack),
                       '--model-variant', head, '--offline', '--punctuation', 'off', '--itn', 'off',
                       '--pool-size', '1', '--encoder-intra-threads', '6', '--format', 'json']
            with stem.with_suffix('.log').open('w') as log:
                subprocess.run(command, env=environment, stdout=log, stderr=log, timeout=1800, check=True)
            wall, rss = stem.with_suffix('.time').read_text().split()
            result = dict(wall_seconds=float(wall), peak_rss_kib=int(rss), details=[])
            for sample in samples:
                text = json.loads((stem / Path(sample['filename']).with_suffix('.json')).read_text())['text']
                row = dict(file=sample['filename'], text=text)
                for prefix, scorer in [('', compute_wer), ('raw_', compute_wer_naive)]:
                    _, row[prefix + 'errors'], row[prefix + 'reference_words'] = scorer(sample['reference'], text)
                result['details'].append(row)
            for prefix in ('', 'raw_'):
                for key in ('errors', 'reference_words'):
                    result[prefix + key] = sum(row[prefix + key] for row in result['details'])
                result[prefix + 'wer'] = result[prefix + 'errors'] / result[prefix + 'reference_words']
            report['results'][head][arm] = result
            (output / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
            print(head, arm, result['errors'], result['reference_words'], flush=True)


if __name__ == '__main__':
    main()
