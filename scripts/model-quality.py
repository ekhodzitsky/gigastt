#!/usr/bin/env python3
"""Compare two model packs on the frozen long-form corpus using one native binary.

Preserves transcripts, cache-cold/cache-warm timings and peak RSS. An alignment
of reference words measures consecutive deletions separately from aggregate
WER. This is not the timestamp-only lost-run metric from private recordings.
"""
import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import shutil
import statistics
import subprocess
import sys

spec = importlib.util.spec_from_file_location('longform', Path(__file__).with_name('longform_stitch.py'))
longform = importlib.util.module_from_spec(spec)
spec.loader.exec_module(longform)


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def score(reference, text):
    """Minimum word edit distance, with deterministic S/D/I tie breaking."""
    ref, hyp = longform.normalize(reference), longform.normalize(text)
    if not ref:
        raise ValueError('empty reference')
    row = list(range(len(hyp) + 1))
    trace = [bytearray([2] * (len(hyp) + 1))]
    for i, word in enumerate(ref, 1):
        nxt, steps = [i], bytearray([1])
        for j, other in enumerate(hyp, 1):
            choices = (row[j - 1] + (word != other), row[j] + 1, nxt[-1] + 1)
            step = min(range(3), key=choices.__getitem__)
            nxt.append(choices[step])
            steps.append(step)
        row = nxt
        trace.append(steps)
    i, j, deleted, runs = len(ref), len(hyp), 0, []
    while i or j:
        step = 2 if i == 0 else 1 if j == 0 else trace[i][j]
        if step == 1:
            deleted += 1
            i -= 1
        else:
            if deleted >= 3:
                runs.append(deleted)
            deleted = 0
            if step == 0:
                i -= 1
            j -= 1
    if deleted >= 3:
        runs.append(deleted)
    return {'reference_words': len(ref), 'edits': row[-1],
            'deletion_runs': len(runs), 'words_in_deletion_runs': sum(runs)}


def compare(report):
    """Reject missing measurements or any quality/resource regression."""
    if report.get('schema') != 1 or len(report.get('cases', [])) != 7:
        raise ValueError('expected all seven frozen long-form cases')
    failures = []
    limits = {'size': 1.10, 'rss': 1.10}
    release = json.loads((Path(__file__).resolve().parents[1] / 'benchmark/model-release.json').read_text())
    pair = release['resource_transition']['encoder_pairs'].get(report.get('head'))
    if pair and all(report['models'][arm]['files'][f"v3_{report['head']}_encoder_int8.onnx"] == pair[arm]
                    for arm in ('baseline', 'candidate')):
        limits = {'size': release['resource_transition']['max_encoder_ratio'],
                  'rss': release['resource_transition']['max_peak_rss_ratio']}
    totals = {'baseline': [0, 0, 0], 'candidate': [0, 0, 0]}
    ids = set()
    for case in report['cases']:
        if case['id'] in ids:
            raise ValueError('duplicate case')
        ids.add(case['id'])
        base, candidate = case['baseline'], case['candidate']
        for arm in (base, candidate):
            if len(arm) != 3:
                raise ValueError('expected one cold and two warm measurements')
            for sample in arm:
                for key in ('wall_seconds', 'peak_rss_kib', 'reference_words'):
                    value = sample[key]
                    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                        raise ValueError(f'invalid {key}')
                for key in ('edits', 'deletion_runs', 'words_in_deletion_runs'):
                    if type(sample[key]) is not int or sample[key] < 0:
                        raise ValueError(f'invalid {key}')
        for b, c in zip(base, candidate):
            if b['reference_words'] != c['reference_words']:
                raise ValueError('reference mismatch')
            for key in ('deletion_runs', 'words_in_deletion_runs'):
                if c[key] > b[key]:
                    failures.append(f"{case['id']}: {key} {b[key]} -> {c[key]}")
        for arm in ('baseline', 'candidate'):
            for index, sample in enumerate(case[arm]):
                totals[arm][index] += sample['edits']
        for key, limit in [('wall_seconds', 1.05), ('peak_rss_kib', limits['rss'])]:
            ratio = statistics.median(c[key] / b[key] for b, c in zip(base[1:], candidate[1:]))
            if ratio > limit:
                failures.append(f"{case['id']}: warm {key} {ratio:.3f}x > {limit}x")
            cold_limit = 1.10 if key == 'wall_seconds' else limits['rss']
            if candidate[0][key] / base[0][key] > cold_limit:
                failures.append(f"{case['id']}: cold {key} exceeds {cold_limit}x")
    if any(c > b for b, c in zip(totals['baseline'], totals['candidate'])):
        failures.append('aggregate word error rate increased')
    for head in ('baseline', 'candidate'):
        if report['models'][head]['encoder_bytes'] <= 0:
            raise ValueError('invalid encoder size')
    if report['models']['candidate']['encoder_bytes'] > limits['size'] * report['models']['baseline']['encoder_bytes']:
        failures.append(f"encoder size exceeds {limits['size']}x")
    return failures


def run(args):
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    binary = args.binary.resolve()
    manifest = json.loads(longform.MANIFEST.read_text())
    encoder = f'v3_{args.head}_encoder_int8.onnx'
    names = [encoder, f'v3_{args.head}_decoder.onnx', f'v3_{args.head}_joint.onnx',
             'v3_vocab.txt' if args.head == 'rnnt' else 'v3_e2e_rnnt_vocab.txt']
    models = {}
    for label in ('baseline', 'candidate'):
        directory = getattr(args, label).resolve()
        models[label] = {'files': {name: digest(directory / name) for name in names},
                         'encoder_bytes': (directory / encoder).stat().st_size}
    if any(models['baseline']['files'][name] != models['candidate']['files'][name] for name in names[1:]):
        raise ValueError('decoder, joiner and vocabulary must match')
    report = {'schema': 1, 'head': args.head, 'binary_sha256': digest(binary),
              'manifest_sha256': digest(longform.MANIFEST), 'models': models,
              'platform': platform.platform(), 'cpu': Path('/proc/cpuinfo').read_text(),
              'encoder_threads': 6, 'cases': []}
    environment = {k: v for k, v in os.environ.items() if not k.startswith('GIGASTT_')}
    environment['RUST_LOG'] = 'warn'
    for sample in manifest['samples']:
        audio = args.audio / sample['file']
        longform.checked_audio(audio, sample['sha256'])
        case = {'id': sample['id'], 'audio_sha256': sample['sha256'], 'baseline': [], 'candidate': []}
        report['cases'].append(case)
        for repeat in range(3):
            labels = ('baseline', 'candidate') if repeat % 2 == 0 else ('candidate', 'baseline')
            for label in labels:
                stem = output / f"{sample['id']}-{label}-{repeat}"
                # A separate cache per case makes repeat zero cold for both arms.
                pack = output / f"pack-{sample['id']}-{label}"
                pack.mkdir(exist_ok=True)
                for name in names:
                    target = pack / name
                    if not target.exists():
                        target.symlink_to((getattr(args, label) / name).resolve())
                command = ['/usr/bin/time', '-f', '%e %M', '-o', str(stem.with_suffix('.time')),
                           str(binary), 'transcribe', str(audio.resolve()), '--offline',
                           '--model-dir', str(pack), '--model-variant', args.head,
                           '--punctuation', 'off', '--itn', 'off',
                           '--encoder-intra-threads', '6', '--file-window-concurrency', '1',
                           '--format', 'json', '--output', str(stem.with_suffix('.json'))]
                with stem.with_suffix('.log').open('w') as log:
                    subprocess.run(command, env=environment, stdout=log, stderr=log, timeout=300, check=True)
                result = json.loads(stem.with_suffix('.json').read_text())
                wall, rss = stem.with_suffix('.time').read_text().split()
                case[label].append({**score(sample['reference'], result['text']),
                                    'wall_seconds': float(wall), 'peak_rss_kib': int(rss),
                                    'transcript_sha256': digest(stem.with_suffix('.json'))})
                (output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
                print(sample['id'], label, repeat, case[label][-1], flush=True)
        # These are generated caches and model symlinks, not the source packs.
        # Reclaim them between cases to keep release runners within disk limits.
        for label in ('baseline', 'candidate'):
            shutil.rmtree(output / f"pack-{sample['id']}-{label}")
    report['failures'] = compare(report)
    (output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    return bool(report['failures'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', required=True, type=Path)
    parser.add_argument('--baseline', required=True, type=Path)
    parser.add_argument('--candidate', required=True, type=Path)
    parser.add_argument('--audio', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--head', choices=['rnnt', 'e2e_rnnt'], required=True)
    return run(parser.parse_args())


if __name__ == '__main__':
    sys.exit(main())
