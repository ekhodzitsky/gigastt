#!/usr/bin/env python3
"""Prepare frozen FLEURS inputs and score matched punctuation probe outputs.

Never downloads models or dispatches CI. Restoration input uses lowercase text
with sentence punctuation removed, retaining internal hyphens and decimal dots.
Punctuation/case is scored at alphanumeric character boundaries, so adding marks
or changing whitespace cannot shift all subsequent evaluation positions.
"""
import argparse
from collections import Counter
import hashlib
import json
import math
import os
import resource
import subprocess
import time
import wave
from pathlib import Path

MARKS = set('.,!?;:—…«»"()')
DROP = set('.,?!:;«»"()[]—…')


def asr_text(text):
    output = []
    for i, c in enumerate(text):
        if c == '.' and not (i and i + 1 < len(text) and text[i-1].isdigit() and text[i+1].isdigit()):
            continue
        if c in DROP and c != '.':
            continue
        output.append(c.lower())
    return ' '.join(''.join(output).split())


def events(text):
    letters, punctuation, capitals = [], Counter(), Counter()
    for c in text:
        if c.isalnum():
            if c.isalpha() and c.isupper():
                capitals[(len(letters), c.lower())] += 1
            letters.append(c.lower())
        elif c in MARKS:
            punctuation[(len(letters), c)] += 1
    return ''.join(letters), punctuation, capitals


def score_counts(reference, hypothesis):
    tp = sum((reference & hypothesis).values())
    return {'tp': tp, 'fp': sum(hypothesis.values()) - tp, 'fn': sum(reference.values()) - tp}


def aggregate(rows):
    totals = {k: sum(r[k] for r in rows) for k in ['tp', 'fp', 'fn']}
    tp, fp, fn = (totals[k] for k in ['tp', 'fp', 'fn'])
    totals['precision'] = tp / (tp + fp) if tp + fp else 0
    totals['recall'] = tp / (tp + fn) if tp + fn else 0
    totals['f1'] = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0
    return totals


def score(cases, report):
    hypotheses = {r['id']: r['text'] for r in report['rows']}
    if len(report['rows']) != len(cases) or len(hypotheses) != len(cases) or set(hypotheses) != {r['id'] for r in cases}:
        raise ValueError('missing, duplicate or unexpected outputs')
    punctuation, casing, failures, exact = [], [], [], 0
    for case in cases:
        hyp = hypotheses[case['id']]
        ref_words, ref_p, ref_c = events(case['reference'])
        hyp_words, hyp_p, hyp_c = events(hyp)
        if ref_words != hyp_words:
            failures.append(case['id'])
            # Never silently discard lexical mutations or score shifted anchors.
            punctuation.append({'tp': 0, 'fp': sum(hyp_p.values()), 'fn': sum(ref_p.values())})
            casing.append({'tp': 0, 'fp': sum(hyp_c.values()), 'fn': sum(ref_c.values())})
        else:
            punctuation.append(score_counts(ref_p, hyp_p))
            casing.append(score_counts(ref_c, hyp_c))
        exact += ' '.join(case['reference'].split()) == ' '.join(hyp.split())
    return {'count': len(cases), 'exact': exact, 'exact_fraction': exact / len(cases),
            'punctuation': aggregate(punctuation), 'casing': aggregate(casing),
            'lexical_failures': failures}


def prepare(root, destination):
    destination.mkdir(parents=True, exist_ok=True)
    manifest = root / 'benchmark/manifests/fleurs_ru_punct.json'
    samples = json.loads(manifest.read_text())['samples']
    cases = [{'id': row['filename'], 'reference': row['reference'], 'input': asr_text(row['reference'])} for row in samples]
    (destination / 'quality-inputs.json').write_text(json.dumps(cases, ensure_ascii=False, indent=2))
    joined = ' '.join(r['input'] for r in cases).split()
    timing = [{'id': f'words-{n}', 'input': ' '.join(joined[:n])} for n in [5, 30, 250, 1000]]
    (destination / 'timing-inputs.json').write_text(json.dumps(timing, ensure_ascii=False, indent=2))
    references = json.loads((root / 'crates/gigastt/tests/fixtures/manifest.json').read_text())
    asr = [{'id': row['filename'], 'audio': str(root / 'crates/gigastt/tests/fixtures' / row['filename'])} for row in references]
    (destination / 'asr-inputs.json').write_text(json.dumps(asr, indent=2))
    combined = destination / 'golos-concatenated.wav'
    with wave.open(str(combined), 'wb') as output:
        for index, row in enumerate(asr):
            with wave.open(row['audio'], 'rb') as source:
                if (source.getnchannels(), source.getsampwidth(), source.getframerate()) != (1, 2, 16000):
                    raise ValueError('ASR concatenation requires mono PCM16 at 16 kHz')
                if index == 0:
                    output.setparams(source.getparams())
                output.writeframes(source.readframes(source.getnframes()))
    (destination / 'long-asr-inputs.json').write_text(json.dumps([
        {'id': 'golos-concatenated', 'audio': str(combined.resolve())}]))

    (destination / 'input-protocol.json').write_text(json.dumps({
        'source': 'google/fleurs ru_ru test, CC BY 4.0, Conneau et al. (2022)',
        'manifest_sha256': hashlib.sha256(manifest.read_bytes()).hexdigest(),
        'cases': len(cases), 'normalization': 'asr_text in scripts/compare-punctuation.py',
        'scoring': 'punctuation character events/capital letters at alphanumeric boundaries; lexical changes conservatively fail the entire row',
    }, indent=2))


def measure(probe, small, big, destination, cpus, concurrency_values):
    profiles = [('onnx-small', 'onnx', small, '1'),
                ('native-small-one', 'native', small, '1'),
                ('native-big-one', 'native', big, '1'),
                ('native-small-auto', 'native', small, None),
                ('native-big-auto', 'native', big, None)]
    process_report = destination / 'timing-processes.json'
    runs = json.loads(process_report.read_text()) if process_report.exists() else {}
    for concurrency in concurrency_values:
        order = profiles if concurrency == 1 else [profiles[0], *reversed(profiles[1:])]
        for name, backend, model, rayon in [*order, profiles[0]]:
            key = f'{name}-c{concurrency}'
            if key in runs:
                key += '-repeat'
            environment = dict(os.environ)
            environment.pop('RAYON_NUM_THREADS', None)
            if rayon:
                environment['RAYON_NUM_THREADS'] = rayon
            output = destination / (key + '.json')
            command = ['taskset', '-c', cpus, str(probe.resolve()), 'timing', backend,
                       str(model.resolve()), str(destination / 'timing-inputs.json'), str(output), str(concurrency)]
            before = resource.getrusage(resource.RUSAGE_CHILDREN)
            start = time.monotonic()
            process = subprocess.Popen(command, env=environment)
            peak_threads = 0
            while process.poll() is None:
                try:
                    status = Path(f'/proc/{process.pid}/status').read_text()
                    peak_threads = max(peak_threads, next(int(line.split()[1]) for line in status.splitlines() if line.startswith('Threads:')))
                except (OSError, StopIteration):
                    pass
                time.sleep(0.01)
            if process.returncode:
                raise RuntimeError(f'{key}: probe exit {process.returncode}')
            elapsed = time.monotonic() - start
            after = resource.getrusage(resource.RUSAGE_CHILDREN)
            report = json.loads(output.read_text())
            if len(report['rows']) != 4 or any(len(row['rounds']) != 7 for row in report['rows']):
                raise ValueError('incomplete timing report')
            for row in report['rows']:
                for round_ in row['rounds']:
                    values = [round_['wall_seconds'], *round_['request_seconds']]
                    if len(round_['request_seconds']) != concurrency or any(not math.isfinite(x) or x <= 0 for x in values):
                        raise ValueError('invalid timing report')
            runs[key] = {'wall_seconds': elapsed, 'cpu_seconds': after.ru_utime + after.ru_stime - before.ru_utime - before.ru_stime,
                         'observed_peak_threads': peak_threads, 'rayon_threads': rayon or 'auto', 'report': output.name}
            print(key, round(elapsed, 2), 's', flush=True)
            (destination / 'timing-processes.json').write_text(json.dumps(runs, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare', type=Path)
    parser.add_argument('--measure', type=Path, help='prepared input/output directory')
    parser.add_argument('--probe', type=Path)
    parser.add_argument('--small', type=Path)
    parser.add_argument('--big', type=Path)
    parser.add_argument('--cpus', default='0-3', help='equal-class physical CPUs for Linux taskset')
    parser.add_argument('--concurrency', nargs='+', type=int, choices=[1, 2, 4], default=[1, 2, 4])
    parser.add_argument('--score', nargs=2, metavar=('INPUTS', 'OUTPUTS'))
    args = parser.parse_args()
    if args.prepare:
        prepare(Path(__file__).resolve().parents[1], args.prepare)
    if args.measure:
        if not all([args.probe, args.small, args.big]):
            parser.error('--measure requires --probe, --small and --big')
        measure(args.probe, args.small, args.big, args.measure.resolve(), args.cpus, args.concurrency)
    if args.score:
        print(json.dumps(score(*[json.loads(Path(p).read_text()) for p in args.score]), ensure_ascii=False, indent=2))
    if not args.prepare and not args.score and not args.measure:
        parser.error('specify --prepare or --score')


if __name__ == '__main__':
    main()
