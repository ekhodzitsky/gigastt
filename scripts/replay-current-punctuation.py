#!/usr/bin/env python3
"""Re-evaluate native punctuation on verified active RNN-T hypotheses.

Run after the serial ASR timing/accuracy phases. Composite RTF is a sum of
separately measured components on the same 40 files, not an integrated-server run.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import statistics
import subprocess


def load(path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    for name in ['probe', 'small', 'big']:
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    metrics = load(root / 'scripts/score-current-asr.py')
    previous = load(root / 'scripts/compare-punctuation.py')
    cases = json.loads((args.directory / 'cases.json').read_text())
    timing_cases = json.loads((args.directory / 'timing-cases.json').read_text())
    gold = [r for r in cases if r['dataset'] == 'fleurs_ru_punct']
    ids = {r['id'] for r in gold}
    output = args.directory / 'punctuation'
    output.mkdir(exist_ok=False)
    environment = dict(os.environ, RAYON_NUM_THREADS='1')
    summary = dict(cases=len(gold), cpu_affinity='0-3', rayon_threads=1,
                   probe_sha256=hashlib.sha256(args.probe.read_bytes()).hexdigest(), paths={}, composites={})
    profiles = [('onnx', 'onnx', args.small), ('small', 'native', args.small), ('big', 'native', args.big)]

    def run(mode, backend, model, inputs, destination, *extra):
        command = ['taskset', '-c', '0-3', str(args.probe), mode, backend, str(model), str(inputs), str(destination), *extra]
        print('RUN', destination.name, flush=True)
        subprocess.run(command, env=environment, check=True)
        return json.loads(destination.read_text())

    def check(report, inputs):
        original = {r['id']: r['input'] for r in inputs}
        rows = report['rows']
        # The punctuation probe itself also rejects any failed restore/window.
        result = metrics.punctuation.score(gold, rows)
        result['lexical_mutations'] = sum(''.join(c.lower() for c in r['text'] if c.isalnum()) !=
                                          ''.join(c.lower() for c in original[r['id']] if c.isalnum()) for r in rows)
        result['punctuation_seconds'] = sum(r['seconds'] for r in rows)
        result['punctuation_peak_rss_kib'] = report['active']['peak_rss_kib']
        return result

    for head in ['rnnt', 'e2e_rnnt']:
        _, rows, _ = metrics.read_report(args.directory / 'runs' / f'{head}-quality.jsonl', cases)
        frozen = [r for r in rows if r['id'] in ids]
        summary['paths'][head + '-raw'] = metrics.punctuation.score(gold, frozen)
        for mode in (['product'] if head == 'rnnt' else ['direct', 'clean']):
            inputs = [dict(id=r['id'], input=r['itn_text'] if head == 'rnnt' else
                           previous.asr_text(r['text']) if mode == 'clean' else r['text']) for r in frozen]
            source = output / f'{head}-{mode}-inputs.json'
            source.write_text(json.dumps(inputs, ensure_ascii=False, indent=2))
            for label, backend, model in profiles:
                if head == 'e2e_rnnt' and mode == 'direct' and label == 'onnx':continue
                key = f'{head}-{mode}-{label}'
                report = run('quality', backend, model, source, output / (key + '.json'))
                summary['paths'][key] = check(report, inputs)
        if head != 'rnnt':continue
        # Actual serial ASR timing hypotheses, including the default product ITN.
        _, timed, _ = metrics.read_report(args.directory / 'runs' / 'rnnt-timing-1.jsonl', timing_cases)
        source = output / 'rnnt-timing-inputs.json'
        source.write_text(json.dumps([dict(id=r['id'], input=r['itn_text']) for r in timed], ensure_ascii=False, indent=2))
        duration = sum(r['duration'] for r in timing_cases)
        for label, backend, model in profiles:
            report = run('timing', backend, model, source, output / f'rnnt-timing-{label}.json', '1')
            round_times, rtfs = [], []
            for i in range(1,7):
                post = sum(r['rounds'][i]['request_seconds'][0] for r in report['rows'])
                _, raw, _ = metrics.read_report(args.directory / 'runs' / f'rnnt-timing-{i}.jsonl', timing_cases)
                total = sum(r['seconds'] for r in raw) + post
                round_times.append(post)
                rtfs.append(total/duration)
            summary['composites'][label] = dict(rounds=list(range(1,7)), punctuation_seconds=round_times,
                                                reconstructed_rtf_median=statistics.median(rtfs),
                                                reconstructed_rtf_range=[min(rtfs),max(rtfs)])
    (output / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':main()
