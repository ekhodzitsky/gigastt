#!/usr/bin/env python3
"""Summarize the reports emitted by compare-punctuation-heads.py."""
import importlib.util
import json
from pathlib import Path
import statistics
import sys

spec = importlib.util.spec_from_file_location('metrics', Path(__file__).with_name('punctuation-heads.py'))
metrics = importlib.util.module_from_spec(spec)
spec.loader.exec_module(metrics)


def summarize(root):
    cases = json.loads((root / 'cases.json').read_text())
    duration = sum(r['duration'] for r in cases)
    summary = {'cases': len(cases), 'audio_seconds': duration, 'paths': {}, 'heads': {}}
    for head in ['rnnt', 'e2e_rnnt']:
        records = [json.loads(line) for line in (root / (head + '.jsonl')).read_text().splitlines()]
        metadata, rows, memory = records[0], records[1:-1], records[-1]['memory']
        if len(rows) != len(cases):
            raise ValueError('incomplete decode')
        normalized = metrics.aggregate([metrics.score_pair(r['reference_itn'], r['itn_text']) for r in rows])
        seconds = sum(r['seconds'] for r in rows)
        summary['heads'][head] = dict(metadata, raw_wer=metrics.score(cases, rows)['wer'],
                                     common_itn_wer=normalized['wer'], seconds=seconds, rtf=seconds/duration,
                                     memory={line.split(':')[0]: line.split(':')[1].strip() for line in memory.splitlines()
                                             if line.startswith(('VmRSS:', 'VmHWM:', 'Threads:'))})
        subset = {r['id'] for r in cases if not any(c.isdigit() for c in r['reference'])}
        summary['heads'][head]['references_without_digits'] = {
            'cases': len(subset),
            'raw': metrics.score([r for r in cases if r['id'] in subset], [r for r in rows if r['id'] in subset])['wer'],
            'common_itn': metrics.aggregate([metrics.score_pair(r['reference_itn'], r['itn_text']) for r in rows if r['id'] in subset])['wer']}
        if head == 'e2e_rnnt':
            summary['paths']['e2e-built-in'] = dict(metrics.score(cases, rows), punctuation_seconds=0, total_rtf=seconds/duration)
        modes = ['product'] if head == 'rnnt' else ['direct', 'clean']
        for mode in modes:
            inputs = {r['id']: r['input'] for r in json.loads((root / f'{head}-{mode}-inputs.json').read_text())}
            for label in ['onnx', 'small', 'big']:
                path = root / f'{head}-{mode}-{label}.json'
                if not path.exists():
                    if head == 'e2e_rnnt' and mode == 'direct' and label == 'onnx':
                        continue
                    raise ValueError(f'missing {path}')
                report = json.loads(path.read_text())
                result = metrics.score(cases, report['rows'])
                times = [r['seconds'] for r in report['rows']]
                result.update(punctuation_seconds=sum(times), punctuation_median_ms=1000*statistics.median(times),
                              total_rtf=(seconds+sum(times))/duration, load_seconds=report['load_seconds'],
                              punctuation_memory=report['active'], lexical_mutations=sum(
                                  ''.join(c.lower() for c in r['text'] if c.isalnum()) !=
                                  ''.join(c.lower() for c in inputs[r['id']] if c.isalnum()) for r in report['rows']))
                summary['paths'][f'{head}-{mode}-{label}'] = result
    previous_spec = importlib.util.spec_from_file_location('previous', Path(__file__).with_name('compare-punctuation.py'))
    previous = importlib.util.module_from_spec(previous_spec)
    previous_spec.loader.exec_module(previous)
    legacy = Path(__file__).resolve().parents[1] / 'docs/measurements/rupunct-20261022'
    gold_cases = [dict(r, input=previous.asr_text(r['reference'])) for r in cases]
    summary['gold_input_strata'] = {}
    for name, has_hyphen in [('hyphen', True), ('no_hyphen', False)]:
        subset = [r for r in gold_cases if ('-' in r['input']) == has_hyphen]
        ids = {r['id'] for r in subset}
        scores = {'cases': len(subset)}
        for label in ['onnx-small', 'native-small']:
            rows = [json.loads(line) for line in (legacy / (label + '-quality.jsonl')).read_text().splitlines()]
            scores[label] = previous.score(subset, {'rows': [r for r in rows if r['id'] in ids]})
        summary['gold_input_strata'][name] = scores
    return summary


if __name__ == '__main__':
    print(json.dumps(summarize(Path(sys.argv[1])), ensure_ascii=False, indent=2))
