#!/usr/bin/env python3
"""Verify published hashes and rescore saved text; no audio/models/network needed."""
import hashlib
import json
from pathlib import Path

from rapidfuzz.distance import Levenshtein
from wer_unicode import normalize, word_edit_distance

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / 'benchmark/results'
READING = RESULTS / 'multilingual_1000_20261001'
CONVERSATION = RESULTS / 'multilingual_conversation_20261001'
COMPARISON = RESULTS / 'multilingual_model_comparison_20261001'
COUNTS = ('reference_words', 'word_errors', 'substitutions', 'deletions',
          'insertions', 'reference_chars', 'char_errors')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def totals(rows):
    values = {k: sum(r['scores'][k] for r in rows) for k in COUNTS}
    values.update(n=len(rows), empty_hypotheses=sum(not normalize(r['hypothesis']) for r in rows),
                  failed_requests=sum(r['status'] == 'failed' for r in rows))
    for metric, errors, references in [('wer_pct', 'word_errors', 'reference_words'),
                                       ('cer_pct', 'char_errors', 'reference_chars')]:
        values[metric] = 100 * values[errors] / values[references] if values[references] else None
    return values


def verify_result(result, manifest):
    rows = result['details']
    samples = {s['id']: s for s in manifest['samples']}
    require(result.get('completed') is True, 'Incomplete result')
    require(result['n'] == len(rows) == len(samples) == len(manifest['samples']), 'Wrong population')
    require(len({r['id'] for r in rows}) == len(rows), 'Duplicate result IDs')
    for row in rows:
        sample = samples.get(row['id'])
        require(sample is not None and all(row[k] == sample[k] for k in
                ('id', 'reference', 'sha256', 'language', 'dataset', 'split')), 'Changed input identity')
        require(row['status'] in ('ok', 'failed'), 'Unknown request status')
        require(row['status'] != 'failed' or row['hypothesis'] == '', 'Failed request contains text')
        ref, hyp = normalize(row['reference']), normalize(row['hypothesis'])
        ref_chars, hyp_chars = ' '.join(ref), ' '.join(hyp)
        saved = row['scores']
        expected = dict(reference_words=len(ref), word_errors=word_edit_distance(ref, hyp),
                        reference_chars=len(ref_chars), char_errors=Levenshtein.distance(ref_chars, hyp_chars))
        require(all(saved[k] == v for k, v in expected.items()), 'Saved edit distance differs')
        require(all(isinstance(saved[k], int) and saved[k] >= 0 for k in COUNTS), 'Invalid count')
        require(saved['word_errors'] == sum(saved[k] for k in ('substitutions', 'deletions', 'insertions')),
                'Edit components do not sum to distance')
    summary = result['summary']
    if isinstance(summary, dict):
        require(summary == totals(rows), 'Aggregate differs')
    else:
        require(any(s['split'] is None and s['subset'] == 'full' for s in summary), 'No full summary')
        for saved in summary:
            require(saved['subset'] in ('full', 'digit_free', 'eligible_official_style'), 'Unknown subset')
            selected = [r for r in rows if (saved['split'] is None or r['split'] == saved['split'])
                        and (saved['subset'] == 'full' or not any(c.isdigit() for c in r['reference']))
                        and (saved['subset'] != 'eligible_official_style' or r['duration_s'] <= 30)]
            require(all(saved[k] == v for k, v in totals(selected).items()), 'Aggregate differs')
    return totals(rows)


def result_pairs():
    for language in ('ru', 'en', 'kk', 'ky', 'uz'):
        for model in ('ml_ctc', 'ml_ctc_large'):
            yield READING / f'{model}_{language}.json', READING / f'{language}_manifest.json'
    for corpus in ('kk', 'kk_conversations_expanded', 'uz'):
        yield CONVERSATION / f'{corpus}_large.json', CONVERSATION / f'{corpus}_manifest.json'
    for model in ('whisper', 'omni'):
        for corpus in ('kk', 'kk_conversations_expanded', 'uz', 'ky'):
            if model == 'whisper' and corpus == 'ky':
                continue
            directory = READING if corpus == 'ky' else CONVERSATION
            yield COMPARISON / f'{model}_{corpus}.json', directory / f'{corpus}_manifest.json'


def main():
    index = json.loads((RESULTS / 'multilingual_publication.json').read_text())
    for relative, checksum in index['sha256'].items():
        path = (ROOT / relative).resolve()
        require(path.is_relative_to(ROOT), 'Path escapes checkout')
        require(hashlib.sha256(path.read_bytes()).hexdigest() == checksum, 'Changed file: ' + relative)
    count = 0
    for result_path, manifest_path in result_pairs():
        require(str(result_path.relative_to(ROOT)) in index['sha256'], 'Unsealed result')
        require(str(manifest_path.relative_to(ROOT)) in index['sha256'], 'Unsealed manifest')
        values = verify_result(json.loads(result_path.read_text()), json.loads(manifest_path.read_text()))
        count += values['n']
        print(f"{result_path.name}: n={values['n']} WER={values['wer_pct']:.4f}% CER={values['cer_pct']:.4f}%")
    require(count == 15989, 'Wrong total recognition count')
    print(f'PASS: {len(index["sha256"])} file hashes and {count} saved recognitions. No inference performed.')


if __name__ == '__main__':
    main()
