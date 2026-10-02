#!/usr/bin/env python3
"""Audit all paired 1000-recording multilingual G.711 runs without changing scores."""
import argparse
import json
from pathlib import Path

import jiwer
from rapidfuzz.distance import Levenshtein
import soundfile as sf

from audit_multilingual_1000 import LANGUAGES, MODEL_HASHES, VOCAB_HASH, digest, require
from wer_unicode import normalize, word_edit_distance

COUNTS = ('reference_words', 'word_errors', 'substitutions', 'deletions', 'insertions',
          'reference_chars', 'char_errors')


def verify_scores(rows):
    for row in rows:
        require(row['status'] in {'ok', 'failed'}, 'Invalid status')
        require(row['status'] != 'failed' or not row['hypothesis'], 'Failed request has hypothesis')
        ref, hyp = normalize(row['reference']), normalize(row['hypothesis'])
        rt, ht = ' '.join(ref), ' '.join(hyp)
        alignment = jiwer.process_words(rt, ht)
        computed = dict(reference_words=len(ref), word_errors=word_edit_distance(ref, hyp),
                        substitutions=alignment.substitutions, deletions=alignment.deletions,
                        insertions=alignment.insertions, reference_chars=len(rt),
                        char_errors=Levenshtein.distance(rt, ht))
        require(all(row['scores'][k] == v for k, v in computed.items()), f"Score mismatch {row['id']}")
        require(sum(computed[k] for k in ('substitutions', 'deletions', 'insertions')) == computed['word_errors'], 'Alignment distance mismatch')


def totals(rows):
    result = {k: sum(row['scores'][k] for row in rows) for k in COUNTS}
    result.update(n=len(rows), empty_hypotheses=sum(not normalize(r['hypothesis']) for r in rows),
                  failed_requests=sum(r['status'] == 'failed' for r in rows))
    result['wer_pct'] = 100 * result['word_errors'] / result['reference_words'] if result['reference_words'] else None
    result['cer_pct'] = 100 * result['char_errors'] / result['reference_chars'] if result['reference_chars'] else None
    return result


def verify_summary(result):
    for saved in result['summary']:
        rows = [r for r in result['details'] if (saved['split'] is None or r['split'] == saved['split'])
                and (saved['subset'] == 'full' or not any(c.isdigit() for c in r['reference']))
                and (saved['subset'] != 'eligible_official_style' or r['duration_s'] <= 30)]
        require(all(saved[k] == v for k, v in totals(rows).items()), 'Aggregate summary mismatch')
    require(result['failed_requests'] == sum(r['status'] == 'failed' for r in result['details']), 'Top-level failure mismatch')


def load_result(path, manifest, manifest_path, condition):
    result = json.loads(path.read_text())
    rows = result['details']
    require(result.get('completed') is True and result['n'] == len(rows) == 1000, f'Incomplete result {path}')
    require(len({r['id'] for r in rows}) == 1000, 'Duplicate result IDs')
    require(result['metadata']['identity']['manifest_sha256'] == digest(manifest_path), 'Manifest hash mismatch')
    for row, sample in zip(rows, manifest['samples']):
        require(row['condition'] == condition, 'Wrong condition')
        require(all(row[k] == sample[k] for k in ('id', 'sha256', 'reference', 'language', 'split', 'dataset')), 'Result sample identity mismatch')
    verify_scores(rows)
    verify_summary(result)
    return result


def compare(base, degraded):
    groups = []
    for subset in ('full', 'digit_free', 'digit_bearing'):
        pairs = [(a, b) for a, b in zip(base, degraded) if subset == 'full' or
                 any(c.isdigit() for c in a['reference']) == (subset == 'digit_bearing')]
        original, telephone = totals([a for a, _ in pairs]), totals([b for _, b in pairs])
        changed = [b['scores']['word_errors'] - a['scores']['word_errors'] for a, b in pairs]
        groups.append(dict(subset=subset, original=original, telephone=telephone,
                           delta={k: telephone[k] - original[k] for k in COUNTS},
                           wer_delta_pp=telephone['wer_pct'] - original['wer_pct'] if original['wer_pct'] is not None else None,
                           cer_delta_pp=telephone['cer_pct'] - original['cer_pct'] if original['cer_pct'] is not None else None,
                           improved=sum(d < 0 for d in changed), worsened=sum(d > 0 for d in changed), unchanged=sum(d == 0 for d in changed),
                           newly_empty=sum(bool(normalize(a['hypothesis'])) and not normalize(b['hypothesis']) for a, b in pairs),
                           recovered_empty=sum(not normalize(a['hypothesis']) and bool(normalize(b['hypothesis'])) for a, b in pairs),
                           increased_deletions=sum(b['scores']['deletions'] > a['scores']['deletions'] for a, b in pairs),
                           increased_substitutions=sum(b['scores']['substitutions'] > a['scores']['substitutions'] for a, b in pairs)))
    return groups


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    audits, runner_hashes = [], set()
    for language in sorted(LANGUAGES):
        baseline_path = args.baseline / f'{language}_manifest.json'
        baseline_manifest = json.loads(baseline_path.read_text())
        baseline_samples = baseline_manifest['samples']
        require(len(baseline_samples) == len({s['id'] for s in baseline_samples}) == 1000, 'Invalid baseline sample count')
        for s in baseline_samples:
            require(digest(s['path']) == s['sha256'], 'Source audio checksum mismatch')
        bases = {v: load_result(args.baseline / f'{v}_{language}.json', baseline_manifest, baseline_path, 'original') for v in MODEL_HASHES}
        for codec in ('alaw', 'mulaw'):
            manifest_path = args.directory / f'{language}_{codec}_manifest.json'
            manifest = json.loads(manifest_path.read_text())
            samples = manifest['samples']
            condition = f'simulated_telephone_{codec}'
            require(manifest['baseline_manifest_sha256'] == digest(baseline_path), 'Baseline manifest link mismatch')
            require(len(samples) == len({s['id'] for s in samples}) == 1000, 'Invalid telephone sample count')
            for original, sample in zip(baseline_samples, samples):
                require(all(sample[k] == original[k] for k in ('id', 'reference', 'language', 'split', 'sentence_id', 'row_index')), 'Unpaired source sample')
                require(sample['source_audio_sha256'] == original['sha256'], 'Source audio hash mismatch')
                require(sample['condition'] == condition and digest(sample['path']) == sample['sha256'], 'Prepared audio identity mismatch')
                info = sf.info(sample['path'])
                require(info.samplerate == 8000 and info.channels == 1 and info.subtype == {'alaw': 'ALAW', 'mulaw': 'ULAW'}[codec], 'Wrong telephone format')
                require(abs(info.duration - original['num_samples'] / 16000) <= 1/8000 + 1e-9, 'Duration mismatch')
            for variant, baseline in bases.items():
                path = args.directory / f'{variant}_{language}_{codec}.json'
                result = load_result(path, manifest, manifest_path, condition)
                before, after = baseline['metadata']['identity'], result['metadata']['identity']
                require(before['variant'] == after['variant'] == variant, 'Wrong model variant')
                require(set(after['model_sha256'].values()) == {MODEL_HASHES[variant], VOCAB_HASH}, 'Non-original model bytes')
                for key in ('binary_sha256', 'model_sha256', 'jiwer', 'health_configuration', 'model_configuration'):
                    require(before[key] == after[key], f'Changed baseline identity: {key}')
                for name in ('benchmark_multilingual_public.py', 'wer_unicode.py'):
                    require(before['scorer_sha256'][name] == after['scorer_sha256'][name] == digest(Path(__file__).with_name(name)), 'Changed scoring code')
                runner_hashes.add(after['scorer_sha256']['benchmark_multilingual_1000.py'])
                audits.append(dict(language=language, variant=variant, codec=codec, n=1000,
                                   result_sha256=digest(path), manifest_sha256=digest(manifest_path),
                                   baseline_result_sha256=digest(args.baseline / f'{variant}_{language}.json'),
                                   baseline_runner_sha256=before['scorer_sha256']['benchmark_multilingual_1000.py'],
                                   subsets=compare(baseline['details'], result['details'])))
                print('Verified', variant, language, codec, flush=True)
    require(len(audits) == 20 and len(runner_hashes) == 1, 'Need consistent runner for all 20 telephone runs')
    require(next(iter(runner_hashes)) == digest(Path(__file__).with_name('benchmark_multilingual_1000.py')), 'Runner changed since inference')
    output = dict(completed=True, samples_per_language=1000, telephone_recognitions=20000,
                  baseline_recognitions_reaudited=10000, independent_word_dp_verified=True,
                  character_distances_recomputed=True, source_and_prepared_audio_sha256_verified=True,
                  scorer_and_model_and_binary_unchanged=True, runner_sha256=next(iter(runner_hashes)),
                  runner_change='Preserve explicit manifest condition and validate it on ledger resume; scoring unchanged.',
                  limitations='Simulated telephone read speech, not real calls. Digit subsets identify formatting exposure, not numeric error attribution. No code-switch labels are available. Deletions are alignment events and do not by themselves establish VAD speech loss. Counts use jiwer alignment; independent DP verifies total distance.',
                  runs=audits)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2) + '\n')


if __name__ == '__main__':
    main()
