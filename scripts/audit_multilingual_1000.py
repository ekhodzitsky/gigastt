#!/usr/bin/env python3
"""Independently audit the fixed five-language original-model benchmark.

Reconstructs selection from pinned source files, verifies selected audio bytes,
recomputes all WER distances with the repository's independent Python DP, and
all CER distances with RapidFuzz (plus 50 Python-DP CER checks per language/run).
Only emits a final validation summary when all five languages and both original
model variants are complete. --manifests-only produces a clearly partial audit.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import unicodedata

import pyarrow.parquet as pq
from rapidfuzz.distance import Levenshtein
import soundfile as sf

from wer_unicode import normalize, word_edit_distance, _wer_ci

LANGUAGES = {'en', 'ru', 'kk', 'ky', 'uz'}
MODEL_HASHES = {
    'ml_ctc': 'e08e27ae5669b39f0c378fae101bbbb9a80505f74f9b66719c309bf5b894a480',
    'ml_ctc_large': 'b2ad9c38fc04197ba758105d33f7404fd13d977958722e0f49e3f3e22521f1c6',
}
VOCAB_HASH = '4d130287892e1099fedfb3f93c4b4cf8a263151158801680b28977d1be4133f4'
REVISION = '168de341b3db6859a9bac1c50a2ef5e3b47647e0'


def digest(path):
    with open(path, 'rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def audit_manifest(path, cache, vocab):
    m = json.loads(path.read_text())
    require(m['language'] in LANGUAGES and m['revision'] == REVISION, 'Unexpected language/revision')
    exclusion = m.get('kazakh_prior_adaptation_exclusion') or {}
    excluded_ids = set(exclusion.get('sentence_ids', []))
    excluded_hashes = set(exclusion.get('audio_sha256', []))
    pools = {}
    for source in m['sources']:
        split = source['split']
        require(split in {'test', 'validation'}, 'Training split is prohibited')
        parquet = cache / f"fleurs_{m['config']}_{split}.parquet"
        require(digest(parquet) == source['sha256'], f'Source checksum mismatch: {parquet}')
        rows = []
        index = 0
        for batch in pq.ParquetFile(parquet).iter_batches(batch_size=16, columns=['id', 'audio', 'transcription', 'num_samples']):
            for row in batch.to_pylist():
                sha = hashlib.sha256(row['audio']['bytes']).hexdigest()
                if (row['id'] not in excluded_ids and sha not in excluded_hashes
                        and normalize(row['transcription']) and row['num_samples'] > 0):
                    rows.append(dict(id=f"fleurs_{m['config']}_{split}_{index:04d}", split=split,
                                     row_index=index, sentence_id=row['id'], sha256=sha,
                                     reference=row['transcription'], num_samples=row['num_samples']))
                index += 1
        pools[split] = rows
    validation = pools['validation'].copy()
    random.Random(42).shuffle(validation)
    selected, seen = [], set()
    for row in pools['test'] + validation:
        if row['sha256'] not in seen:
            selected.append(row)
            seen.add(row['sha256'])
        if len(selected) == 1000:
            break
    selected.sort(key=lambda row: (row['split'] != 'test', row['row_index']))
    samples = m['samples']
    require(len(samples) == len(selected) == 1000, f"Incomplete selection: {m['language']}")
    require(len({s['id'] for s in samples}) == 1000, 'Duplicate sample IDs')
    require(len({s['sha256'] for s in samples}) == 1000, 'Duplicate selected audio')
    for actual, expected in zip(samples, selected):
        require(all(actual[k] == v for k, v in expected.items()), f"Selection mismatch: {actual['id']}")
        require(digest(actual['path']) == actual['sha256'], f"Audio mismatch: {actual['id']}")
        info = sf.info(actual['path'])
        require(info.samplerate == 16000 and info.frames == actual['num_samples'], f"Header mismatch: {actual['id']}")
    counts = Counter(s['split'] for s in samples)
    require(dict(counts) == m['selected_split_counts'], 'Split count metadata mismatch')
    oov = Counter(c for s in samples for c in ' '.join(normalize(s['reference'])) if c not in vocab)
    facts = dict(language=m['language'], manifest_sha256=digest(path), n=1000,
                 splits=dict(counts), unique_sentence_ids=len({s['sentence_id'] for s in samples}),
                 audio_s=sum(s['num_samples']/16000 for s in samples),
                 digit_bearing_references=sum(any(c.isdigit() for c in s['reference']) for s in samples),
                 non_nfc_references=sum(unicodedata.normalize('NFC', s['reference']) != s['reference'] for s in samples),
                 normalized_out_of_vocabulary_characters=dict(sorted(oov.items())),
                 source_selection_and_audio_hashes_verified=True)
    return m, facts


def audit_result(path, manifests):
    result = json.loads(path.read_text())
    require(result.get('completed') is True, f'Incomplete run: {path}')
    identity = result['metadata']['identity']
    variant = identity['variant']
    require(variant in MODEL_HASHES, 'Unexpected/adapted model variant')
    require(set(identity['model_sha256'].values()) == {MODEL_HASHES[variant], VOCAB_HASH}, 'Unexpected model/vocab bytes')
    configuration = identity['health_configuration']
    require(configuration['variant'] == variant and configuration['punctuation'] is False
            and configuration['itn'] is False, 'Unexpected server configuration')
    rows = result['details']
    languages = {r['language'] for r in rows}
    require(len(languages) == 1, 'Expect one language per run artifact')
    language = next(iter(languages))
    manifest, manifest_hash = manifests[language]
    require(identity['manifest_sha256'] == manifest_hash, 'Manifest identity mismatch')
    samples = manifest['samples']
    require(len(rows) == result['n'] == 1000, 'Expected 1000 results')
    require([r['id'] for r in rows] == [s['id'] for s in samples], 'Result ordering/IDs differ from manifest')
    character_checks = set(random.Random(42).sample(range(1000), 50))
    computed = []
    for index, (row, sample) in enumerate(zip(rows, samples)):
        require(row['condition'] == 'original', 'Only original read-speech inputs belong in this run')
        require(all(row[k] == sample[k] for k in ['id','sha256','reference','language','split']), 'Result/sample mismatch')
        require(row['status'] in {'ok','failed'}, 'Invalid request status')
        require(row['status'] != 'failed' or row['hypothesis'] == '', 'Failed request has nonempty hypothesis')
        ref, hyp = normalize(row['reference']), normalize(row['hypothesis'])
        ref_text, hyp_text = ' '.join(ref), ' '.join(hyp)
        errors = word_edit_distance(ref, hyp)
        cer_errors = Levenshtein.distance(ref_text, hyp_text)
        saved = row['scores']
        require(saved['reference_words'] == len(ref) and saved['word_errors'] == errors, 'Independent WER mismatch')
        require(saved['reference_chars'] == len(ref_text) and saved['char_errors'] == cer_errors, 'CER mismatch')
        require(sum(saved[k] for k in ['substitutions','deletions','insertions']) == errors, 'Error components mismatch')
        if index in character_checks:
            require(word_edit_distance(list(ref_text), list(hyp_text)) == cer_errors, 'Independent character-DP mismatch')
        computed.append(dict(row=row, words=len(ref), word_errors=errors, chars=len(ref_text), char_errors=cer_errors))
    for summary in result['summary']:
        group = [x for x in computed if (summary['split'] is None or x['row']['split'] == summary['split'])
                 and (summary['subset'] == 'full' or not any(c.isdigit() for c in x['row']['reference']))
                 and (summary['subset'] != 'eligible_official_style' or x['row']['duration_s'] <= 30)]
        words = sum(x['words'] for x in group)
        chars = sum(x['chars'] for x in group)
        wer = 100*sum(x['word_errors'] for x in group)/words if words else None
        cer = 100*sum(x['char_errors'] for x in group)/chars if chars else None
        require(summary['n'] == len(group) and summary['wer_pct'] == wer and summary['cer_pct'] == cer, 'Aggregate score mismatch')
    point, low, high = _wer_ci([(x['words'], x['word_errors']) for x in computed])
    return dict(language=language, variant=variant, result_sha256=digest(path), n=1000,
                model_sha256=MODEL_HASHES[variant], binary_sha256=identity['binary_sha256'],
                failed_requests=sum(r['status']=='failed' for r in rows),
                independent_all_word_distances_verified=True, all_character_distances_recomputed=True,
                independent_python_character_dp_checks=50, all_summary_groups_verified=True,
                wer_percent=point, wer_utterance_bootstrap_95_percent=[low,high],
                cer_percent=100*sum(x['char_errors'] for x in computed)/sum(x['chars'] for x in computed)), identity


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--directory', type=Path, required=True)
    p.add_argument('--cache', type=Path, required=True)
    p.add_argument('--vocab', type=Path, required=True)
    p.add_argument('--results', nargs='*', type=Path, default=[])
    p.add_argument('--manifests-only', action='store_true')
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    require(digest(args.vocab) == VOCAB_HASH, 'Vocabulary differs from original')
    vocab = {s.rsplit(' ',1)[0] for s in args.vocab.read_text().splitlines()} | {' '}
    manifests, facts = {}, []
    for language in sorted(LANGUAGES):
        path = args.directory / (language + '_manifest.json')
        m, f = audit_manifest(path, args.cache, vocab)
        manifests[language] = (m, digest(path))
        facts.append(f)
        print('Verified manifest', language, flush=True)
    audits, identities = [], []
    if not args.manifests_only:
        for path in args.results:
            audit, identity = audit_result(path, manifests)
            audits.append(audit)
            identities.append(identity)
            print('Verified results', audit['variant'], audit['language'], flush=True)
        require({(r['variant'],r['language']) for r in audits} == {(v,l) for v in MODEL_HASHES for l in LANGUAGES}, 'Need both models × five languages')
        require(len(audits) == 10, 'Duplicate result artifacts')
        require(len({i['binary_sha256'] for i in identities}) == 1, 'Paired runs use different binaries')
        require(len({json.dumps(i['scorer_sha256'],sort_keys=True) for i in identities}) == 1, 'Paired runs use different scorers')
        require(len({i['jiwer'] for i in identities}) == 1, 'Paired runs use different scorer dependency versions')
    output = dict(completed=not args.manifests_only, scope='Manifest-only audit' if args.manifests_only else 'All 5,000 original recordings × two unadapted INT8 models',
                  revision=REVISION, no_training_split=True,
                  original_model_hashes_verified=None if args.manifests_only else True,
                  bootstrap=dict(resampling_unit='utterance',seed=42,resamples=1000,coverage='95%',
                                 caveat='Descriptive selected-set interval; repeated sentences/speakers violate independent-speaker interpretation; not a language-wide estimate'),
                  manifests=facts, runs=audits)
    args.output.write_text(json.dumps(output,ensure_ascii=False,indent=2)+'\n')


if __name__ == '__main__':
    main()
