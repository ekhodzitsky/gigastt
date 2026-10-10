#!/usr/bin/env python3
"""Validate fixed native small-model controls and explicitly require baseline parity."""
import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf

from audit_multilingual_telephony import compare, digest, require, verify_scores
from audit_multilingual_1000 import MODEL_HASHES, VOCAB_HASH
from wer_unicode import normalize


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--work', type=Path, required=True)
    parser.add_argument('--inspect-audio', type=Path, required=True)
    args = parser.parse_args()
    cases_path = args.directory / 'cases.json'
    cases = json.loads(cases_path.read_text())
    result_path = args.directory / 'native_input_controls.json'
    result = json.loads(result_path.read_text())
    provenance = json.loads((args.directory / 'provenance.json').read_text())
    prepared = json.loads((args.work / 'prepared.json').read_text())
    require(result['completed'] is True and result['n'] == len(result['rows']) == 444, 'Incomplete request count')
    require(result['metadata'] == prepared['metadata'], 'Preparation metadata changed')
    require(result['metadata']['cases_sha256'] == digest(cases_path), 'Cases changed')
    require(result['metadata']['inspect_audio_sha256'] == digest(args.inspect_audio), 'Inspect binary changed')
    for item in provenance['identity'].values():
        require(digest(item['path']) == item['sha256'], f"Provenance changed: {item['path']}")
    require(provenance['identity']['binary']['sha256'] == cases['baseline_binary_sha256'], 'Server binary changed')
    require(provenance['identity']['int8']['sha256'] == MODEL_HASHES['ml_ctc'], 'Non-original small weights')
    require(provenance['identity']['vocab']['sha256'] == VOCAB_HASH, 'Non-original vocabulary')
    expected = {}
    by_case = {(r['id'], r['condition']): r for r in cases['rows']}
    for key in by_case:
        for variant in ('source_file', 'decoded_float') + (() if key[1] == 'original' else ('length_only_pad4', 'delay_only_shift256', 'flush_trim256')):
            expected[(*key, variant)] = by_case[key]
    actual = { (r['id'], r['condition'], r['variant']): r for r in result['rows'] }
    require(len(actual) == len(expected) == 444 and actual.keys() == expected.keys(), 'Wrong intervention selection')
    require(len(prepared['rows']) == 444, 'Wrong prepared count')
    for row, prep in zip(result['rows'], prepared['rows']):
        require(all(row[k] == v for k, v in prep.items()), 'Prepared/result mismatch')
        case = expected[row['id'], row['condition'], row['variant']]
        require(all(row[k] == v for k, v in case.items()), 'Changed case metadata')
        require(digest(row['input_path']) == row['input_sha256'], 'Input hash mismatch')
        require(digest(row['feature_path']) == row['feature_sha256'], 'Feature hash mismatch')
        features = np.fromfile(row['feature_path'], dtype='<f4')
        require(features.size == 64 * row['frames'] and np.isfinite(features).all(), 'Invalid features')
        require(row['exact_baseline_match'] == (row['hypothesis'].strip() == row['baseline_hypothesis'].strip()), 'Incorrect parity flag')
    verify_scores([{**r, 'status': 'ok'} for r in result['rows']])
    parity = [r for r in result['rows'] if r['variant'] in ('source_file', 'decoded_float')]
    require(len(parity) == 222 and all(r['exact_baseline_match'] for r in parity), 'Native baseline parity failed despite completed requests')
    for key, case in by_case.items():
        directory = args.work / (case['id'] + '_' + case['condition'])
        native = np.fromfile(directory / 'native.pcm.f32', dtype='<f4')
        source = actual[(*key, 'source_file')]
        decoded = actual[(*key, 'decoded_float')]
        require(source['feature_sha256'] == decoded['feature_sha256'], 'Decoded float frontend mismatch')
        wanted = {'decoded_float': native}
        if case['condition'] != 'original':
            samples, rate = sf.read(case['path'], dtype='float32')
            require(rate == 8000 and samples.ndim == 1, 'Wrong phone source')
            expected_length = len(samples) * 2
            require(expected_length - len(native) == 4, 'Unexpected native length')
            padded, padded_rate = sf.read(directory / 'source_with_flush.wav', dtype='float32')
            require(padded_rate == 8000 and np.array_equal(padded, np.pad(samples, (0, 512))), 'Wrong flush input')
            flushed = np.fromfile(directory / 'flushed.pcm.f32', dtype='<f4')
            require(np.array_equal(flushed[:len(native)], native), 'Flush changed existing prefix')
            require(len(flushed) >= 256 + expected_length, 'Insufficient flush')
            wanted.update(length_only_pad4=np.pad(native, (0, 4)),
                          delay_only_shift256=np.pad(native[256:], (0, 256)),
                          flush_trim256=flushed[256:256 + expected_length])
        for variant, array in wanted.items():
            row = actual[(*key, variant)]
            audio, rate = sf.read(row['input_path'], dtype='float32')
            require(rate == 16000 and np.array_equal(array, audio), 'Wrong float WAV intervention')
            dumped = np.fromfile(directory / (variant + '.pcm.f32'), dtype='<f4')
            require(np.array_equal(array, dumped), 'Float WAV decode changed samples')
    groups = []
    for selection in ('failure', 'control'):
        for condition in ('original', 'alaw', 'mulaw'):
            for variant in ('source_file', 'decoded_float', 'length_only_pad4', 'delay_only_shift256', 'flush_trim256'):
                rows = [r for r in result['rows'] if r['selection'] == selection and r['condition'] == condition and r['variant'] == variant]
                if not rows:
                    continue
                pairs = [(actual[(r['id'], r['condition'], 'source_file')], r) for r in rows]
                before = [{**a, 'status': 'ok'} for a, _ in pairs]
                after = [{**b, 'status': 'ok'} for _, b in pairs]
                summary = compare(before, after)[0]
                summary.update(selection=selection, condition=condition, variant=variant,
                               worsened_ids=[b['id'] for a, b in pairs if b['scores']['word_errors'] > a['scores']['word_errors']],
                               newly_empty_ids=[b['id'] for a, b in pairs if normalize(a['hypothesis']) and not normalize(b['hypothesis'])],
                               recovered_empty_ids=[b['id'] for a, b in pairs if not normalize(a['hypothesis']) and normalize(b['hypothesis'])])
                groups.append(summary)
    output = dict(completed=True, request_count=444, explicit_native_parity_count=222,
                  parity_required_separately_from_request_completion=True,
                  cases_sha256=digest(cases_path), result_sha256=digest(result_path),
                  prepared_input_and_feature_hashes_verified=True,
                  cached_pcm_transformations_and_float_roundtrips_verified=True,
                  flush_prefix_and_lengths_verified=True, all_word_distances_independently_verified=True,
                  provenance_files_match_recorded_hashes=True,
                  limitations='Failure-enriched selected diagnostics, not independent language accuracy. Cache transformations verified; this audit does not rerun feature extraction or inference.',
                  groups=groups)
    (args.directory / 'native_controls_validation.json').write_text(json.dumps(output, ensure_ascii=False, indent=2) + '\n')
    print('Verified 444 controls, explicit 222 baseline parity, cached transformations and provenance')
    for g in groups:
        if g['variant'] not in ('source_file', 'decoded_float'):
            print(g['selection'], g['condition'], g['variant'], 'WER', round(g['original']['wer_pct'], 2), '->', round(g['telephone']['wer_pct'], 2),
                  'improved/worse/same', g['improved'], g['worsened'], g['unchanged'], 'empty',g['original']['empty_hypotheses'],'->',g['telephone']['empty_hypotheses'])


if __name__ == '__main__':
    main()
