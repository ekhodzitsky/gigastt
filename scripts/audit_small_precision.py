#!/usr/bin/env python3
"""Independently audit saved paired native precision logits and HTTP parity."""
import argparse
import json
from pathlib import Path

import jiwer
import numpy as np

from audit_multilingual_1000 import MODEL_HASHES, VOCAB_HASH, digest, require
from wer_unicode import normalize, word_edit_distance


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--directory', type=Path, required=True)
    p.add_argument('--work', type=Path, required=True)
    p.add_argument('--probe', type=Path, required=True)
    p.add_argument('--vocab', type=Path, required=True)
    args = p.parse_args()
    public_path = args.directory / 'native_precision.json'
    public = json.loads(public_path.read_text())
    private = json.loads((args.work / 'native_precision_private.json').read_text())
    prepared_path = args.work.parent / 'inputs/prepared.json'
    prepared = json.loads(prepared_path.read_text())
    samples = {(r['id'], r['condition'], r['variant']): r for r in prepared['rows'] if r['variant'] != 'source_file'}
    http = json.loads((args.directory / 'native_input_controls.json').read_text())
    http_rows = {(r['id'], r['condition'], r['variant']): r for r in http['rows']}
    require(public['completed'] is True and private['completed'] is True and len(public['rows']) == len(private['rows']) == 666, 'Incomplete precision runs')
    require(public['metadata'] == private['metadata'], 'Public/private metadata mismatch')
    metadata = public['metadata']
    require(metadata['prepared_sha256'] == digest(prepared_path), 'Preparation changed')
    require(metadata['probe_sha256'] == digest(args.probe), 'Probe changed')
    require(metadata['script_sha256'] == digest(Path(__file__).with_name('analyze_small_precision.py')), 'Runner changed')
    require(metadata['vocab_sha256'] == digest(args.vocab) == VOCAB_HASH, 'Vocabulary changed')
    require(metadata['models_sha256']['int8'] == MODEL_HASHES['ml_ctc'], 'Not original INT8')
    vocab = [line.rsplit(' ', 1)[0] for line in args.vocab.read_text().splitlines()]
    require(len(vocab) == 71 and vocab[70] == '<blk>', 'Unexpected token mapping')
    for precision, sha in metadata['models_sha256'].items():
        model_paths = list((args.work / ('models_' + precision)).glob('*.onnx'))
        require(len(model_paths) == 1 and digest(model_paths[0]) == sha, 'Model hash mismatch')
    seen, groups, parity, baseline = set(), {}, 0, 0
    exact_http, whitespace_differences = 0, []
    for row, pub in zip(private['rows'], public['rows']):
        require({k: v for k, v in row.items() if k not in ('reference', 'hypothesis', 'raw_ctc_hypothesis')} == pub, 'Public/private row mismatch')
        key = (row['id'], row['condition'], row['variant'])
        precision = row['precision']
        require(precision in ('int8', 'fp32') and key in samples and (*key, precision) not in seen, 'Invalid or duplicate row')
        seen.add((*key, precision))
        sample = samples[key]
        require(all(row[k] == sample[k] for k in ('id', 'language', 'selection', 'condition', 'variant', 'reference')), 'Changed row identity')
        prefix = args.work / precision / '_'.join(key)
        meta = json.loads(Path(str(prefix) + '.json').read_text())
        require(digest(Path(str(prefix) + '.features.f32')) == digest(sample['feature_path']) == sample['feature_sha256'] == row['feature_sha256'], 'Matched feature checksum failed')
        require(meta['feature_length'] == row['feature_frames'] == sample['frames'], 'Feature length mismatch')
        require(meta['logit_shape'] == row['logit_shape'] and meta['ort_build_info'] == row['ort_build_info'], 'Logit metadata mismatch')
        logits = np.fromfile(str(prefix) + '.logits.f32', dtype='<f4').reshape(meta['logit_shape'])
        require(np.isfinite(logits).all() and logits.shape[0] == 1 and logits.shape[2] == 71, 'Nonfinite/wrong-shaped logits')
        frames = meta['output_lengths'][0]
        require(0 < frames <= logits.shape[1] and frames == row['frames'], 'Invalid output length')
        ids = np.argmax(logits[0, :frames], axis=1)
        collapsed = ids[np.r_[True, ids[1:] != ids[:-1]]]
        hypothesis = ''.join(vocab[int(i)] for i in collapsed if i != 70).replace('▁', ' ').strip()
        require(hypothesis == row['raw_ctc_hypothesis'], 'Independent raw CTC collapse mismatch')
        raw_hypothesis = hypothesis
        hypothesis = ' '.join(hypothesis.split())
        require(hypothesis == row['hypothesis'], 'Production word-join mismatch')
        ref, hyp = normalize(row['reference']), normalize(hypothesis)
        alignment = jiwer.process_words(' '.join(ref), ' '.join(hyp))
        counts = dict(reference_words=len(ref), word_errors=word_edit_distance(ref, hyp),
                      substitutions=alignment.substitutions, deletions=alignment.deletions,
                      insertions=alignment.insertions, empty=not bool(hypothesis),
                      all_blank=bool(np.all(ids == 70)), blank_frames=int(np.count_nonzero(ids == 70)), frames=frames)
        require(all(row[k] == v for k, v in counts.items()), 'Metric or blank-count mismatch')
        expected_match = hypothesis == sample['baseline_hypothesis'].strip() if row['variant'] == 'decoded_float' else None
        require(row['matches_original_product'] == expected_match, 'Incorrect baseline match flag')
        if precision == 'int8':
            http_hypothesis = http_rows[key]['hypothesis'].strip()
            require(' '.join(hypothesis.split()) == ' '.join(http_hypothesis.split()), 'Native INT8/HTTP word mismatch')
            require(hypothesis == http_hypothesis, 'Production word-join HTTP mismatch')
            exact_http += 1
            if raw_hypothesis != http_hypothesis:
                whitespace_differences.append(dict(id=row['id'], condition=row['condition'], variant=row['variant']))
            parity += 1
            if row['variant'] == 'decoded_float':
                require(expected_match, 'Baseline mismatch')
                baseline += 1
        group_key = '/'.join(row[k] for k in ('precision', 'selection', 'condition', 'variant'))
        g = groups.setdefault(group_key, {'n': 0, **{k: 0 for k in counts}})
        g['n'] += 1
        for k, value in counts.items():
            g[k] += value
    require(len(seen) == 666 and len(samples) == 333 and parity == 333 and baseline == 111, 'Incomplete paired set/parity')
    for g in groups.values():
        g['wer_percent'] = 100 * g['word_errors'] / g['reference_words']
        g['blank_fraction'] = g['blank_frames'] / g['frames']
    require(groups == public['groups'] == private['groups'], 'Group metrics differ')
    require(public['baseline_parity'] == private['baseline_parity'] == dict(checked=111, exact_matches_except_outer_whitespace=111), 'Baseline parity metadata mismatch')
    output = dict(completed=True, executions=666, matched_feature_inputs=333,
                  baseline_int8_exact_matches=111, int8_http_exact_matches=exact_http,
                  int8_http_whitespace_normalized_matches=parity,
                  raw_ctc_http_exact_matches=parity - len(whitespace_differences),
                  internal_whitespace_only_differences=whitespace_differences,
                  all_saved_logits_finite=True, output_lengths_and_ctc_collapse_verified=True,
                  all_word_distances_independently_verified=True, blank_counts_and_group_fractions_verified=True,
                  model_and_vocab_and_probe_and_feature_hashes_verified=True,
                  result_sha256=digest(public_path),
                  limitations='Post-selection diagnostic set enriched for INT8 failures; precision differences here are not an unbiased language-level quantization penalty. Saved outputs audited; inference not rerun.',
                  groups=groups)
    (args.directory / 'native_precision_audit.json').write_text(json.dumps(output, ensure_ascii=False, indent=2) + '\n')
    print('PASS: 666 saved logits/metrics, 333 matched features and exact production-word-join HTTP parity, 111 baseline parity')


if __name__ == '__main__':
    main()
