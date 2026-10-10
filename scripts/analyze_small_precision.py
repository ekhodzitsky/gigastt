#!/usr/bin/env python3
"""Run native INT8/FP32 on identical frozen small-model diagnostic features.

FP32 is an offline control only. Private model/cache folders isolate the
production installation. Previous benchmark artifacts are never overwritten.
"""
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import subprocess

from wer_unicode import normalize


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def save(path, value):
    temporary = path.with_suffix('.partial')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prepared', type=Path, required=True)
    p.add_argument('--probe', type=Path, required=True)
    p.add_argument('--int8', type=Path, required=True)
    p.add_argument('--fp32', type=Path, required=True)
    p.add_argument('--vocab', type=Path, required=True)
    p.add_argument('--work', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--reuse-native', action='store_true', help='Rescore existing logits after verifying probe/model/input identity')
    args = p.parse_args()
    import numpy as np
    import jiwer
    args.work.mkdir(parents=True, exist_ok=True)
    samples = [r for r in json.loads(args.prepared.read_text())['rows'] if r['variant'] != 'source_file']
    manifest = []
    for row in samples:
        if digest(row['feature_path']) != row['feature_sha256']:
            raise ValueError('Feature checksum mismatch')
        key = row['id'] + '_' + row['condition'] + '_' + row['variant']
        manifest.append({'id': key, 'input': row['feature_path'], 'mode': 'mel'})
    if len({r['id'] for r in manifest}) != len(manifest):
        raise ValueError('Duplicate diagnostic IDs')
    manifest_path = args.work / 'native_manifest.json'
    save(manifest_path, manifest)
    vocab = [line.rsplit(' ', 1)[0] for line in args.vocab.read_text().splitlines()]
    if len(vocab) != 71 or vocab[-1] != '<blk>':
        raise ValueError('Unexpected multilingual vocabulary')
    metadata = {'prepared_sha256': digest(args.prepared), 'probe_sha256': digest(args.probe),
                'script_sha256': digest(__file__), 'vocab_sha256': digest(args.vocab),
                'models_sha256': {'int8': digest(args.int8), 'fp32': digest(args.fp32)},
                'input_count': len(samples), 'intra_threads': 3, 'inter_threads': 1,
                'factory': 'production_factory with isolated cache for each precision',
                'restriction': 'FP32 laboratory control; no production model/pipeline change'}
    metadata['text_assembly'] = 'CTC collapse then whitespace-separated word join, matching production ctc_tokens_to_words; raw CTC strings retained privately'
    if args.reuse_native:
        previous = json.loads((args.work / 'native_precision_private.json').read_text())
        for key in ['prepared_sha256', 'probe_sha256', 'models_sha256', 'vocab_sha256']:
            if previous['metadata'][key] != metadata[key]:
                raise ValueError('Cannot reuse native logits with changed ' + key)
    rows = []

    def finish(complete):
        groups = {}
        for row in rows:
            key = '/'.join(row[k] for k in ['precision', 'selection', 'condition', 'variant'])
            g = groups.setdefault(key, {'n': 0, 'reference_words': 0, 'word_errors': 0, 'substitutions': 0, 'deletions': 0, 'insertions': 0, 'empty': 0, 'all_blank': 0, 'blank_frames': 0, 'frames': 0})
            g['n'] += 1
            for k in ['reference_words', 'word_errors', 'substitutions', 'deletions', 'insertions', 'empty', 'all_blank', 'blank_frames', 'frames']:
                g[k] += row[k]
        for g in groups.values():
            g['wer_percent'] = 100 * g['word_errors'] / g['reference_words']
            g['blank_fraction'] = g['blank_frames'] / g['frames']
        baseline = [r for r in rows if r['precision'] == 'int8' and r['variant'] == 'decoded_float']
        common = {'metadata': metadata, 'completed': complete, 'groups': groups,
                  'baseline_parity': {'checked': len(baseline), 'exact_matches_except_outer_whitespace': sum(r['matches_original_product'] for r in baseline)}}
        save(args.work / 'native_precision_private.json', {**common, 'rows': rows})
        public = [{k: v for k, v in row.items() if k not in ['reference', 'hypothesis', 'raw_ctc_hypothesis']} for row in rows]
        save(args.output, {**common, 'rows': public})

    for precision, model in [('int8', args.int8), ('fp32', args.fp32)]:
        model_dir = args.work / ('models_' + precision)
        model_dir.mkdir(exist_ok=True)
        isolated_model = model_dir / model.name
        if not isolated_model.exists():
            isolated_model.symlink_to(model.resolve())
        if digest(isolated_model) != metadata['models_sha256'][precision]:
            raise ValueError('Isolated model checksum mismatch')
        output = args.work / precision
        if not args.reuse_native:
            subprocess.run([str(args.probe.resolve()), str(isolated_model), str(manifest_path), str(output), 'production', '3'], check=True)
        for sample, entry in zip(samples, manifest):
            key = entry['id']
            meta = json.loads((output / (key + '.json')).read_text())
            feature_hash = digest(output / (key + '.features.f32'))
            if feature_hash != sample['feature_sha256'] or meta['feature_length'] != sample['frames']:
                raise ValueError('Native feature input mismatch')
            logits = np.fromfile(output / (key + '.logits.f32'), dtype='<f4').reshape(meta['logit_shape'])
            if not np.isfinite(logits).all():
                raise ValueError('Non-finite logits')
            frames = meta['output_lengths'][0]
            if not 0 < frames <= logits.shape[1]:
                raise ValueError('Invalid output length')
            ids = logits[0, :frames].argmax(-1)
            raw_text = ''.join(vocab[i] for i, _ in itertools.groupby(ids) if i != 70).replace('▁', ' ').strip()
            text = ' '.join(raw_text.split())
            ref = normalize(sample['reference'])
            score = jiwer.process_words(' '.join(ref), ' '.join(normalize(text)))
            rows.append({k: sample[k] for k in ['id', 'language', 'selection', 'condition', 'variant', 'reference']} | {
                'precision': precision, 'feature_sha256': feature_hash, 'feature_frames': sample['frames'],
                'logit_shape': meta['logit_shape'], 'frames': frames, 'ort_build_info': meta['ort_build_info'],
                'hypothesis': text, 'raw_ctc_hypothesis': raw_text, 'reference_words': len(ref), 'word_errors': score.substitutions + score.deletions + score.insertions,
                'substitutions': score.substitutions, 'deletions': score.deletions, 'insertions': score.insertions,
                'empty': not bool(text), 'all_blank': bool((ids == 70).all()), 'blank_frames': int((ids == 70).sum()),
                'matches_original_product': text == sample['baseline_hypothesis'].strip() if sample['variant'] == 'decoded_float' else None})
        finish(precision == 'fp32')
        print(precision, 'completed', len(samples), 'native inputs', flush=True)


if __name__ == '__main__':
    main()
