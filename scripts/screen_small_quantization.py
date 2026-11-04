#!/usr/bin/env python3
"""Screen deterministic quantization candidates on frozen native features."""
import argparse
import itertools
import json
import subprocess
from pathlib import Path
import jiwer
import numpy as np
from analyze_small_precision import digest, save
from wer_unicode import normalize


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('candidate')
    args = p.parse_args()
    root = Path(__file__).resolve().parents[1]
    report = root / 'benchmark/results/multilingual_small_quantization_20261001'
    protocol = report / 'protocol.json'
    config = json.loads(protocol.read_text())
    if args.candidate not in config['candidates']:
        raise ValueError('Candidate was not declared')
    model = json.loads((report / (args.candidate + '.json')).read_text())
    if digest(model['path']) != model['sha256']:
        raise ValueError('Model identity mismatch')
    prepared = Path.home() / '.cache/gigastt-small-diagnosis/inputs/prepared.json'
    source = json.loads(prepared.read_text())
    assert source['metadata']['cases_sha256'] == config['cases_sha256']
    samples = [r for r in source['rows'] if r['variant'] == 'decoded_float']
    assert len(samples) == 111
    work = Path.home() / '.cache/gigastt-small-quantization' / args.candidate / 'screen'
    work.mkdir(parents=True, exist_ok=True)
    manifest = []
    for r in samples:
        assert digest(r['feature_path']) == r['feature_sha256']
        manifest.append({'id': r['id'] + '_' + r['condition'], 'input': r['feature_path'], 'mode': 'mel'})
    assert len({r['id'] for r in manifest}) == 111
    save(work / 'manifest.json', manifest)
    probe = Path.home() / '.cache/gigastt-small-diagnosis/bin/native_ctc_probe_5c2f64d2cb09c50e'
    assert digest(probe) == '5c2f64d2cb09c50eb978e8161d04151dcd3c8775e30c3d975104ea18554866d3'
    metadata = {'candidate': args.candidate, 'model_sha256': model['sha256'], 'protocol_sha256': digest(protocol), 'prepared_sha256': digest(prepared), 'probe_sha256': digest(probe), 'script_sha256': digest(__file__)}
    save(work / 'identity.json', metadata)
    with (work / 'native.log').open('w') as log:
        subprocess.run([str(probe), model['path'], str(work / 'manifest.json'), str(work / 'native'), 'production', '3'], stdout=log, stderr=subprocess.STDOUT, check=True)
    vocab = [line.rsplit(' ', 1)[0] for line in (Path.home() / '.gigastt/models/multilingual_vocab.txt').read_text().splitlines()]
    rows = []
    groups = {}
    new_empty, new_catastrophic = [], []
    for sample, entry in zip(samples, manifest):
        out = work / 'native' / entry['id']
        meta = json.loads(out.with_suffix('.json').read_text())
        assert digest(out.with_suffix('.features.f32')) == sample['feature_sha256']
        assert meta['feature_length'] == sample['frames']
        logits = np.fromfile(out.with_suffix('.logits.f32'), dtype='<f4').reshape(meta['logit_shape'])
        assert np.isfinite(logits).all()
        length = meta['output_lengths'][0]
        assert 0 < length <= logits.shape[1]
        ids = logits[0, :length].argmax(-1)
        text = ' '.join(''.join(vocab[i] for i, _ in itertools.groupby(ids) if i != 70).replace('▁', ' ').split())
        ref = normalize(sample['reference'])
        score = jiwer.process_words(' '.join(ref), ' '.join(normalize(text)))
        row = {k: sample[k] for k in ['id', 'language', 'selection', 'condition', 'feature_sha256']}
        row.update(reference_words=len(ref), word_errors=score.substitutions+score.deletions+score.insertions, deletions=score.deletions, substitutions=score.substitutions, insertions=score.insertions, empty=not bool(text), hypothesis=text, reference=sample['reference'], baseline_scores=sample['baseline_scores'], ort_build_info=meta['ort_build_info'])
        row['new_empty'] = row['empty'] and bool(sample['baseline_hypothesis'].strip())
        row['new_catastrophic'] = 5*row['deletions'] >= 4*len(ref) and 5*sample['baseline_scores']['deletions'] < 4*len(ref)
        if row['new_empty']: new_empty.append(entry['id'])
        if row['new_catastrophic']: new_catastrophic.append(entry['id'])
        rows.append(row)
        g = groups.setdefault(sample['selection']+'/'+sample['condition'], {k: 0 for k in ['n', 'reference_words', 'word_errors', 'deletions', 'empty']})
        g['n'] += 1
        for k in ['reference_words', 'word_errors', 'deletions', 'empty']: g[k] += row[k]
    save(work / 'private_results.json', {'metadata': metadata, 'groups': groups, 'rows': rows})
    public = [{k:v for k,v in r.items() if k not in ['reference','hypothesis']} for r in rows]
    save(report / (args.candidate+'_screen.json'), {'metadata': metadata, 'complete': True, 'groups': groups, 'new_empty': new_empty, 'new_catastrophic': new_catastrophic, 'rows': public})
    print(json.dumps({'candidate': args.candidate, 'groups': groups, 'new_empty': new_empty, 'new_catastrophic': new_catastrophic}, indent=2), flush=True)


if __name__ == '__main__':
    main()
