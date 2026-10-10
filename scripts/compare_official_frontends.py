#!/usr/bin/env python3
"""Fixed 15-input PCM/frontend/precision diagnostic; never changes inference.

Three chains isolate PCM and mel changes. Source, FP32 ONNX and INT8 consume
the exact same saved arrays. See frontend_protocol.md for the fixed protocol.
"""
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import subprocess

from wer_unicode import normalize, word_edit_distance


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline', type=Path, required=True)
    p.add_argument('--degraded', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--fp32', type=Path, required=True)
    p.add_argument('--int8', type=Path, required=True)
    p.add_argument('--inspect-audio', type=Path, required=True)
    p.add_argument('--work', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    import numpy as np
    import torch
    import gigaam
    import onnxruntime as ort
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    args.work.mkdir(parents=True, exist_ok=True)
    model = gigaam.load_model('multilingual_ctc', device='cpu', fp16_encoder=False, download_root=str(args.checkpoint.parent))
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    options.inter_op_num_threads = 1
    sessions = {name: ort.InferenceSession(str(path), options, providers=['CPUExecutionProvider']) for name, path in [('fp32', args.fp32), ('int8', args.int8)]}
    vocab = list(model.decoding.tokenizer.vocab)
    metadata = {
        'protocol': 'frontend_protocol.md: first one entry per language, three conditions, three chains, three precisions',
        'checkpoint_sha256': digest(args.checkpoint), 'fp32_sha256': digest(args.fp32), 'int8_sha256': digest(args.int8),
        'inspect_audio_sha256': digest(args.inspect_audio), 'script_sha256': digest(__file__),
        'rust_source_sha256': {name: digest(name) for name in ['crates/gigastt-core/src/inference/audio/resample.rs', 'crates/gigastt-core/src/inference/features.rs', 'crates/gigastt-core/examples/inspect_audio.rs']},
        'torch': torch.__version__, 'onnxruntime': ort.__version__, 'threads': 2,
        'ffmpeg': subprocess.check_output(['ffmpeg', '-version'], text=True).splitlines()[0],
    }
    scores, numeric = [], []

    def difference(a, b):
        shape = tuple(min(x, y) for x, y in zip(a.shape, b.shape))
        slices = tuple(slice(0, n) for n in shape)
        delta = np.abs(a[slices] - b[slices])
        return {'left_shape': list(a.shape), 'right_shape': list(b.shape), 'aligned_shape': list(shape), 'mean_absolute_difference': float(delta.mean()), 'max_absolute_difference': float(delta.max()), 'identical_aligned_fraction': float((delta == 0).mean())}

    def finish():
        groups = {}
        for r in scores:
            key = '/'.join([r['condition'], r['chain'], r['precision']])
            g = groups.setdefault(key, {'n': 0, 'errors': 0, 'ref_words': 0, 'empty': 0})
            g['n'] += 1
            g['errors'] += r['errors']
            g['ref_words'] += r['ref_words']
            g['empty'] += not bool(r['hypothesis'].strip())
        for g in groups.values():
            g['wer_percent'] = 100 * g['errors'] / g['ref_words']
        common = {'metadata': metadata, 'groups': groups, 'numerical_inputs': numeric}
        save(args.work / 'frontend_private.json', {**common, 'rows': scores})
        save(args.output, {**common, 'rows': [{k: v for k, v in r.items() if k not in ['reference', 'hypothesis']} for r in scores]})

    with torch.inference_mode():
        for language in ['ru', 'en', 'kk', 'ky', 'uz']:
            original = json.loads((args.baseline / f'{language}_manifest.json').read_text())['samples'][0]
            for condition in ['original', 'alaw', 'mulaw']:
                if condition == 'original':
                    row = original
                else:
                    rows = json.loads((args.degraded / f'{language}_{condition}_manifest.json').read_text())['samples']
                    row = next(r for r in rows if r['id'] == original['id'])
                if row['reference'] != original['reference'] or digest(row['path']) != row['sha256']:
                    raise ValueError('Input/reference mismatch')
                prefix = args.work / f"{row['id']}_{condition}"
                subprocess.run([str(args.inspect_audio.resolve()), row['path'], str(prefix)], check=True)
                rust_pcm = np.fromfile(str(prefix) + '.pcm.f32', dtype='<f4')
                rust_mel = np.fromfile(str(prefix) + '.mel.f32', dtype='<f4').reshape(1, 64, -1)
                wav, length = model.prepare_wav(row['path'])
                official_mel, official_length = model.preprocessor(wav, length)
                cross_mel, cross_length = model.preprocessor(torch.from_numpy(rust_pcm)[None], torch.tensor([len(rust_pcm)]))
                chains = [('official_pcm_official_mel', official_mel.numpy(), official_length),
                          ('rust_pcm_official_mel', cross_mel.numpy(), cross_length),
                          ('rust_pcm_rust_mel', rust_mel, torch.tensor([rust_mel.shape[2]]))]
                numeric.append({'id': row['id'], 'language': language, 'condition': condition, 'audio_sha256': row['sha256'],
                                'pcm_official_vs_rust': difference(wav.numpy().ravel(), rust_pcm),
                                'mel_official_vs_rust_pcm_official': difference(official_mel.numpy(), cross_mel.numpy()),
                                'mel_rust_pcm_official_vs_rust': difference(cross_mel.numpy(), rust_mel)})
                for chain, features, feature_length in chains:
                    feature_path = Path(str(prefix) + '_' + chain + '.npy')
                    np.save(feature_path, features)
                    encoded, elen = model.encoder(torch.from_numpy(features), feature_length)
                    source_logits = model.head(encoded).numpy()
                    source_text = model.decoding.decode(model.head, encoded, elen)[0][0]
                    values = [('source', source_logits, source_text)]
                    for precision, session in sessions.items():
                        logits = session.run(None, {'features': features, 'feature_lengths': feature_length.numpy()})[0]
                        ids = logits.argmax(-1).ravel()
                        text = ''.join(vocab[i] for i, _ in itertools.groupby(ids) if i != len(vocab)).replace('▁', ' ').strip()
                        values.append((precision, logits, text))
                    ref = normalize(row['reference'])
                    for precision, logits, text in values:
                        if logits.shape != source_logits.shape:
                            raise ValueError('Source/ONNX logit shape mismatch')
                        delta = np.abs(logits - source_logits)
                        scores.append({'id': row['id'], 'language': language, 'condition': condition,
                                       'chain': chain, 'precision': precision, 'audio_sha256': row['sha256'],
                                       'feature_sha256': digest(feature_path), 'frames': int(feature_length[0]),
                                       'reference': row['reference'], 'hypothesis': text,
                                       'ref_words': len(ref), 'errors': word_edit_distance(ref, normalize(text)),
                                       'source_logits_mae': float(delta.mean()), 'source_logits_max_error': float(delta.max()),
                                       'source_frame_argmax_agreement': float((logits.argmax(-1) == source_logits.argmax(-1)).mean()),
                                       'source_normalized_hypothesis_matches': normalize(text) == normalize(source_text)})
                    finish()
                print(language, condition, f'{len(scores)}/135', flush=True)


if __name__ == '__main__':
    main()
