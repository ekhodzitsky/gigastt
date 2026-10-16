#!/usr/bin/env python3
"""Verify original source against native FP32 on five frozen empty-output cases.

The five IDs come from the completed original benchmark, not new diagnostic
outcomes. All three conditions use identical saved product feature arrays.
"""
import argparse
import hashlib
import json
from pathlib import Path


IDS = {'fleurs_kk_kz_test_0121', 'fleurs_kk_kz_test_0162',
       'fleurs_kk_kz_test_0295', 'fleurs_kk_kz_validation_0253',
       'fleurs_ky_kg_test_0578'}


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prepared', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--native-fp32', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    import numpy as np
    import torch
    import gigaam
    torch.set_num_threads(3)
    torch.set_num_interop_threads(1)
    rows = [r for r in json.loads(args.prepared.read_text())['rows'] if r['id'] in IDS and r['variant'] == 'decoded_float']
    if len(rows) != 15 or {r['id'] for r in rows} != IDS:
        raise ValueError('Frozen control selection mismatch')
    model = gigaam.load_model('multilingual_ctc', device='cpu', fp16_encoder=False, download_root=str(args.checkpoint.parent))
    results = []
    with torch.inference_mode():
        for row in rows:
            if digest(row['feature_path']) != row['feature_sha256']:
                raise ValueError('Feature checksum mismatch')
            features = np.fromfile(row['feature_path'], dtype='<f4').reshape(1, 64, -1)
            length = torch.tensor([features.shape[-1]])
            encoded, encoded_length = model.encoder(torch.from_numpy(features), length)
            logits = model.head(encoded).numpy()
            key = row['id'] + '_' + row['condition'] + '_decoded_float'
            meta = json.loads((args.native_fp32 / (key + '.json')).read_text())
            native = np.fromfile(args.native_fp32 / (key + '.logits.f32'), dtype='<f4').reshape(meta['logit_shape'])
            if logits.shape != native.shape or int(encoded_length[0]) != meta['output_lengths'][0]:
                raise ValueError('Output shape/length mismatch')
            frames = int(encoded_length[0])
            a, b = logits[:, :frames], native[:, :frames]
            delta = np.abs(a - b)
            results.append({'id': row['id'], 'condition': row['condition'], 'feature_sha256': row['feature_sha256'],
                            'feature_frames': features.shape[-1], 'output_frames': frames,
                            'frame_argmax_identical': bool(np.array_equal(a.argmax(-1), b.argmax(-1))),
                            'frame_argmax_agreement': float((a.argmax(-1) == b.argmax(-1)).mean()),
                            'logits_max_absolute_difference': float(delta.max()),
                            'logits_mean_absolute_difference': float(delta.mean()),
                            'source_all_blank': bool((a.argmax(-1) == 70).all())})
    args.output.write_text(json.dumps({'metadata': {'selection': 'All five unique utterances with empty original telephone output, fixed before these diagnostics, times all three conditions', 'checkpoint_sha256': digest(args.checkpoint), 'prepared_sha256': digest(args.prepared), 'script_sha256': digest(__file__), 'torch': torch.__version__, 'threads': 3, 'pipeline': 'Original source encoder/head on exact saved product features; not an audio-loader or long-form test'}, 'rows': results}, indent=2) + '\n')
    print(f"Source/native FP32 identical argmax: {sum(r['frame_argmax_identical'] for r in results)}/15", flush=True)


if __name__ == '__main__':
    main()
