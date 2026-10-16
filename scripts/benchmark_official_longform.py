#!/usr/bin/env python3
"""Evaluate unmodified official long-form ASR on private telephone manifests.

Requires authorized segmentation-model access configured in the environment.
Never pass credentials in arguments or save them to artifacts. Full reference
and hypothesis text stays under --work; public output is aggregate metadata.
"""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import time

from wer_unicode import normalize, word_edit_distance


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phone-manifest', type=Path, action='append', required=True)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--work', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    import torch
    import gigaam
    from huggingface_hub import snapshot_download
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    args.work.mkdir(parents=True, exist_ok=True)
    samples = [r for path in args.phone_manifest for r in json.loads(path.read_text())['samples']]
    for row in samples:
        if digest(row['path']) != row['sha256']:
            raise ValueError('Telephone input checksum mismatch')
    model = gigaam.load_model('multilingual_ctc', device='cpu', fp16_encoder=False, download_root=str(args.checkpoint.parent))
    metadata = {'checkpoint_sha256': digest(args.checkpoint), 'script_sha256': digest(__file__),
                'model': 'multilingual_ctc', 'precision': 'original FP32 CPU', 'threads': 2,
                'pipeline': 'Unmodified GigaAM transcribe_longform, official default Pyannote VAD/segmentation; fr_batch_size=1',
                'packages': {name: importlib.metadata.version(name) for name in ['gigaam', 'torch', 'torchaudio', 'pyannote.audio', 'torchcodec']}}
    rows = []
    for row in samples:
        start = time.monotonic()
        result = model.transcribe_longform(row['path'], fr_batch_size=1)
        text = ' '.join(segment.text for segment in result.segments)
        ref = normalize(row['reference'])
        rows.append({'id': row['id'], 'sha256': row['sha256'], 'reference': row['reference'], 'hypothesis': text,
                     'ref_words': len(ref), 'errors': word_edit_distance(ref, normalize(text)),
                     'elapsed_s': time.monotonic() - start,
                     'segments': [{'start': s.start, 'end': s.end} for s in result.segments]})
        metadata['segmentation_snapshot'] = Path(snapshot_download('pyannote/segmentation-3.0', local_files_only=True)).name
        local = Path(snapshot_download('pyannote/segmentation-3.0', local_files_only=True))
        metadata['segmentation_files_sha256'] = {str(path.relative_to(local)): digest(path) for path in sorted(local.rglob('*')) if path.is_file()}
        public = [{k: v for k, v in r.items() if k not in ['reference', 'hypothesis']} for r in rows]
        for r in public:
            r['wer_percent'] = 100 * r['errors'] / r['ref_words']
        complete = len(rows) == len(samples)
        (args.work / 'official_longform_private.json').write_text(json.dumps({'metadata': metadata, 'completed': complete, 'rows': rows}, ensure_ascii=False, indent=2) + '\n')
        args.output.write_text(json.dumps({'metadata': metadata, 'completed': complete, 'rows': public}, indent=2) + '\n')
        print(f"{row['id']}: {rows[-1]['errors']}/{len(ref)} word errors, {len(result.segments)} automatic segments", flush=True)


if __name__ == '__main__':
    main()
