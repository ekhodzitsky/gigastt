#!/usr/bin/env python3
"""Run the official short API on a fixed matched multilingual telephone subset.

Keep private results outside the repository: telephone reference text and ASR
hypotheses must not be redistributed. Public output contains scores and hashes.
The optional 20-second call control is not the official long-form pipeline.
"""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import time

from wer_unicode import normalize, word_edit_distance


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.partial')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline', type=Path, required=True)
    p.add_argument('--degraded', type=Path, required=True)
    p.add_argument('--phone-manifest', type=Path, action='append', default=[])
    p.add_argument('--work', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--threads', type=int, default=2)
    args = p.parse_args()
    import torch
    import gigaam
    import soundfile as sf
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    args.work.mkdir(parents=True, exist_ok=True)
    rows = []
    for language in ['ru', 'en', 'kk', 'ky', 'uz']:
        original = json.loads((args.baseline / f'{language}_manifest.json').read_text())['samples'][:10]
        for condition in ['original', 'alaw', 'mulaw']:
            candidates = original if condition == 'original' else json.loads((args.degraded / f'{language}_{condition}_manifest.json').read_text())['samples']
            by_id = {x['id']: x for x in candidates}
            for ref in original:
                row = by_id[ref['id']]
                if row['reference'] != ref['reference']:
                    raise ValueError('Reference mismatch')
                rows.append({**row, 'language': language, 'condition': condition})
    phones = [row for path in args.phone_manifest for row in json.loads(path.read_text())['samples']]
    for row in rows + phones:
        if digest(row['path']) != row['sha256']:
            raise ValueError(f"Audio checksum mismatch: {row['id']}")
    metadata = {
        'source_revision': subprocess.check_output(['git', '-C', str(args.source), 'rev-parse', 'HEAD'], text=True).strip(),
        'checkpoint_sha256': digest(args.checkpoint), 'model': 'multilingual_ctc',
        'precision': 'original checkpoint FP32 CPU', 'threads': args.threads,
        'packages': {name: importlib.metadata.version(name) for name in ['torch', 'torchaudio', 'gigaam', 'numpy', 'soundfile']},
        'selection': 'First ten manifest entries per language, unchanged across original/A-law/mu-law; no replacement or training',
        'ffmpeg': subprocess.check_output(['ffmpeg', '-version'], text=True).splitlines()[0],
        'script_sha256': digest(__file__),
    }
    model = gigaam.load_model('multilingual_ctc', device='cpu', fp16_encoder=False, download_root=str(args.checkpoint.parent))
    results, attempts, excluded = [], [], []

    def finish():
        groups = {}
        for row in results:
            key = row['language'] + '/' + row['condition']
            g = groups.setdefault(key, {'n': 0, 'errors': 0, 'ref_words': 0, 'empty': 0})
            g['n'] += 1
            g['errors'] += row['errors']
            g['ref_words'] += row['ref_words']
            g['empty'] += not bool(row['hypothesis'].strip())
        for g in groups.values():
            g['wer_percent'] = 100 * g['errors'] / g['ref_words']
        common = {'metadata': metadata, 'groups': groups, 'official_longform_attempts': attempts, 'excluded_short_inputs': excluded}
        save(args.work / 'results_private.json', {**common, 'rows': results})
        public = [{k: v for k, v in row.items() if k not in ['reference', 'hypothesis', 'path']} for row in results]
        save(args.output / 'source.json', {**common, 'rows': public})

    for row in phones:
        try:
            model.transcribe_longform(row['path'], fr_batch_size=1)
            attempts.append({'id': row['id'], 'status': 'completed'})
        except (ImportError, RuntimeError, OSError) as error:
            attempts.append({'id': row['id'], 'status': 'blocked', 'error_type': type(error).__name__, 'error': str(error)})
    finish()
    for row in rows + [{**r, 'language': 'kk_real_call', 'condition': 'automatic_nonoverlapping20s_short_api_control'} for r in phones]:
        start = time.monotonic()
        info = sf.info(row['path'])
        if row['language'] == 'kk_real_call':
            texts = []
            for index, offset in enumerate(range(0, info.frames, 20 * info.samplerate)):
                path = args.work / f"{row['id']}_{index}.wav"
                audio, rate = sf.read(row['path'], start=offset, stop=min(offset + 20 * info.samplerate, info.frames), dtype='int16')
                sf.write(path, audio, rate, subtype='PCM_16')
                texts.append(model.transcribe(str(path)).text)
            text = ' '.join(texts)
        else:
            # Never bypass or replace the official 25-second short API guard.
            try:
                text = model.transcribe(row['path']).text
            except ValueError as error:
                if 'Too long wav file' not in str(error):
                    raise
                excluded.append({'id': row['id'], 'language': row['language'], 'condition': row['condition'], 'duration_s': info.duration, 'sha256': row['sha256'], 'reason': str(error)})
                finish()
                continue
        reference = normalize(row['reference'])
        out = {k: row[k] for k in ['id', 'language', 'condition', 'reference', 'path', 'sha256']}
        out.update(hypothesis=text, ref_words=len(reference), errors=word_edit_distance(reference, normalize(text)), duration_s=info.duration, elapsed_s=time.monotonic()-start)
        results.append(out)
        finish()
        print(f"{len(results)}/{len(rows)+len(phones)} {row['language']} {row['condition']} {out['errors']}/{len(reference)}", flush=True)


if __name__ == '__main__':
    main()
