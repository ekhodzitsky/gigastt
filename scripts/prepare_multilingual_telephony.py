#!/usr/bin/env python3
"""Prepare paired G.711 FLEURS conditions from the fixed 1000/language baseline.

Requires ffmpeg and soundfile. Source and output hashes are checked on resume;
libsoundfile independently decodes every prepared WAV before manifest publication.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import subprocess

import soundfile as sf

FILTER = 'highpass=f=300,lowpass=f=3400'
CODECS = {'alaw': ('pcm_alaw', 'ALAW'), 'mulaw': ('pcm_mulaw', 'ULAW')}


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def atomic_json(path, value):
    temporary = path.with_name(path.name + '.partial')
    with temporary.open('w') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def verify_audio(path, codec, expected_seconds):
    info = sf.info(path)
    if (info.samplerate, info.channels, info.subtype, info.format) != (8000, 1, CODECS[codec][1], 'WAV'):
        raise ValueError(f'Wrong audio format: {path}: {info}')
    samples, rate = sf.read(path, dtype='int16')
    if len(samples) != info.frames or rate != 8000 or abs(info.duration - expected_seconds) > 1 / 8000:
        raise ValueError(f'Audio duration/decode mismatch: {path}')
    return info.frames, info.duration


def prepare(sample, codec, cache, version):
    source = Path(sample['path'])
    actual_source_hash = digest(source)
    if actual_source_hash != sample['sha256']:
        raise ValueError(f'Source audio changed: {source}')
    destination = cache / sample['language'] / codec / (sample['id'] + '.wav')
    destination.parent.mkdir(parents=True, exist_ok=True)
    sidecar = destination.with_suffix('.json')
    identity = {'source_audio_sha256': actual_source_hash, 'ffmpeg_version': version,
                'filter': FILTER, 'sample_rate': 8000, 'channels': 1, 'codec': CODECS[codec][0]}
    previous = json.loads(sidecar.read_text()) if sidecar.exists() else {}
    reusable = (destination.exists() and previous.get('identity') == identity
                and previous.get('sha256') == digest(destination))
    if not reusable:
        temporary = destination.with_name(destination.stem + '.partial.wav')
        command = ['ffmpeg', '-nostdin', '-v', 'error', '-y', '-threads', '1', '-i', str(source),
                   '-map_metadata', '-1', '-ac', '1', '-af', FILTER, '-ar', '8000',
                   '-c:a', CODECS[codec][0], '-threads', '1', str(temporary)]
        subprocess.run(command, check=True, capture_output=True)
        verify_audio(temporary, codec, sample['duration_s'])
        temporary.replace(destination)
        atomic_json(sidecar, {'identity': identity, 'sha256': digest(destination)})
    frames, seconds = verify_audio(destination, codec, sample['duration_s'])
    return {**sample, 'dataset': sample['dataset'] + '_' + codec,
            'condition': 'simulated_telephone_' + codec, 'path': str(destination),
            'source_path': str(source), 'source_audio_sha256': actual_source_hash,
            'sha256': digest(destination), 'duration_s': seconds, 'num_samples': frames,
            'source_num_samples': sample['num_samples'], 'original_sample_rate': sample['source_sample_rate'],
            'source_sample_rate': 8000, 'codec': CODECS[codec][0]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, default=Path('benchmark/results/multilingual_1000_20261001'))
    parser.add_argument('--output', type=Path, default=Path('benchmark/results/multilingual_telephony_20261001'))
    parser.add_argument('--cache', type=Path, default=Path('~/.cache/gigastt-multilingual-telephony'))
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--languages', nargs='+', default=['ru', 'en', 'kk', 'ky', 'uz'])
    args = parser.parse_args()
    if not 1 <= args.workers <= 8 or len(set(args.languages)) != len(args.languages):
        raise ValueError('Use 1..8 workers and distinct languages')
    args.cache = args.cache.expanduser().resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    version = subprocess.check_output(['ffmpeg', '-version'], text=True).splitlines()[0]
    provenance = {'ffmpeg_version': version, 'soundfile_version': sf.__version__,
                  'libsndfile_version': sf.__libsndfile_version__,
                  'transform': '300 Hz high-pass then 3400 Hz low-pass; mono; resample 8000 Hz; G.711 WAV',
                  'command_template': 'ffmpeg -nostdin -v error -y -threads 1 -i SOURCE -map_metadata -1 -ac 1 -af highpass=f=300,lowpass=f=3400 -ar 8000 -c:a pcm_{alaw,mulaw} -threads 1 OUTPUT.wav',
                  'normalization': 'No gain, loudness normalization, denoising, or clipping normalization.',
                  'verification': 'Every source SHA256, output SHA256, WAV format/subtype/channels/rate/frames, full libsndfile PCM16 decode, duration within one 8000 Hz sample; IDs/order/reference/split preserved.',
                  'limitations': 'Simulated telephone degradation of read speech, not real calls. Bandpass filters are FFmpeg defaults (not brick-wall). Original baseline sampling and exposure limitations apply.',
                  'manifests': []}
    for language in args.languages:
        source_manifest = args.baseline / f'{language}_manifest.json'
        baseline = json.loads(source_manifest.read_text())
        samples = baseline['samples']
        if len(samples) != 1000 or len({s['id'] for s in samples}) != 1000 or any(s['language'] != language for s in samples):
            raise ValueError(f'Invalid baseline: {source_manifest}')
        for codec in CODECS:
            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                rows = list(pool.map(lambda sample: prepare(sample, codec, args.cache, version), samples))
            for original, row in zip(samples, rows):
                for field in ('id', 'reference', 'language', 'split', 'row_index', 'sentence_id'):
                    if row[field] != original[field]:
                        raise ValueError(f'Pairing changed: {field}')
            output = args.output / f'{language}_{codec}_manifest.json'
            manifest = {**baseline, 'selection': 'Exactly the baseline manifest samples and order; no exclusions or reselection.',
                        'baseline_manifest': str(source_manifest.resolve()),
                        'baseline_manifest_sha256': digest(source_manifest),
                        'condition': 'simulated_telephone_' + codec, 'codec': CODECS[codec][0],
                        'sample_rate': 8000, 'limitations': provenance['limitations'], 'preparation': {k: v for k, v in provenance.items() if k != 'manifests'},
                        'audio_seconds': sum(r['duration_s'] for r in rows), 'samples': rows}
            atomic_json(output, manifest)
            provenance['manifests'].append({'path': output.name, 'sha256': digest(output), 'language': language,
                                            'codec': codec, 'count': len(rows), 'audio_seconds': manifest['audio_seconds']})
            atomic_json(args.output / 'telephony_dataset_provenance.json', provenance)
            print(f'Published {output}: {len(rows)} verified pairs', flush=True)

    provenance['complete'] = True
    provenance['total_prepared_recordings'] = sum(m['count'] for m in provenance['manifests'])
    provenance['unique_baseline_recordings'] = 1000 * len(args.languages)
    atomic_json(args.output / 'telephony_dataset_provenance.json', provenance)


if __name__ == '__main__':
    main()
