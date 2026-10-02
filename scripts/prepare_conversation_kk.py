#!/usr/bin/env python3
"""Restore all31 pinned, human-annotated Kazakh interview/stand-up clips."""
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
import urllib.request
import wave

REVISION = '6575c5cd9e9e481edcb4d9928fda2b2f60d3e5ad'
SOURCE = 'https://github.com/Tim2190/Kaz-ASR-codeswitch-benchmark'


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    cache = Path.home() / '.cache/gigastt-conversation/kk/source'
    output = Path('benchmark/results/multilingual_conversation_20261001')
    old = json.loads(Path('benchmark/results/multilingual_public_20261001/kazakh_manifest.json').read_text())
    expected = {row['id']: row for row in old['samples']}
    base = f'https://raw.githubusercontent.com/Tim2190/Kaz-ASR-codeswitch-benchmark/{REVISION}/'
    for name in ['metadata.csv', 'annotation_methodology.md', 'README.md', 'LICENSE',
                 *('audio/' + key + '.wav' for key in expected)]:
        path = cache / name
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            temporary = path.with_suffix(path.suffix + '.partial')
            with urllib.request.urlopen(base + name, timeout=90) as src:
                temporary.write_bytes(src.read())
            temporary.replace(path)
    samples = []
    for row in csv.DictReader((cache / 'metadata.csv').open()):
        key = row['audio_id']
        path = cache / 'audio' / (key + '.wav')
        if digest(path) != expected[key]['sha256']:
            raise ValueError('Pinned audio differs: ' + key)
        raw = row['transcript_verbatim']
        secondary = row['transcript_normalized_written']
        if raw != expected[key]['reference'] or secondary != expected[key]['reference_normalized']:
            raise ValueError('Pinned reference differs: ' + key)
        if '[unclear]' in raw:
            raise ValueError('Unanticipated unclear span requires an explicit scoring decision')
        with wave.open(str(path)) as audio:
            duration = audio.getnframes() / audio.getframerate()
            format_info = {'sample_rate': audio.getframerate(), 'channels': audio.getnchannels(),
                           'sample_width_bytes': audio.getsampwidth()}
        samples.append({'id': key, 'path': str(path), 'sha256': digest(path), 'language': 'kk',
                        'dataset': 'kazakh_russian_natural', 'split': 'labelled_all',
                        'reference': raw.replace('[false_start]', ''), 'reference_verbatim_raw': raw,
                        'reference_normalized': secondary, 'source': row['audio_source_link'],
                        'speech_type': 'standup' if 'd0R8AHlugCE' in row['audio_source_link'] else 'interview',
                        'condition': 'original', 'duration_s': duration, **format_info,
                        'flags': {k: row[k] for k in row if k.startswith('has_')}})
    if len(samples) != 31 or len({s['id'] for s in samples}) != 31:
        raise ValueError('Expected all31 unique publisher clips')
    metadata = {'source': SOURCE, 'revision': REVISION, 'annotation_author': 'Timur Seidalin',
                'license': 'Annotations and code MIT; source audio CC BY per publisher attribution.',
                'selection': 'All31 publisher clips in metadata order, no output-based selection or new segmentation.',
                'primary_reference': 'Publisher transcript_verbatim with literal [false_start] markup removed; spoken fragments retained. No [unclear] occurs.',
                'secondary_reference': 'Publisher normalized_written, scored separately; never select best/min reference.',
                'human_reference_evidence': [SOURCE + '/blob/' + REVISION + '/annotation_methodology.md#7-annotation-quality-control',
                                             SOURCE + '/blob/' + REVISION + '/README.md#author--contributions'],
                'audio_handling': 'Original publisher WAV bytes; no external resampling, channel editing, denoising or VAD.',
                'limitation': 'Only two source videos: performance stand-up and interview; not representative everyday dyadic conversation. Previously evaluated corpus, not blind holdout.',
                'n': len(samples), 'duration_s': sum(s['duration_s'] for s in samples),
                'hours': sum(s['duration_s'] for s in samples) / 3600,
                'speech_type_counts': dict(Counter(s['speech_type'] for s in samples)),
                'source_counts': dict(Counter(s['source'] for s in samples)),
                'russian_insertion_flag_count': sum(s['flags']['has_barbarisms'] == '1' for s in samples),
                'false_start_markup_removed_count': sum(s['reference_verbatim_raw'].count('[false_start]') for s in samples),
                'source_file_sha256': {name: digest(cache / name) for name in ['metadata.csv', 'annotation_methodology.md', 'README.md', 'LICENSE']},
                'preparation_script_sha256': digest(Path(__file__))}
    output.mkdir(parents=True, exist_ok=True)
    for name, value in [('kk_manifest.json', {**metadata, 'samples': samples}), ('kk_provenance.json', metadata)]:
        path = output / name
        text = json.dumps(value, ensure_ascii=False, indent=2) + '\n'
        if path.exists() and path.read_text() != text:
            raise FileExistsError('Refusing to replace frozen manifest: ' + str(path))
        path.write_text(text)
    print(json.dumps({'n': len(samples), 'duration_s': metadata['duration_s'], 'hours': metadata['hours'],
                      'manifest_sha256': digest(output / 'kk_manifest.json')}, indent=2))


if __name__ == '__main__':
    main()
