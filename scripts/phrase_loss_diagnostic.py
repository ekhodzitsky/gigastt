#!/usr/bin/env python3
"""Local CPU phrase-loss diagnostic. No upload; output contains PRIVATE transcripts.

Requires a CPU gigastt binary and mono 16 kHz PCM16 WAV. Uses the standard
24-second / 2-second-overlap file path, without VAD or postprocessing.
Only interior windows are supported. Comparisons are not ground-truth scoring.
"""
import argparse
from difflib import SequenceMatcher
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import time
import wave


RATE = 16000


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def save(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def window_starts(positions, frames):
    if not 1 <= len(positions) <= 8 or any(not math.isfinite(t) or t < 0 for t in positions):
        raise ValueError('provide 1..8 finite nonnegative timestamps')
    starts = sorted({int(t // 22) * 22 for t in positions})
    # Deliberately reject the adjusted/absorbed tail, rather than approximate it.
    if any((start + 52) * RATE >= frames for start in starts):
        raise ValueError('selected window is too close to the adjusted file tail; use an interior position')
    return starts


def extract(source, output, positions):
    with wave.open(str(source), 'rb') as src:
        if (src.getnchannels(), src.getframerate(), src.getsampwidth(), src.getcomptype()) != (1, RATE, 2, 'NONE'):
            raise ValueError('expected mono 16 kHz uncompressed PCM16 WAV; no resampling is performed')
        starts = window_starts(positions, src.getnframes())
        output.mkdir(parents=True, exist_ok=False)
        rows = []
        for start in starts:
            src.setpos(start * RATE)
            pcm = src.readframes(24 * RATE)
            if len(pcm) != 24 * RATE * 2:
                raise ValueError('truncated source PCM')
            name = f'window-{start}-{start+24}.wav'
            with wave.open(str(output/name), 'wb') as dst:
                dst.setparams((1, 2, RATE, 0, 'NONE', 'not compressed'))
                dst.writeframes(pcm)
            rows.append({'file': name, 'start': start, 'end': start + 24,
                         'pcm_sha256': hashlib.sha256(pcm).hexdigest()})
    return rows


def validate_words(response):
    if not isinstance(response, dict):
        raise ValueError('response must be a JSON object')
    words = response.get('words')
    if not isinstance(words, list):
        raise ValueError('response must contain a words array')
    previous = -1.
    for word in words:
        if not isinstance(word, dict) or not isinstance(word.get('word'), str):
            raise ValueError('invalid word record')
        start, end = word.get('start'), word.get('end')
        if any(not isinstance(x, (int, float)) or isinstance(x, bool) or not math.isfinite(x) for x in (start, end)):
            raise ValueError('word timestamps must be finite numbers')
        if not 0 <= start <= end or start < previous:
            raise ValueError('word timestamps must be ordered and nonnegative')
        previous = start
    return words


def tokens(words):
    # Diagnostic normalization only: no number expansion or semantic equivalence.
    return [token for word in words for token in re.findall(r'\w+', word['word'].casefold())]


def compare(full, local, offset):
    shifted = [{**w, 'start': w['start'] + offset, 'end': w['end'] + offset} for w in local]
    # Exclude the overlap edges from alignment; raw output retains every word.
    left, right = offset + 1, offset + 23
    a = [w for w in shifted if left <= w['start'] < right]
    b = [w for w in full if left <= w['start'] < right]
    at, bt = tokens(a), tokens(b)
    opcodes = [{'operation': tag, 'local_tokens': at[i:j], 'full_tokens': bt[k:l]}
               for tag, i, j, k, l in SequenceMatcher(None, at, bt, autojunk=False).get_opcodes()
               if tag != 'equal']
    # Coverage ignores identity, intentionally separate from the text comparison.
    uncovered = [w['start'] for w in a if not any(abs(w['start'] - v['start']) <= .5 for v in full)]
    return {'interval': [left, right], 'local_tokens': at, 'full_tokens': bt,
            'text_opcodes': opcodes, 'local_starts_without_full_start': uncovered}


def clean_environment():
    # Clap env defaults must not silently enable VAD, hotwords or other settings.
    return {k: v for k, v in os.environ.items() if not k.startswith('GIGASTT_') and k != 'RUST_LOG'}


def verify_execution(log, cached):
    log = re.sub(r'\x1b\[[0-9;]*m', '', log)
    if 'execution provider bound execution_provider="cpu"' not in log or 'falling back to CPU execution provider' in log:
        raise ValueError('CPU execution not confirmed; use a current CPU-only binary and inspect the log')
    marker = ('loaded fresh optimized graph cache' if cached else
              'loading source model and refreshing optimized graph cache')
    if marker not in log:
        raise ValueError('expected graph-cache path not confirmed; inspect the log before comparing')


def transcribe(binary, audio, models, variant, threads, destination, timeout, environment):
    command = [str(binary), '--offline', '--log-level', 'info', 'transcribe', str(audio),
               '--model-dir', str(models), '--model-variant', variant,
               '--encoder-intra-threads', str(threads), '--file-window-concurrency', '1',
               '--punctuation', 'off', '--itn', 'off', '--format', 'json', '--output', str(destination)]
    started = time.monotonic()
    with destination.with_suffix('.log').open('w') as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, env=environment,
                       check=True, timeout=timeout)
    response = json.loads(destination.read_text())
    validate_words(response)
    return response, {'command': command, 'wall_seconds': time.monotonic() - started}


def run(args):
    binary_name = shutil.which(str(args.binary))
    if binary_name is None:
        raise ValueError('gigastt binary not found')
    binary = Path(binary_name).resolve()
    source, models, output = args.audio.resolve(), args.model_dir.resolve(), args.output.resolve()
    prefix = 'v3_e2e_rnnt' if args.variant == 'e2e_rnnt' else 'v3_rnnt'
    names = [f'{prefix}_encoder_int8.onnx', f'{prefix}_decoder.onnx', f'{prefix}_joint.onnx',
             'v3_e2e_rnnt_vocab.txt' if args.variant == 'e2e_rnnt' else 'v3_vocab.txt']
    for name in names:
        if not (models/name).is_file():
            raise ValueError(f'missing model file: {name}')
    if output.exists():
        raise ValueError('output already exists; choose a new directory for each run')
    # Validate before creating the output directory.
    with wave.open(str(source), 'rb') as src:
        if (src.getnchannels(), src.getframerate(), src.getsampwidth(), src.getcomptype()) != (1, RATE, 2, 'NONE'):
            raise ValueError('expected mono 16 kHz uncompressed PCM16 WAV')
        window_starts(args.at, src.getnframes())
    output.mkdir(parents=True, mode=0o700)
    windows = extract(source, output/'windows', args.at)
    private_models = output/'models'
    private_models.mkdir()
    for name in names:
        (private_models/name).symlink_to(models/name)
    environment = clean_environment()
    provenance = {'schema_version': 1, 'platform': platform.platform(), 'machine': platform.machine(),
                  'logical_cpus': os.cpu_count(), 'encoder_threads_requested': args.threads,
                  'binary_sha256': digest(binary), 'audio_sha256': digest(source),
                  'binary_version': subprocess.check_output([str(binary), '--version'], env=environment,
                                                            text=True, timeout=10).strip(),
                  'model_sha256': {name: digest(models/name) for name in names},
                  'variant': args.variant, 'positions': args.at, 'windows': windows, 'runs': []}
    save(output/'provenance.json', provenance)
    responses = {}
    # Same isolated cache, cold full-file then warm full-file; compare warm windows
    # to both rather than attributing any cold/warm change to stitching.
    cases = [('full-cold', source), ('full-warm', source)]
    cases += [(Path(w['file']).stem, output/'windows'/w['file']) for w in windows]
    for name, audio in cases:
        print(f'Running {name} (private results stay in the output directory)', flush=True)
        response, record = transcribe(binary, audio, private_models, args.variant, args.threads,
                                     output/f'{name}.json', args.timeout, environment)
        verify_execution((output/f'{name}.log').read_text(), cached=name != 'full-cold')
        responses[name] = response
        provenance['runs'].append({'name': name, **record})
        save(output/'provenance.json', provenance)
    if digest(source) != provenance['audio_sha256'] or digest(binary) != provenance['binary_sha256']:
        raise ValueError('source audio or binary changed during the run')
    if any(digest(models/n) != sha for n, sha in provenance['model_sha256'].items()):
        raise ValueError('model files changed during the run')
    comparisons = []
    for w in windows:
        local = responses[Path(w['file']).stem]['words']
        comparisons.append({'start': w['start'], 'end': w['end'], 'comparisons': {
            arm: compare(responses[arm]['words'], local, w['start']) for arm in ('full-cold', 'full-warm')}})
    save(output/'comparison-private.json', comparisons)
    summary = {k: provenance[k] for k in ('schema_version', 'machine', 'encoder_threads_requested',
                                         'binary_version', 'binary_sha256', 'model_sha256', 'variant')}
    summary.update({'complete': True, 'ground_truth_available': False,
                    'cold_warm_response_equal': responses['full-cold'] == responses['full-warm'],
                    'full_word_counts': {arm: len(responses[arm]['words']) for arm in ('full-cold', 'full-warm')},
                    'windows': [{'start': c['start'], 'comparisons': {
                        arm: {'text_equal': not r['text_opcodes'],
                              'local_token_count': len(r['local_tokens']),
                              'full_token_count': len(r['full_tokens']),
                              'timestamp_uncovered_count': len(r['local_starts_without_full_start'])}
                        for arm, r in c['comparisons'].items()}} for c in comparisons]})
    save(output/'summary.json', summary)
    print('Complete. Review summary.json before sharing; all other outputs may contain private content.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audio', type=Path, required=True)
    parser.add_argument('--binary', type=Path, required=True)
    parser.add_argument('--model-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--at', type=float, nargs='+', required=True)
    parser.add_argument('--variant', choices=['rnnt', 'e2e_rnnt'], default='e2e_rnnt')
    parser.add_argument('--threads', type=int, default=6)
    parser.add_argument('--timeout', type=float, default=300)
    args = parser.parse_args()
    if not 1 <= args.threads <= 64 or not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error('threads must be 1..64 and timeout must be finite and positive')
    try:
        run(args)
    except (OSError, ValueError, wave.Error, subprocess.SubprocessError) as error:
        parser.exit(1, f'Diagnostic failed: {error}\nPartial output is not a completed report.\n')


if __name__ == '__main__':
    main()
