#!/usr/bin/env python3
"""Resumable manifest-driven gigastt evaluation, exactly 1000 samples per language.

A single-writer JSONL ledger commits each result with a newline; an interrupted
last line is discarded on resume. Use --fsync for per-result durable writes.
Failures are retained and scored as empty hypotheses; no retries or exclusions.
"""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
import fcntl
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import urllib.request
import wave

from benchmark_multilingual_public import score, sha256
from wer_unicode import normalize


SCORING = ('wer_unicode.normalize + jiwer; lowercase; apostrophe variants deleted; '
           'punctuation deleted (hyphens join words); no NFC, ITN or yo-folding; '
           'CER includes spaces between normalized tokens. Primary uses reference as supplied. '
           'Digit-free excludes any str.isdigit character in original reference; '
           'eligible_official_style additionally requires audio <=30s. This is not a model-card replication.')


def atomic_json(path, value):
    temporary = path.with_name(path.name + '.partial')
    with temporary.open('w') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def duration(path):
    try:
        with wave.open(str(path), 'rb') as audio:
            return audio.getnframes() / audio.getframerate()
    except (wave.Error, EOFError):
        result = subprocess.check_output(['ffprobe', '-v', 'error', '-show_entries', 'format=duration',
                                          '-of', 'default=noprint_wrappers=1:nokey=1', str(path)], text=True)
        return float(result.strip())


def preflight(manifest_path, expected_per_language=1000):
    manifest = json.loads(manifest_path.read_text())
    samples = manifest['samples']
    if not samples or len({s['id'] for s in samples}) != len(samples):
        raise ValueError('Sample IDs must be nonempty and globally unique')
    counts = Counter(s['language'] for s in samples)
    if any(n != expected_per_language for n in counts.values()):
        raise ValueError(f'Expected {expected_per_language} samples per language, got {dict(counts)}')
    prepared = []
    for sample in samples:
        for key in ('id', 'path', 'reference', 'dataset', 'language', 'split', 'sha256'):
            if not isinstance(sample.get(key), str) or not sample[key]:
                raise ValueError(f'Missing/non-string sample field {key}')
        if not isinstance(sample.get('condition', 'original'), str) or not sample.get('condition', 'original'):
            raise ValueError(f'Invalid condition: {sample["id"]}')
        if not normalize(sample['reference']):
            raise ValueError(f'Empty normalized reference: {sample["id"]}')
        path = Path(sample['path']).expanduser()
        if not path.is_absolute():
            path = manifest_path.resolve().parent / path
        if sha256(path) != sample['sha256']:
            raise ValueError(f'Audio checksum mismatch: {sample["id"]}')
        seconds = duration(path)
        if not math.isfinite(seconds) or seconds <= 0:
            raise ValueError(f'Invalid duration: {sample["id"]}')
        prepared.append({**sample, '_resolved_path': path, 'duration_s': seconds})
    return prepared


def summarize(details):
    result = []
    for language in sorted({d['language'] for d in details}):
        splits = sorted({d['split'] for d in details if d['language'] == language})
        for split in [None, *splits]:
            group = [d for d in details if d['language'] == language and (split is None or d['split'] == split)]
            for subset in ('full', 'digit_free', 'eligible_official_style'):
                rows = [d for d in group if subset == 'full' or
                        (not any(c.isdigit() for c in d['reference']) and
                         (subset != 'eligible_official_style' or d['duration_s'] <= 30))]
                totals = {k: sum(d['scores'][k] for d in rows) for k in
                          ('reference_words', 'word_errors', 'substitutions', 'deletions', 'insertions', 'reference_chars', 'char_errors')}
                audio_s = sum(d['duration_s'] for d in rows)
                elapsed_s = sum(d['elapsed_s'] for d in rows)
                result.append({'language': language, 'split': split, 'subset': subset, 'n': len(rows),
                               **totals, 'wer_pct': 100 * totals['word_errors'] / totals['reference_words'] if totals['reference_words'] else None,
                               'cer_pct': 100 * totals['char_errors'] / totals['reference_chars'] if totals['reference_chars'] else None,
                               'empty_hypotheses': sum(not normalize(d['hypothesis']) for d in rows),
                               'failed_requests': sum(d['status'] == 'failed' for d in rows),
                               'audio_s': audio_s, 'request_elapsed_s': elapsed_s,
                               'aggregate_request_rtf': elapsed_s / audio_s if audio_s else None})
    return result


def read_ledger(path, samples):
    by_id = {s['id']: s for s in samples}
    rows = {}
    if not path.exists():
        return rows
    data = path.read_bytes()
    committed = data.rfind(b'\n') + 1
    if committed != len(data):
        with path.open('r+b') as stream:
            stream.truncate(committed)
            stream.flush()
            os.fsync(stream.fileno())
        print('Recovered interrupted final JSONL line', file=sys.stderr)
    for line in data[:committed].splitlines():
        row = json.loads(line)
        sample = by_id.get(row['id'])
        if sample is None or row['id'] in rows:
            raise ValueError('Ledger contains an unknown or duplicate sample ID')
        for key in ('sha256', 'reference', 'language', 'split', 'dataset', 'duration_s'):
            if row.get(key) != sample[key]:
                raise ValueError(f'Ledger sample identity mismatch: {row["id"]}/{key}')
        if row.get('condition') != sample.get('condition', 'original'):
            raise ValueError(f'Ledger sample identity mismatch: {row["id"]}/condition')
        if row['status'] not in ('ok', 'failed') or row['scores'] != score(row['reference'], row['hypothesis']):
            raise ValueError(f'Invalid persisted result: {row["id"]}')
        if row['status'] == 'failed' and row['hypothesis']:
            raise ValueError('A failed request must be scored as empty')
        rows[row['id']] = row
    return rows


def run(args, expected_per_language=1000):
    args.output.parent.mkdir(parents=True, exist_ok=True)
    lock_path = args.output.with_suffix('.lock')
    with lock_path.open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return run_locked(args, expected_per_language)


def run_locked(args, expected_per_language):
    samples = preflight(args.manifest, expected_per_language)
    with urllib.request.urlopen(args.url.rstrip('/') + '/health', timeout=args.timeout) as response:
        health = json.load(response)
    if health.get('variant') != args.variant or health.get('punctuation') is not False or health.get('itn') is not False:
        raise ValueError(f'Expected variant={args.variant}, punctuation=false, itn=false: {health}')
    with urllib.request.urlopen(args.url.rstrip('/') + '/v1/models', timeout=args.timeout) as response:
        model_info = json.load(response)
    if model_info.get('variant') != args.variant or model_info.get('punctuation') is not False or model_info.get('itn') is not False:
        raise ValueError('Model endpoint disagrees with required ASR configuration')
    scorer_paths = [Path(__file__), Path(__file__).with_name('benchmark_multilingual_public.py'), Path(__file__).with_name('wer_unicode.py')]
    identity = {'schema': 1, 'manifest_sha256': sha256(args.manifest), 'variant': args.variant,
                'binary_sha256': sha256(args.binary),
                'model_sha256': {str(p.resolve()): sha256(p) for p in args.model},
                'scorer_sha256': {p.name: sha256(p) for p in scorer_paths},
                'jiwer': importlib.metadata.version('jiwer'), 'workers': args.workers,
                'timeout_s': args.timeout, 'url': args.url.rstrip('/'),
                'health_configuration': {k: health.get(k) for k in ('variant', 'punctuation', 'itn', 'execution_provider', 'backend', 'vad')},
                'model_configuration': {k: model_info.get(k) for k in ('variant', 'version', 'execution_provider', 'encoder', 'pool_size', 'sample_rate', 'vocab_size')},
                'expected_samples_per_language': expected_per_language}
    metadata_path = args.output.with_suffix('.meta.json')
    ledger_path = args.output.with_suffix('.jsonl')
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text())
        if metadata['identity'] != identity:
            raise ValueError('Resume identity mismatch; use a new output path for different inputs/model/binary/scorer/configuration')
    else:
        if ledger_path.exists() or args.output.exists():
            raise ValueError('Existing result without identity metadata; refusing stale reuse')
        metadata = {'identity': identity, 'health': health, 'model_info': model_info, 'scoring': SCORING, 'python': sys.version,
                    'platform': platform.platform(), 'timing_caveat': 'Contended CPU timing; summed request times are not throughput or headline performance', 'runs': []}
        atomic_json(metadata_path, metadata)
    rows = read_ledger(ledger_path, samples)

    def request(sample):
        tick = time.perf_counter()
        hypothesis = ''
        error = None
        try:
            audio = sample['_resolved_path'].read_bytes()
            import hashlib
            if hashlib.sha256(audio).hexdigest() != sample['sha256']:
                raise ValueError('Audio changed after preflight')
            req = urllib.request.Request(args.url.rstrip('/') + '/v1/transcribe', data=audio,
                                         headers={'Content-Type': 'application/octet-stream'}, method='POST')
            with urllib.request.urlopen(req, timeout=args.timeout) as response:
                result = json.load(response)
            hypothesis = result['text']
            if not isinstance(hypothesis, str):
                raise ValueError('Non-string transcription')
        except Exception as exception:
            error = f'{type(exception).__name__}: {exception}'
            hypothesis = ''
        row = {k: v for k, v in sample.items() if k != '_resolved_path'}
        return {**row, 'condition': sample.get('condition', 'original'), 'status': 'failed' if error else 'ok', 'error': error,
                'hypothesis': hypothesis, 'elapsed_s': time.perf_counter() - tick,
                'scores': score(sample['reference'], hypothesis)}

    warm = request(samples[0])
    run_record = {'started_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), 'resumed_results': len(rows),
                  'warmup': {'status': warm['status'], 'elapsed_s': warm['elapsed_s'], 'error': warm['error']}, 'completed': False}
    metadata['runs'].append(run_record)
    atomic_json(metadata_path, metadata)
    if warm['status'] != 'ok':
        raise RuntimeError(f'Unmeasured warm-up failed: {warm["error"]}')
    pending_samples = iter(s for s in samples if s['id'] not in rows)
    started = time.perf_counter()
    executor = ThreadPoolExecutor(max_workers=args.workers)
    futures = {}
    descriptor = os.open(ledger_path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o600)
    try:
        for _ in range(args.workers):
            sample = next(pending_samples, None)
            if sample is not None:
                futures[executor.submit(request, sample)] = sample['id']
        while futures:
            done, _ = wait(futures, return_when=FIRST_COMPLETED)
            for future in done:
                del futures[future]
                row = future.result()
                encoded = (json.dumps(row, ensure_ascii=False, separators=(',', ':')) + '\n').encode()
                remaining = memoryview(encoded)
                while remaining:
                    written = os.write(descriptor, remaining)
                    if written <= 0:
                        raise OSError('Failed to append result')
                    remaining = remaining[written:]
                if args.fsync:
                    os.fsync(descriptor)
                rows[row['id']] = row
                print(f'{len(rows)}/{len(samples)} {row["id"]} {row["status"]}', flush=True)
                sample = next(pending_samples, None)
                if sample is not None:
                    futures[executor.submit(request, sample)] = sample['id']
    finally:
        os.close(descriptor)
        executor.shutdown(wait=True, cancel_futures=True)
    ordered = [rows[s['id']] for s in samples]
    run_record.update({'completed': True, 'measured_wall_s': time.perf_counter() - started,
                       'finished_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())})
    atomic_json(metadata_path, metadata)
    failures = sum(r['status'] == 'failed' for r in ordered)
    atomic_json(args.output, {'metadata': metadata, 'completed': True, 'n': len(ordered), 'failed_requests': failures,
                              'details': ordered, 'summary': summarize(ordered)})
    return 1 if failures else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('manifest', 'output', 'binary'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--model', type=Path, action='append', required=True, help='Repeat for encoder and vocabulary files')
    parser.add_argument('--variant', required=True)
    parser.add_argument('--url', default='http://127.0.0.1:19876')
    parser.add_argument('--workers', type=int, choices=(1, 2), default=1)
    parser.add_argument('--timeout', type=float, default=300)
    parser.add_argument('--fsync', action='store_true')
    args = parser.parse_args()
    if args.timeout <= 0:
        parser.error('timeout must be positive')
    raise SystemExit(run(args))


if __name__ == '__main__':
    main()
