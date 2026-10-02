#!/usr/bin/env python3
"""Durable matched-corpus evaluation of frozen ready ASR backends, without training."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
import fcntl
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import resource
import queue
import threading
import subprocess
import time

import numpy as np
import psutil
import soundfile as sf

from benchmark_multilingual_public import score, sha256
from benchmark_multilingual_1000 import atomic_json
from wer_unicode import normalize


def totals(rows):
    counts = ('reference_words', 'word_errors', 'substitutions', 'deletions', 'insertions',
              'reference_chars', 'char_errors')
    result = {key:sum(row['scores'][key] for row in rows) for key in counts}
    result.update(n=len(rows), empty_hypotheses=sum(not normalize(r['hypothesis']) for r in rows),
                  failed_requests=sum(r['status'] == 'failed' for r in rows))
    result['wer_pct'] = 100 * result['word_errors'] / result['reference_words'] if result['reference_words'] else None
    result['cer_pct'] = 100 * result['char_errors'] / result['reference_chars'] if result['reference_chars'] else None
    return result


def decode(path):
    """Preserve floating-point samples; only mono conversion and required resampling."""
    raw = subprocess.check_output(['ffmpeg', '-v', 'error', '-threads', '1', '-i', str(path),
                                   '-ac', '1', '-ar', '16000', '-f', 'f32le', '-'], timeout=120)
    audio = np.frombuffer(raw, dtype='<f4').copy()
    if not len(audio) or not np.isfinite(audio).all():
        raise ValueError('Invalid decoded audio')
    return audio


def memory():
    process = psutil.Process()
    full = process.memory_full_info()
    return dict(rss_bytes=full.rss, pss_bytes=getattr(full, 'pss', None),
                uss_bytes=getattr(full, 'uss', None),
                process_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backend', required=True, choices=['whisper', 'omni'])
    parser.add_argument('--model-dir', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--threads', type=int, default=3)
    parser.add_argument('--workers', type=int, default=1)
    args = parser.parse_args()
    if args.threads < 1 or args.workers < 1:
        parser.error('threads and workers must be positive')
    if args.backend != 'whisper' and args.workers != 1:
        parser.error('Only independent Whisper instances support parallel quality workers')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.with_suffix('.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run(args)


def run(args):
    manifest = json.loads(args.manifest.read_text())
    samples = manifest['samples']
    if not samples or len({s['id'] for s in samples}) != len(samples):
        raise ValueError('Empty corpus or duplicate IDs')
    languages = {s['language'] for s in samples}
    if len(languages) != 1:
        raise ValueError('One corpus language required')
    language = next(iter(languages))
    paths = []
    durations = []
    for sample in samples:
        path = Path(sample['path']).expanduser()
        if not path.is_absolute():
            path = args.manifest.resolve().parent / path
        if sha256(path) != sample['sha256']:
            raise ValueError(f"Audio hash mismatch: {sample['id']}")
        paths.append(path)
        durations.append(sf.info(path).duration)
    os.environ['OMP_NUM_THREADS'] = str(args.threads)
    os.environ['MKL_NUM_THREADS'] = str(args.threads)
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    backend = importlib.import_module('ready_asr_' + args.backend)
    load_start = time.perf_counter()
    model = backend.ReadyASR(args.model_dir, language, threads=args.threads)
    load_s = time.perf_counter() - load_start
    identity = dict(backend=args.backend, model=model.metadata, language=language,
                    manifest_sha256=sha256(args.manifest), n=len(samples), threads=args.threads,
                    quality_workers=args.workers,
                    worker_policy='Independent identical model instances; source-order durable commits; timing not comparative',
                    script_sha256=sha256(Path(__file__)),
                    backend_script_sha256=sha256(Path(backend.__file__)),
                    scorer_sha256={name:sha256(Path(__file__).with_name(name)) for name in
                                   ['benchmark_multilingual_public.py', 'wer_unicode.py']},
                    jiwer=importlib.metadata.version('jiwer'),
                    ffmpeg=subprocess.check_output(['ffmpeg', '-version'],text=True).splitlines()[0],
                    decode='FFmpeg mono 16 kHz float32; no filtering, VAD, PCM16 requantization or telephone simulation',
                    scoring='One fixed reference, existing Unicode normalization, micro WER/CER including spaces; all failures retained as empty',
                    platform=platform.platform(), python=platform.python_version(),
                    warmup='First manifest audio, one unmeasured inference per independent instance; same fixed settings',
                    timing_scope='Exploratory quality run, potentially concurrent workloads; not an isolated speed benchmark')
    meta_path = args.output.with_suffix('.meta.json')
    ledger_path = args.output.with_suffix('.jsonl')
    if meta_path.exists():
        if json.loads(meta_path.read_text()) != identity:
            raise ValueError('Frozen identity differs; refuse mixed run')
    else:
        if ledger_path.exists():
            raise ValueError('Ledger exists without frozen identity')
        atomic_json(meta_path, identity)
    rows = []
    if ledger_path.exists():
        data = ledger_path.read_bytes()
        if data and not data.endswith(b'\n'):
            raise ValueError('Uncommitted ledger tail; inspect before resuming')
        rows = [json.loads(line) for line in data.splitlines()]
        if len(rows) > len(samples):
            raise ValueError('Oversized ledger')
        for sample, row in zip(samples, rows):
            if any(row[k] != sample[k] for k in ['id','path','reference','sha256','language','dataset','split']):
                raise ValueError('Ledger sample identity mismatch')
            if row.get('condition') != sample.get('condition') or row.get('status') not in {'ok', 'failed'}:
                raise ValueError('Ledger condition or status mismatch')
            if row['status'] == 'failed' and row['hypothesis']:
                raise ValueError('Failed ledger row has a hypothesis')
            if row['scores'] != score(sample['reference'], row['hypothesis']):
                raise ValueError('Ledger score mismatch')
    if len(rows) == len(samples):
        if args.output.exists():
            saved = json.loads(args.output.read_text())
            if (saved.get('completed') is not True or saved.get('n') != len(rows)
                    or saved.get('metadata') != identity or saved.get('details') != rows
                    or saved.get('summary') != totals(rows)):
                raise ValueError('Final result differs from complete ledger')
        else:
            atomic_json(args.output, dict(completed=True, n=len(rows), metadata=identity, details=rows,
                                         summary=totals(rows), reconstructed_from_complete_ledger=True))
        print('Complete ledger already present', flush=True)
        return
    before_warmup = memory()
    model.transcribe(decode(paths[0]))
    after_warmup = memory()
    run_id = time.time_ns()
    worker_models = queue.Queue()
    worker_models.put(model)
    for _ in range(1, args.workers):
        other = backend.ReadyASR(args.model_dir, language, threads=args.threads)
        if other.metadata != model.metadata:
            raise ValueError('Quality worker model identity differs')
        other.transcribe(decode(paths[0]))
        worker_models.put(other)
    local = threading.local()

    def initialize_worker():
        local.model = worker_models.get_nowait()

    def recognize(index):
        sample, path = samples[index], paths[index]
        start = time.perf_counter()
        status, error, hypothesis, diagnostics = 'ok', None, '', {}
        decode_s, infer_s = 0.0, 0.0
        try:
            audio = decode(path)
            decode_s = time.perf_counter() - start
            before = time.perf_counter()
            response = local.model.transcribe(audio)
            infer_s = time.perf_counter() - before
            hypothesis = response['hypothesis']
            if not isinstance(hypothesis, str):
                raise ValueError('Backend hypothesis must be text')
            diagnostics = {k:v for k,v in response.items() if k != 'hypothesis'}
        except Exception as exc:
            status, error, hypothesis = 'failed', f'{type(exc).__name__}: {exc}', ''
        return dict(sample, status=status, error=error, hypothesis=hypothesis,
                    scores=score(sample['reference'], hypothesis), duration_s=durations[index],
                    elapsed_s=time.perf_counter()-start, decode_s=decode_s, inference_s=infer_s,
                    diagnostics=diagnostics, process_peak_rss_bytes=memory()['process_peak_rss_bytes'],
                    run_id=run_id)

    pool = (ThreadPoolExecutor(max_workers=args.workers, initializer=initialize_worker)
            if args.workers > 1 else nullcontext(None))
    with ledger_path.open('a') as stream, pool as executor:
        indices = range(len(rows), len(samples))
        if executor is None:
            initialize_worker()
            results = map(recognize, indices)
        else:
            results = executor.map(recognize, indices)
        for row in results:
            stream.write(json.dumps(row, ensure_ascii=False)+'\n')
            stream.flush()
            os.fsync(stream.fileno())
            rows.append(row)
            print(f"{args.backend} {len(rows)}/{len(samples)} {row['id']} {row['status']} {row['elapsed_s']:.2f}s", flush=True)
    atomic_json(args.output, dict(completed=True, n=len(rows), metadata=identity,
                primary_instance_load_s=load_s, memory_before_primary_warmup=before_warmup,
                memory_after_primary_warmup=after_warmup,
                memory_final=memory(), current_process_run_id=run_id, details=rows, summary=totals(rows)))
    print(json.dumps(totals(rows)), flush=True)


if __name__ == '__main__':
    main()
