#!/usr/bin/env python3
"""Balanced single-job CPU/resource comparison using the same frozen server binary.

Run only after other benchmark inference has stopped. No recognition scores are
computed. Both model directories must be prepared and loadable offline.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import socket
import statistics
import subprocess
import threading
import time
import urllib.request

ROOT = Path(__file__).resolve().parent.parent


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def proc_snapshot(pid):
    # Linux stat comm can contain spaces and parentheses; fields after final ')'.
    fields = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
    result = {'cpu_s': (int(fields[11]) + int(fields[12])) / os.sysconf('SC_CLK_TCK')}
    for filename in ('status', 'smaps_rollup'):
        for line in Path(f'/proc/{pid}/{filename}').read_text().splitlines():
            key, _, value = line.partition(':')
            if key in ('VmRSS', 'VmHWM', 'Rss', 'Pss', 'Private_Clean', 'Private_Dirty', 'Swap'):
                result[f'{filename}_{key}_kib'] = int(value.split()[0])
    return result


def fetch(url, data=None):
    req = urllib.request.Request(url, data=data,
                                 headers={'Content-Type': 'application/octet-stream'})
    with urllib.request.urlopen(req, timeout=600) as response:
        return json.load(response)


def samples():
    rows = []
    for language in ('ru', 'en', 'kk', 'ky', 'uz'):
        for condition in ('original', 'alaw', 'mulaw'):
            directory = 'multilingual_1000_20261001' if condition == 'original' else 'multilingual_telephony_20261001'
            name = f'{language}_manifest.json' if condition == 'original' else f'{language}_{condition}_manifest.json'
            manifest = ROOT / 'benchmark/results' / directory / name
            row = json.loads(manifest.read_text())['samples'][0]
            path = Path(row['path'])
            if not path.is_absolute():
                path = manifest.parent / path
            payload = path.read_bytes()
            if hashlib.sha256(payload).hexdigest() != row['sha256']:
                raise ValueError(f'Audio hash mismatch: {path}')
            rows.append({'id': row['id'], 'language': language, 'condition': condition,
                         'duration_s': row['duration_s'], 'sha256': row['sha256'], '_payload': payload})
    return rows


def atomic_write(path, data):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def run_round(args, label, index, rows, cache):
    model_dir = getattr(args, f'{label}_model_dir').resolve()
    # Each launch has an empty optimized-graph cache. OS page cache is not cleared.
    optimized = cache / f'optimized_{index}'
    if optimized.exists():
        raise RuntimeError(f'Refusing stale optimized cache: {optimized}')
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        port = sock.getsockname()[1]
    url = f'http://127.0.0.1:{port}'
    command = [str(args.binary.resolve()), '--offline', 'serve', '--model-dir', str(model_dir),
               '--model-variant', 'ml_ctc', '--execution-provider', 'cpu', '--pool-size', '2',
               '--encoder-intra-threads', '3', '--punctuation', 'off', '--itn', 'off',
               '--optimized-cache-dir', str(optimized), '--port', str(port)]
    env = {k: v for k, v in os.environ.items() if not k.startswith('GIGASTT_')}
    env['RUST_LOG'] = 'warn'
    started = time.perf_counter()
    with (cache / f'server_{index}_{label}.log').open('w') as log:
        proc = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env)
        stopped = threading.Event()
        monitoring = []
        def monitor():
            while not stopped.is_set():
                try:
                    monitoring.append(proc_snapshot(proc.pid))
                except (FileNotFoundError, ProcessLookupError):
                    break
                stopped.wait(0.2)
        thread = None
        try:
            while True:
                if proc.poll() is not None:
                    raise RuntimeError(f'Server exited: {cache / f"server_{index}_{label}.log"}')
                try:
                    health = fetch(url + '/health')
                    model_info = fetch(url + '/v1/models')
                    break
                except (OSError, ValueError):
                    if time.perf_counter() - started > 600:
                        raise TimeoutError('Server startup exceeded 600 seconds')
                    time.sleep(0.1)
            startup_s = time.perf_counter() - started
            startup_resources = proc_snapshot(proc.pid)
            # Two sequential warmups use the normal checkout policy. They do not prove
            # every slot or internal allocation is warmed; same protocol both arms.
            for row in rows[:2]:
                reply = fetch(url + '/v1/transcribe', row['_payload'])
                if not isinstance(reply.get('text'), str):
                    raise ValueError('Warmup did not return transcription')
            before = proc_snapshot(proc.pid)
            load_before = os.getloadavg()
            thread = threading.Thread(target=monitor, daemon=True)
            thread.start()
            measured = []
            block_started = time.perf_counter()
            for row in rows:
                cpu_before = proc_snapshot(proc.pid)['cpu_s']
                tick = time.perf_counter()
                reply = fetch(url + '/v1/transcribe', row['_payload'])
                elapsed = time.perf_counter() - tick
                cpu_after = proc_snapshot(proc.pid)['cpu_s']
                if not isinstance(reply.get('text'), str):
                    raise ValueError('Request did not return transcription')
                measured.append({k: v for k, v in row.items() if k != '_payload'} | {
                    'wall_s': elapsed, 'server_cpu_s': cpu_after - cpu_before,
                    'rtf': elapsed / row['duration_s']})
            block_s = time.perf_counter() - block_started
            after = proc_snapshot(proc.pid)
            return {'label': label, 'round': index, 'command': command,
                    'startup_s': startup_s, 'startup_resources': startup_resources,
                    'health': health, 'model_info': model_info,
                    'warm_resources': before, 'end_resources': after,
                    'sampled_peak_resources': {key: max(x[key] for x in monitoring) for key in before} if monitoring else {},
                    'memory_sampling_interval_s': 0.2, 'memory_samples': len(monitoring),
                    'block_wall_s': block_s, 'request_wall_s': sum(x['wall_s'] for x in measured),
                    'server_cpu_s': after['cpu_s'] - before['cpu_s'],
                    'loadavg_before': load_before, 'loadavg_after': os.getloadavg(), 'rows': measured}
        finally:
            stopped.set()
            if thread:
                thread.join()
            proc.terminate()
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', type=Path, required=True)
    parser.add_argument('--baseline-model-dir', type=Path, required=True)
    parser.add_argument('--candidate-model-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    rows = samples()
    identity = {'binary_sha256': sha(args.binary), 'platform': platform.platform(),
                'cpu_count': os.cpu_count(), 'models': {},
                'selection': 'First manifest recording per language, all three conditions; 15 inputs fixed before timing',
                'order': ['baseline', 'candidate', 'candidate', 'baseline'],
                'scope': 'Sequential one-job REST wall latency; CPU pool=2, encoder threads=3; no quality claim',
                'startup_policy': 'Fresh optimized cache per launch, OS page cache uncontrolled; startup is not disk-cold boot',
                'memory_policy': 'Linux process RSS/PSS sampled every 200ms; VmHWM includes startup; sampled peaks are lower bounds',
                'contention_policy': 'Run after quality workers stop; unrelated host load may remain, ABBA reduces linear drift but is not isolation',
                'samples': [{k: v for k, v in row.items() if k != '_payload'} for row in rows]}
    for label in ('baseline', 'candidate'):
        directory = getattr(args, f'{label}_model_dir')
        identity['models'][label] = {str(p.name): {'sha256': sha(p), 'bytes': p.stat().st_size}
                                    for p in sorted(directory.iterdir()) if p.is_file() and p.suffix in ('.onnx', '.json', '.txt', '.toml')}
    if args.dry_run:
        print(json.dumps(identity, indent=2))
        return
    args.cache.mkdir(parents=True, exist_ok=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise RuntimeError('Refusing to overwrite existing resource result')
    report = {'identity': identity, 'completed': False, 'rounds': []}
    atomic_write(args.output, report)
    for index, label in enumerate(identity['order']):
        report['rounds'].append(run_round(args, label, index, rows, args.cache))
        atomic_write(args.output, report)
    report['summary'] = {}
    for label in ('baseline', 'candidate'):
        rounds = [r for r in report['rounds'] if r['label'] == label]
        measured = [r for block in rounds for r in block['rows']]
        audio_s = sum(r['duration_s'] for r in measured)
        report['summary'][label] = {'n': len(measured), 'audio_s': audio_s,
            'aggregate_request_rtf': sum(r['wall_s'] for r in measured) / audio_s,
            'aggregate_cpu_rtf': sum(r['server_cpu_s'] for r in measured) / audio_s,
            'median_request_s': statistics.median(r['wall_s'] for r in measured),
            'max_sampled_pss_kib': max(r['sampled_peak_resources']['smaps_rollup_Pss_kib'] for r in rounds),
            'max_sampled_rss_kib': max(r['sampled_peak_resources']['smaps_rollup_Rss_kib'] for r in rounds)}
    report['completed'] = True
    atomic_write(args.output, report)


if __name__ == '__main__':
    main()
