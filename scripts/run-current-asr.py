#!/usr/bin/env python3
"""Run frozen current-head quality and six steady rounds after one conditioning round.

Quality may use separate disjoint CPU sets; only serial timing runs qualify
speed comparisons. Existing timing evidence can precede a separate quality phase.
"""
import argparse
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import resource
import subprocess
from threading import Lock
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--probe', type=Path, required=True)
    parser.add_argument('--cpus', default='0-3', help='serial timing affinity')
    parser.add_argument('--phase', choices=['all', 'timing', 'quality'], default='all')
    parser.add_argument('--start-round', type=int, default=0, help='continue a timing phase from this round')
    parser.add_argument('--quality-cpus', nargs='+', help='disjoint affinity sets for parallel accuracy only')
    args = parser.parse_args()
    protocol = json.loads((args.directory / 'protocol.json').read_text())
    spec = importlib.util.spec_from_file_location('scoring', Path(__file__).with_name('score-current-asr.py'))
    scoring = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(scoring)
    catalog = {row['head']: row['assets'] for row in json.loads(subprocess.check_output([str(args.probe), 'catalog']))}
    for profile in protocol['profiles']:
        expected = catalog.get(profile['head'], {name: row['sha256'] for name, row in profile['assets'].items()})
        actual = scoring.verify_assets(Path(profile['model_dir']), expected)
        if actual != profile['assets']:
            raise ValueError('model protocol no longer matches the active asset catalog')
        if (Path(profile['model_dir']) / 'manifest.toml').read_text() != profile['manifest']:
            raise ValueError('model pack changed after freezing')
    output = args.directory / 'runs'
    output.mkdir(exist_ok=args.phase == 'quality' or args.start_round > 0)
    environment = dict(os.environ, RAYON_NUM_THREADS='1')
    log_path = args.directory / 'run-protocol.json'
    log = json.loads(log_path.read_text()) if args.phase == 'quality' or args.start_round > 0 else {
        'cpus': args.cpus, 'rayon_threads': 1,
        'probe_sha256': hashlib.sha256(args.probe.read_bytes()).hexdigest(), 'runs': []}
    if log['probe_sha256'] != hashlib.sha256(args.probe.read_bytes()).hexdigest():
        raise ValueError('probe changed between phases')
    log['active_asset_preflight'] = 'passed'
    log['active_catalog'] = catalog
    lock = Lock()

    def run(profile, mode, inputs, cpus, parallel=False):
        destination = output / f"{profile['head']}-{mode}.jsonl"
        command = ['taskset', '-c', cpus, str(args.probe), profile['head'],
                   profile['architecture'], profile['model_dir'], str(inputs), str(destination)]
        print('START', profile['head'], mode, cpus, flush=True)
        before = resource.getrusage(resource.RUSAGE_CHILDREN)
        start = time.monotonic()
        subprocess.run(command, env=environment, check=True)
        after = resource.getrusage(resource.RUSAGE_CHILDREN)
        entry = dict(head=profile['head'], mode=mode, command=command, cpus=cpus,
                     wall_seconds=time.monotonic()-start, parallel_quality=parallel,
                     process_cpu_seconds=None if parallel else after.ru_utime+after.ru_stime-before.ru_utime-before.ru_stime)
        with lock:
            log['runs'].append(entry)
            log_path.write_text(json.dumps(log, indent=2))
        print('DONE', profile['head'], mode, flush=True)

    profiles = protocol['profiles']
    if args.phase != 'quality':
        for round_ in range(args.start_round, 7):
            offset = round_ % len(profiles)
            for profile in profiles[offset:] + profiles[:offset]:
                run(profile, f'timing-{round_}', args.directory / 'timing-cases.json', args.cpus)
    if args.phase == 'timing':
        return
    cpus = args.quality_cpus or [args.cpus]
    if len(cpus) > len(profiles):
        raise ValueError('more quality workers than models')
    weights = {}
    for profile in profiles:
        records = [json.loads(s) for s in (output / f"{profile['head']}-timing-0.jsonl").read_text().splitlines()]
        weights[profile['head']] = sum(r['seconds'] for r in records[1:-1])
    ordered = sorted(profiles, key=lambda p: weights[p['head']], reverse=True)
    pending = deque(ordered[len(cpus):])

    def worker(first, affinity):
        profile = first
        while profile is not None:
            run(profile, 'quality', args.directory / 'cases.json', affinity, parallel=len(cpus) > 1)
            with lock:
                profile = pending.popleft() if pending else None

    with ThreadPoolExecutor(max_workers=len(cpus)) as executor:
        futures = [executor.submit(worker, first, affinity) for first, affinity in zip(ordered, cpus)]
        for future in futures:
            future.result()


if __name__ == '__main__':
    main()
