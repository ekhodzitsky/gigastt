#!/usr/bin/env python3
"""Opt-in paired Criterion experiment, not a replacement for the regression gate."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import statistics
import subprocess
import time

CASES = {
    'tokenizer': ('tokenizer_decode/decode_64_ids', 'tokenizer_decode/decode_1024_ids'),
    'decode': ('decode_audio/pcm16_wav_3s', 'decode_telephony_raw/raw_pcmu_8k',
               'decode_telephony_raw/raw_g722_3s'),
    'resample': ('resample/48k_to_16k_1s', 'resample/8k_to_16k_1s'),
}
PAIR_COUNT = 7
LIMIT = 0.05
MODES = {'aa': ('base', 'base'), 'ab': ('base', 'candidate')}
FLAGS = ['--bench', '--noplot', '--save-baseline', 'main', '--exact',
         '--sample-size', '10', '--warm-up-time', '1', '--measurement-time', '1']


def positive(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError(f'invalid positive measurement: {value!r}')
    return value


def summarize(pairs):
    if len(pairs) != PAIR_COUNT:
        raise ValueError('incomplete paired measurement')
    ratios = [positive(b) / positive(a) for a, b in pairs]
    # Seven independent pairs yield a two-sided, distribution-free interval
    # for the median with 98.4375% nominal coverage (min/max; sign test).
    # Independence is an assumption, not established on hosted runners.
    return {'ratios': ratios, 'median_ratio': statistics.median(ratios),
            'median_interval': [min(ratios), max(ratios)]}


def assess(controls, comparison):
    aa, ab = summarize(controls), summarize(comparison)
    low, high = aa['median_interval']
    if low < 1 - LIMIT or high > 1 + LIMIT:
        outcome = 'unstable_control'
    elif ab['median_interval'][0] > 1 + LIMIT:
        outcome = 'regression'
    elif ab['median_interval'][1] > 1 + LIMIT:
        outcome = 'inconclusive'
    else:
        outcome = 'within_limit'
    return {'aa': aa, 'ab': ab, 'outcome': outcome}


def binary_hash(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def validated_binaries(provenance):
    binaries = {}
    for side in ('base', 'candidate'):
        record = provenance[side]
        artifacts = [a for a in record['executables'] if a['kind'] == ['bench']]
        for target in CASES:
            matches = [a for a in artifacts if a['target'] == target]
            if len(matches) != 1:
                raise ValueError(f'expected one {side}/{target} benchmark executable')
            artifact = matches[0]
            if not Path(artifact['path']).resolve().is_relative_to(Path(record['target_dir']).resolve()):
                raise ValueError(f'{side}/{target} executable outside recorded target directory')
            if binary_hash(artifact['path']) != artifact['sha256']:
                raise ValueError(f'{side}/{target} executable hash mismatch')
            binaries[side, target] = artifact
    return binaries


def read_measurement(home, case):
    baseline = home / case / 'main'
    sample = json.loads((baseline / 'sample.json').read_text())
    for field in ('iters', 'times'):
        if len(sample[field]) != 10:
            raise ValueError(f'expected ten {field} in {baseline}')
        for value in sample[field]:
            positive(value)
    estimate = json.loads((baseline / 'estimates.json').read_text())
    return positive(estimate['mean']['point_estimate'])


def workload_identity(provenance, cwd):
    # The benchmark source contains the deterministic synthetic inputs and
    # embeds these fixture files; keep identities separate from the binary hash.
    paths = [f'crates/gigastt-core/benches/{target}.rs' for target in CASES]
    if 'decode' in CASES:
        paths.append('crates/gigastt-core/tests/fixtures/telephony')
    identities = {}
    for side in ('base', 'candidate'):
        revision = provenance[side]['source_revision']
        if not re.fullmatch(r'[0-9a-f]{40}', revision):
            raise ValueError(f'invalid {side} source revision')
        rows = subprocess.check_output(['git', 'ls-tree', revision, '--', *paths],
                                       cwd=cwd, text=True).splitlines()
        entries = {}
        for line in rows:
            metadata, path = line.split('\t', 1)
            entries[path] = metadata
        if set(entries) != set(paths):
            raise ValueError(f'incomplete {side} benchmark workload identity')
        identities[side] = entries
    return {'trees': identities, 'identical': identities['base'] == identities['candidate']}


def host_metadata():
    affinity = sorted(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else None
    return {'machine': platform.machine(), 'logical_cpus': os.cpu_count(),
            'cpu_affinity': affinity}


def cpu_ticks():
    # Linux exposes cumulative times for every logical CPU, including siblings
    # outside our affinity. These diagnose contention but cannot prove isolation.
    proc_stat = Path('/proc/stat')
    if not proc_stat.is_file():
        return None
    return parse_cpu_ticks(proc_stat.read_text())


def parse_cpu_ticks(text):
    rows = (line.split() for line in text.splitlines())
    return {row[0]: [int(value) for value in row[1:]] for row in rows
            if row and row[0].startswith('cpu') and row[0][3:].isdigit()}


def measure(binary, case, home, cwd):
    home.mkdir(parents=True)
    argv = [binary['path'], *FLAGS, case]
    ticks_before = cpu_ticks()
    start = time.time()
    with (home / 'process.log').open('w') as log:
        subprocess.run(argv, cwd=cwd, env={**os.environ, 'CRITERION_HOME': str(home)},
                       stdout=log, stderr=subprocess.STDOUT, check=True, timeout=60)
    finish = time.time()
    ticks_after = cpu_ticks()
    return {'mean_ns': read_measurement(home, case), 'started_epoch': start,
            'finished_epoch': finish, 'binary_sha256': binary['sha256'], 'argv': argv,
            'cpu_ticks_before': ticks_before, 'cpu_ticks_after': ticks_after}


def run(provenance_path, out, cwd):
    # The regular gate recursively discovers Criterion directories under its
    # root; never put pilot measurements inside it.
    provenance_path, out = provenance_path.resolve(), out.resolve()
    if out.is_relative_to(provenance_path.parent):
        raise ValueError('pilot output must be separate from normal Criterion reports')
    provenance = json.loads(provenance_path.read_text())
    binaries = validated_binaries(provenance)
    out.mkdir(parents=True, exist_ok=False)
    (out / 'build-provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
    start = time.monotonic()
    report = {'platform': platform.platform(), 'host': host_metadata(),
              'workload_identity': workload_identity(provenance, cwd),
              'pairs_per_mode': PAIR_COUNT,
              'limit': LIMIT, 'cases': {}, 'complete': False}

    def save():
        report['elapsed_seconds'] = time.monotonic() - start
        (out / 'report.json').write_text(json.dumps(report, indent=2) + '\n')

    save()
    for target, cases in CASES.items():
        for case in cases:
            case_report = report['cases'][case] = {'measurements': []}
            paired = {mode: [] for mode in MODES}
            for i in range(PAIR_COUNT):
                # Alternate mode order as well as arm order, so a control is
                # close in time to every comparison. No adaptive retries.
                modes = ('aa', 'ab') if i % 2 == 0 else ('ab', 'aa')
                order = ('a', 'b') if i % 2 == 0 else ('b', 'a')
                for mode in modes:
                    block = {'index': i, 'mode': mode, 'order': order, 'arms': {}}
                    case_report['measurements'].append(block)
                    for arm in order:
                        side = MODES[mode][('a', 'b').index(arm)]
                        home = out / case / f'{i}-{mode}' / arm
                        block['arms'][arm] = measure(binaries[side, target], case, home, cwd)
                        save()
                    paired[mode].append(tuple(block['arms'][arm]['mean_ns'] for arm in ('a', 'b')))
            case_report.update(assess(paired['aa'], paired['ab']))
            # Uniform synthetic slowdown on measured controls; does not model
            # the variance of a real slower executable. Never a gate result.
            case_report['synthetic_10pct_slowdown'] = assess(
                paired['aa'], [(a, b * 1.10) for a, b in paired['aa']])['outcome']
            print(case, case_report['outcome'], flush=True)
            save()
    validated_binaries(provenance)
    report['complete'] = True
    save()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--provenance', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    run(args.provenance, args.output, Path(__file__).resolve().parent.parent)


if __name__ == '__main__':
    main()
