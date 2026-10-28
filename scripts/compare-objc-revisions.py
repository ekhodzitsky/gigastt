#!/usr/bin/env python3
"""Compare the real Core ML bridge with locked dependencies on one macOS runner.

Runs only when Objective-C dependencies or production bridge sources change.
CPUAndNeuralEngine is a requested policy, not proof of physical ANE execution.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import statistics
import subprocess
import sys
import tarfile
import tempfile
import tomllib
import urllib.request

ROOT = Path(__file__).resolve().parent.parent
TEST = 'runtime::coreml::bridge::revision_probe::test_objc_revision_probe'
MODEL = 'gigaam_v3_encoder_768.mlpackage'
MODEL_SHA = '111bd2722c46d41c0984e246752782f05892017990f50837ee6342b0dc41b5be'
MODEL_URL = 'https://github.com/ekhodzitsky/gigastt/releases/download/ane-v3-2026-06-24/' + MODEL + '.tar'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def require_model_host(version):
    require(re.fullmatch(r'\d+(?:\.\d+)*', version) is not None
            and int(version.split('.')[0]) >= 15,
            'Pinned CoreML8 encoder requires macOS 15 or newer; got ' + version)


def validate(report):
    require(all(len(report[a]) == 7 for a in ('base', 'candidate')), 'seven paired rounds required')
    reference = report['base'][0]
    for arm in ('base', 'candidate'):
        for row in report[arm]:
            require(row['fixtures'] == reference['fixtures'] and len(row['fixtures']) == 3, 'fixture mismatch')
            require(math.isfinite(row['peak_rss_bytes']) and row['peak_rss_bytes'] > 0, 'invalid RSS')
            for mode in ('cpu', 'cpu_and_ne'):
                data = row['modes'][mode]
                require(data['cached'] is True, 'mixed cold/compiled-cache measurement')
                require(len(data['outputs']) == 3, 'missing encoder output')
                require(data['outputs'] == reference['modes'][mode]['outputs'], 'encoder output changed')
                for output in data['outputs']:
                    require(re.fullmatch('[0-9a-f]{64}', output['sha256']) is not None, 'invalid output hash')
                    require(output['shape'] and all(type(v) is int and v > 0 for v in output['shape']), 'invalid output shape')
                for key in ('seconds', 'load_seconds'):
                    require(math.isfinite(data[key]) and data[key] > 0, 'invalid timing')
    summary = {}
    for mode in ('cpu', 'cpu_and_ne'):
        for key in ('seconds', 'load_seconds'):
            ratios = [b['modes'][mode][key] / a['modes'][mode][key]
                      for a, b in zip(report['base'], report['candidate'])]
            median = statistics.median(ratios)
            summary[f'{mode}/{key}'] = {'paired_ratios': ratios, 'median_ratio': median}
            require(median <= 1.05, f'{mode}/{key}: median regression exceeds 5%: {median}')
    ratios = [b['peak_rss_bytes'] / a['peak_rss_bytes'] for a, b in zip(report['base'], report['candidate'])]
    summary['peak_rss'] = {'paired_ratios': ratios, 'median_ratio': statistics.median(ratios)}
    require(statistics.median(ratios) <= 1.10, 'peak RSS regression exceeds 10%')
    return summary


def run(*args, **kwargs):
    return subprocess.run(args, cwd=ROOT, check=True, text=True, **kwargs)


def git(*args):
    return run('git', *args, capture_output=True).stdout.strip()


def affected(base, candidate):
    def versions(ref):
        lock = tomllib.loads(git('show', ref + ':Cargo.lock'))
        return sorted((p['name'], p['version'], p.get('source', ''), p.get('checksum', '')) for p in lock['package']
                      if p['name'].startswith('objc2'))
    paths = git('diff', '--name-only', base, candidate, '--',
                'crates/gigastt-core/src/runtime/coreml').splitlines()
    return versions(base) != versions(candidate) or any('/tests' not in p for p in paths)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', required=True)
    parser.add_argument('--affected-only', action='store_true',
                        help='Emit a GitHub Actions changed output without building or downloading')
    parser.add_argument('--output', type=Path, default=ROOT / 'target/objc-revisions')
    args = parser.parse_args()
    require(re.fullmatch('[0-9a-f]{40}', args.base) is not None, 'immutable base SHA required')
    base = git('rev-parse', '--verify', args.base + '^{commit}')
    candidate = git('rev-parse', 'HEAD')
    require(base != candidate, 'base must differ from candidate')
    run('git', 'merge-base', '--is-ancestor', base, candidate)
    changed = affected(base, candidate)
    if args.affected_only:
        print('changed=' + str(changed).lower())
        return
    if not changed:
        print('Objective-C dependencies and bridge unchanged; no model download or benchmark needed.')
        return
    require(sys.platform == 'darwin', 'Core ML comparison requires macOS')
    require_model_host(run('sw_vers', '-productVersion', capture_output=True).stdout.strip())
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    harness = (ROOT / 'scripts/objc-revision-probe.rs').read_bytes()
    report = {'base_revision': base, 'candidate_revision': candidate, 'model_sha256': MODEL_SHA,
              'harness_sha256': hashlib.sha256(harness).hexdigest(),
              'rustc': run('rustc', '-Vv', capture_output=True).stdout,
              'macos': run('sw_vers', capture_output=True).stdout,
              'machine': run('uname', '-m', capture_output=True).stdout.strip(),
              'compute_units_note': 'Requested policies; physical ANE use is not asserted.',
              'prime': {}, 'base': [], 'candidate': []}

    def save():
        (output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')

    save()
    with tempfile.TemporaryDirectory(prefix='gigastt-objc-revisions-') as temp:
        temp = Path(temp)
        archive = temp / (MODEL + '.tar')
        with urllib.request.urlopen(MODEL_URL, timeout=120) as source, archive.open('wb') as dest:
            shutil.copyfileobj(source, dest)
        with archive.open('rb') as source:
            require(hashlib.file_digest(source, 'sha256').hexdigest() == MODEL_SHA, 'model checksum mismatch')
        with tarfile.open(archive) as bundle:
            bundle.extractall(temp / 'model', filter='data')
        packages = list((temp / 'model').rglob(MODEL))
        require(len(packages) == 1, 'expected exactly one pinned model package')
        worktrees, binaries, models = [], {}, {}
        try:
            for arm in ('base', 'candidate'):
                tree = temp / arm
                run('git', 'worktree', 'add', '--detach', str(tree), report[arm + '_revision'])
                worktrees.append(tree)
                module = tree / 'crates/gigastt-core/src/runtime/coreml/bridge.rs'
                (module.parent / 'objc_revision_probe.rs').write_bytes(harness)
                with module.open('a') as target:
                    target.write('\n#[cfg(test)]\n#[path = "objc_revision_probe.rs"]\nmod revision_probe;\n')
                build = subprocess.run(['cargo', 'test', '--manifest-path', str(tree / 'Cargo.toml'),
                    '--target-dir', str(ROOT / 'target'), '-p', 'gigastt-core', '--features', 'ane',
                    '--release', '--locked', '--lib', '--no-run', '--message-format=json'],
                    cwd=ROOT, text=True, capture_output=True, check=False)
                (output / f'{arm}-build.jsonl').write_text(build.stdout)
                (output / f'{arm}-build.stderr').write_text(build.stderr)
                if build.returncode:
                    print(build.stderr, flush=True)
                    for line in build.stdout.splitlines():
                        message = json.loads(line)
                        if message.get('reason') == 'compiler-message':
                            print(message['message'].get('rendered', ''), flush=True)
                    build.check_returncode()
                artifacts = [json.loads(line) for line in build.stdout.splitlines() if line.startswith('{')]
                tests = [a['executable'] for a in artifacts if a.get('reason') == 'compiler-artifact'
                         and a.get('executable') and a.get('profile', {}).get('test')]
                require(len(tests) == 1, 'expected one test executable')
                binaries[arm] = temp / (arm + '-probe')
                shutil.copy2(tests[0], binaries[arm])
                listing = run(str(binaries[arm]), TEST, '--ignored', '--exact', '--list', capture_output=True)
                require(f'{TEST}: test' in listing.stdout.splitlines(), 'probe missing')
                model_dir = temp / (arm + '-model')
                model_dir.mkdir()
                models[arm] = model_dir / MODEL
                models[arm].symlink_to(packages[0], target_is_directory=True)

            def measure(arm, label):
                env = dict(os.environ, LC_ALL='C', GIGASTT_OBJC_PROBE_MODEL=str(models[arm]))
                result = subprocess.run(['/usr/bin/time', '-l', str(binaries[arm]), TEST,
                    '--ignored', '--exact', '--test-threads=1', '--nocapture'], cwd=ROOT,
                    env=env, text=True, capture_output=True, timeout=300, check=False)
                log = result.stdout + result.stderr
                (output / f'{arm}-{label}.log').write_text(log)
                result.check_returncode()
                records = [json.loads(line.partition('objc revision ')[2]) for line in log.splitlines()
                           if 'objc revision {' in line]
                require(len(records) == 1, 'missing or duplicate probe record')
                peak = re.findall(r'^\s*(\d+)\s+maximum resident set size\s*$', result.stderr, re.MULTILINE)
                require(len(peak) == 1, 'missing peak RSS from macOS time')
                records[0]['peak_rss_bytes'] = int(peak[0])
                return records[0]

            # Each arm compiles its own cache before either arm is timed.
            # Prime load/compile timings are retained but are descriptive only.
            for arm in ('base', 'candidate'):
                report['prime'][arm] = measure(arm, 'prime')
                save()
            for index in range(7):
                for arm in (('base', 'candidate') if index % 2 == 0 else ('candidate', 'base')):
                    report[arm].append(measure(arm, str(index)))
                    save()
                    print(f'{arm} paired round {index + 1} complete', flush=True)
            report['summary'] = validate(report)
            save()
            print(json.dumps(report['summary'], indent=2))
        finally:
            for tree in reversed(worktrees):
                run('git', 'worktree', 'remove', '--force', str(tree))


if __name__ == '__main__':
    main()
