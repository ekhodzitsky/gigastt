#!/usr/bin/env python3
"""Opt-in Core ML load-only experiment. Never replaces the required revision gate."""
import argparse
import importlib.util
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
import urllib.request

from objc_cached_load_pilot import copy_cache, file_sha256, require, schedule, snapshot, validate

SPEC = importlib.util.spec_from_file_location('objc_gate', Path(__file__).with_name('compare-objc-revisions.py'))
gate = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(gate)
ROOT = gate.ROOT
TEST = 'runtime::coreml::bridge::cached_load_pilot::test_cached_load_pilot'


def build_env():
    # ANE links an older static ORT CoreML archive that needs compiler-rt.
    # Supply the same selected Xcode library to both revisions without
    # changing either production source tree.
    resource = gate.run('xcrun', '--sdk', 'macosx', 'clang', '--print-resource-dir',
                        capture_output=True).stdout.strip()
    require(resource and not any(c in resource for c in ('\n', '\r', '\x1f')), 'invalid clang resource directory')
    archive = Path(resource) / 'lib/darwin/libclang_rt.osx.a'
    require(archive.is_file(), f'missing compiler runtime: {archive}')
    env = dict(os.environ)
    flags = env['CARGO_ENCODED_RUSTFLAGS'].split('\x1f') if env.get('CARGO_ENCODED_RUSTFLAGS') else (
        [] if 'CARGO_ENCODED_RUSTFLAGS' in env else env.get('RUSTFLAGS', '').split())
    env['CARGO_ENCODED_RUSTFLAGS'] = '\x1f'.join([*flags, '-C', 'link-arg=' + str(archive)])
    return env, archive


def build_probe(arm, tree, temp, output, harness, env):
    bridge = tree / 'crates/gigastt-core/src/runtime/coreml/bridge.rs'
    (bridge.parent / 'objc_cached_load_pilot.rs').write_bytes(harness)
    with bridge.open('a') as target:
        target.write('\n#[cfg(test)]\n#[path = "objc_cached_load_pilot.rs"]\nmod cached_load_pilot;\n')
    result = subprocess.run(['cargo', 'test', '--manifest-path', str(tree / 'Cargo.toml'),
        '--target-dir', str(ROOT / 'target'), '-p', 'gigastt-core', '--features', 'ane',
        '--release', '--locked', '--lib', '--no-run', '--message-format=json'],
        cwd=tree, env=env, text=True, capture_output=True, check=False)
    (output / f'{arm}-build.jsonl').write_text(result.stdout)
    (output / f'{arm}-build.stderr').write_text(result.stderr)
    artifacts = [json.loads(line) for line in result.stdout.splitlines() if line.startswith('{')]
    if result.returncode:
        print(result.stderr, flush=True)
        for artifact in artifacts:
            if artifact.get('reason') == 'compiler-message':
                print(artifact['message'].get('rendered', ''), flush=True)
        result.check_returncode()
    tests = [a for a in artifacts if a.get('reason') == 'compiler-artifact'
             and a.get('executable') and a.get('profile', {}).get('test')]
    require(len(tests) == 1, 'expected one release test executable')
    artifact = tests[0]
    require(not artifact['fresh'], 'pilot executable must be built from this worktree')
    require(Path(artifact['manifest_path']).resolve() ==
            (tree / 'crates/gigastt-core/Cargo.toml').resolve(), 'unexpected Cargo manifest')
    require(Path(artifact['target']['src_path']).resolve() ==
            (tree / 'crates/gigastt-core/src/lib.rs').resolve(), 'unexpected Cargo test source')
    binary = temp / f'{arm}-probe'
    executable = Path(tests[0]['executable']).resolve(strict=True)
    require(executable.is_relative_to((ROOT / 'target').resolve()), 'unexpected Cargo artifact path')
    digest = file_sha256(executable)
    shutil.copy2(executable, binary)
    require(file_sha256(binary) == digest, 'copied executable differs')
    listing = gate.run(str(binary), TEST, '--ignored', '--exact', '--list', capture_output=True)
    require(f'{TEST}: test' in listing.stdout.splitlines(), 'pilot probe missing')
    return binary, {'source_revision': subprocess.check_output(
                        ['git', '-C', str(tree), 'rev-parse', 'HEAD'], text=True).strip(),
                    'source_tree': subprocess.check_output(
                        ['git', '-C', str(tree), 'rev-parse', 'HEAD^{tree}'], text=True).strip(),
                    'cargo_lock_sha256': file_sha256(tree / 'Cargo.lock'),
                    'tracked_diff_sha256': hashlib.sha256(subprocess.check_output(
                        ['git', '-C', str(tree), 'diff', '--binary', 'HEAD'])).hexdigest(),
                    'executable': str(executable), 'executable_sha256': digest,
                    'cargo_fresh': tests[0]['fresh']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', required=True)
    parser.add_argument('--output', type=Path, default=ROOT / 'target/objc-cached-load-pilot')
    args = parser.parse_args()
    require(re.fullmatch('[0-9a-f]{40}', args.base) is not None, 'immutable base SHA required')
    base = gate.git('rev-parse', '--verify', args.base + '^{commit}')
    candidate = gate.git('rev-parse', 'HEAD')
    require(base != candidate, 'base must differ from candidate')
    gate.run('git', 'merge-base', '--is-ancestor', base, candidate)
    require(sys.platform == 'darwin', 'Core ML comparison requires macOS')
    gate.require_model_host(gate.run('sw_vers', '-productVersion', capture_output=True).stdout.strip())
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    harness_path = ROOT / 'scripts/objc-cached-load-probe.rs'
    harness = harness_path.read_bytes()
    report = {'base_revision': base, 'candidate_revision': candidate, 'model_sha256': gate.MODEL_SHA,
              'harness_sha256': file_sha256(harness_path),
              'driver_sha256': file_sha256(Path(__file__)),
              'validation_sha256': file_sha256(ROOT / 'scripts/objc_cached_load_pilot.py'),
              'rustc': gate.run('rustc', '-Vv', capture_output=True).stdout,
              'macos': gate.run('sw_vers', capture_output=True).stdout,
              'machine': gate.run('uname', '-m', capture_output=True).stdout.strip(),
              'compute_units_note': 'Requested policies; physical ANE use is not asserted.',
              'diagnostic_only': True, 'complete': False,
              'builds': {}, 'binary_sha256': {}, 'prime': {}, 'samples': [],
              'cache_before': {}, 'cache_after': {}}
    started = time.monotonic()

    def save():
        report['elapsed_seconds'] = time.monotonic() - started
        (output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')

    save()
    env, archive = build_env()
    report['common_compiler_runtime'] = str(archive)
    report['common_compiler_runtime_sha256'] = file_sha256(archive)
    report['common_encoded_rustflags'] = env['CARGO_ENCODED_RUSTFLAGS']
    save()
    with tempfile.TemporaryDirectory(prefix='gigastt-objc-cached-load-pilot-') as temporary:
        temp = Path(temporary)
        bundle_path = temp / (gate.MODEL + '.tar')
        with urllib.request.urlopen(gate.MODEL_URL, timeout=120) as source, bundle_path.open('wb') as dest:
            shutil.copyfileobj(source, dest)
        require(file_sha256(bundle_path) == gate.MODEL_SHA, 'model checksum mismatch')
        with tarfile.open(bundle_path) as bundle:
            bundle.extractall(temp / 'model', filter='data')
        packages = list((temp / 'model').rglob(gate.MODEL))
        require(len(packages) == 1, 'expected exactly one pinned model package')
        binaries, models, worktrees = {}, {}, []
        try:
            for arm, revision in [('base', base), ('candidate', candidate)]:
                tree = temp / arm
                gate.run('git', 'worktree', 'add', '--detach', str(tree), revision)
                worktrees.append(tree)
                binary, provenance = build_probe(arm, tree, temp, output, harness, env)
                require(provenance['source_revision'] == revision, 'build source revision changed')
                report['builds'][arm] = provenance
                binaries['ab_' + arm] = binary
                save()
            for arm in ('aa_base', 'aa_candidate'):
                binaries[arm] = binaries['ab_base']
            for arm, binary in binaries.items():
                report['binary_sha256'][arm] = file_sha256(binary)
                model = temp / (arm + '-model') / gate.MODEL
                model.parent.mkdir()
                model.symlink_to(packages[0], target_is_directory=True)
                models[arm] = model
            save()

            def measure(arm, phase, label):
                binary = binaries[arm]
                binary_sha = file_sha256(binary)
                require(binary_sha == report['binary_sha256'][arm], 'probe executable changed')
                start = time.monotonic()
                result = subprocess.run([str(binary), TEST, '--ignored', '--exact',
                    '--test-threads=1', '--nocapture'], cwd=ROOT,
                    env=dict(os.environ, LC_ALL='C', GIGASTT_OBJC_PROBE_MODEL=str(models[arm]),
                             GIGASTT_OBJC_PILOT_PHASE=phase),
                    text=True, capture_output=True, check=False, timeout=300)
                log = result.stdout + result.stderr
                (output / f'{arm}-{label}.log').write_text(log)
                result.check_returncode()
                records = [json.loads(line.partition('objc cached load ')[2]) for line in log.splitlines()
                           if 'objc cached load {' in line]
                require(len(records) == 1, 'missing or duplicate pilot record')
                return {'data': records[0], 'binary_sha256': binary_sha,
                        'started_offset_seconds': start - started,
                        'wall_seconds': time.monotonic() - start}

            def cache_dir(arm):
                return models[arm].parent / 'compiled_cache'

            for arm in ('ab_base', 'ab_candidate'):
                report['prime'][arm] = measure(arm, 'prime', 'prime')
                save()
            for arm in ('aa_base', 'aa_candidate'):
                copy_cache(cache_dir('ab_base'), cache_dir(arm))
            report['cache_before'] = {arm: snapshot(cache_dir(arm)) for arm in binaries}
            save()
            for phase, index, group, arm in schedule():
                sample = measure(f'{group}_{arm}', phase, f'{phase}-{index}')
                report['samples'].append(dict(sample, phase=phase, round=index, group=group, arm=arm))
                save()
                print(f'{phase}: {group} {arm} pair {index + 1} complete', flush=True)
            report['cache_after'] = {arm: snapshot(cache_dir(arm)) for arm in binaries}
            save()
            report['summary'] = validate(report)
            report['complete'] = True
            save()
            print(json.dumps(report['summary'], indent=2))
            # Qualification is diagnostic only; it cannot change a required
            # revision gate failure, nor select a rerun because of a result.
        finally:
            for tree in reversed(worktrees):
                gate.run('git', 'worktree', 'remove', '--force', str(tree))


if __name__ == '__main__':
    main()
