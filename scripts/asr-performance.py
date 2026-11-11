#!/usr/bin/env python3
"""Paired Linux CPU ASR gate: identical assets/probe, two immutable Rust revisions."""
import argparse
from contextlib import contextmanager
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
import tempfile
import wave

ROOT = Path(__file__).resolve().parents[1]
ASSETS_FILE = ROOT / 'scripts/asr-performance-assets.json'
ASSETS = json.loads(ASSETS_FILE.read_text())
REVIEW_FILE = ROOT / 'scripts/asr-reviewed-text.json'
HEADS = ('rnnt', 'e2e_rnnt')
CASES = ('golos_00.wav', 'golos_01.wav', 'golos_02.wav', 'long.wav')
TEST = 'inference::engine::asr_revision_probe::test_asr_revision_probe'
ROUNDS = 7
LIMITS = {'load_seconds': 1.20, 'first_seconds': 1.20, 'rss_kib': 1.10, 'peak_rss_kib': 1.10}
WARM_LIMIT = 1.10


def positive(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError(f'invalid positive measurement: {value}')
    return value


def affinity_label(cpus):
    if not isinstance(cpus, list) or len(cpus) not in (1, 2) or any(type(c) is not int or c < 0 for c in cpus):
        raise ValueError('invalid CPU affinity')
    if cpus != sorted(set(cpus)):
        raise ValueError('invalid CPU affinity order/duplicates')
    if len(cpus) == 2 and cpus[1] == cpus[0] + 1:
        return f'{cpus[0]}-{cpus[1]}'
    return ','.join(map(str, cpus))


@contextmanager
def pinned_cpus():
    """Keep both binaries on the same cores, including on hybrid CPU hosts."""
    original = os.sched_getaffinity(0)
    selected = sorted(original)[:2]
    affinity_label(selected)
    try:
        os.sched_setaffinity(0, selected)
        yield selected
    finally:
        os.sched_setaffinity(0, original)


def reviewed_texts(report):
    """Only protocol 2 can use an exact, source-controlled quality review."""
    if report['protocol'] == 1:
        return []
    if report.get('reviewed_text_sha256') != digest(REVIEW_FILE):
        raise ValueError('missing or changed reviewed-text fingerprint')
    document = json.loads(REVIEW_FILE.read_text())
    if document.get('schema') != 1 or not isinstance(document.get('reviews'), list):
        raise ValueError('invalid reviewed-text document')
    reviews = document['reviews']
    seen = set()
    for review in reviews:
        if (review['head'] not in HEADS or review['case'] not in CASES
                or not re.fullmatch('[a-f0-9]{40}', review['base_revision'])
                or review['assets_sha256'] != digest(ASSETS_FILE)
                or not isinstance(review['rationale'], str) or not review['rationale'].strip()):
            raise ValueError('invalid reviewed-text scope')
        before, after = review['base_text'], review['candidate_text']
        if not isinstance(before, str) or not isinstance(after, str) or not before.strip() or not after.strip() or before == after:
            raise ValueError('invalid reviewed-text pair')
        # This validates the review, never normalizes measured transcripts.
        if re.findall(r'[^\W_]+', before) != re.findall(r'[^\W_]+', after):
            raise ValueError('reviewed-text pair changes lexical content')
        identity = (review['base_revision'], review['head'], review['case'], before, after)
        if identity in seen:
            raise ValueError('duplicate reviewed-text pair')
        seen.add(identity)
    return reviews


def check(report):
    """Fail closed; reviewed differences never relax per-side stability or limits."""
    try:
        cpus = report['cpu_affinity']
        expected_affinity = affinity_label(cpus)
        if report.get('inputs_verified') is not True:
            raise ValueError('input verification did not complete')
        if report['audio_sha256'] != {n: a['sha256'] for n, a in ASSETS['workload'].items()}:
            raise ValueError('audio fingerprints missing or changed')
        if not re.fullmatch('[a-f0-9]{64}', report['harness_sha256']):
            raise ValueError('missing probe fingerprint')
        if type(report['protocol']) is not int or report['protocol'] not in (1, 2) or report['assets'] != ASSETS or set(report['heads']) != set(HEADS):
            raise ValueError('incompatible ASR protocol/assets/heads')
        reviews = reviewed_texts(report)
        applied_reviews = []
        if report['protocol'] == 2:
            report['applied_text_reviews'] = applied_reviews
        for side in ('base', 'candidate'):
            if not re.fullmatch('[a-f0-9]{40}', report['revisions'][side]):
                raise ValueError('missing immutable revision')
            if not re.fullmatch('[a-f0-9]{64}', report['binary_sha256'][side]):
                raise ValueError('missing binary fingerprint')
        ratios = {}
        for head in HEADS:
            base, candidate = (report['heads'][head][s] for s in ('base', 'candidate'))
            if len(base) != ROUNDS or len(candidate) != ROUNDS:
                raise ValueError('require seven complete pairs for each head')
            reference = base[0]
            for samples in (base, candidate):
                for sample in samples:
                    if sample['first_text'] != samples[0]['first_text']:
                        raise ValueError('first-call repetition changed')
                    for name in CASES:
                        if sample['cases'][name]['text'] != samples[0]['cases'][name]['text']:
                            raise ValueError(f'ASR repetition changed: {head}/{name}')
            for name in CASES:
                before, after = base[0]['cases'][name]['text'], candidate[0]['cases'][name]['text']
                if before != after:
                    matches = [r for r in reviews if r['base_revision'] == report['revisions']['base']
                               and r['head'] == head and r['case'] == name
                               and r['base_text'] == before and r['candidate_text'] == after]
                    if len(matches) != 1:
                        raise ValueError(f'ASR transcript changed: {head}/{name}')
                    applied_reviews.append(dict(matches[0]))
            for sample in base + candidate:
                if sample['provider'] != 'cpu' or type(sample['pool_size']) is not int or sample['pool_size'] != 1 or type(sample['threads']) is not int or sample['threads'] not in (1, 2):
                    raise ValueError('unexpected ASR execution configuration')
                if sample['cpu_affinity'] != expected_affinity or sample['threads'] > len(cpus):
                    raise ValueError('ASR CPU affinity/threads changed')
                if sample['threads'] != reference['threads'] or set(sample['cases']) != set(CASES):
                    raise ValueError('incomparable ASR execution configuration')
                if not isinstance(sample['first_text'], str) or not sample['first_text'].strip() or sample['first_text'] != reference['first_text']:
                    raise ValueError('first-call transcript changed')
                for metric in LIMITS: positive(sample[metric])
                if sample['peak_rss_kib'] < sample['rss_kib']:
                    raise ValueError('peak RSS below current RSS')
                for name in CASES:
                    case, expected = sample['cases'][name], reference['cases'][name]
                    positive(case['seconds']); positive(case['duration'])
                    if type(case['iterations']) is not int or case['iterations'] != ASSETS['workload'][name]['iterations']:
                        raise ValueError('warm iteration count differs from pinned workload')
                    if abs(case['duration'] - ASSETS['workload'][name]['duration']) > 0.001:
                        raise ValueError('audio duration differs from pinned workload')
                    if not isinstance(case['text'], str) or not case['text'].strip():
                        raise ValueError(f'ASR transcript changed: {head}/{name}')
                    if abs(case['duration'] - expected['duration']) > 0.001:
                        raise ValueError('ASR duration changed')
            for metric, limit in LIMITS.items():
                ratio = statistics.median(c[metric] / b[metric] for b, c in zip(base, candidate))
                ratios[f'{head}/{metric}'] = ratio
                if ratio > limit: raise ValueError(f'ASR regression {head}/{metric}: {ratio:.3f}x > {limit:.2f}x')
            for name in CASES:
                ratio = statistics.median(c['cases'][name]['seconds'] / b['cases'][name]['seconds']
                                          for b, c in zip(base, candidate))
                ratios[f'{head}/{name}'] = ratio
                if ratio > WARM_LIMIT: raise ValueError(f'ASR regression {head}/{name}: {ratio:.3f}x > {WARM_LIMIT:.2f}x')
        return ratios
    except (KeyError, TypeError, IndexError) as error:
        raise ValueError(f'missing/invalid ASR evidence: {error}') from error


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def command(*args, **kwargs):
    return subprocess.run(args, cwd=ROOT, text=True, check=True, **kwargs)


def prepare_assets(models, audio):
    """Stage only pinned files: ambient manifests/sidecars cannot change the workload."""
    if not os.environ.get('GIGASTT_ASR_MODELS'):
        raise ValueError('set GIGASTT_ASR_MODELS to a directory containing rnnt/ and e2e_rnnt/')
    for head in HEADS:
        destination = models / head
        destination.mkdir()
        source = Path(os.environ['GIGASTT_ASR_MODELS']) / head
        for name, checksum in ASSETS['models'][head].items():
            if digest(source / name) != checksum: raise ValueError(f'unpinned model: {head}/{name}')
            shutil.copyfile(source / name, destination / name)
    audio.mkdir()
    frames = b''
    for name, checksum in ASSETS['audio'].items():
        source = ROOT / 'crates/gigastt/tests/fixtures' / name
        if digest(source) != checksum: raise ValueError(f'unpinned audio: {name}')
        shutil.copyfile(source, audio / name)
        with wave.open(str(source)) as wav:
            if (wav.getnchannels(), wav.getsampwidth(), wav.getframerate()) != (1, 2, 16000):
                raise ValueError('unexpected fixture PCM format')
            frames += wav.readframes(wav.getnframes()) + bytes(ASSETS['silence_frames'] * 2)
    with wave.open(str(audio / 'long.wav'), 'wb') as wav:
        wav.setparams((1, 2, 16000, 0, 'NONE', 'not compressed'))
        wav.writeframes(frames * ASSETS['long_repeats'])


def compare(args):
    if sys.platform != 'linux': raise ValueError('RSS protocol requires Linux')
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    # A failed rerun must not leave a previous green report behind.
    report = {'protocol': 2, 'reviewed_text_sha256': digest(REVIEW_FILE),
              'applied_text_reviews': [], 'assets': ASSETS, 'revisions': {}, 'binary_sha256': {},
              'heads': {h: {'base': [], 'candidate': []} for h in HEADS}}
    def save(): (output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    save()
    for side, ref in [('base', args.base), ('candidate', args.candidate)]:
        report['revisions'][side] = command('git', 'rev-parse', '--verify', f'{ref}^{{commit}}', capture_output=True).stdout.strip()
    command('git', 'merge-base', '--is-ancestor', report['revisions']['base'], report['revisions']['candidate'])
    harness = (ROOT / 'scripts/asr-revision-probe.rs').read_bytes()
    report['harness_sha256'] = hashlib.sha256(harness).hexdigest()
    report['rustc'] = command('rustc', '--version', capture_output=True).stdout.strip()
    report['platform'] = os.uname()._asdict() if hasattr(os.uname(), '_asdict') else list(os.uname())
    save()
    with tempfile.TemporaryDirectory(prefix='gigastt-asr-revisions-') as temporary:
        temp = Path(temporary)
        models = temp / 'models'; models.mkdir()
        prepare_assets(models, temp / 'audio')
        report['audio_sha256'] = {n: digest(temp / 'audio' / n) for n in CASES}
        binaries, trees = {}, []
        try:
            for side in ('base', 'candidate'):
                tree = temp / side
                command('git', 'worktree', 'add', '--detach', str(tree), report['revisions'][side])
                trees.append(tree)
                module = tree / 'crates/gigastt-core/src/inference/engine/mod.rs'
                (module.parent / 'asr_revision_probe.rs').write_bytes(harness)
                with module.open('a') as stream:
                    stream.write('\n#[cfg(all(test, target_os = "linux", feature = "file-decode"))]\nmod asr_revision_probe;\n')
                with (output / f'{side}-build.stderr').open('w') as errors:
                    built = subprocess.run(['cargo', 'test', '--manifest-path', str(tree / 'Cargo.toml'),
                             '--target-dir', str(args.target_dir.resolve() / 'asr-builds' / side), '-p', 'gigastt-core',
                             '--release', '--locked', '--lib', '--no-run', '--message-format=json'],
                             cwd=tree, text=True, stdout=subprocess.PIPE, stderr=errors, timeout=1800)
                (output / f'{side}-build.jsonl').write_text(built.stdout)
                built.check_returncode()
                artifacts = [json.loads(line) for line in built.stdout.splitlines() if line.startswith('{')]
                executables = [a['executable'] for a in artifacts if a.get('reason') == 'compiler-artifact'
                               and a.get('executable') and a.get('profile', {}).get('test')]
                if len(executables) != 1: raise ValueError('missing/ambiguous ASR executable')
                binary = temp / f'{side}-probe'; shutil.copyfile(executables[0], binary); binary.chmod(0o700)
                listing = command(str(binary), TEST, '--ignored', '--exact', '--list', capture_output=True)
                if f'{TEST}: test' not in listing.stdout.splitlines(): raise ValueError('ASR probe not discovered')
                binaries[side] = binary
                report['binary_sha256'][side] = digest(binary)
            # Build both first; filesystem pages are warm, graph caches are cold.
            with pinned_cpus() as cpus:
                report['cpu_affinity'] = cpus
                environment = {k: v for k, v in os.environ.items() if not k.startswith(('GIGASTT_', 'ORT_', 'OMP_', 'POLYVOICE_'))}
                for head in HEADS:
                    for index in range(ROUNDS):
                        for side in (('base', 'candidate') if index % 2 == 0 else ('candidate', 'base')):
                            env = {**environment, 'GIGASTT_ASR_PROBE_MODEL': str(models / head),
                                   'GIGASTT_ASR_PROBE_HEAD': head, 'GIGASTT_ASR_PROBE_AUDIO': str(temp / 'audio'),
                                   'GIGASTT_ASR_PROBE_CACHE': str(temp / f'cache-{head}-{side}-{index}')}
                            result = subprocess.run([str(binaries[side]), TEST, '--ignored', '--exact', '--test-threads=1', '--nocapture'],
                                                    cwd=ROOT, env=env, text=True, capture_output=True, timeout=300)
                            log = result.stdout + result.stderr
                            (output / f'{head}-{side}-{index}.log').write_text(log)
                            result.check_returncode()
                            records = [json.loads(line.split('asr revision ', 1)[1]) for line in log.splitlines() if line.startswith('asr revision ')]
                            if len(records) != 1: raise ValueError('missing/duplicate ASR measurement')
                            report['heads'][head][side].append(records[0]); save()
                            print(f'{head} {side} {index+1}/{ROUNDS}: {records[0]["cases"]["long.wav"]["seconds"]:.3f}s long file', flush=True)
            # Detect input mutation during execution before accepting measurements.
            for head in HEADS:
                for name, checksum in ASSETS['models'][head].items():
                    if digest(models / head / name) != checksum:
                        raise ValueError(f'model changed during measurement: {head}/{name}')
            if {n: digest(temp / 'audio' / n) for n in CASES} != report['audio_sha256']:
                raise ValueError('audio changed during measurement')
            report['inputs_verified'] = True
            save()
            try:
                ratios = check(report)
            finally:
                save()
            print(json.dumps({'ratios': ratios, 'applied_text_reviews': report['applied_text_reviews']}, indent=2))
        finally:
            for tree in reversed(trees): command('git', 'worktree', 'remove', '--force', str(tree))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', type=Path, help='validate an existing complete report')
    parser.add_argument('--base')
    parser.add_argument('--candidate', default='HEAD')
    parser.add_argument('--output', type=Path, default=ROOT / 'target/asr-revisions')
    parser.add_argument('--target-dir', type=Path, default=ROOT / 'target')
    args = parser.parse_args()
    if args.check:
        report = json.loads(args.check.read_text())
        ratios = check(report)
        print(json.dumps({'ratios': ratios, 'applied_text_reviews': report.get('applied_text_reviews', [])}, indent=2))
    elif not args.base: parser.error('--base is required for measurement')
    else: compare(args)


if __name__ == '__main__':
    try: main()
    except (OSError, ValueError, subprocess.SubprocessError) as error: sys.exit(f'ASR gate failed: {error}')
