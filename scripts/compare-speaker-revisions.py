#!/usr/bin/env python3
"""Build the same production-path probe in immutable base/candidate checkouts."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parent.parent
TEST = 'inference::diarization::revision_probe::test_speaker_revision_probe'


def run(*args, **kwargs):
    return subprocess.run(args, cwd=ROOT, check=True, text=True, **kwargs)


def revision(ref):
    return run('git', 'rev-parse', '--verify', f'{ref}^{{commit}}', capture_output=True).stdout.strip()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', required=True)
    parser.add_argument('--candidate', default='HEAD')
    parser.add_argument('--output', type=Path, default=ROOT / 'target/speaker-revisions')
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    harness = (ROOT / 'scripts/speaker-revision-probe.rs').read_bytes()
    report = {'base_revision': revision(args.base), 'candidate_revision': revision(args.candidate),
              'harness_sha256': hashlib.sha256(harness).hexdigest(),
              'rustc': run('rustc', '--version', capture_output=True).stdout.strip(),
              'base': [], 'candidate': []}
    (output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    # Both builds finish before timing. Copy the statically linked test executables
    # so Cargo rebuilding the second checkout cannot replace the first binary.
    with tempfile.TemporaryDirectory(prefix='gigastt-speaker-revisions-') as tmp:
        worktrees = []
        binaries = {}
        try:
            for label in ['base', 'candidate']:
                tree = Path(tmp) / label
                run('git', 'worktree', 'add', '--detach', str(tree), report[f'{label}_revision'])
                worktrees.append(tree)
                module = tree / 'crates/gigastt-core/src/inference/diarization.rs'
                probe = module.parent / 'speaker_revision_probe.rs'
                probe.write_bytes(harness)
                with module.open('a') as target:
                    target.write('\n#[cfg(all(test, feature = "file-decode", target_os = "linux"))]\n'
                                 '#[path = "speaker_revision_probe.rs"]\nmod revision_probe;\n')
                build = subprocess.run(['cargo', 'test', '--manifest-path', str(tree / 'Cargo.toml'),
                            '--target-dir', str(ROOT / 'target'), '-p', 'gigastt-core',
                            '--release', '--locked', '--lib', '--no-run', '--message-format=json'],
                            cwd=ROOT, text=True, stdout=subprocess.PIPE, check=False)
                (output / f'{label}-build.jsonl').write_text(build.stdout)
                artifacts = [json.loads(line) for line in build.stdout.splitlines() if line.startswith('{')]
                if build.returncode:
                    for artifact in artifacts:
                        if artifact.get('reason') == 'compiler-message':
                            print(artifact['message'].get('rendered', ''), flush=True)
                    build.check_returncode()
                tests = [a['executable'] for a in artifacts if a.get('reason') == 'compiler-artifact'
                         and a.get('executable') and a.get('profile', {}).get('test')]
                if len(tests) != 1:
                    raise ValueError(f'expected one test executable for {label}, got {tests}')
                binary = Path(tmp) / f'{label}-probe'
                shutil.copy2(tests[0], binary)
                listing = run(str(binary), TEST, '--ignored', '--exact', '--list', capture_output=True)
                if f'{TEST}: test' not in listing.stdout.splitlines():
                    raise ValueError(f'missing revision probe in {label}')
                binaries[label] = binary
            for round_index in range(7):
                order = ['base', 'candidate'] if round_index % 2 == 0 else ['candidate', 'base']
                for label in order:
                    measured = subprocess.run([str(binaries[label]), TEST, '--ignored', '--exact',
                                               '--test-threads=1', '--nocapture'], cwd=ROOT,
                                              text=True, capture_output=True, timeout=180, check=False)
                    log = measured.stdout + measured.stderr
                    (output / f'{label}-{round_index}.log').write_text(log)
                    measured.check_returncode()
                    records = [json.loads(line.partition('speaker revision ')[2])
                               for line in log.splitlines() if 'speaker revision {' in line]
                    if len(records) != 1:
                        raise ValueError(f'missing or duplicate speaker measurement: {label}')
                    report[label].append(records[0])
                    print(f'{label} round {round_index + 1}: '
                          f'{records[0]["embedding_seconds"]:.4f} s, '
                          f'peak RSS {records[0]["peak_rss_kib"]} KiB', flush=True)
                    (output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
            run('python3', str(ROOT / 'scripts/performance-gates.py'), 'speaker-revisions',
                str(output / 'report.json'))
        finally:
            for tree in reversed(worktrees):
                run('git', 'worktree', 'remove', '--force', str(tree))


if __name__ == '__main__':
    main()
