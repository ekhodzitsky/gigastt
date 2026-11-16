#!/usr/bin/env python3
"""Run Criterion on separately compiled base/candidate revisions; share only measurements."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import tempfile

from revision_build import build, executable_record, target_directory

ROOT = Path(__file__).resolve().parent.parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', required=True)
    args = parser.parse_args()
    base = subprocess.check_output(['git', 'rev-parse', '--verify', f'{args.base}^{{commit}}'],
                                   cwd=ROOT, text=True).strip()
    candidate = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    reports = Path(os.environ['CRITERION_HOME']).resolve()
    if reports.exists() and any(reports.iterdir()):
        raise ValueError('Criterion comparison requires an empty measurement directory')
    reports.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, 'CRITERION_HOME': str(reports)}
    records = {}
    with tempfile.TemporaryDirectory(prefix='gigastt-benchmark-revisions-') as tmp:
        tree = Path(tmp) / 'base'
        subprocess.run(['git', 'worktree', 'add', '--detach', str(tree), base], cwd=ROOT, check=True)
        try:
            for label, source, revision in [('base', tree, base), ('candidate', ROOT, candidate)]:
                target = target_directory(ROOT / 'target/benchmark-builds', label, revision)
                records[label] = build(source, target,
                                       ['bench', '-p', 'gigastt-core', '--locked', '--features',
                                        '__internals', '--no-run'], reports / f'{label}-build.jsonl')
                if records[label]['source_revision'] != revision:
                    raise ValueError('source revision changed during comparison')
                (reports / 'build-provenance.json').write_text(json.dumps(records, indent=2) + '\n')
                baseline = '--save-baseline' if label == 'base' else '--baseline'
                subprocess.run(['cargo', 'bench', '--manifest-path', str(source / 'Cargo.toml'),
                                '--target-dir', str(target), '-p', 'gigastt-core', '--locked',
                                '--features', '__internals', '--', baseline, 'main'],
                               cwd=source, env=env, check=True)
                for artifact in records[label]['executables']:
                    if executable_record(artifact['path'], target)['sha256'] != artifact['sha256']:
                        raise ValueError('benchmark executable changed after provenance capture')
        finally:
            subprocess.run(['git', 'worktree', 'remove', '--force', str(tree)], cwd=ROOT, check=True)


if __name__ == '__main__':
    main()
