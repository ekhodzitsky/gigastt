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


def benchmark_executables(record):
    benches = {}
    for artifact in record['executables']:
        if artifact['kind'] != ['bench']:
            continue
        name = artifact['target']
        if name in benches:
            raise ValueError(f'duplicate benchmark target: {name}')
        benches[name] = artifact['path']
    if 'tokenizer' not in benches:
        raise ValueError('missing tokenizer benchmark executable')
    return benches


def measure_pair(first, second, reports, *, short_tokenizer=False):
    env = {**os.environ, 'CRITERION_HOME': str(reports)}
    selection = ['--exact', 'tokenizer_decode/decode_64_ids'] if short_tokenizer else []
    for (binary, source), operation in [(first, '--save-baseline'), (second, '--baseline')]:
        subprocess.run([binary, '--bench', '--noplot', operation, 'main', *selection],
                       cwd=source, env=env, check=True)


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
            base_benches = benchmark_executables(records['base'])
            candidate_benches = benchmark_executables(records['candidate'])
            if base_benches.keys() != candidate_benches.keys():
                raise ValueError('benchmark target sets differ between revisions')

            base_tokenizer = (base_benches.pop('tokenizer'), tree)
            candidate_tokenizer = (candidate_benches.pop('tokenizer'), ROOT)
            measure_pair(base_tokenizer, candidate_tokenizer, reports)
            measure_pair(candidate_tokenizer, base_tokenizer, reports / 'order-check' / 'reverse',
                         short_tokenizer=True)
            measure_pair(base_tokenizer, base_tokenizer, reports / 'order-check' / 'control',
                         short_tokenizer=True)
            for name in sorted(base_benches):
                measure_pair((base_benches[name], tree), (candidate_benches[name], ROOT), reports)

            for record in records.values():
                for artifact in record['executables']:
                    if executable_record(artifact['path'], record['target_dir'])['sha256'] != artifact['sha256']:
                        raise ValueError('benchmark executable changed after provenance capture')
        finally:
            subprocess.run(['git', 'worktree', 'remove', '--force', str(tree)], cwd=ROOT, check=True)


if __name__ == '__main__':
    main()
