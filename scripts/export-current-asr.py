#!/usr/bin/env python3
"""Freeze auditable current-ASR evidence without copying weights or graph caches."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('destination', type=Path)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location('scoring', repo / 'scripts/score-current-asr.py')
    metrics = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(metrics)
    protocol = json.loads((args.directory / 'protocol.json').read_text())
    catalog = {row['head']: row['assets'] for row in protocol['active_catalog']}
    for profile in protocol['profiles']:
        expected = catalog.get(profile['head'], {k: v['sha256'] for k, v in profile['assets'].items()})
        if metrics.verify_assets(Path(profile['model_dir']), expected) != profile['assets']:
            raise ValueError('source model changed during campaign')
    cases = json.loads((args.directory / 'cases.json').read_text())
    for row in cases:
        if hashlib.sha256(Path(row['audio']).read_bytes()).hexdigest() != row['sha256']:
            raise ValueError('source audio changed during campaign')
    summary = metrics.summarize(args.directory)
    args.destination.mkdir(parents=True, exist_ok=False)
    for name in ['protocol.json', 'run-protocol.json', 'smoke-cases.json', 'v3_ctc-smoke.jsonl', 'v3_e2e_ctc-smoke.jsonl']:
        shutil.copy2(args.directory / name, args.destination / name)
    for name in ['cases.json', 'timing-cases.json']:
        rows = json.loads((args.directory / name).read_text())
        (args.destination / name).write_text('[\n' + ',\n'.join(json.dumps(r, ensure_ascii=False, separators=(',', ':')) for r in rows) + '\n]\n')
    target = args.destination / 'runs'
    target.mkdir()
    for source in sorted((args.directory / 'runs').glob('*.jsonl')):
        rows = [json.loads(line) for line in source.read_text().splitlines()]
        for row in rows:
            row.pop('reference_itn', None)  # Derived duplicate; original gold remains in cases.json.
        (target / source.name).write_text(''.join(json.dumps(r, ensure_ascii=False, separators=(',', ':')) + '\n' for r in rows))
    if (args.directory / 'punctuation').exists():
        shutil.copytree(args.directory / 'punctuation', args.destination / 'punctuation')
    (args.destination / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
    paths = [repo / 'Cargo.lock', repo / 'benchmark/common.py', repo / 'crates/gigastt-core/src/model/variant.rs',
             repo / 'crates/gigastt-core/examples/current_asr_probe.rs',
             repo / 'crates/gigastt-quantize/examples/inspect_ops.rs',
             *repo.glob('scripts/*current-asr.py'), repo / 'scripts/replay-current-punctuation.py']
    changed = subprocess.check_output(['git', 'diff', '--name-only', 'HEAD'], cwd=repo, text=True).splitlines()
    paths.extend(repo / name for name in changed if (repo / name).is_file())
    identity = dict(base_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip(),
                    rustc=subprocess.check_output(['rustc', '--version', '--verbose'], text=True).strip(),
                    source_sha256={str(p.relative_to(repo)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(set(paths))})
    (args.destination / 'source-identity.json').write_text(json.dumps(identity, ensure_ascii=False, indent=2) + '\n')
    print(args.destination)


if __name__ == '__main__':main()
