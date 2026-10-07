#!/usr/bin/env python3
"""Fail closed on missing measurements, regressions, or unverified release commits."""
import argparse
import json
import math
import os
from pathlib import Path
import re
import statistics
import subprocess
import sys

REQUIRED_JOBS = {'Format', 'Clippy', 'Unit Tests', 'Model smoke',
                 'Speaker performance', 'Benchmarks (regression gate)',
                 'Quality Gate (WER/RTF/RSS)'}


def number(value, positive=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f'invalid measurement: {value!r}')
    if positive and value <= 0:
        raise ValueError(f'nonpositive measurement: {value}')
    return value


def check_criterion(root):
    root = Path(root)
    baseline = {p.parent.parent for p in root.glob('**/main/estimates.json')}
    current = {p.parent.parent for p in root.glob('**/new/estimates.json')}
    if not baseline or baseline != current:
        raise ValueError(f'missing benchmark measurements: baseline-only={baseline-current}, current-only={current-baseline}')
    regressions = []
    for benchmark in sorted(baseline):
        try:
            for version in ['main', 'new']:
                estimate = json.loads((benchmark / version / 'estimates.json').read_text())
                number(estimate['mean']['point_estimate'], positive=True)
            change = json.loads((benchmark / 'change/estimates.json').read_text())
            interval = change['mean']['confidence_interval']
            lower = number(interval['lower_bound'])
            upper = number(interval['upper_bound'])
            if lower > upper:
                raise ValueError('reversed confidence interval')
        except (OSError, KeyError, TypeError, ValueError) as error:
            raise ValueError(f'invalid or missing benchmark data for {benchmark}: {error}') from error
        if lower > 0.05:
            regressions.append(f'{benchmark.relative_to(root)}: lower confidence bound +{lower:.1%}')
    if regressions:
        raise ValueError('performance regressions above 5%:\n' + '\n'.join(regressions))
    return len(baseline)


def latest_main_run(runs, commit):
    matches = [run for run in runs if run['head_sha'] == commit and run['event'] == 'push'
               and run['head_branch'] == 'main']
    if not matches:
        raise ValueError(f'no main-push CI run for release commit {commit}')
    return max(matches, key=lambda run: (run['id'], run.get('run_attempt', 1)))


def check_speaker_revisions(report):
    """Compare separate base/candidate processes, including each linked runtime."""
    limits = {'load_seconds': 1.20, 'embedding_seconds': 1.20,
              'offline_seconds': 1.20, 'streaming_seconds': 1.20,
              'rss_kib': 1.10, 'peak_rss_kib': 1.10}
    try:
        base, candidate = report['base'], report['candidate']
        if len(base) != 7 or len(candidate) != 7:
            raise ValueError('require seven complete paired speaker measurements')
        reference = base[0]
        for sample in base + candidate:
            for key in limits:
                number(sample[key], positive=True)
            if sample['peak_rss_kib'] < sample['rss_kib']:
                raise ValueError('peak RSS below current RSS')
            for key in ['model_sha256', 'fixture_sha256', 'pool_size', 'logical_cpus']:
                if sample[key] != reference[key]:
                    raise ValueError(f'incomparable speaker protocol: {key}')
            if sample['pool_size'] != 4 or number(sample['logical_cpus'], positive=True) < 1:
                raise ValueError('invalid speaker pool/CPU configuration')
            hashes = [sample['model_sha256'], *sample['fixture_sha256']]
            if len(hashes) != 4 or not all(re.fullmatch(r'[0-9a-f]{64}', h) for h in hashes):
                raise ValueError('missing speaker asset fingerprints')
            vectors = sample['embeddings']
            if len(vectors) != 3 or any(len(v) != 256 for v in vectors):
                raise ValueError('missing speaker embeddings')
            for actual, expected in zip(vectors, reference['embeddings']):
                if any(abs(number(a) - number(b)) >= 0.001 for a, b in zip(actual, expected)):
                    raise ValueError('speaker embedding quality regression')
            for key in ['offline_turns', 'streaming_turns']:
                turns = sample[key]
                if not turns or turns != reference[key]:
                    raise ValueError(f'speaker quality changed: {key}')
                for start, end, speaker in turns:
                    if number(start) < 0 or number(end) <= start or number(speaker) < 0:
                        raise ValueError('invalid speaker turn')
        ratios = {key: statistics.median(c[key] / b[key] for b, c in zip(base, candidate))
                  for key in limits}
        failures = [f'{key}: {ratios[key]:.3f}x (limit {limit:.2f}x)'
                    for key, limit in limits.items() if ratios[key] > limit]
        if failures:
            raise ValueError('speaker revision regressions: ' + ', '.join(failures))
        return ratios
    except (KeyError, TypeError, IndexError) as error:
        raise ValueError(f'missing or invalid speaker revision data: {error}') from error


def check_release(runs, jobs, commit):
    run = latest_main_run(runs, commit)
    if run['status'] != 'completed' or run['conclusion'] != 'success':
        raise ValueError(f'latest CI for {commit} must complete successfully before release')
    succeeded = {job['name'] for job in jobs if job['conclusion'] == 'success'}
    if missing := REQUIRED_JOBS - succeeded:
        raise ValueError('required release gates missing or unsuccessful: ' + ', '.join(sorted(missing)))
    return run['id']


def api_pages(endpoint, key):
    # gh handles authentication; pagination prevents unrelated jobs hiding gates.
    result = subprocess.check_output(['gh', 'api', '--paginate', '--slurp', endpoint], text=True)
    return [item for page in json.loads(result) for item in page[key]]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('criterion').add_argument('directory', type=Path)
    sub.add_parser('speaker-revisions').add_argument('report', type=Path)
    sub.add_parser('release').add_argument('commit')
    args = parser.parse_args()
    if args.command == 'criterion':
        print(f'Validated {check_criterion(args.directory)} benchmark comparisons')
    elif args.command == 'speaker-revisions':
        print(json.dumps(check_speaker_revisions(json.loads(args.report.read_text())), indent=2))
    else:
        repo = os.environ['GITHUB_REPOSITORY']
        runs = api_pages(f'repos/{repo}/actions/workflows/ci.yml/runs?event=push&head_sha={args.commit}&per_page=100', 'workflow_runs')
        run = latest_main_run(runs, args.commit)
        jobs = api_pages(f'repos/{repo}/actions/runs/{run["id"]}/attempts/{run.get("run_attempt", 1)}/jobs?per_page=100', 'jobs')
        print(f'Release commit {args.commit} validated by CI run {check_release(runs, jobs, args.commit)}')


if __name__ == '__main__':
    try:
        main()
    except (ValueError, KeyError, OSError, subprocess.CalledProcessError) as error:
        sys.exit(f'performance/release gate failed: {error}')
