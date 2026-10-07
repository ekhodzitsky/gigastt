#!/usr/bin/env python3
"""Fail closed on missing measurements, regressions, or unverified release commits."""
import argparse
import json
import math
import os
from pathlib import Path
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
    sub.add_parser('release').add_argument('commit')
    args = parser.parse_args()
    if args.command == 'criterion':
        print(f'Validated {check_criterion(args.directory)} benchmark comparisons')
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
