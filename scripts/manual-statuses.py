#!/usr/bin/env python3
"""Report actual manual job outcomes through GitHub's commit status API.

GitHub excludes workflow_dispatch job checks from protected PR evaluation.
Commit statuses preserve the same required contexts and GitHub Actions identity.
"""
import argparse
import json
import os
import re
import subprocess

CONTEXTS = {
    'ci': {'fmt': 'Format', 'clippy': 'Clippy', 'unit-tests': 'Unit Tests',
           'model-smoke': 'Model smoke'},
    'publication': {'public-content': 'Public content'},
}


def updates(group, needs):
    if group not in CONTEXTS:
        raise ValueError(f'Unknown status group: {group}')
    return {
        context: ('pending' if needs is None else
                  'success' if needs.get(job, {}).get('result') == 'success' else 'failure')
        for job, context in CONTEXTS[group].items()
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('group', choices=CONTEXTS)
    parser.add_argument('--start', action='store_true')
    args = parser.parse_args()
    sha = os.environ['GITHUB_SHA']
    repo = os.environ['GITHUB_REPOSITORY']
    run_id = os.environ['GITHUB_RUN_ID']
    if not re.fullmatch(r'[0-9a-f]{40}', sha) or not run_id.isdigit():
        parser.error('Expected an exact GitHub commit SHA and numeric run ID')
    needs = None if args.start else json.loads(os.environ['JOB_RESULTS'])
    for context, state in updates(args.group, needs).items():
        payload = {
            'context': context, 'state': state,
            'description': f'Manual workflow: {state}',
            'target_url': f'{os.environ["GITHUB_SERVER_URL"]}/{repo}/actions/runs/{run_id}',
        }
        subprocess.run(
            ['gh', 'api', '--method', 'POST', f'repos/{repo}/statuses/{sha}', '--input', '-'],
            input=json.dumps(payload), text=True, stdout=subprocess.DEVNULL, check=True)
        print(f'{context}: {state}')


if __name__ == '__main__':
    main()
