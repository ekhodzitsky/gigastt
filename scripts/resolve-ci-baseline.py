#!/usr/bin/env python3
"""Require an immutable, distinct ancestor for manual CI comparisons."""
import argparse
from pathlib import Path
import re
import subprocess
import sys


def resolve(revision, repository=Path('.')):
    if not re.fullmatch(r'[0-9a-f]{40}', revision):
        raise ValueError('baseline_sha must be a full lowercase commit SHA')

    def git(*args):
        return subprocess.check_output(['git', *args], cwd=repository,
                                       stderr=subprocess.PIPE, text=True).strip()

    try:
        commit = git('rev-parse', '--verify', '--end-of-options', revision + '^{commit}')
        if commit != revision or commit == git('rev-parse', 'HEAD'):
            raise ValueError('baseline must be a distinct commit, not the candidate')
        git('merge-base', '--is-ancestor', commit, 'HEAD')
    except subprocess.CalledProcessError as error:
        raise ValueError('baseline must be an available ancestor of the candidate') from error
    return commit


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('baseline_sha')
    args = parser.parse_args()
    try:
        print(resolve(args.baseline_sha))
    except (ValueError, OSError) as error:
        sys.exit(f'Invalid CI baseline: {error}')
