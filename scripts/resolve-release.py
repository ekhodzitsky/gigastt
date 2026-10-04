#!/usr/bin/env python3
"""Resolve and validate release source without creating tags or publishing."""
import json
import os
import re
import subprocess
import sys
import tomllib
from pathlib import Path


def git(*args):
    return subprocess.check_output(['git', *args], text=True).strip()


def resolve(tag):
    # Docker tags cannot contain SemVer build metadata (+suffix).
    number = r'(?:0|[1-9][0-9]*)'
    if not re.fullmatch(rf'v{number}\.{number}\.{number}(?:-[0-9A-Za-z.-]+)?', tag):
        raise ValueError('expected vMAJOR.MINOR.PATCH with optional prerelease suffix')
    if '-' in tag:
        for part in tag.split('-', 1)[1].split('.'):
            if not part or (part.isdigit() and len(part) > 1 and part[0] == '0'):
                raise ValueError('invalid SemVer prerelease identifier')
    if len(tag[1:] + '-cuda') > 128:
        raise ValueError('release version exceeds Docker tag length')
    git('fetch', '--no-tags', '--depth=1', 'origin', f'refs/tags/{tag}')
    commit = git('rev-parse', '--verify', 'FETCH_HEAD^{commit}')
    version = workspace_version(commit)
    if version != tag[1:]:
        raise ValueError(f'tag {tag} disagrees with workspace version {version}')
    return {'tag': tag, 'commit': commit, 'version': version}


def workspace_version(commit):
    """Check that all built workspace crates share the source version."""
    manifest = tomllib.loads(git('show', f'{commit}:Cargo.toml'))
    version = manifest['workspace']['package']['version']
    for member in manifest['workspace']['members']:
        package = tomllib.loads(git('show', f'{commit}:{member}/Cargo.toml'))['package']
        effective = package['version']
        if effective == {'workspace': True}:
            effective = version
        if effective != version:
            raise ValueError(f'{member} version {effective} disagrees with {version}')
    return version


def provenance(source, env):
    """SLSA v1 workflow build type, including the manual source selection.

    The default action predicate records only the invocation commit. Keep that
    dependency and add the exact source consumed by every artifact producer.
    """
    repository = env['GITHUB_SERVER_URL'] + '/' + env['GITHUB_REPOSITORY']
    workflow_path = env['GITHUB_WORKFLOW_REF'].removesuffix('@' + env['GITHUB_REF']).removeprefix(env['GITHUB_REPOSITORY'] + '/')
    external = {'workflow': {
        'ref': env['GITHUB_REF'], 'repository': repository, 'path': workflow_path,
    }}
    if env['GITHUB_EVENT_NAME'] == 'workflow_dispatch':
        external['inputs'] = {'tag': source['tag']}
    return {
        'buildDefinition': {
            'buildType': 'https://actions.github.io/buildtypes/workflow/v1',
            'externalParameters': external,
            # Preserve the GitHub context carried by the standard provenance
            # generator; the attestations API validates these fields too.
            'internalParameters': {'github': {
                'event_name': env['GITHUB_EVENT_NAME'],
                'repository_id': env['GITHUB_REPOSITORY_ID'],
                'repository_owner_id': env['GITHUB_REPOSITORY_OWNER_ID'],
                'runner_environment': env['RUNNER_ENVIRONMENT'],
            }},
            'resolvedDependencies': [
                {'uri': 'git+' + repository + '@' + env['GITHUB_REF'],
                 'digest': {'gitCommit': env['GITHUB_SHA']}},
                {'uri': 'git+' + repository + '@refs/tags/' + source['tag'],
                 'digest': {'gitCommit': source['commit']}},
            ],
        },
        'runDetails': {
            'builder': {'id': env['GITHUB_SERVER_URL'] + '/' + env['GITHUB_WORKFLOW_REF']},
            'metadata': {'invocationId': repository + '/actions/runs/' + env['GITHUB_RUN_ID']
                         + '/attempts/' + env['GITHUB_RUN_ATTEMPT']},
        },
    }


def main():
    result = resolve(os.environ['RELEASE_TAG'])
    result['predicate'] = json.dumps(provenance(result, os.environ), separators=(',', ':'))
    metadata = {
        'schema': 1, 'tag': result['tag'], 'commit': result['commit'],
        'repository': os.environ['GITHUB_REPOSITORY'],
        'run_id': int(os.environ['GITHUB_RUN_ID']),
        'run_attempt': int(os.environ['GITHUB_RUN_ATTEMPT']),
    }
    Path(os.environ['RELEASE_METADATA_PATH']).write_text(json.dumps(metadata) + '\n')
    # Publish outputs only after every validation succeeds.
    with Path(os.environ['GITHUB_OUTPUT']).open('a') as output:
        for key, value in result.items():
            output.write(f'{key}={value}\n')
    print(json.dumps(result))


if __name__ == '__main__':
    try:
        main()
    except (KeyError, ValueError, OSError, subprocess.CalledProcessError) as error:
        sys.exit(f'release source validation failed: {error}')
