"""Cargo artifact provenance for the two revision-comparison scripts."""
import hashlib
import json
from pathlib import Path
import re
import subprocess


def digest(data):
    return hashlib.sha256(data).hexdigest()


def target_directory(root, label, revision):
    if label not in ('base', 'candidate') or not re.fullmatch(r'[0-9a-f]{40}', revision):
        raise ValueError('expected comparison side and full commit SHA')
    return Path(root).resolve() / label / revision


def executable_record(binary, target):
    binary = Path(binary).resolve(strict=True)
    target = Path(target).resolve()
    if not binary.is_relative_to(target):
        raise ValueError(f'Cargo executable outside revision target: {binary} (expected {target})')
    return {'path': str(binary), 'sha256': digest(binary.read_bytes())}


def build(source, target, cargo_args, log):
    source, target = Path(source).resolve(), Path(target).resolve()
    def git(*args):
        return subprocess.check_output(['git', '-C', str(source), *args])
    record = {'source_revision': git('rev-parse', 'HEAD').decode().strip(),
              'source_tree': git('rev-parse', 'HEAD^{tree}').decode().strip(),
              'tracked_diff_sha256': digest(git('diff', '--binary', 'HEAD')),
              'target_dir': str(target)}
    command = ['cargo', *cargo_args, '--manifest-path', str(source / 'Cargo.toml'),
               '--target-dir', str(target), '--message-format=json']
    result = subprocess.run(command, cwd=source, text=True, stdout=subprocess.PIPE, check=False)
    Path(log).write_text(result.stdout)
    artifacts = [json.loads(line) for line in result.stdout.splitlines() if line.startswith('{')]
    if result.returncode:
        for artifact in artifacts:
            if artifact.get('reason') == 'compiler-message':
                print(artifact['message'].get('rendered', ''), flush=True)
        result.check_returncode()
    record['executables'] = [
        {**executable_record(a['executable'], target), 'target': a['target']['name'],
         'kind': a['target']['kind'], 'test': a.get('profile', {}).get('test', False),
         'fresh': a['fresh']}
        for a in artifacts if a.get('reason') == 'compiler-artifact' and a.get('executable')]
    if not record['executables']:
        raise ValueError('Cargo produced no executable artifacts')
    return record
