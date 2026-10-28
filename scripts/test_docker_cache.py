#!/usr/bin/env python3
"""Parse real Docker dependency-cache workspaces without building an image."""
import json
from pathlib import Path
import shlex
import shutil
import subprocess
import tempfile
import tomllib
import unittest

ROOT = Path(__file__).resolve().parent.parent


def prepare_cache(dockerfile, directory):
    # These Dockerfiles use simple COPY instructions before one dummy build.
    # Execute only its preparation, never apt, cargo build, or cleanup commands.
    for line in (ROOT / dockerfile).read_text().replace('\\\n', '').splitlines():
        if line.startswith('COPY '):
            sources = shlex.split(line)[1:]
            destination = sources.pop()
            if destination.startswith('/') or not destination.endswith(('/', '.')):
                raise ValueError('expected relative dependency-cache COPY directory')
            target = directory / destination
            target.mkdir(parents=True, exist_ok=True)
            for source in sources:
                if source.startswith('--'):
                    raise ValueError('unexpected COPY option before dependency cache')
                original = ROOT / source
                if original.is_dir():
                    shutil.copytree(original, target, dirs_exist_ok=True)
                else:
                    shutil.copy2(original, target)
        elif line.startswith('RUN ') and 'cargo build --release' in line:
            preparation = line[4:].split('cargo build --release', 1)[0].rstrip()
            if not preparation.endswith('&&'):
                raise ValueError('expected preparation before Cargo dependency build')
            subprocess.run(['sh', '-eu', '-c', preparation[:-2]], cwd=directory, check=True)
            return
    raise ValueError('Dockerfile has no dependency-cache build')


def metadata(directory):
    return subprocess.run(['cargo', 'metadata', '--no-deps', '--locked', '--offline',
                           '--format-version', '1'], cwd=directory, text=True, capture_output=True)


def missing_targets(directory, result):
    data = json.loads(result.stdout)
    missing = [target['src_path'] for package in data['packages']
               for target in package['targets'] if not Path(target['src_path']).is_file()]
    # Cargo metadata can silently omit missing explicit targets, while cargo
    # build rejects them. Check declarations independently of that behavior.
    workspace = tomllib.loads((directory / 'Cargo.toml').read_text())
    for patches in workspace.get('patch', {}).values():
        for name, patch in patches.items():
            if 'path' in patch and not (directory / patch['path'] / 'Cargo.toml').is_file():
                missing.append(f"patch {name}: {patch['path']}/Cargo.toml")
    for member in workspace['workspace']['members']:
        crate = directory / member
        manifest = tomllib.loads((crate / 'Cargo.toml').read_text())
        for kind, folder in [('bench', 'benches'), ('test', 'tests'), ('example', 'examples'), ('bin', 'src/bin')]:
            for target in manifest.get(kind, []):
                if 'path' in target:
                    candidates = [crate / target['path']]
                else:
                    candidates = [crate / folder / (target['name'] + '.rs'),
                                  crate / folder / target['name'] / 'main.rs']
                    if kind == 'bin' and target['name'] == manifest['package']['name']:
                        candidates.append(crate / 'src/main.rs')
                if not any(path.is_file() for path in candidates):
                    missing.append(f"{member}: {kind} {target['name']}")
    return missing


class DockerCacheTests(unittest.TestCase):
    def test_cpu_and_cuda_cache_manifests_parse(self):
        for dockerfile in ('Dockerfile', 'Dockerfile.cuda'):
            with self.subTest(dockerfile=dockerfile), tempfile.TemporaryDirectory() as temp:
                directory = Path(temp)
                prepare_cache(dockerfile, directory)
                result = metadata(directory)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(missing_targets(directory, result), [])

    def test_new_explicit_target_without_stub_is_detected(self):
        for dockerfile in ('Dockerfile', 'Dockerfile.cuda'):
            with self.subTest(dockerfile=dockerfile), tempfile.TemporaryDirectory() as temp:
                directory = Path(temp)
                prepare_cache(dockerfile, directory)
                manifest = directory / 'crates/gigastt-core/Cargo.toml'
                with manifest.open('a') as output:
                    output.write('\n[[bench]]\nname="future_cache_probe"\nharness=false\n')
                result = metadata(directory)
                if result.returncode:
                    self.assertIn('future_cache_probe', result.stderr)
                else:
                    self.assertTrue(any('future_cache_probe' in path for path in missing_targets(directory, result)))


if __name__ == '__main__':
    unittest.main()
