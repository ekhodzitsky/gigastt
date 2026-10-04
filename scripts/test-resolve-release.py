#!/usr/bin/env python3
"""Local Git fixtures; never contact GitHub or publish release artifacts."""
import os
import json
from pathlib import Path
import subprocess
import tempfile
import unittest

SCRIPT = Path(__file__).with_name('resolve-release.py').resolve()


class ReleaseSourceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.repo = self.root / 'source'
        self.repo.mkdir()
        self.git('init', '-q')
        self.git('config', 'user.email', 'release-test@example.invalid')
        self.git('config', 'user.name', 'Release test')
        self.write_version('2.22.0')
        self.release = self.git('rev-parse', 'HEAD')
        self.git('tag', '-a', 'v2.22.0', '-m', 'Release')
        self.write_version('2.23.0')
        self.dispatch = self.git('rev-parse', 'HEAD')
        self.checkout = self.root / 'checkout'
        subprocess.run(['git', 'clone', '-q', str(self.repo), str(self.checkout)], check=True)
        self.output = self.root / 'outputs'

    def git(self, *args):
        return subprocess.check_output(['git', '-C', str(self.repo), *args], text=True).strip()

    def write_version(self, version, member=None):
        (self.repo / 'Cargo.toml').write_text(f'[workspace]\nmembers=["crate"]\n[workspace.package]\nversion="{version}"\n')
        (self.repo / 'crate').mkdir(exist_ok=True)
        field = f'version="{member}"' if member else 'version.workspace=true'
        (self.repo / 'crate/Cargo.toml').write_text(f'[package]\nname="example"\n{field}\n')
        self.git('add', '.')
        self.git('commit', '-qm', 'Version fixture')

    def run_resolver(self, tag, **github):
        return subprocess.run(['python3', str(SCRIPT)], cwd=self.checkout,
                              env={**os.environ, 'RELEASE_TAG': tag, 'GITHUB_OUTPUT': str(self.output),
                                   'RELEASE_METADATA_PATH': str(self.root / 'metadata.json'),
                                   'GITHUB_SERVER_URL': 'https://github.com', 'GITHUB_REPOSITORY': 'example/repo',
                                   'GITHUB_WORKFLOW_REF': 'example/repo/.github/workflows/release.yml@refs/heads/main',
                                   'GITHUB_REF': 'refs/heads/main', 'GITHUB_SHA': self.dispatch,
                                   'GITHUB_RUN_ID': '123', 'GITHUB_RUN_ATTEMPT': '2',
                                   'GITHUB_REPOSITORY_ID': '1234', 'GITHUB_REPOSITORY_OWNER_ID': '5678',
                                   'RUNNER_ENVIRONMENT': 'github-hosted',
                                   'GITHUB_EVENT_NAME': 'workflow_dispatch', **github},
                              text=True, capture_output=True)

    def test_manual_dispatch_uses_annotated_tag_not_workflow_head(self):
        result = self.run_resolver('v2.22.0')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn(f'commit={self.release}\n', self.output.read_text())
        metadata = json.loads((self.root / 'metadata.json').read_text())
        self.assertEqual(metadata, {'schema': 1, 'tag': 'v2.22.0', 'commit': self.release,
                                   'repository': 'example/repo', 'run_id': 123, 'run_attempt': 2})
        outputs = dict(line.split('=', 1) for line in self.output.read_text().splitlines())
        predicate = json.loads(outputs['predicate'])
        self.assertEqual(predicate['buildDefinition']['externalParameters']['inputs'], {'tag': 'v2.22.0'})
        self.assertEqual(predicate['buildDefinition']['internalParameters']['github'], {
            'event_name': 'workflow_dispatch', 'repository_id': '1234',
            'repository_owner_id': '5678', 'runner_environment': 'github-hosted',
        })
        dependencies = predicate['buildDefinition']['resolvedDependencies']
        self.assertEqual(dependencies[0]['digest']['gitCommit'], self.dispatch)
        self.assertEqual(dependencies[1]['digest']['gitCommit'], self.release)
        self.assertTrue(dependencies[1]['uri'].endswith('@refs/tags/v2.22.0'))
        self.assertTrue(predicate['runDetails']['metadata']['invocationId'].endswith('/123/attempts/2'))

    def test_push_resolves_same_commit_and_records_tag_invocation(self):
        result = self.run_resolver('v2.22.0', GITHUB_REF='refs/tags/v2.22.0', GITHUB_SHA=self.release, GITHUB_EVENT_NAME='push',
                                   GITHUB_WORKFLOW_REF='example/repo/.github/workflows/release.yml@refs/tags/v2.22.0')
        self.assertEqual(result.returncode, 0, result.stderr)
        outputs = dict(line.split('=', 1) for line in self.output.read_text().splitlines())
        predicate = json.loads(outputs['predicate'])
        self.assertEqual(predicate['buildDefinition']['externalParameters']['workflow']['ref'], 'refs/tags/v2.22.0')
        self.assertNotIn('inputs', predicate['buildDefinition']['externalParameters'])
        self.assertEqual(outputs['commit'], self.release)
        # Moving the tag afterwards cannot change the exported checkout identity.
        self.git('tag', '-f', 'v2.22.0', self.dispatch)
        actual = subprocess.check_output(['git', 'show', outputs['commit'] + ':Cargo.toml'], cwd=self.checkout, text=True)
        self.assertIn('version="2.22.0"', actual)

    def test_prerelease_lightweight_tag_is_supported(self):
        self.write_version('2.23.0-rc.1')
        self.git('tag', 'v2.23.0-rc.1')
        result = self.run_resolver('v2.23.0-rc.1')
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_missing_or_invalid_tags_emit_no_outputs(self):
        for tag in ['v2.99.0', 'main', 'v02.22.0', 'v2.22.0-01', 'v2.22.0-rc..1', 'v2.22.0+build', 'v2.22.0\ninjected=yes']:
            with self.subTest(tag=tag):
                self.assertNotEqual(self.run_resolver(tag).returncode, 0)
                self.assertFalse(self.output.exists())

    def test_workspace_version_mismatch_emits_no_outputs(self):
        self.git('tag', 'v2.24.0')
        result = self.run_resolver('v2.24.0')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('disagrees with workspace', result.stderr)
        self.assertFalse(self.output.exists())

    def test_explicit_member_version_mismatch_emits_no_outputs(self):
        self.write_version('2.24.0', member='2.23.0')
        self.git('tag', 'v2.24.0')
        result = self.run_resolver('v2.24.0')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('crate version', result.stderr)
        self.assertFalse(self.output.exists())


if __name__ == '__main__':
    unittest.main()
