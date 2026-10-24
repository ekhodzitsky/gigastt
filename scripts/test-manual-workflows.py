#!/usr/bin/env python3
"""Keep GitHub Actions opt-in and manual performance comparisons meaningful."""
import importlib.util
import json
import os
import sys
import textwrap
from pathlib import Path
import re
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class WorkflowPolicyTests(unittest.TestCase):
    def test_all_workflows_require_explicit_dispatch(self):
        workflows = [path for path in (ROOT / '.github/workflows').iterdir()
                     if path.suffix in {'.yml', '.yaml'}]
        self.assertTrue(workflows)
        for path in workflows:
            with self.subTest(workflow=path.name):
                block = re.search(r'^on:\n((?:[ \t].*\n|\n|#[^\n]*\n)*)', path.read_text(), re.M)
                self.assertIsNotNone(block)
                events = re.findall(r'^  ([a-z_]+):', block[1], re.M)
                self.assertEqual(events, ['workflow_dispatch'])

    def test_manual_ci_does_not_depend_on_automatic_event_payloads(self):
        workflow = (ROOT / '.github/workflows/ci.yml').read_text()
        self.assertNotIn('github.event.before', workflow)
        self.assertNotIn('github.event.pull_request', workflow)
        self.assertNotRegex(workflow, r"github.event_name\s*[!=]=\s*'(push|pull_request)'")
        self.assertIn('inputs.baseline_sha', workflow)
        self.assertIn('inputs.full', workflow)


class PublicationSourceTests(unittest.TestCase):
    def run_source_guard(self, head_repo, base_repo, head_sha):
        workflow = (ROOT / '.github/workflows/publication.yml').read_text()
        snippet = textwrap.dedent(workflow.split("<<'PYCODE'\n", 1)[1].split('          PYCODE', 1)[0])
        event = {'pull_request': {
            'head': {'sha': head_sha, 'repo': {'full_name': head_repo}},
            'base': {'repo': {'full_name': base_repo}},
        }}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'event.json'
            path.write_text(json.dumps(event))
            return subprocess.run([sys.executable, '-c', snippet, str(path)],
                                  env={**os.environ, 'GITHUB_SHA': 'a' * 40,
                                       'GITHUB_REPOSITORY': 'owner/project'},
                                  capture_output=True, text=True)

    def test_accepts_exact_head_from_same_repo_or_reviewed_fork_mirror(self):
        for head_repo in ['owner/project', 'contributor/project']:
            with self.subTest(head_repo=head_repo):
                result = self.run_source_guard(head_repo, 'owner/project', 'a' * 40)
                self.assertEqual(result.returncode, 0, result.stderr)

    def test_rejects_stale_head_even_from_the_same_repository(self):
        result = self.run_source_guard('owner/project', 'owner/project', 'b' * 40)
        self.assertNotEqual(result.returncode, 0)

    def test_rejects_another_target_repository_even_with_matching_head(self):
        result = self.run_source_guard('owner/project', 'elsewhere/project', 'a' * 40)
        self.assertNotEqual(result.returncode, 0)


class BaselineTests(unittest.TestCase):
    def setUp(self):
        spec = importlib.util.spec_from_file_location('baseline', ROOT / 'scripts/resolve-ci-baseline.py')
        self.baseline = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.baseline)
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.repo = Path(self.tmp.name)
        self.git('init', '-q')
        self.git('config', 'user.email', 'test@example.invalid')
        self.git('config', 'user.name', 'Test')
        self.git('commit', '--allow-empty', '-qm', 'base')
        self.base = self.git('rev-parse', 'HEAD')
        self.git('commit', '--allow-empty', '-qm', 'candidate')
        self.head = self.git('rev-parse', 'HEAD')

    def git(self, *args):
        return subprocess.check_output(['git', *args], cwd=self.repo, text=True).strip()

    def test_immutable_ancestor_passes(self):
        self.assertEqual(self.baseline.resolve(self.base, self.repo), self.base)

    def test_missing_moving_same_and_nonancestor_baselines_fail(self):
        self.git('checkout', '-q', '--detach', self.base)
        self.git('commit', '--allow-empty', '-qm', 'other branch')
        other = self.git('rev-parse', 'HEAD')
        self.git('checkout', '-q', '--detach', self.head)
        for revision in ['', 'HEAD^', '--help', 'a' * 40, self.head, other]:
            with self.subTest(revision=revision), self.assertRaises(ValueError):
                self.baseline.resolve(revision, self.repo)


if __name__ == '__main__':
    unittest.main()
