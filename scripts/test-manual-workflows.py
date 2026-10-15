#!/usr/bin/env python3
"""Keep GitHub Actions opt-in and manual performance comparisons meaningful."""
import importlib.util
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
