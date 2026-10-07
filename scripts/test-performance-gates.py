#!/usr/bin/env python3
"""Model-free regression tests for fail-closed performance and release checks."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location('gates', Path(__file__).with_name('performance-gates.py'))
gates = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gates)


class CriterionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.write('mel/main/estimates.json', {'mean': {'point_estimate': 100.0}})
        self.write('mel/new/estimates.json', {'mean': {'point_estimate': 101.0}})
        self.change(0.0)

    def write(self, path, data):
        path = self.root / path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data))

    def change(self, lower):
        self.write('mel/change/estimates.json', {'mean': {'confidence_interval': {
            'lower_bound': lower, 'upper_bound': max(lower, 0.03)}}})

    def test_normal_measurement_passes(self):
        self.assertEqual(gates.check_criterion(self.root), 1)

    def test_regression_fails(self):
        self.change(0.06)
        with self.assertRaises(ValueError):
            gates.check_criterion(self.root)

    def test_missing_and_invalid_measurements_fail(self):
        for path in ['mel/main/estimates.json', 'mel/new/estimates.json', 'mel/change/estimates.json']:
            with self.subTest(path=path):
                file = self.root / path
                original = file.read_text()
                file.unlink()
                with self.assertRaises(ValueError):
                    gates.check_criterion(self.root)
                file.write_text(original)
        for value in [float('nan'), float('inf'), -1.0, 0.0, True]:
            with self.subTest(value=value):
                self.write('mel/new/estimates.json', {'mean': {'point_estimate': value}})
                with self.assertRaises(ValueError):
                    gates.check_criterion(self.root)

    def test_empty_report_fails(self):
        with self.assertRaises(ValueError):
            gates.check_criterion(self.root / 'absent')

    def test_new_benchmark_without_baseline_fails(self):
        self.write('speaker/new/estimates.json', {'mean': {'point_estimate': 1.0}})
        with self.assertRaises(ValueError):
            gates.check_criterion(self.root)

    def test_nonfinite_confidence_bounds_fail(self):
        self.change(float('nan'))
        with self.assertRaises(ValueError):
            gates.check_criterion(self.root)


class ReleaseTests(unittest.TestCase):
    def run_record(self, **kw):
        return {'id': 10, 'head_sha': 'a' * 40, 'event': 'push', 'head_branch': 'main',
                'status': 'completed', 'conclusion': 'success', **kw}

    def jobs(self):
        return [{'name': name, 'conclusion': 'success'} for name in gates.REQUIRED_JOBS]

    def test_only_successful_main_ci_for_exact_commit_is_accepted(self):
        self.assertEqual(gates.check_release([self.run_record()], self.jobs(), 'a' * 40), 10)
        for change in [{'head_sha': 'b' * 40}, {'event': 'pull_request'}, {'head_branch': 'branch'},
                       {'status': 'in_progress'}, {'conclusion': 'failure'}, {'conclusion': 'cancelled'}]:
            with self.subTest(change=change), self.assertRaises(ValueError):
                gates.check_release([self.run_record(**change)], self.jobs(), 'a' * 40)

    def test_latest_failed_run_cannot_reuse_older_green(self):
        with self.assertRaises(ValueError):
            gates.check_release([self.run_record(), self.run_record(id=11, conclusion='failure')],
                                self.jobs(), 'a' * 40)

    def test_missing_or_skipped_gate_fails(self):
        with self.assertRaises(ValueError):
            gates.check_release([], self.jobs(), 'a' * 40)
        with self.assertRaises(ValueError):
            gates.check_release([self.run_record()], self.jobs()[:-1], 'a' * 40)
        jobs = self.jobs()
        jobs[-1]['conclusion'] = 'skipped'
        with self.assertRaises(ValueError):
            gates.check_release([self.run_record()], jobs, 'a' * 40)


if __name__ == '__main__':
    unittest.main()
