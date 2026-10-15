"""Manual jobs must report real, fail-closed commit statuses for protected PRs."""
import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location(
    'manual_statuses', Path(__file__).with_name('manual-statuses.py'))
statuses = importlib.util.module_from_spec(spec)
spec.loader.exec_module(statuses)


class ManualStatusTests(unittest.TestCase):
    def test_start_marks_all_required_contexts_pending(self):
        self.assertEqual(statuses.updates('ci', None), {
            'Format': 'pending', 'Clippy': 'pending',
            'Unit Tests': 'pending', 'Model smoke': 'pending',
        })

    def test_only_successful_jobs_report_success(self):
        needs = {'fmt': {'result': 'success'}, 'clippy': {'result': 'failure'},
                 'unit-tests': {'result': 'cancelled'}, 'model-smoke': {'result': 'skipped'}}
        self.assertEqual(statuses.updates('ci', needs), {
            'Format': 'success', 'Clippy': 'failure',
            'Unit Tests': 'failure', 'Model smoke': 'failure',
        })

    def test_missing_or_unknown_results_fail_closed(self):
        for needs in [{}, {'public-content': {}},
                      {'public-content': {'result': 'neutral'}}]:
            self.assertEqual(statuses.updates('publication', needs),
                             {'Public content': 'failure'})

    def test_publication_success_and_unknown_group(self):
        self.assertEqual(statuses.updates('publication', {
            'public-content': {'result': 'success'}}), {'Public content': 'success'})
        with self.assertRaises(ValueError):
            statuses.updates('other', None)


if __name__ == '__main__':
    unittest.main()
