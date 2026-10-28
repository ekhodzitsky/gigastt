#!/usr/bin/env python3
"""Protect comparison metrics against lexical alignment and metadata mistakes."""
import io
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch
from urllib.error import HTTPError

import benchmark_stitch_case as benchmark

from benchmark_stitch_case import casing, delta


class ComparisonTests(unittest.TestCase):
    def test_optimized_python_still_rejects_lexical_changes(self):
        code = """
from benchmark_stitch_case import delta
base = {'text': 'да', 'words': [{'word': 'да', 'start': 0.1}]}
candidate = {'text': 'нет', 'words': [{'word': 'нет', 'start': 0.1}]}
delta(base, candidate)
"""
        result = subprocess.run([sys.executable, '-O', '-c', code],
                                cwd=Path(__file__).parent, capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('lexical/punctuation change', result.stderr)

    def test_timing_warmups_compare_arms_and_preserve_failed_evidence(self):
        base = {'text': 'да', 'words': [{'word': 'да', 'start': 0.1}]}
        candidate = {'text': 'нет', 'words': [{'word': 'нет', 'start': 0.1}]}
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            with patch.object(benchmark, 'request', side_effect=[(base, 1), (candidate, 1)]):
                with self.assertRaisesRegex(ValueError, 'lexical/punctuation change'):
                    benchmark.warmup_pair({'baseline': 1, 'candidate': 2}, b'audio',
                                          output, 'rnnt', 'sample')
            self.assertEqual(len(list(output.glob('warmup-*.json'))), 2)

    def test_timing_warmups_record_allowed_case_changes(self):
        base = {'text': 'Да', 'words': [{'word': 'Да', 'start': 0.1}]}
        candidate = {'text': 'да', 'words': [{'word': 'да', 'start': 0.1}]}
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            with patch.object(benchmark, 'request', side_effect=[(base, 1), (candidate, 1)]):
                expected = benchmark.warmup_pair({'baseline': 1, 'candidate': 2}, b'audio',
                                                output, 'rnnt', 'sample')
            self.assertEqual(expected, {'baseline': base, 'candidate': candidate})
            changes = json.loads((output / 'warmup-rnnt-sample-changes.json').read_text())
            self.assertEqual(changes[0]['candidate'], 'да')

    def test_startup_waits_for_model_readiness(self):
        ready = MagicMock()
        ready.__enter__.return_value.status = 200
        not_ready = HTTPError('http://127.0.0.1:1/ready', 503, 'loading', {}, None)
        with patch.object(benchmark.subprocess, 'Popen') as popen, \
                patch.object(benchmark.HTTP, 'open', side_effect=[not_ready, ready]) as opener, \
                patch.object(benchmark.time, 'sleep'):
            popen.return_value.poll.return_value = None
            proc, _, _ = benchmark.start_server(
                Path('/binary'), 1, 'rnnt', SimpleNamespace(models=Path('/models')), io.StringIO())
        self.assertIs(proc, popen.return_value)
        self.assertEqual(opener.call_count, 2)
        self.assertTrue(all(call.args[0].endswith('/ready') for call in opener.call_args_list))
        popen.return_value.kill.assert_not_called()

    def test_insertion_does_not_shift_capitalization(self):
        result = casing('Мы приехали в Москву.', 'Мы уже приехали в москву.')
        self.assertEqual(result['aligned_words'], 4)
        self.assertEqual(result['initial_case_errors'], 1)
        self.assertEqual(result['capitalization_tp_fp_fn'], [1, 0, 1])
        self.assertEqual(sum(result['case_sensitive_sdi']), 2)

    def test_capitalization_false_positive(self):
        result = casing('Он сказал и ушёл.', 'Он сказал И ушёл.')
        self.assertEqual(result['initial_case_errors'], 1)
        self.assertEqual(result['capitalization_tp_fp_fn'], [1, 1, 0])

    def test_lexical_substitution_is_not_a_case_error(self):
        result = casing('Мы встретили Ивана.', 'Мы встретили петра.')
        self.assertEqual(result['aligned_words'], 2)
        self.assertEqual(result['initial_case_errors'], 0)
        self.assertEqual(sum(result['case_sensitive_sdi']), 1)

    def test_delta_accepts_only_case_and_preserves_context(self):
        a = {'text': 'И', 'duration': 1, 'words': [{'word': 'И', 'start': 0.1, 'end': 0.2}]}
        b = {'text': 'и', 'duration': 1, 'words': [{'word': 'и', 'start': 0.1, 'end': 0.2}]}
        self.assertEqual(len(delta(a, b)), 1)
        b['words'][0]['end'] = 0.3
        with self.assertRaisesRegex(ValueError, 'metadata'):
            delta(a, b)

    def test_delta_rejects_punctuation_change(self):
        a = {'text': 'Да.', 'words': [{'word': 'Да.', 'start': 0.1}]}
        b = {'text': 'Да!', 'words': [{'word': 'Да!', 'start': 0.1}]}
        with self.assertRaisesRegex(ValueError, 'punctuation'):
            delta(a, b)


if __name__ == '__main__':
    unittest.main()
