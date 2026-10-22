import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
spec = importlib.util.spec_from_file_location('current', Path(__file__).with_name('score-current-asr.py'))
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

class CurrentAsrTests(unittest.TestCase):
    def test_case_punctuation_and_yo_do_not_count_as_word_errors(self):
        result = module.measure_pair('Ёлка, здесь!', 'елка здесь')
        self.assertEqual(result['word_errors'], 0)
        self.assertEqual(result['char_errors'], 0)

    def test_spoken_numbers_have_separate_symmetric_score(self):
        result = module.measure_pair('В 2020 году', 'в две тысячи двадцатом году')
        self.assertGreater(result['word_errors'], 0)
        self.assertEqual(result['normalized_errors'], 0)
        self.assertEqual(result['normalized_char_errors'], 0)

    def test_numeral_free_slice_excludes_spoken_numbers_using_reference_only(self):
        for reference in ['В 2020 году', 'в две тысячи двадцатом году', 'первое место', 'три с половиной литра']:
            self.assertFalse(module.numeral_free(reference))
        self.assertTrue(module.numeral_free('Покажи передачу Fashion TV'))

    def test_word_errors_separate_substitution_deletion_and_insertion(self):
        for hypothesis, expected in [('первое другое', (1, 0, 0)), ('первое', (0, 1, 0)),
                                     ('первое второе третье', (0, 0, 1))]:
            result = module.measure_pair('первое второе', hypothesis)
            self.assertEqual(tuple(result[k] for k in ['substitutions', 'deletions', 'insertions']), expected)
            self.assertEqual(result['word_errors'], sum(expected))

    def test_failed_request_stays_in_denominator_as_deletions(self):
        cases = [{'id': 'a', 'reference': 'два слова', 'dataset': 'x'},
                 {'id': 'b', 'reference': 'верно', 'dataset': 'x'}]
        rows = [{'id': 'a', 'text': '', 'error': 'failed'}, {'id': 'b', 'text': 'верно', 'error': None}]
        result = module.score_cases(cases, rows)
        self.assertEqual(result['word_errors'], 2)
        self.assertEqual(result['words'], 3)
        self.assertEqual(result['failures'], 1)

    def test_missing_duplicate_or_unknown_hypothesis_rejected(self):
        cases = [{'id': 'a', 'reference': 'слово', 'dataset': 'x'}]
        for rows in [[], [{'id': 'x', 'text': 'слово'}], [{'id': 'a', 'text': 'слово'}] * 2]:
            with self.assertRaises(ValueError):module.score_cases(cases, rows)

    def test_stale_named_bundle_is_rejected_before_benchmark(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'encoder.onnx').write_bytes(b'old cached weights')
            with self.assertRaises(ValueError):
                module.verify_assets(root, {'encoder.onnx': '0' * 64})
            digest = hashlib.sha256(b'old cached weights').hexdigest()
            self.assertEqual(module.verify_assets(root, {'encoder.onnx': digest})['encoder.onnx']['sha256'], digest)

    def test_runner_rejects_stale_cache_before_starting_inference(self):
        spec = importlib.util.spec_from_file_location('runner', Path(__file__).with_name('run-current-asr.py'))
        runner = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(runner)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            weights = b'older local quantization'
            (root / 'encoder.onnx').write_bytes(weights)
            profile = {'head': 'rnnt', 'model_dir': str(root), 'manifest': 'unused',
                       'assets': {'encoder.onnx': {'sha256': hashlib.sha256(weights).hexdigest(), 'bytes': len(weights)}}}
            (root / 'protocol.json').write_text(json.dumps({'profiles': [profile]}))
            catalog = json.dumps([{'head': 'rnnt', 'assets': {'encoder.onnx': '0' * 64}}]).encode()
            with patch('sys.argv', ['runner', str(root), '--probe', str(root / 'probe')]), \
                 patch.object(runner.subprocess, 'check_output', return_value=catalog), \
                 patch.object(runner.subprocess, 'run') as inference:
                with self.assertRaisesRegex(ValueError, 'nonmatching model asset'):
                    runner.main()
                inference.assert_not_called()
            self.assertFalse((root / 'runs').exists())

    def test_empty_or_duplicated_reference_cohort_rejected(self):
        case = {'id': 'a', 'reference': 'слово', 'dataset': 'x'}
        for cases in [[], [case, case]]:
            with self.assertRaises(ValueError):module.score_cases(cases, [])

    def test_truncated_and_invalid_timing_report_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'report.jsonl'
            cases = [{'id': 'a'}]
            for records in [[{'head': 'x'}, {'id': 'a', 'text': '', 'seconds': 1}],
                            [{'head': 'x'}, {'id': 'a', 'text': '', 'seconds': -1}, {'complete': True, 'count': 1}]]:
                path.write_text(''.join(json.dumps(r) + '\n' for r in records))
                with self.assertRaises(ValueError):module.read_report(path, cases)

if __name__ == '__main__':unittest.main()
