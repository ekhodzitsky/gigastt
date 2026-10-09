#!/usr/bin/env python3
import copy
import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location('quality', Path(__file__).with_name('model-quality.py'))
quality = importlib.util.module_from_spec(spec)
spec.loader.exec_module(quality)


class QualityTests(unittest.TestCase):
    def test_phrase_deletions_are_not_hidden_by_aggregate_wer(self):
        result = quality.score('one two three four five six seven', 'one six seven')
        self.assertEqual(result, dict(reference_words=7, edits=4, deletion_runs=1, words_in_deletion_runs=4))
        self.assertEqual(quality.score('one two three four', '')['words_in_deletion_runs'], 4)
        self.assertEqual(quality.score('one two three four', 'one other three four')['deletion_runs'], 0)

    def report(self):
        sample = dict(wall_seconds=1.0, peak_rss_kib=100, reference_words=20,
                      edits=2, deletion_runs=0, words_in_deletion_runs=0)
        return dict(schema=1, cases=[dict(id=str(i), baseline=[copy.deepcopy(sample) for _ in range(3)],
                                         candidate=[copy.deepcopy(sample) for _ in range(3)]) for i in range(7)],
                    models={arm: dict(encoder_bytes=1000) for arm in ('baseline', 'candidate')})

    def test_empty_reference_is_rejected(self):
        with self.assertRaises(ValueError):
            quality.score('', 'some text')

    def test_memory_exception_is_bound_to_exact_reviewed_models(self):
        import json
        release = json.loads((Path(__file__).resolve().parents[1] / 'benchmark/model-release.json').read_text())
        recipe = release['heads']['rnnt']
        report = self.report()
        report['head'] = 'rnnt'
        for arm, key in [('baseline', 'baseline_encoder_sha256'), ('candidate', 'candidate_encoder_sha256')]:
            report['models'][arm]['files'] = {'v3_rnnt_encoder_int8.onnx': recipe[key]}
        report['models']['candidate']['encoder_bytes'] = 1400
        for case in report['cases']:
            for sample in case['candidate']:
                sample['peak_rss_kib'] = 140
        self.assertEqual(quality.compare(report), [])
        report['models']['candidate']['files']['v3_rnnt_encoder_int8.onnx'] = 'a' * 64
        self.assertTrue(quality.compare(report))

    def test_identical_metrics_pass(self):
        self.assertEqual(quality.compare(self.report()), [])

    def test_each_regression_fails(self):
        for key, value in dict(wall_seconds=2, peak_rss_kib=200, edits=3,
                               deletion_runs=1, words_in_deletion_runs=3).items():
            with self.subTest(key=key):
                report = self.report()
                report['cases'][0]['candidate'][1][key] = value
                self.assertTrue(quality.compare(report))
        report = self.report()
        report['models']['candidate']['encoder_bytes'] = 1200
        self.assertTrue(quality.compare(report))

    def test_incomplete_or_invalid_evidence_fails(self):
        for mutation in [lambda r: r['cases'].pop(),
                         lambda r: r['cases'][0]['candidate'].pop(),
                         lambda r: r['cases'][0]['candidate'][0].update(wall_seconds=float('nan')),
                         lambda r: r['cases'][0]['candidate'][0].update(edits=-1),
                         lambda r: r['cases'][1].update(id='0')]:
            report = self.report()
            mutation(report)
            with self.assertRaises(ValueError):
                quality.compare(report)


if __name__ == '__main__':
    unittest.main()
