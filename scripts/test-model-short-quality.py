#!/usr/bin/env python3
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location('quality', Path(__file__).with_name('model-short-quality.py'))
quality = importlib.util.module_from_spec(spec)
spec.loader.exec_module(quality)


class ShortQualityTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.manifest = Path(self.directory.name) / 'manifest.json'
        self.manifest.write_text(json.dumps({'samples': [
            {'filename': 'a.wav', 'reference': 'one two'},
            {'filename': 'empty.wav', 'reference': ''}]}))
        self.release = {'heads': {'rnnt': {
            'baseline_encoder_sha256': 'a' * 64, 'candidate_encoder_sha256': 'b' * 64,
            'source_files': {'v3_rnnt_encoder.onnx': 'c' * 64, 'v3_vocab.txt': 'd' * 64}}}}
        sample = dict(file='a.wav', text='one two', errors=0, reference_words=2,
                      raw_errors=0, raw_reference_words=2)
        result = dict(errors=0, reference_words=2, wer=0, raw_errors=0,
                      raw_reference_words=2, raw_wer=0, details=[sample])
        self.report = dict(schema=1, wer_unit='fraction', samples=1, nominal_samples=2,
                           skipped_empty_references=1,
                           manifest_sha256=hashlib.sha256(self.manifest.read_bytes()).hexdigest(),
                           audio_sha256={'a.wav': 'e' * 64},
                           models={'rnnt': {arm: {'files': {'v3_rnnt_encoder_int8.onnx': value * 64,
                                                           'v3_vocab.txt': 'd' * 64}}
                                            for arm, value in [('baseline', 'a'), ('candidate', 'b')]}},
                           results={'rnnt': {arm: copy.deepcopy(result) for arm in ('baseline', 'candidate')}})

    def check(self):
        return quality.check(self.report, self.manifest, self.release)

    def test_identical_transcripts_pass(self):
        self.assertEqual(self.check(), [])

    def test_short_form_regression_blocks_publication(self):
        candidate = self.report['results']['rnnt']['candidate']
        candidate.update(errors=1, raw_errors=1, wer=0.5, raw_wer=0.5)
        candidate['details'][0].update(text='one other', errors=1, raw_errors=1)
        self.assertEqual(self.check(), ['rnnt: Golos word errors increased from 0 to 1'])

    def test_report_cannot_hide_transcript_errors(self):
        self.report['results']['rnnt']['candidate']['details'][0]['text'] = 'one other'
        with self.assertRaises(ValueError):
            self.check()

    def test_unmeasured_model_or_companion_is_rejected(self):
        for name in ('v3_rnnt_encoder_int8.onnx', 'v3_vocab.txt'):
            with self.subTest(name=name):
                original = self.report['models']['rnnt']['candidate']['files'][name]
                self.report['models']['rnnt']['candidate']['files'][name] = 'f' * 64
                with self.assertRaises(ValueError):
                    self.check()
                self.report['models']['rnnt']['candidate']['files'][name] = original

    def test_incomplete_or_changed_corpus_is_rejected(self):
        for mutate in [lambda r: r['results']['rnnt']['candidate']['details'].clear(),
                       lambda r: r['audio_sha256'].clear(),
                       lambda r: r.update(manifest_sha256='f' * 64),
                       lambda r: r['results']['rnnt']['candidate'].update(wer=float('nan'))]:
            original = copy.deepcopy(self.report)
            mutate(self.report)
            with self.assertRaises(ValueError):
                self.check()
            self.report = original


if __name__ == '__main__':
    unittest.main()
