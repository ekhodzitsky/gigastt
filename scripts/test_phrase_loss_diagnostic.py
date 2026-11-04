#!/usr/bin/env python3
"""Regression tests for the local-only phrase-loss diagnostic."""
import argparse
import importlib.util
import json
import os
import subprocess
import sys
from unittest.mock import patch
from pathlib import Path
import tempfile
import unittest
import wave

spec = importlib.util.spec_from_file_location('diagnostic', Path(__file__).with_name('phrase_loss_diagnostic.py'))
diag = importlib.util.module_from_spec(spec)
spec.loader.exec_module(diag)


class DiagnosticTests(unittest.TestCase):
    def test_issue_offsets_select_exact_interior_windows(self):
        self.assertEqual(diag.window_starts([823.5, 1087.8, 1172.7], 2040.8 * 16000), [814, 1078, 1166])

    def test_invalid_positions_and_tail_are_rejected(self):
        for positions in [[], [float('nan')], [float('inf')], [-1], [99], list(range(9))]:
            with self.subTest(positions=positions), self.assertRaises(ValueError):
                diag.window_starts(positions, 100 * 16000)

    def test_deduplicate_same_window(self):
        self.assertEqual(diag.window_starts([23, 24, 23], 100 * 16000), [22])

    def test_extract_preserves_nonzero_pcm_offset(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pcm = b''.join((i % 32749).to_bytes(2, 'little') for i in range(100 * 16000))
            source = root/'source.wav'
            with wave.open(str(source), 'wb') as dst:
                dst.setparams((1, 2, 16000, 0, 'NONE', 'not compressed'))
                dst.writeframes(pcm)
            rows = diag.extract(source, root/'windows', [23])
            with wave.open(str(root/'windows'/rows[0]['file']), 'rb') as src:
                self.assertEqual(src.readframes(src.getnframes()), pcm[22*32000:46*32000])
            with self.assertRaises(FileExistsError):
                diag.extract(source, root/'windows', [23])

    def test_wrong_audio_format_is_rejected_without_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with wave.open(str(root/'source.wav'), 'wb') as dst:
                dst.setparams((2, 2, 16000, 0, 'NONE', 'not compressed'))
                dst.writeframes(b'\0' * 64000)
            with self.assertRaises(ValueError):
                diag.extract(root/'source.wav', root/'windows', [0])
            self.assertFalse((root/'windows').exists())

    def test_timestamp_shift_is_not_text_deletion(self):
        local = [{'word': 'Привет,', 'start': 2., 'end': 2.5}]
        full = [{'word': 'привет', 'start': 25., 'end': 25.5}]
        result = diag.compare(full, local, 22)
        self.assertEqual(result['text_opcodes'], [])
        self.assertEqual(result['local_starts_without_full_start'], [24.])

    def test_text_difference_is_separate_from_timestamp_coverage(self):
        local = [{'word': 'привет', 'start': 2., 'end': 2.5}]
        full = [{'word': 'пока', 'start': 24., 'end': 24.5}]
        result = diag.compare(full, local, 22)
        self.assertTrue(result['text_opcodes'])
        self.assertEqual(result['local_starts_without_full_start'], [])

    def test_invalid_word_times_fail_instead_of_empty_success(self):
        for words in [[{'word': 'x', 'start': float('nan'), 'end': 1}], [{'word': 'x', 'start': 2, 'end': 1}]]:
            with self.assertRaises(ValueError):
                diag.validate_words({'words': words})
        with self.assertRaises(ValueError):
            diag.validate_words({})
        with self.assertRaises(ValueError):
            diag.validate_words([])

    def test_timeout_does_not_produce_a_transcript(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            binary = root/'slow-gigastt'
            binary.write_text('#!' + sys.executable + '\nimport time\ntime.sleep(30)\n')
            binary.chmod(0o755)
            destination = root/'output.json'
            with self.assertRaises(subprocess.TimeoutExpired):
                diag.transcribe(binary, root/'audio.wav', root/'models', 'e2e_rnnt',
                                6, destination, .05, diag.clean_environment())
            self.assertFalse(destination.exists())

    def test_provider_and_cache_must_be_confirmed(self):
        cpu = 'execution provider bound execution_provider="cpu"\n'
        diag.verify_execution(cpu + 'loaded fresh optimized graph cache', True)
        for log in ['', cpu, cpu + 'falling back to CPU execution provider loaded fresh optimized graph cache']:
            with self.assertRaises(ValueError):
                diag.verify_execution(log, True)

    def test_runner_keeps_private_content_out_of_summary_and_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            audio = root/'private-meeting.wav'
            with wave.open(str(audio), 'wb') as dst:
                dst.setparams((1, 2, 16000, 0, 'NONE', 'not compressed'))
                dst.writeframes(b'\0' * (100 * 32000))
            models = root/'source-models'
            models.mkdir()
            for suffix in ['encoder_int8.onnx', 'decoder.onnx', 'joint.onnx', 'vocab.txt']:
                (models/f'v3_e2e_rnnt_{suffix}').write_bytes(b'model fixture')
            binary = root/'fake-gigastt'
            binary.write_text('#!' + sys.executable + '\n' + """
import json, os, pathlib, sys
if '--version' in sys.argv:
    print('fixture binary'); sys.exit(0)
assert not any(k.startswith('GIGASTT_') for k in os.environ)
out = pathlib.Path(sys.argv[sys.argv.index('--output')+1])
assert '--offline' in sys.argv
assert sys.argv[sys.argv.index('--punctuation')+1] == 'off'
assert sys.argv[sys.argv.index('--itn')+1] == 'off'
assert sys.argv[sys.argv.index('--file-window-concurrency')+1] == '1'
start = 24 if out.stem.startswith('full') else 2
out.write_text(json.dumps({'text':'PRIVATE', 'words':[{'word':'PRIVATE','start':start,'end':start+.5}]}))
print('execution provider bound execution_provider="cpu"')
print('loading source model and refreshing optimized graph cache' if out.stem == 'full-cold' else 'loaded fresh optimized graph cache')
""")
            binary.chmod(0o755)
            args = argparse.Namespace(audio=audio, binary=binary, model_dir=models,
                                      output=root/'result', at=[23], variant='e2e_rnnt', threads=6, timeout=10)
            with patch.dict(os.environ, {'GIGASTT_VAD':'true'}):
                diag.run(args)
            summary = (args.output/'summary.json').read_text()
            self.assertNotIn('PRIVATE', summary)
            self.assertNotIn(str(root), summary)
            self.assertTrue(json.loads(summary)['windows'][0]['comparisons']['full-warm']['text_equal'])
            with self.assertRaises(ValueError):
                diag.run(args)
            binary.write_text('#!' + sys.executable + '\nimport sys\nif "--version" in sys.argv: print("fake")\nelse: sys.exit(7)\n')
            args.output = root/'failed'
            with self.assertRaises(subprocess.CalledProcessError):
                diag.run(args)
            self.assertFalse((args.output/'summary.json').exists())


if __name__ == '__main__':
    unittest.main()
