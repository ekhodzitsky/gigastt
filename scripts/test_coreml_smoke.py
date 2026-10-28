#!/usr/bin/env python3
"""The CoreML gate must retain diagnostics even under GitHub's bash -e."""
from pathlib import Path
import subprocess
import tempfile
import unittest

SCRIPT = Path(__file__).with_name('coreml-smoke.sh').resolve()


class CoreMLSmokeTests(unittest.TestCase):
    def run_smoke(self, text='шестьдесят тысяч тенге', stderr='diagnostic sentinel', status=0):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            binary = root/'gigastt'
            binary.write_text(f'#!/bin/sh\nprintf "%s\\n" "{text}"\nprintf "%s\\n" "{stderr}" >&2\nexit {status}\n')
            binary.chmod(0o755)
            result = subprocess.run(['bash', '-e', str(SCRIPT), str(binary), 'fixture.wav'],
                                    cwd=root, text=True, capture_output=True)
            return result

    def test_failure_keeps_stderr_and_transcript_under_errexit(self):
        result = self.run_smoke(text='partial transcript', status=7)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('diagnostic sentinel', result.stdout)
        self.assertIn('partial transcript', result.stdout)
        self.assertIn('exit 7', result.stdout)

    def test_success_requires_reference_text(self):
        self.assertEqual(self.run_smoke().returncode, 0)
        self.assertNotEqual(self.run_smoke(text='unexpected').returncode, 0)

    def test_cpu_fallback_is_not_success(self):
        result = self.run_smoke(stderr='falling back to CPU execution provider')
        self.assertNotEqual(result.returncode, 0)

    def test_cpu_fallback_on_stdout_is_not_success(self):
        result = self.run_smoke(text='falling back to CPU execution provider: шестьдесят тысяч тенге')
        self.assertNotEqual(result.returncode, 0)


if __name__ == '__main__':
    unittest.main()
