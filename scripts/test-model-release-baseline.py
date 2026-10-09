#!/usr/bin/env python3
import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location('baseline', Path(__file__).with_name('model-release-baseline.py'))
baseline = importlib.util.module_from_spec(spec)
spec.loader.exec_module(baseline)


class BaselineTests(unittest.TestCase):
    def test_release_must_compare_with_current_runtime_download(self):
        source = 'pub(super) const PREQUANT_RELEASE_BASE: &str =\n    "https://example.com/current";'
        baseline.check({'baseline_release': 'https://example.com/current'}, source)
        with self.assertRaises(ValueError):
            baseline.check({'baseline_release': 'https://example.com/older'}, source)

    def test_missing_runtime_pin_fails_closed(self):
        with self.assertRaises(ValueError):
            baseline.check({'baseline_release': 'https://example.com/current'}, '')


if __name__ == '__main__':
    unittest.main()
