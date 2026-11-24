"""Model-free tests for the WER recomputation command's output paths."""

import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class RecomputeWerTest(unittest.TestCase):
    def setUp(self):
        source = Path(__file__).resolve().parents[1]
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.benchmark = self.root / "benchmark"
        self.benchmark.mkdir()
        for name in ("recompute_wer.py", "common.py"):
            shutil.copyfile(source / name, self.benchmark / name)
        self.archive = self.benchmark / "results_full"
        self.archive.mkdir()
        self.default_output = self.benchmark / "results_local" / "recompute_wer"

    def run_recompute(self, *args):
        subprocess.run(
            [sys.executable, str(self.benchmark / "recompute_wer.py"), *args],
            cwd=self.root,
            check=True,
            capture_output=True,
        )

    def write_source(self, directory):
        report = {
            "dataset": "fixture",
            "runners": [{
                "name": "example",
                "details": [{"reference": "hello world", "hypothesis": "hello"}],
            }],
        }
        (directory / "sample.json").write_text(json.dumps(report), encoding="utf-8")

    def assert_report(self, output):
        result = json.loads((output / "sample_renorm.json").read_text(encoding="utf-8"))
        self.assertEqual(result["runners"][0]["wer"], 50.0)
        self.assertTrue((output / "renorm_summary.md").is_file())

    def test_default_output_preserves_source_and_archive(self):
        self.write_source(self.archive)
        (self.archive / "sample_renorm.json").write_bytes(b"archived report")
        (self.archive / "renorm_summary.md").write_bytes(b"archived summary")
        before = {path.name: path.read_bytes() for path in self.archive.iterdir()}

        self.run_recompute()

        self.assertEqual({path.name: path.read_bytes() for path in self.archive.iterdir()}, before)
        self.assert_report(self.default_output)

    def test_explicit_input_and_output_directories(self):
        selected = self.root / "selected"
        selected.mkdir()
        self.write_source(selected)
        output = self.root / "custom" / "output"

        self.run_recompute("--input-dir", str(selected), "--output-dir", str(output))

        self.assert_report(output)
        self.assertFalse(self.default_output.exists())
        self.assertFalse((selected / "sample_renorm.json").exists())

    def test_empty_input_writes_summary_to_fresh_default_output(self):
        self.run_recompute()

        self.assertEqual(list(self.archive.iterdir()), [])
        self.assertTrue((self.default_output / "renorm_summary.md").is_file())
