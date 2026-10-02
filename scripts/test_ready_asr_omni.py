"""Adapter boundary checks without loading the multi-gigabyte model."""
import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import ready_asr_omni as backend


class AdapterTests(unittest.TestCase):
    def test_corrupt_file_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / "weights").write_bytes(b"good")
            expected = hashlib.sha256(b"good").hexdigest()
            with patch.object(backend, "FILES", {"weights": (4, expected)}):
                self.assertEqual(backend.verify_files(path)[0]["sha256"], expected)
                (path / "weights").write_bytes(b"evil")
                with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
                    backend.verify_files(path)

    def test_audio_not_truncated_and_no_language_conditioning(self):
        class Pipeline:
            def transcribe(self, inputs, *, batch_size):
                self.inputs = inputs
                self.batch_size = batch_size
                return ["output"]

        model = object.__new__(backend.ReadyASR)
        model.pipeline = Pipeline()
        audio = np.zeros(31 * 16000, dtype=np.float32)
        result = model.transcribe(audio)
        self.assertIs(model.pipeline.inputs[0]["waveform"], audio)
        self.assertEqual(model.pipeline.batch_size, 1)
        self.assertTrue(result["exceeds_recommended_30_seconds"])
        self.assertFalse(result["language_conditioning"])
        self.assertEqual(result["hypothesis"], "output")

    def test_invalid_audio_rejected_before_inference(self):
        model = object.__new__(backend.ReadyASR)
        for audio in [np.zeros(1, dtype=np.float64), np.zeros((1, 2), dtype=np.float32),
                      np.array([np.nan], dtype=np.float32), np.array([], dtype=np.float32),
                      np.zeros(40 * 16000 + 1, dtype=np.float32)]:
            with self.assertRaises(ValueError):
                model.transcribe(audio)


if __name__ == "__main__":
    unittest.main()
