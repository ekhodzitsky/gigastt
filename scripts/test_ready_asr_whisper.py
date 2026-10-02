"""Input and provenance guards without inference or model downloads."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock
import numpy as np
from ready_asr_whisper import ReadyASR, validate_audio, verify_model


class Guards(unittest.TestCase):
    def test_reject_invalid_audio_without_coercion(self):
        for audio in [np.zeros(8, dtype=np.int16), np.zeros((8, 2), dtype=np.float32),
                      np.array([np.nan], dtype=np.float32), np.zeros(0, dtype=np.float32)]:
            with self.assertRaises(ValueError):
                validate_audio(audio)
        x = np.array([0.000001, 0.123456], dtype=np.float32)
        self.assertIs(validate_audio(x), x)

    def test_missing_provenance_and_unsupported_language_fail_before_load(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileNotFoundError):
                verify_model(Path(tmp))
            with self.assertRaises(ValueError):
                ReadyASR(Path(tmp), 'ky')

    def test_array_passed_unchanged_and_segments_consumed(self):
        backend = object.__new__(ReadyASR)
        backend.language = 'kk'
        backend.options = {'beam_size': 5}
        backend.model = Mock()
        backend.model.transcribe.return_value = (iter([Mock(text=' қазақ'), Mock(text=' тілі ')]), Mock(language='kk', language_probability=1.0))
        x = np.ones(16, dtype=np.float32)
        self.assertEqual(backend.transcribe(x)['hypothesis'], 'қазақ тілі')
        self.assertIs(backend.model.transcribe.call_args.args[0], x)


if __name__ == '__main__':
    unittest.main()
