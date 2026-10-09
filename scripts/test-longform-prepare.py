#!/usr/bin/env python3
import hashlib
import importlib.util
import io
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch
import urllib.error

spec = importlib.util.spec_from_file_location('longform', Path(__file__).with_name('longform_stitch.py'))
longform = importlib.util.module_from_spec(spec)
spec.loader.exec_module(longform)


class PrepareTests(unittest.TestCase):
    def test_metadata_is_batched_and_verified_audio_is_reused(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            wav = root / 'source.wav'
            longform.write_wav(wav, b'\x01\x00' * 160)
            part_hash = hashlib.sha256(wav.read_bytes()).hexdigest()
            joined = root / 'joined.wav'
            longform.write_wav(joined, b'\x01\x00' * 320)
            sample = dict(id='sample', file='sample.wav', sha256=hashlib.sha256(joined.read_bytes()).hexdigest(),
                          parts=[dict(row=i, reference=f'words {i}', sha256=part_hash) for i in (4, 5)])
            metadata = {'rows': [dict(row_idx=i, row={'text': f'words {i}', 'audio': [{'src': 'audio-url'}]})
                                 for i in (4, 5)]}
            def response(url, **kwargs):
                return io.BytesIO(wav.read_bytes() if url == 'audio-url' else json.dumps(metadata).encode())
            args = SimpleNamespace(audio=root / 'corpus')
            with patch.object(longform.urllib.request, 'urlopen', side_effect=response) as request:
                longform.prepare(args, {'samples': [sample]})
                urls = [call.args[0] for call in request.call_args_list]
                self.assertEqual(sum('datasets-server' in url for url in urls), 1)
                self.assertIn('&offset=4&length=2', urls[0])
                request.reset_mock()
                longform.prepare(args, {'samples': [sample]})
                request.assert_not_called()
            (args.audio / sample['file']).write_bytes(b'corrupt')
            with self.assertRaises(ValueError):
                longform.prepare(args, {'samples': [sample]})

    def test_rate_limit_retry_respects_server_delay(self):
        error = urllib.error.HTTPError('url', 429, 'limited', {'Retry-After': '2'}, None)
        with patch.object(longform.urllib.request, 'urlopen', side_effect=[error, io.BytesIO(b'ok')]), \
             patch.object(longform.time, 'sleep') as sleep:
            with longform.open_url('url') as response:
                self.assertEqual(response.read(), b'ok')
            sleep.assert_called_once_with(2)

    def test_non_transient_http_error_is_not_retried(self):
        error = urllib.error.HTTPError('url', 404, 'missing', {}, None)
        with patch.object(longform.urllib.request, 'urlopen', side_effect=error) as request:
            with self.assertRaises(urllib.error.HTTPError):
                longform.open_url('url')
            self.assertEqual(request.call_count, 1)


if __name__ == '__main__':
    unittest.main()
