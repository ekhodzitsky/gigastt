"""Contract checks for the opt-in FP32 pack installer (no model downloads)."""
import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import install_e2e_fp32 as installer


class InstallTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / 'source'
        self.source.mkdir()
        self.destination = self.root / 'installed'
        self.files = {}
        for name in installer.FILES:
            data = name.encode()
            (self.source / name).write_bytes(data)
            self.files[name] = (len(data), hashlib.sha256(data).hexdigest())
        self.mock_files = patch.object(installer, 'FILES', self.files)
        self.mock_files.start()
        self.addCleanup(self.mock_files.stop)

    def test_local_install_selects_fp32_without_network(self):
        with patch.object(installer.urllib.request, 'urlopen', side_effect=AssertionError('network')):
            installer.install(self.destination, self.source)
        self.assertEqual((self.destination / 'manifest.toml').read_text(), installer.MANIFEST)
        self.assertIn('precision = "fp32"', installer.MANIFEST)
        self.assertNotIn('encoder_int8', installer.MANIFEST)
        for name in self.files:
            self.assertEqual((self.destination / name).read_bytes(), (self.source / name).read_bytes())
            self.assertFalse((self.destination / name).is_symlink())

    def test_corrupt_input_is_not_published(self):
        name = next(iter(self.files))
        (self.source / name).write_bytes(b'x' * self.files[name][0])
        with self.assertRaisesRegex(ValueError, 'SHA-256'):
            installer.install(self.destination, self.source)
        self.assertFalse(self.destination.exists())
        self.assertEqual(sorted(p.name for p in self.root.iterdir()), ['source'])

    def test_oversized_input_is_not_published(self):
        name = next(iter(self.files))
        (self.source / name).write_bytes(b'x' * (self.files[name][0] + 1))
        with self.assertRaisesRegex(ValueError, 'size'):
            installer.install(self.destination, self.source)
        self.assertFalse(self.destination.exists())

    def test_missing_input_is_not_published(self):
        (self.source / next(iter(self.files))).unlink()
        with self.assertRaises(FileNotFoundError):
            installer.install(self.destination, self.source)
        self.assertFalse(self.destination.exists())

    def test_existing_installation_is_preserved(self):
        self.destination.mkdir()
        marker = self.destination / 'keep'
        marker.write_text('existing user file')
        with self.assertRaises(FileExistsError):
            installer.install(self.destination, self.source)
        self.assertEqual(marker.read_text(), 'existing user file')
        self.assertEqual(list(self.destination.iterdir()), [marker])

    def test_download_uses_pinned_revision_and_verifies_payload(self):
        import io
        def open_url(url, timeout):
            self.assertIn('/resolve/' + installer.REVISION + '/', url)
            self.assertGreater(timeout, 0)
            return io.BytesIO(url.rsplit('/', 1)[1].encode())
        with patch.object(installer.urllib.request, 'urlopen', side_effect=open_url) as download:
            installer.install(self.destination)
        self.assertEqual(download.call_count, len(self.files))
        self.assertTrue((self.destination / 'manifest.toml').is_file())


if __name__ == '__main__':
    unittest.main()
