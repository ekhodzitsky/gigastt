#!/usr/bin/env python3
import hashlib
import importlib.util
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location('bundle', Path(__file__).with_name('prepare-model-bundle.py'))
bundle = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bundle)


class BundleTests(unittest.TestCase):
    def test_corrupt_source_cannot_replace_existing_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, destination = root / 'source', root / 'target'
            source.write_bytes(b'wrong')
            expected = hashlib.sha256(b'correct').hexdigest()
            with self.assertRaises(ValueError):
                bundle.fetch(source.as_uri(), destination, expected)
            self.assertFalse(destination.exists())
            self.assertFalse(destination.with_suffix('.partial').exists())
            destination.write_bytes(b'also wrong')
            with self.assertRaises(ValueError):
                bundle.fetch(source.as_uri(), destination, expected)
            self.assertEqual(destination.read_bytes(), b'also wrong')

    def test_verified_source_is_promoted_and_cached(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, destination = root / 'source', root / 'target'
            source.write_bytes(b'correct')
            expected = bundle.digest(source)
            bundle.fetch(source.as_uri(), destination, expected)
            source.unlink()
            bundle.fetch(source.as_uri(), destination, expected)
            self.assertEqual(bundle.digest(destination), expected)


if __name__ == '__main__':
    unittest.main()
