#!/usr/bin/env python3
"""Safety and reproducibility checks using tiny synthetic model payloads."""
import io
from pathlib import Path
import tarfile
import tempfile
import unittest

import package_small_multilingual as pack


class PackageTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.encoder = self.root / 'source.onnx'
        self.encoder.write_bytes(b'synthetic test weights\x00\x01')
        self.vocab = self.root / 'vocab.txt'
        self.vocab.write_bytes(b'a 0\n<blk> 1\n')
        self.archive = self.root / 'pack.tar.gz'
        pack.build_archive(self.encoder, self.vocab, self.archive,
                           pack.digest(self.encoder), pack.digest(self.vocab))

    def install(self, archive=None, target=None, expected=None):
        archive = archive or self.archive
        return pack.install_archive(archive, expected or pack.digest(archive), target or self.root / 'installed')

    def modified(self, edit):
        with tarfile.open(self.archive, 'r:gz') as original:
            entries = [(m, original.extractfile(m).read()) for m in original]
        edit(entries)
        path = self.root / 'modified.tar.gz'
        with tarfile.open(path, 'w:gz') as output:
            for member, content in entries:
                output.addfile(member, io.BytesIO(content) if member.isfile() else None)
        return path

    def test_roundtrip_installs_only_three_runtime_files(self):
        self.install()
        target = self.root / 'installed'
        self.assertEqual(set(p.name for p in target.iterdir()), set(pack.RUNTIME_FILES))
        self.assertEqual((target / pack.ENCODER).read_bytes(), self.encoder.read_bytes())
        self.assertEqual((target / 'manifest.toml').read_bytes(), pack.MANIFEST)

    def test_rebuild_is_byte_identical(self):
        second = self.root / 'second.tar.gz'
        pack.build_archive(self.encoder, self.vocab, second, pack.digest(self.encoder), pack.digest(self.vocab))
        self.assertEqual(self.archive.read_bytes(), second.read_bytes())

    def test_archive_metadata_is_fixed_and_contains_no_companion(self):
        with tarfile.open(self.archive, 'r:gz') as bundle:
            members = bundle.getmembers()
        self.assertEqual({m.name for m in members}, set(pack.RUNTIME_FILES) | {'SHA256SUMS'})
        self.assertTrue(all(m.isfile() and m.uid == m.gid == m.mtime == 0 and m.mode == 0o644 for m in members))

    def test_builder_rejects_unpinned_source_and_existing_archive(self):
        with self.assertRaises(ValueError):
            pack.build_archive(self.encoder, self.vocab, self.root / 'wrong.tar.gz')
        before = self.archive.read_bytes()
        with self.assertRaises(FileExistsError):
            pack.build_archive(self.encoder, self.vocab, self.archive, pack.digest(self.encoder), pack.digest(self.vocab))
        self.assertEqual(before, self.archive.read_bytes())

    def test_wrong_archive_digest_leaves_no_destination(self):
        with self.assertRaises(ValueError):
            self.install(expected='0' * 64)
        self.assertFalse((self.root / 'installed').exists())

    def test_corrupted_archive_with_matching_outer_digest_rejected(self):
        data = bytearray(self.archive.read_bytes())
        data[len(data) // 2] ^= 0xff
        self.archive.write_bytes(data)
        with self.assertRaises((ValueError, tarfile.TarError, OSError, EOFError)):
            self.install()
        self.assertFalse((self.root / 'installed').exists())

    def test_internal_digest_mismatch_rejected(self):
        def edit(entries):
            for i, (member, content) in enumerate(entries):
                if member.name == pack.ENCODER:
                    entries[i] = (member, b'X' * len(content))
        with self.assertRaises(ValueError):
            self.install(self.modified(edit))
        self.assertFalse((self.root / 'installed').exists())

    def test_traversal_and_absolute_names_rejected(self):
        for name in ['../escape', '/absolute', 'dir/file', 'dir\\file']:
            def edit(entries):
                member = tarfile.TarInfo(name)
                entries.append((member, b''))
            with self.subTest(name=name), self.assertRaises(ValueError):
                self.install(self.modified(edit))
        self.assertFalse((self.root / 'escape').exists())

    def test_symbolic_and_hard_links_rejected(self):
        for kind in [tarfile.SYMTYPE, tarfile.LNKTYPE]:
            def edit(entries):
                member = entries[0][0]
                member.type = kind
                member.linkname = '/tmp/outside'
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                self.install(self.modified(edit))

    def test_duplicate_member_rejected(self):
        with self.assertRaises(ValueError):
            self.install(self.modified(lambda entries: entries.append(entries[0])))

    def test_missing_member_rejected(self):
        with self.assertRaises(ValueError):
            self.install(self.modified(lambda entries: entries.pop()))

    def test_existing_nonempty_destination_preserved(self):
        target = self.root / 'installed'
        target.mkdir()
        (target / 'keep').write_text('unchanged')
        with self.assertRaises(FileExistsError):
            self.install()
        self.assertEqual(list(target.iterdir()), [target / 'keep'])

    def test_symlink_destination_rejected(self):
        target = self.root / 'installed'
        target.symlink_to(self.root / 'absent', target_is_directory=True)
        with self.assertRaises(FileExistsError):
            self.install()
        self.assertTrue(target.is_symlink())

    def test_empty_destination_supported(self):
        (self.root / 'installed').mkdir()
        self.install()
        self.assertTrue((self.root / 'installed' / pack.ENCODER).is_file())

    def test_http_url_rejected(self):
        with self.assertRaises(ValueError):
            pack.download_archive('http://example.com/pack.tar.gz', self.root / 'download')

    def test_https_redirect_cannot_downgrade_to_http(self):
        with self.assertRaises(ValueError):
            pack.HTTPSRedirects().redirect_request(None, None, 302, '', {}, 'http://example.com/model')


if __name__ == '__main__':
    unittest.main()
