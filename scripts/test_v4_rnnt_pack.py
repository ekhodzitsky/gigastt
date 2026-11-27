"""Bounded installer/packager tests: no real model, network, or signing key."""
from hashlib import sha256
import gzip
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import package_v4_rnnt as package


class TestPack(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.parts = {"file/encoder.onnx": b"tiny encoder", "native/step.onnx": b"tiny step",
                      "LICENSE-GigaAM": b"fixture license\n"}
        self.plan = {name: package.payload(value) for name, value in self.parts.items()}
        self.out = self.root / "installed"

    def packed(self, name="download.tar.gz", *, plan=None):
        paths = self.root / name
        with paths.open("xb") as stream:
            package.write_bundle(stream, self.plan if plan is None else plan)
        return paths, sha256(paths.read_bytes()).hexdigest()

    def mutated(self, path, mutate):
        payload = gzip.decompress(path.read_bytes())
        path.write_bytes(gzip.compress(mutate(payload), mtime=0))
        return sha256(path.read_bytes()).hexdigest()

    def test_catalogue_is_exact_fp32_dual_profile(self):
        catalogue, plan = package.qualified_plan()
        self.assertEqual(catalogue["profiles"]["file"]["mode"], "full_context")
        self.assertEqual(catalogue["profiles"]["native"]["mode"], "streaming")
        self.assertEqual(len(catalogue["profiles"]["file"]["artifacts"]), 7)
        self.assertEqual(len(catalogue["profiles"]["native"]["artifacts"]), 11)
        self.assertIn("file/manifest.toml", plan)
        self.assertIn("native/manifest.toml", plan)
        self.assertNotIn("libmkl_core.so.2", plan)
        self.assertEqual(len(plan), 23)

    def test_canonical_rebuilds_and_offline_install(self):
        a, sha = self.packed()
        b, other = self.packed("again.tar.gz", plan=dict(reversed(list(self.plan.items()))))
        self.assertEqual(sha, other)
        self.assertEqual(a.read_bytes(), b.read_bytes())
        with patch.object(package, "download_archive", side_effect=AssertionError("network forbidden")):
            package.install_archive(a, sha, self.out, self.plan)
        self.assertEqual({str(f.relative_to(self.out)) for f in self.out.rglob("*") if f.is_file()}, set(self.parts))
        for rel, data in self.parts.items():
            self.assertEqual((self.out / rel).read_bytes(), data)

    def test_tamper_fails_even_when_archive_digest_matches(self):
        a, _ = self.packed()
        sha = self.mutated(a, lambda b: b.replace(b"tiny encoder", b"fake encoder"))
        with self.assertRaisesRegex(ValueError, "digest"):
            package.install_archive(a, sha, self.out, self.plan)
        self.assertFalse(self.out.exists())
        self.assertEqual(list(self.root.iterdir()), [a])

    def test_noncanonical_headers_and_early_truncation(self):
        for label, mutate in (
            ("traversal", lambda b: b.replace(b"file/encoder.onnx", b"../x/encoder.onnx")),
            ("symlink", lambda b: b[:156] + b"2" + b[157:]),
            ("extension", lambda b: b[:156] + b"x" + b[157:]),
            ("evil-append", lambda b: b + b"extra member"),
            ("truncated", lambda b: b[:660]),
        ):
            with self.subTest(label=label):
                a, _ = self.packed(label)
                sha = self.mutated(a, mutate)
                with self.assertRaisesRegex(ValueError, "(?i)header|trailing|truncated"):
                    package.install_archive(a, sha, self.out, self.plan)
                self.assertFalse(self.out.exists())
                self.assertEqual(list(self.root.glob(".installed.*")), [])

    def test_no_second_gzip_member_or_trailing_bytes(self):
        for extra in (gzip.compress(b"", mtime=0), b"\0" * 512):
            with self.subTest(extra=len(extra)):
                filename = f"extra-{len(extra)}.gz"
                a, _ = self.packed(filename)
                data = a.read_bytes() + extra
                a.write_bytes(data)
                with self.assertRaisesRegex(ValueError, "trailing compressed bytes"):
                    package.install_archive(a, sha256(data).hexdigest(), self.out, self.plan)
                self.assertFalse(self.out.exists())

    def test_wrong_signature_cannot_publish(self):
        a, sha = self.packed()
        with patch.object(package, "verify_signature", side_effect=ValueError("bad signature")):
            with self.assertRaisesRegex(ValueError, "bad signature"):
                package.install_archive(a, sha, self.out, self.plan, signature=self.root / "signature",
                                        public_key=self.root / "public-key")
        self.assertFalse(self.out.exists())

    def test_same_inode_rewrite_detected_during_extract(self):
        a, sha = self.packed()
        before = a.stat().st_ino

        def change_mtime_header(*_args):
            with a.open("r+b") as writer:
                writer.seek(4)
                writer.write(b"\1")
            self.assertEqual(a.stat().st_ino, before)

        with patch.object(package, "verify_signature", side_effect=change_mtime_header):
            with self.assertRaisesRegex(ValueError, "SHA-256 changed during extraction"):
                package.install_archive(a, sha, self.out, self.plan, signature=self.root / "signature",
                                        public_key=self.root / "public-key")
        self.assertFalse(self.out.exists())

    def test_wrong_outer_sha_refuses_to_install(self):
        a, _ = self.packed()
        with self.assertRaisesRegex(ValueError, "Archive SHA"):
            package.install_archive(a, "0"*64, self.out, self.plan)
        self.assertFalse(self.out.exists())

    def test_existing_empty_regular_directory_or_symlink_never_clobbered(self):
        a, sha = self.packed()
        self.out.mkdir()
        ino = self.out.stat().st_ino
        with self.assertRaises(FileExistsError):
            package.install_archive(a, sha, self.out, self.plan)
        self.assertEqual(ino, self.out.stat().st_ino)
        self.out.rmdir()
        self.out.symlink_to(self.root / "absent")
        with self.assertRaises(FileExistsError):
            package.install_archive(a, sha, self.out, self.plan)
        self.assertTrue(self.out.is_symlink())

    def test_atomic_publication_primitive_refuses_racing_empty_target(self):
        ready = self.root / "ready"
        ready.mkdir()
        self.out.mkdir()
        inode = self.out.stat().st_ino
        with self.assertRaises(FileExistsError):
            package.publish_noreplace(ready, self.out)
        self.assertEqual(self.out.stat().st_ino, inode)
        self.assertTrue(ready.is_dir())

    def test_input_symlink_and_nonregular_file_refused(self):
        a, sha = self.packed()
        link = self.root / "link"
        link.symlink_to(a)
        with self.assertRaises((ValueError, OSError)):
            package.install_archive(link, sha, self.out, self.plan)
        fifo = self.root / "fifo"
        os.mkfifo(fifo)
        with self.assertRaises(ValueError):
            package.install_archive(fifo, sha, self.out, self.plan)

    def test_explicit_https_and_no_offline_url(self):
        self.assertEqual(package.require_https("https://example.test/path"), "https://example.test/path")
        for url in ("file:///tmp/data", "http://example.test/path", "https://user@example.test/path",
                    "https:///absent", "https://example.test/abc#fragment"):
            with self.subTest(url=url), self.assertRaises(ValueError):
                package.require_https(url)
        with patch.object(package, "download_archive", side_effect=AssertionError("network forbidden")):
            with self.assertRaises(SystemExit):
                package.main(["--offline", "download", "--url", "https://example.test/a",
                              "--sha256", "0"*64, "--destination", str(self.out)])

    def test_signature_options_must_be_paired(self):
        with self.assertRaises(SystemExit):
            package.main(["install", "--archive", "x", "--sha256", "0"*64, "--signature", "x.sig",
                          "--destination", str(self.out)])


if __name__ == "__main__":
    unittest.main()
