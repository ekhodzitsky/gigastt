#!/usr/bin/env python3
"""Exercise the Apple link setup without requiring an Apple SDK on Linux."""
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class AppleRuntimeLinkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.directory = Path(cls.temp.name)
        cls.binary = cls.directory / 'build-script'
        subprocess.run(['rustc', '--edition=2024', str(ROOT / 'crates/gigastt-core/build.rs'),
                        '-o', str(cls.binary)], check=True)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def run_setup(self, target='macos', coreml=True, dynamic=False, missing=False, fail=False):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / 'SDK with spaces' / 'lib/darwin/libclang_rt.osx.a'
            archive.parent.mkdir(parents=True)
            if not missing:
                archive.write_bytes(b'archive fixture')
            xcrun = root / 'xcrun'
            xcrun.write_text('#!/bin/sh\nprintf "%s\\n" "$*" > "$CALLS"\n'
                             'if [ "$FAIL" = 1 ]; then echo "SDK unavailable" >&2; exit 1; fi\n'
                             'printf "%s\\n" "$RESOURCE_DIR"\n')
            xcrun.chmod(0o755)
            env = {k: v for k, v in os.environ.items() if not k.startswith('CARGO_FEATURE_')}
            env.update(PATH=f'{root}:{env["PATH"]}', CARGO_CFG_TARGET_OS=target,
                       CALLS=str(root/'calls'), RESOURCE_DIR=str(root / "SDK with spaces"), FAIL=str(int(fail)))
            if coreml:
                env['CARGO_FEATURE_COREML'] = '1'
            if dynamic:
                env['CARGO_FEATURE_ORT_LOAD_DYNAMIC'] = '1'
            result = subprocess.run([str(self.binary)], env=env, text=True, capture_output=True)
            calls = (root/'calls').read_text() if (root/'calls').exists() else ''
            return result, calls, archive

    def test_static_coreml_links_selected_sdk_runtime(self):
        result, calls, archive = self.run_setup()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('--sdk macosx clang --print-resource-dir', calls)
        self.assertIn(f'cargo:rustc-link-search=native={archive.parent}', result.stdout)
        self.assertIn('cargo:rustc-link-lib=static=clang_rt.osx', result.stdout)

    def test_unaffected_builds_do_not_probe_sdk(self):
        for options in ({'target': 'linux'}, {'coreml': False}, {'dynamic': True}):
            with self.subTest(options=options):
                result, calls, _ = self.run_setup(**options)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(calls, '')
                self.assertNotIn('rustc-link-lib', result.stdout)

    def test_missing_archive_fails_instead_of_emitting_invalid_link_flags(self):
        result, _, _ = self.run_setup(missing=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertNotIn('rustc-link-lib', result.stdout)

    def test_sdk_failure_is_reported(self):
        result, _, _ = self.run_setup(fail=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('SDK unavailable', result.stderr)


if __name__ == '__main__':
    unittest.main()
