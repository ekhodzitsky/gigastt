#!/usr/bin/env python3
"""Offline runtime compatibility tests; no SDK or Maven download required."""
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

spec = importlib.util.spec_from_file_location('runtime', Path(__file__).with_name('check-android-runtime.py'))
runtime = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runtime)


class AndroidRuntimeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.archive = Path(self.temp.name) / 'runtime.aar'
        self.metadata = {'packages': [{'id': 'ort-sys-id', 'name': 'ort-sys'}],
                         'resolve': {'nodes': [{'id': 'ort-sys-id', 'features': ['api-26', 'api-27']}]}}
        self.pin = {'version': '1.27.0', 'sha256': '0' * 64}

    def make_archive(self, api=27, missing=None):
        with zipfile.ZipFile(self.archive, 'w') as archive:
            archive.writestr('headers/onnxruntime_c_api.h', f'#define ORT_API_VERSION {api}\n')
            for abi in runtime.ABIS:
                if abi != missing:
                    archive.writestr(f'jni/{abi}/libonnxruntime.so', b'\x7fELFfixture')
        self.pin['sha256'] = hashlib.sha256(self.archive.read_bytes()).hexdigest()

    def test_matching_api_and_all_abis_pass(self):
        self.make_archive()
        runtime.check(self.metadata, self.pin, self.archive)

    def test_offline_drift_rejects_old_runtime_pin(self):
        self.pin['version'] = '1.24.2'
        with self.assertRaisesRegex(ValueError, 'API 27'):
            runtime.check(self.metadata, self.pin)

    def test_new_dependency_api_requires_pin_update(self):
        self.metadata['resolve']['nodes'][0]['features'].append('api-28')
        with self.assertRaisesRegex(ValueError, 'API 28'):
            runtime.check(self.metadata, self.pin)

    def test_old_rust_api_can_use_newer_runtime(self):
        self.metadata['resolve']['nodes'][0]['features'] = ['api-24']
        runtime.check(self.metadata, self.pin)

    def test_checksum_header_and_abi_mismatches_fail(self):
        self.make_archive()
        self.pin['sha256'] = '0' * 64
        with self.assertRaisesRegex(ValueError, 'checksum'):
            runtime.check(self.metadata, self.pin, self.archive)
        self.make_archive(api=24)
        with self.assertRaisesRegex(ValueError, 'header'):
            runtime.check(self.metadata, self.pin, self.archive)
        self.make_archive(missing='armeabi-v7a')
        with self.assertRaisesRegex(ValueError, 'armeabi-v7a'):
            runtime.check(self.metadata, self.pin, self.archive)

    def test_missing_or_ambiguous_dependency_fails_closed(self):
        self.metadata['packages'] = []
        with self.assertRaisesRegex(ValueError, 'one ort-sys'):
            runtime.check(self.metadata, self.pin)

    def test_repository_pin_supports_current_api_31(self):
        root = Path(__file__).resolve().parent.parent
        pin = json.loads((root / 'packaging/android/onnxruntime.json').read_text())
        self.metadata['resolve']['nodes'][0]['features'].append('api-31')
        runtime.check(self.metadata, pin)

    def test_api_31_requires_matching_header_and_rejects_api_27(self):
        self.metadata['resolve']['nodes'][0]['features'].append('api-31')
        with self.assertRaisesRegex(ValueError, 'API 31'):
            runtime.check(self.metadata, self.pin)
        self.pin['version'] = '1.31.0'
        self.make_archive(api=31)
        runtime.check(self.metadata, self.pin, self.archive)
        self.make_archive(api=27)
        with self.assertRaisesRegex(ValueError, 'header'):
            runtime.check(self.metadata, self.pin, self.archive)

    def test_workflow_checks_archive_before_build_with_trusted_tools(self):
        root = Path(__file__).resolve().parent.parent
        workflow = (root / '.github/workflows/android-aar.yml').read_text()
        self.assertIn('ref: ${{ github.workflow_sha }}', workflow)
        self.assertIn('path: .workflow-tools', workflow)
        self.assertLess(workflow.index('scripts/check-android-runtime.py'), workflow.index('- name: Build native libs'))
        self.assertIn('--aar ort-android.aar', workflow)
        self.assertIn('scripts/check-android-runtime.py', (root / '.github/workflows/ci.yml').read_text())


if __name__ == '__main__':
    unittest.main()
