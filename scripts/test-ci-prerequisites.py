#!/usr/bin/env python3
"""Exercise provisioning retries without installing packages or using sudo."""
import os
from pathlib import Path
import subprocess
import tempfile
import unittest


class ProvisioningTests(unittest.TestCase):
    def run_installer(self, fail_install=False, fail_update=False):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            log = root / 'calls'
            mock = root / 'timeout'
            mock.write_text('''#!/bin/sh
echo "$*" >> "$CALLS"
case "$*" in
  *" update") [ "$FAIL_UPDATE" = 0 ] ;;
  *) if [ "$FAIL_INSTALL" = 1 ] && [ ! -f "$CALLS.failed" ]; then
       touch "$CALLS.failed"; exit 1
     fi ;;
esac
''')
            mock.chmod(0o755)
            for name in ['protoc', 'pkg-config', 'sudo']:
                path = root / name
                path.write_text('#!/bin/sh\nexit 0\n')
                path.chmod(0o755)
            result = subprocess.run(['bash', str(Path(__file__).with_name('install-ci-prerequisites.sh'))],
                                    env={**os.environ, 'PATH': f'{root}:{os.environ["PATH"]}',
                                         'CALLS': str(log), 'FAIL_INSTALL': str(int(fail_install)),
                                         'FAIL_UPDATE': str(int(fail_update))}, check=False)
            return result.returncode, log.read_text().splitlines()

    def test_cached_indexes_do_not_trigger_update(self):
        code, calls = self.run_installer()
        self.assertEqual(code, 0)
        self.assertEqual(len(calls), 1)
        self.assertIn('--kill-after=10s 120s', calls[0])
        self.assertIn('DPkg::Lock::Timeout=30', calls[0])
        self.assertNotIn(' update', calls[0])

    def test_missing_packages_refresh_indexes_once(self):
        code, calls = self.run_installer(fail_install=True)
        self.assertEqual(code, 0)
        self.assertEqual(len(calls), 3)
        self.assertTrue(calls[1].endswith(' update'))

    def test_failed_refresh_fails_job(self):
        code, calls = self.run_installer(fail_install=True, fail_update=True)
        self.assertNotEqual(code, 0)
        self.assertEqual(len(calls), 2)


if __name__ == '__main__':
    unittest.main()
