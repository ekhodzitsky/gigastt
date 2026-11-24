"""Exercise the manual Homebrew tag resolver without API calls or publication."""
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

SCRIPT = Path(__file__).with_name('resolve-homebrew-release.py')


class ReleaseTagTests(unittest.TestCase):
    def run_resolver(self, tag, output):
        return subprocess.run(
            [sys.executable, str(SCRIPT), '--tag', tag, '--output', str(output)],
            capture_output=True, text=True,
        )

    def test_writes_valid_stable_and_prerelease_tags(self):
        with tempfile.TemporaryDirectory() as directory:
            for tag in ('v2.22.0', 'v2.22.0-rc.1'):
                with self.subTest(tag=tag):
                    output = Path(directory) / tag
                    result = self.run_resolver(tag, output)
                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assertEqual(output.read_text(), f'tag={tag}\n')

    def test_invalid_tag_does_not_create_workflow_output(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'output'
            for tag in ('main', 'v02.22.0', 'v2.22.0/invalid', 'v2.22.0\ninvalid',
                        'v2.22.0-rc..1', 'v2.22.0-01'):
                with self.subTest(tag=tag):
                    result = self.run_resolver(tag, output)
                    self.assertNotEqual(result.returncode, 0)
                    self.assertIn('Homebrew release selection failed:', result.stderr)
                    self.assertFalse(output.exists())

    def test_invalid_tag_preserves_existing_workflow_output(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'output'
            output.write_text('existing=1\n')
            result = self.run_resolver('main', output)
            self.assertNotEqual(result.returncode, 0)
            self.assertEqual(output.read_text(), 'existing=1\n')


if __name__ == '__main__':
    unittest.main()
