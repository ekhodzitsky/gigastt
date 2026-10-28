"""Regression coverage for documentation outside the central docs directory."""
import importlib.util
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('drift', Path(__file__).with_name('check-docs-drift.py'))
drift = importlib.util.module_from_spec(spec)
spec.loader.exec_module(drift)


class DocumentationLinksTests(unittest.TestCase):
    def test_workspace_readmes_and_agent_guides_are_checked(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            subprocess.run(['git', 'init', '-q', str(root)], check=True)
            for name in ['README.md', 'AGENTS.md', 'crates/core/README.md', 'sdks/js/README.md',
                         'benchmark/results/report.md', 'packaging/offline/README-OFFLINE.md']:
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text('[Missing guide](missing.md)\n')
            # Ignored local planning is not part of the public documentation.
            (root / '.gitignore').write_text('private/\n')
            (root / 'private').mkdir()
            (root / 'private/notes.md').write_text('[Missing](missing.md)\n')
            with patch.object(drift, 'ROOT', root):
                errors = drift.check_links()
            self.assertEqual(len(errors), 6, errors)
            self.assertFalse(any('private/' in error for error in errors))

    def test_missing_anchor_fails_but_valid_relative_guide_passes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            subprocess.run(['git', 'init', '-q', str(root)], check=True)
            (root / 'README.md').write_text('[Guide](guide.md#setup)\n[Bad](guide.md#gone)\n')
            (root / 'guide.md').write_text('# Setup\n')
            with patch.object(drift, 'ROOT', root):
                errors = drift.check_links()
            self.assertEqual(len(errors), 1, errors)
            self.assertIn('#gone', errors[0])


if __name__ == '__main__':
    unittest.main()
