#!/usr/bin/env python3
"""Revision comparisons must execute their own Cargo artifacts, without models."""
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

import revision_build


class RevisionBuildTests(unittest.TestCase):
    def test_artifact_outside_revision_target_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            binary = root / 'other/probe'
            binary.parent.mkdir()
            binary.write_bytes(b'wrong revision')
            with self.assertRaisesRegex(ValueError, 'outside'):
                revision_build.executable_record(binary, root / 'candidate')

    def test_symlink_escape_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target = root / 'candidate'
            target.mkdir()
            outside = root / 'base-probe'
            outside.write_bytes(b'base')
            (target / 'probe').symlink_to(outside)
            with self.assertRaisesRegex(ValueError, 'outside'):
                revision_build.executable_record(target / 'probe', target)

    def test_same_package_and_mtimes_execute_distinct_sources(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            records = []
            for label, literal in [('base', 'BASE'), ('candidate', 'HEAD')]:
                source = root / label
                (source / 'src').mkdir(parents=True)
                (source / 'Cargo.toml').write_text(
                    '[package]\nname="revision-probe"\nversion="0.1.0"\nedition="2021"\n')
                (source / 'src/main.rs').write_text(f'fn main() {{ println!("{literal}"); }}\n')
                for name in ['Cargo.toml', 'src/main.rs']:
                    os.utime(source / name, (1_600_000_000, 1_600_000_000))
                subprocess.run(['git', 'init', '-q', str(source)], check=True)
                subprocess.run(['git', '-C', str(source), 'add', '.'], check=True)
                subprocess.run(['git', '-C', str(source), '-c', 'user.name=Test',
                                '-c', 'user.email=test@example.invalid', 'commit', '-qm', label], check=True)
                revision = subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip()
                target = revision_build.target_directory(root / 'builds', label, revision)
                record = revision_build.build(source, target, ['build', '--offline'], root / f'{label}.jsonl')
                self.assertEqual(record['source_revision'], revision)
                self.assertEqual(len(record['executables']), 1)
                binary = record['executables'][0]['path']
                self.assertEqual(subprocess.check_output([binary], text=True).strip(), literal)
                self.assertEqual({k: record['executables'][0][k] for k in ('path', 'sha256')},
                                 revision_build.executable_record(Path(binary), target))
                records.append(record)
            self.assertNotEqual(records[0]['target_dir'], records[1]['target_dir'])
            self.assertNotEqual(records[0]['executables'][0]['sha256'], records[1]['executables'][0]['sha256'])

    def test_targets_remain_distinct_for_identical_revisions(self):
        self.assertNotEqual(revision_build.target_directory('/tmp/builds', 'base', 'a' * 40),
                            revision_build.target_directory('/tmp/builds', 'candidate', 'a' * 40))
        with self.assertRaises(ValueError):
            revision_build.target_directory('/tmp/builds', '../other', 'a' * 40)

    def test_speaker_builds_use_revision_targets_and_injected_source_hash(self):
        source = Path(__file__).with_name('compare-speaker-revisions.py').read_text()
        self.assertIn('target_directory(', source)
        self.assertIn("provenance['injected_harness_sha256']", source)
        self.assertIn('executable_record(tests[0], target_dir)', source)
        self.assertNotIn("'--target-dir', str(ROOT / 'target')", source)

    def test_workflow_shares_only_criterion_data(self):
        workflow = (Path(__file__).resolve().parent.parent / '.github/workflows/ci.yml').read_text()
        bench = workflow.split('\n  bench:\n', 1)[1].split('\n  coverage-e2e:', 1)[0]
        self.assertIn('scripts/compare-benchmark-revisions.py', bench)
        self.assertIn('CRITERION_HOME:', bench)
        self.assertNotIn('--target-dir "$GITHUB_WORKSPACE/target"', bench)
        self.assertIn('performance-gates.py criterion target/criterion', bench)


if __name__ == '__main__':
    unittest.main()
