import copy
import importlib.util
from pathlib import Path
import unittest
import os
import re
import subprocess
import tempfile
from unittest import mock

spec = importlib.util.spec_from_file_location('objc_gate', Path(__file__).with_name('compare-objc-revisions.py'))
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def fixture():
    mode = {'seconds': 1.0, 'load_seconds': 0.5, 'cached': True,
            'outputs': [{'sha256': 'a' * 64, 'shape': [1, 1024, 192]} for _ in range(3)]}
    row = {'peak_rss_bytes': 1000, 'fixtures': ['b' * 64] * 3,
           'modes': {'cpu': mode, 'cpu_and_ne': copy.deepcopy(mode)}}
    return {arm: [copy.deepcopy(row) for _ in range(7)] for arm in ['base', 'candidate']}


class GateTests(unittest.TestCase):
    def test_identical_passes(self):
        module.validate(fixture())

    def test_missing_round_fails(self):
        r = fixture(); r['candidate'].pop()
        with self.assertRaises(ValueError): module.validate(r)

    def test_changed_output_fails(self):
        r = fixture(); r['candidate'][2]['modes']['cpu']['outputs'][0]['sha256'] = 'c' * 64
        with self.assertRaises(ValueError): module.validate(r)

    def test_invalid_metrics_fail(self):
        for value in [0, -1, float('nan'), float('inf')]:
            r = fixture(); r['candidate'][0]['modes']['cpu']['seconds'] = value
            with self.subTest(value=value), self.assertRaises(ValueError): module.validate(r)

    def test_slowdown_fails(self):
        r = fixture()
        for row in r['candidate']: row['modes']['cpu']['seconds'] = 1.1
        with self.assertRaises(ValueError): module.validate(r)

    def test_diagnostic_retains_regression_without_admitting_candidate(self):
        r = fixture()
        for row in r['candidate']:
            row['modes']['cpu']['load_seconds'] = 0.6
        self.assertAlmostEqual(module.summarize(r)['cpu/load_seconds']['median_ratio'], 1.2)
        with self.assertRaises(ValueError):
            module.validate(r)

    def test_cache_fingerprint_tracks_contents(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaises(ValueError):
                module.cache_fingerprint(root)
            path = root / 'weights'
            path.write_bytes(b'one')
            before = module.cache_fingerprint(root)
            path.write_bytes(b'two')
            self.assertNotEqual(before, module.cache_fingerprint(root))
            (root / 'link').symlink_to('weights')
            self.assertEqual(module.cache_fingerprint(root)['link'], {'symlink': 'weights'})

    def test_copy_cache_preserves_contents_and_separates_roots(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / 'source'
            target = Path(directory) / 'target'
            source.mkdir()
            (source / 'weights').write_bytes(b'compiled')
            (source / 'model.meta').write_text('source key')
            (source / 'link').symlink_to('weights')
            expected = module.copy_cache(source, target)
            self.assertEqual(expected, module.cache_fingerprint(target))
            (target / 'weights').write_bytes(b'changed')
            self.assertEqual(expected, module.cache_fingerprint(source))
            self.assertNotEqual(expected, module.cache_fingerprint(target))

    def test_key_diagnostic_rejects_invalid_timings(self):
        row = {'key_seconds': 0.01, 'meta_seconds': 0.001}
        report = {arm: [dict(row) for _ in range(7)] for arm in ('base', 'candidate')}
        self.assertEqual(module.summarize_key_timings(report)['key_seconds']['median_ratio'], 1.0)
        for value in (0, -1, float('nan'), float('inf')):
            invalid = copy.deepcopy(report)
            invalid['candidate'][0]['key_seconds'] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                module.summarize_key_timings(invalid)

    def test_memory_regression_fails(self):
        r = fixture()
        for row in r['candidate']: row['peak_rss_bytes'] = 1101
        with self.assertRaises(ValueError): module.validate(r)

    def test_changed_assets_or_cold_cache_fails(self):
        r = fixture(); r['candidate'][0]['fixtures'][0] = 'c' * 64
        with self.assertRaises(ValueError): module.validate(r)
        r = fixture(); r['candidate'][0]['modes']['cpu']['cached'] = False
        with self.assertRaises(ValueError): module.validate(r)


class BuildEnvironmentTests(unittest.TestCase):
    def test_common_runtime_preserves_effective_flags_and_spaces(self):
        with tempfile.TemporaryDirectory(prefix='Xcode Toolchain ') as directory:
            resource = Path(directory)
            archive = resource / 'lib' / 'darwin' / 'libclang_rt.osx.a'
            archive.parent.mkdir(parents=True)
            archive.write_bytes(b'test archive')
            discovery = subprocess.CompletedProcess([], 0, stdout=str(resource) + '\n')
            cases = [
                ({}, []),
                ({'RUSTFLAGS': '-C debuginfo=1'}, ['-C', 'debuginfo=1']),
                ({'RUSTFLAGS': '-C debuginfo=1',
                  'CARGO_ENCODED_RUSTFLAGS': '-C\x1fopt-level=3'}, ['-C', 'opt-level=3']),
                ({'RUSTFLAGS': '-C debuginfo=1',
                  'CARGO_ENCODED_RUSTFLAGS': ''}, []),
            ]
            for source, expected in cases:
                source = {**source, 'MACOSX_DEPLOYMENT_TARGET': '11.0'}
                with self.subTest(source=source), mock.patch.dict(os.environ, source, clear=True), \
                        mock.patch.object(module, 'run', return_value=discovery) as tool:
                    result, observed = module.comparison_build_env()
                    tool.assert_called_once_with('xcrun', '--sdk', 'macosx', 'clang',
                                                 '--print-resource-dir', capture_output=True)
                    self.assertEqual(observed, archive)
                    self.assertEqual(result['CARGO_ENCODED_RUSTFLAGS'].split('\x1f'),
                                     [*expected, '-C', 'link-arg=' + str(archive)])
                    self.assertEqual(result['MACOSX_DEPLOYMENT_TARGET'], '11.0')
                    self.assertEqual(dict(os.environ), source)

    def test_missing_runtime_fails_before_measurement(self):
        with tempfile.TemporaryDirectory() as directory:
            discovery = subprocess.CompletedProcess([], 0, stdout=directory + '\n')
            with mock.patch.object(module, 'run', return_value=discovery), self.assertRaises(ValueError):
                module.comparison_build_env()


class AffectedTests(unittest.TestCase):
    def changed(self, base_source, candidate_source, paths=''):
        def git(*args):
            if args[0] == 'show':
                source = base_source if args[1].startswith('base:') else candidate_source
                return '[[package]]\nname = "objc2"\nversion = "0.6.5"\nsource = "' + source + '"\n'
            pathspecs = args[args.index('--') + 1:]
            return '\n'.join(path for path in paths.splitlines()
                             if any(path == spec or path.startswith(spec + '/') for spec in pathspecs))
        with mock.patch.object(module, 'git', side_effect=git):
            return module.affected('base', 'candidate')

    def test_unchanged_runtime_skips_measurement(self):
        self.assertFalse(self.changed('registry+same', 'registry+same'))

    def test_source_revision_change_is_detected(self):
        self.assertTrue(self.changed('git+repo#old', 'git+repo#new'))

    def test_bridge_change_is_detected(self):
        self.assertTrue(self.changed('registry+same', 'registry+same',
                                    'crates/gigastt-core/src/runtime/coreml/bridge/predict.rs'))

    def test_linker_setup_change_is_detected(self):
        self.assertTrue(self.changed('registry+same', 'registry+same',
                                    'crates/gigastt-core/build.rs'))

    def test_test_only_change_skips_measurement(self):
        self.assertFalse(self.changed('registry+same', 'registry+same',
                                     'crates/gigastt-core/src/runtime/coreml/bridge/tests/objc_lifetime.rs'))


class HostTests(unittest.TestCase):
    def test_model_requires_macos_15_or_newer(self):
        for version in ('14.8.9', '', 'unknown'):
            with self.subTest(version=version), self.assertRaises(ValueError):
                module.require_model_host(version)
        for version in ('15.0', '15.7.1', '26.0'):
            module.require_model_host(version)

    def test_ci_runner_supports_pinned_model(self):
        workflow = (module.ROOT / '.github/workflows/ci.yml').read_text()
        job = workflow.split('  build-ane:\n', 1)[1]
        job = re.split(r'\n  [a-zA-Z0-9_-]+:\n', job, maxsplit=1)[0]
        self.assertIn('runs-on: macos-15', job)


class ProtectedGateTests(unittest.TestCase):
    def test_each_required_failure_blocks_model_smoke(self):
        workflow = (module.ROOT / '.github/workflows/ci.yml').read_text()
        job = workflow.split('  model-smoke:\n', 1)[1]
        job = re.split(r'\n  [a-zA-Z0-9_-]+:\n', job, maxsplit=1)[0]
        command = re.search(r'^        run: (.+)$', job, re.MULTILINE).group(1)
        successful = dict(BENCH_RESULT='success', SPEAKER_RESULT='success',
                          ASR_RESULT='success', OBJC_RESULT='success')
        for dependency in successful:
            for result in ('failure', 'cancelled', 'skipped', '', 'success', None):
                with self.subTest(dependency=dependency, result=result):
                    env = {**os.environ, **successful}
                    if result is None:
                        env.pop(dependency)
                    else:
                        env[dependency] = result
                    run = subprocess.run(['bash', '-c', command], check=False, env=env)
                    self.assertEqual(run.returncode == 0, result == 'success')
        self.assertRegex(job, r'needs: \[[^\]]*build-ane[^\]]*\]')
        self.assertIn('    if: always()', job)
        self.assertIn('OBJC_RESULT: ${{ needs.build-ane.result }}', job)


if __name__ == '__main__': unittest.main()
