#!/usr/bin/env python3
"""Fail-closed whole-ASR comparison contract, without model downloads."""
import copy
import importlib.util
import unittest
from unittest.mock import patch
import tempfile
import json
import subprocess
import sys
from types import SimpleNamespace
import shutil
from pathlib import Path

spec = importlib.util.spec_from_file_location('asr', Path(__file__).with_name('asr-performance.py'))
asr = importlib.util.module_from_spec(spec)
spec.loader.exec_module(asr)


def report():
    sample = {'cpu_affinity': '0-1', 'load_seconds': 1.0, 'first_seconds': 0.4, 'rss_kib': 400000,
              'peak_rss_kib': 500000, 'first_text': 'привет мир', 'provider': 'cpu', 'pool_size': 1, 'threads': 2,
              'cases': {name: {'seconds': 0.2, 'iterations': asr.ASSETS['workload'][name]['iterations'], 'duration': asr.ASSETS['workload'][name]['duration'], 'text': 'привет мир'}
                        for name in asr.CASES}}
    return {'cpu_affinity': [0, 1], 'inputs_verified': True, 'audio_sha256': {n: a['sha256'] for n, a in asr.ASSETS['workload'].items()},
            'harness_sha256': 'e'*64, 'protocol': 1, 'assets': copy.deepcopy(asr.ASSETS),
            'revisions': {'base': 'a'*40, 'candidate': 'b'*40},
            'binary_sha256': {'base': 'c'*64, 'candidate': 'd'*64},
            'heads': {h: {side: [copy.deepcopy(sample) for _ in range(7)]
                          for side in ['base', 'candidate']} for h in asr.HEADS}}


class ComparisonTests(unittest.TestCase):
    def test_equal_pair_passes(self):
        self.assertTrue(asr.check(report()))

    def test_twofold_regression_in_each_phase_fails(self):
        for metric in ['load_seconds', 'first_seconds', 'rss_kib', 'peak_rss_kib', *asr.CASES]:
            with self.subTest(metric=metric):
                r = report()
                for sample in r['heads']['rnnt']['candidate']:
                    if metric in asr.CASES:
                        sample['cases'][metric]['seconds'] *= 2
                    else:
                        sample[metric] *= 2
                        if metric == 'rss_kib':
                            sample['peak_rss_kib'] *= 2
                with self.assertRaisesRegex(ValueError, 'ASR regression'): asr.check(r)

    def test_missing_invalid_or_changed_evidence_fails(self):
        mutations = [lambda r: r.update(cpu_affinity=[0, 0]),
                     lambda r: r['heads']['rnnt']['candidate'][0].update(cpu_affinity='2-3'),
                     lambda r: r['heads']['rnnt']['candidate'][0]['cases'][asr.CASES[0]].update(iterations=1),
                     lambda r: r.pop('inputs_verified'),
                     lambda r: r.update(protocol=True),
                     lambda r: r['heads']['rnnt']['candidate'][0].update(pool_size=True),
                     lambda r: r['heads']['rnnt']['candidate'][0].update(first_text={}),
                     lambda r: r['heads'].pop('e2e_rnnt'),
                     lambda r: r['heads']['rnnt']['base'].pop(),
                     lambda r: r['assets']['models']['rnnt'].clear(),
                     lambda r: r['binary_sha256'].pop('candidate'),
                     lambda r: r['heads']['rnnt']['candidate'][0]['cases'].pop(asr.CASES[0]),
                     lambda r: r['heads']['rnnt']['candidate'][0].update(provider='cuda'),
                     lambda r: r['heads']['rnnt']['candidate'][0]['cases'][asr.CASES[0]].update(text='привет'),
                     lambda r: r['heads']['rnnt']['candidate'][0]['cases'][asr.CASES[0]].update(duration=2)]
        for mutate in mutations:
            r = report(); mutate(r)
            with self.assertRaises(ValueError): asr.check(r)
        for value in [None, True, 0, -1, float('nan'), float('inf')]:
            r = report(); r['heads']['rnnt']['candidate'][0]['load_seconds'] = value
            with self.assertRaises(ValueError): asr.check(r)

    def test_one_timing_outlier_does_not_mask_or_create_regression(self):
        r = report(); r['heads']['rnnt']['candidate'][0]['load_seconds'] *= 5
        asr.check(r)
        for s in r['heads']['rnnt']['candidate'][1:]: s['load_seconds'] *= 2
        with self.assertRaises(ValueError): asr.check(r)

    def test_model_mismatch_fails_before_any_probe(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'source' / 'rnnt'
            source.mkdir(parents=True)
            name = next(iter(asr.ASSETS['models']['rnnt']))
            (source / name).write_bytes(b'corrupt model')
            destination = root / 'destination'
            destination.mkdir()
            with patch.dict(asr.os.environ, {'GIGASTT_ASR_MODELS': str(source.parent)}):
                with self.assertRaisesRegex(ValueError, 'unpinned model'):
                    asr.prepare_assets(destination, root / 'audio')
            self.assertFalse((root / 'audio').exists())

    def test_missing_model_configuration_fails(self):
        with patch.dict(asr.os.environ, {}, clear=True):
            with self.assertRaisesRegex(ValueError, 'GIGASTT_ASR_MODELS'):
                asr.prepare_assets(Path('unused'), Path('unused'))

    def test_affinity_is_restored_after_failure(self):
        with patch.object(asr.os, 'sched_getaffinity', return_value={5, 7, 8}, create=True), \
             patch.object(asr.os, 'sched_setaffinity', create=True) as setter:
            with self.assertRaises(RuntimeError):
                with asr.pinned_cpus() as cpus:
                    self.assertEqual(cpus, [5, 7])
                    raise RuntimeError('failed subprocess')
            self.assertEqual(setter.call_args_list[0].args, (0, [5, 7]))
            self.assertEqual(setter.call_args_list[-1].args, (0, {5, 7, 8}))

    def test_build_uses_revision_config_and_failure_cleans_up(self):
        trees = []
        def command(*args, **kwargs):
            if args[:3] == ('git', 'worktree', 'add'):
                tree = Path(args[4]); trees.append(tree)
                module = tree / 'crates/gigastt-core/src/inference/engine/mod.rs'
                module.parent.mkdir(parents=True); module.write_text('')
                (tree / '.cargo').mkdir()
                (tree / '.cargo/config.toml').write_text('# revision configuration')
            elif args[:3] == ('git', 'worktree', 'remove'):
                shutil.rmtree(args[4])
            return SimpleNamespace(stdout='a' * 40)
        def assets(models, audio):
            audio.mkdir()
            for name in asr.CASES: (audio / name).write_bytes(b'fixture')
        def fail_build(args, **kwargs):
            self.assertEqual((kwargs['cwd'] / '.cargo/config.toml').read_text(), '# revision configuration')
            return subprocess.CompletedProcess(args, 1, stdout='')
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = SimpleNamespace(base='base', candidate='candidate', output=root / 'output', target_dir=root / 'target')
            with patch.object(asr.sys, 'platform', 'linux'), patch.object(asr, 'command', command), \
                 patch.object(asr, 'prepare_assets', assets), patch.object(asr.subprocess, 'run', fail_build):
                with self.assertRaises(subprocess.CalledProcessError): asr.compare(args)
            self.assertEqual(len(trees), 1)
            self.assertFalse(trees[0].exists())
            with self.assertRaises(ValueError): asr.check(json.loads((args.output / 'report.json').read_text()))

    @unittest.skipUnless(sys.platform == "linux", "measurement runner requires Linux")
    def test_failed_rerun_cannot_leave_passing_report(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            evidence = output / 'report.json'
            evidence.write_text(json.dumps(report()))
            result = subprocess.run([sys.executable, str(Path(asr.__file__)),
                                     '--base', 'not a valid revision', '--output', str(output)],
                                    text=True, capture_output=True, timeout=30)
            self.assertNotEqual(result.returncode, 0)
            with self.assertRaises(ValueError): asr.check(json.loads(evidence.read_text()))

    def test_release_rejects_missing_or_unsuccessful_asr_job(self):
        spec = importlib.util.spec_from_file_location('gates', Path(__file__).with_name('performance-gates.py'))
        gates = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(gates)
        run = {'id': 10, 'head_sha': 'a' * 40, 'event': 'workflow_dispatch',
               'head_branch': 'main', 'status': 'completed', 'conclusion': 'success'}
        jobs = [{'name': name, 'conclusion': 'success'} for name in gates.REQUIRED_JOBS
                if name != 'ASR performance']
        with self.assertRaises(ValueError): gates.check_release([run], jobs, 'a' * 40)
        for result in ['failure', 'cancelled', 'skipped', None]:
            with self.subTest(result=result), self.assertRaises(ValueError):
                gates.check_release([run], jobs + [{'name': 'ASR performance', 'conclusion': result}], 'a' * 40)
        self.assertEqual(gates.check_release([run], jobs + [{'name': 'ASR performance', 'conclusion': 'success'}], 'a' * 40), 10)

    def test_ci_and_release_require_comparison(self):
        root = Path(__file__).resolve().parents[1]
        ci = (root / '.github/workflows/ci.yml').read_text()
        smoke = ci.split('  model-smoke:', 1)[1].split('\n  e2e-tests:', 1)[0]
        self.assertIn('asr-performance', smoke)
        self.assertIn('ASR_RESULT: ${{ needs.asr-performance.result }}', smoke)
        self.assertIn('test "$ASR_RESULT" = success', smoke)
        self.assertIn('ASR performance', (root / 'scripts/performance-gates.py').read_text())


if __name__ == '__main__': unittest.main()
