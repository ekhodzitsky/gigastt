#!/usr/bin/env python3
"""Synthetic qualification of paired diagnostics; no model or Criterion needed."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import criterion_pilot as pilot


class PilotTests(unittest.TestCase):
    def test_known_regression(self):
        control = [(100, 100)] * 7
        comparison = [(100, 110)] * 7
        self.assertEqual(pilot.assess(control, comparison)['outcome'], 'regression')

    def test_noise_is_not_pass_or_regression(self):
        control = [(100, 109)] + [(100, 100)] * 6
        self.assertEqual(pilot.assess(control, [(100, 110)] * 7)['outcome'], 'unstable_control')
        comparison = [(100, 107)] + [(100, 100)] * 6
        self.assertEqual(pilot.assess([(100, 100)] * 7, comparison)['outcome'], 'inconclusive')

    def test_complete_stable_control(self):
        result = pilot.assess([(100, 96), (100, 104)] * 3 + [(100, 100)], [(100, 103)] * 7)
        self.assertEqual(result['outcome'], 'within_limit')
        self.assertEqual(result['aa']['median_ratio'], 1)
        self.assertEqual(result['aa']['median_interval'], [.96, 1.04])

    def test_missing_control_or_invalid_sample(self):
        for control in ([], [(100, 100)] * 6, [(100, float('nan'))] * 7, [(0, 100)] * 7):
            with self.subTest(control=control), self.assertRaises(ValueError):
                pilot.assess(control, [(100, 100)] * 7)

    def test_raw_sample_required(self):
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            baseline = home / 'case' / 'main'
            baseline.mkdir(parents=True)
            (baseline / 'estimates.json').write_text(json.dumps({'mean': {'point_estimate': 3}}))
            with self.assertRaises(FileNotFoundError):
                pilot.read_measurement(home, 'case')
            (baseline / 'sample.json').write_text(json.dumps({'iters': [1] * 10, 'times': [3] * 10}))
            self.assertEqual(pilot.read_measurement(home, 'case'), 3)
            (baseline / 'sample.json').write_text(json.dumps({'iters': [1] * 9, 'times': [3] * 10}))
            with self.assertRaises(ValueError):
                pilot.read_measurement(home, 'case')

    def test_fixed_schedule_and_executable_identities(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            provenance = {}
            for side in ('base', 'candidate'):
                target = root / side
                target.mkdir()
                artifacts = []
                for name in pilot.CASES:
                    binary = target / name
                    binary.write_text(side + name)
                    artifacts.append({'target': name, 'kind': ['bench'], 'path': str(binary),
                                      'sha256': pilot.binary_hash(binary)})
                provenance[side] = {'target_dir': str(target), 'executables': artifacts}
            source = root / 'gate' / 'build-provenance.json'
            source.parent.mkdir()
            source.write_text(json.dumps(provenance))
            out = root / 'pilot'
            with patch.object(pilot, 'workload_identity', return_value={'identical': True}), \
                 patch.object(pilot, 'measure', side_effect=lambda binary, *args: {
                     'mean_ns': 100, 'binary_sha256': binary['sha256']
                 }) as measure:
                pilot.run(source, out, root)
            self.assertEqual(measure.call_count, 7 * 7 * 2 * 2)
            report = json.loads((out / 'report.json').read_text())
            self.assertTrue(report['complete'])
            self.assertTrue(report['workload_identity']['identical'])
            for case in report['cases'].values():
                self.assertEqual(case['outcome'], 'within_limit')
                self.assertEqual(case['synthetic_10pct_slowdown'], 'regression')
                for i, block in enumerate(case['measurements']):
                    self.assertEqual(block['index'], i // 2)
                    self.assertEqual(block['order'], ['a', 'b'] if i // 2 % 2 == 0 else ['b', 'a'])
                    self.assertEqual(block['mode'], ['aa', 'ab'][((i // 2) + i) % 2])
                    self.assertEqual(block['arms']['a']['binary_sha256'] == block['arms']['b']['binary_sha256'],
                                     block['mode'] == 'aa')
            Path(provenance['base']['executables'][0]['path']).write_text('changed')
            with self.assertRaisesRegex(ValueError, 'hash mismatch'):
                pilot.validated_binaries(provenance)

    def test_cpu_tick_parser(self):
        data = 'cpu 100 0 0 100\ncpu0 12 0 2 80\ncpu15 8 0 1 85\nintr 10 4 6\n'
        self.assertEqual(pilot.parse_cpu_ticks(data),
                         {'cpu0': [12, 0, 2, 80], 'cpu15': [8, 0, 1, 85]})

    def test_workload_identity(self):
        provenance = {side: {'source_revision': side[0] * 40} for side in ('base', 'candidate')}
        rows = ['100644 blob hash\tcrates/gigastt-core/benches/' + target + '.rs'
                for target in pilot.CASES]
        rows.append('040000 tree hash\tcrates/gigastt-core/tests/fixtures/telephony')
        original = '\n'.join(rows)
        changed = '\n'.join(rows[:-1] + [rows[-1].replace('hash', 'other')])
        with patch.object(pilot.subprocess, 'check_output', side_effect=[original, changed]):
            self.assertFalse(pilot.workload_identity(provenance, Path('.'))['identical'])
        with patch.object(pilot.subprocess, 'check_output', return_value=''):
            with self.assertRaisesRegex(ValueError, 'incomplete'):
                pilot.workload_identity(provenance, Path('.'))

    def test_cannot_contaminate_gate_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaisesRegex(ValueError, 'separate'):
                pilot.run(root / 'build-provenance.json', root / 'pilot', root)


if __name__ == '__main__':
    unittest.main()
