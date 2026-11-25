import copy
from pathlib import Path
import tempfile
import unittest

import objc_cached_load_pilot as pilot


def fixture():
    content = {'.': {'directory': True}, 'model': {'sha256': 'a' * 64}}
    snapshots = {f'{group}_{arm}': {'content': copy.deepcopy(content), 'identity': {'.': [index, 0, 0]}}
                 for index, (group, arm) in enumerate(
                     (group, arm) for group in pilot.GROUPS for arm in pilot.ARMS)}
    report = {'binary_sha256': {'ab_base': 'b' * 64, 'ab_candidate': 'c' * 64,
                                'aa_base': 'b' * 64, 'aa_candidate': 'b' * 64},
              'cache_before': snapshots, 'cache_after': copy.deepcopy(snapshots), 'samples': []}
    for phase, index, group, arm in pilot.schedule():
        data = ({'load_seconds': [1.0] * (pilot.REPEATED_LOADS + 1)}
                if phase in pilot.MODES else {'key_seconds': 0.1, 'meta_seconds': 0.001})
        report['samples'].append({'phase': phase, 'round': index, 'group': group, 'arm': arm,
                                  'data': data, 'binary_sha256': report['binary_sha256'][f'{group}_{arm}']})
    return report


class ProtocolTests(unittest.TestCase):
    def test_fixed_schedule_alternates_and_interleaves(self):
        rows = list(pilot.schedule())
        self.assertEqual(len(rows), 84)
        self.assertEqual(rows[:8], [
            ('cpu', 0, 'ab', 'base'), ('cpu', 0, 'ab', 'candidate'),
            ('cpu', 0, 'aa', 'base'), ('cpu', 0, 'aa', 'candidate'),
            ('cpu', 1, 'aa', 'candidate'), ('cpu', 1, 'aa', 'base'),
            ('cpu', 1, 'ab', 'candidate'), ('cpu', 1, 'ab', 'base'),
        ])
        self.assertTrue(all(row[0] == 'cpu' for row in rows[:28]))
        self.assertTrue(all(row[0] == 'cpu_and_ne' for row in rows[28:56]))
        self.assertTrue(all(row[0] == 'key' for row in rows[56:]))

    def test_identical(self):
        summary = pilot.validate(fixture())
        self.assertTrue(summary['ab_cache_content_equal'])
        self.assertEqual(set(summary['qualification'].values()), {'within_limit'})

    def test_known_slowdown_detected_without_hiding_first_load(self):
        r = fixture()
        for row in r['samples']:
            if (row['phase'], row['group'], row['arm']) == ('cpu', 'ab', 'candidate'):
                row['data']['load_seconds'] = [1.2] + [1.0] * pilot.REPEATED_LOADS
            if (row['phase'], row['group'], row['arm']) == ('cpu_and_ne', 'ab', 'candidate'):
                row['data']['load_seconds'] = [1.0] + [1.10] * pilot.REPEATED_LOADS
        q = pilot.validate(r)['qualification']
        self.assertEqual(q['cpu/first'], 'regression')
        self.assertEqual(q['cpu/repeated_median'], 'within_limit')
        self.assertEqual(q['cpu_and_ne/first'], 'within_limit')
        self.assertEqual(q['cpu_and_ne/repeated_median'], 'regression')

    def test_control_unstable_in_both_directions(self):
        for ratio in (0.90, 1.10):
            r = fixture()
            for row in r['samples']:
                if (row['phase'], row['group'], row['arm']) == ('cpu', 'aa', 'candidate'):
                    row['data']['load_seconds'] = [ratio] * (pilot.REPEATED_LOADS + 1)
            with self.subTest(ratio=ratio):
                q = pilot.validate(r)['qualification']
                self.assertEqual(q['cpu/first'], 'unstable_control')
                self.assertEqual(q['cpu/repeated_median'], 'unstable_control')
                self.assertEqual(q['cpu_and_ne/first'], 'within_limit')

    def test_single_noisy_control_cannot_be_hidden_by_median(self):
        r = fixture()
        row = next(row for row in r['samples'] if
                   (row['phase'], row['group'], row['arm']) == ('cpu', 'aa', 'candidate'))
        row['data']['load_seconds'] = [1.1] * (pilot.REPEATED_LOADS + 1)
        result = pilot.validate(r)
        self.assertEqual(result['metrics']['aa']['cpu/first']['median_ratio'], 1)
        self.assertEqual(result['qualification']['cpu/first'], 'unstable_control')
        self.assertEqual(result['control_pairs_within_limit']['cpu/first'], 6)

    def test_missing_or_selected_round_fails(self):
        r = fixture()
        r['samples'].pop()
        with self.assertRaises(ValueError):
            pilot.validate(r)
        r = fixture()
        r['samples'][0], r['samples'][1] = r['samples'][1], r['samples'][0]
        with self.assertRaises(ValueError):
            pilot.validate(r)
        r = fixture()
        r['samples'][0]['data']['load_seconds'].pop()
        with self.assertRaises(ValueError):
            pilot.validate(r)

    def test_invalid_timing_fails(self):
        for value in (0, -1, float('nan'), float('inf'), True):
            r = fixture()
            r['samples'][0]['data']['load_seconds'][0] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                pilot.validate(r)

    def test_wrong_control_executable_fails(self):
        r = fixture()
        r['binary_sha256']['aa_candidate'] = 'a' * 64
        with self.assertRaises(ValueError):
            pilot.validate(r)
        r = fixture()
        r['samples'][0]['binary_sha256'] = 'a' * 64
        with self.assertRaises(ValueError):
            pilot.validate(r)

    def test_control_cache_mismatch_or_mutation_fails(self):
        r = fixture()
        for key in ('cache_before', 'cache_after'):
            r[key]['aa_candidate']['content']['model']['sha256'] = 'd' * 64
        with self.assertRaises(ValueError):
            pilot.validate(r)
        r = fixture()
        r['cache_after']['aa_candidate']['identity']['.'][0] = 7
        with self.assertRaises(ValueError):
            pilot.validate(r)

    def test_different_ab_caches_reported(self):
        r = fixture()
        for key in ('cache_before', 'cache_after'):
            r[key]['ab_candidate']['content']['model']['sha256'] = 'd' * 64
        summary = pilot.validate(r)
        self.assertFalse(summary['ab_cache_content_equal'])
        self.assertEqual(set(summary['qualification'].values()), {'different_ab_caches'})


class CacheCopyTests(unittest.TestCase):
    def test_copy_matches_byte_content_and_detects_replacement(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / 'src'
            target = Path(tmp) / 'dst'
            source.mkdir()
            (source / 'model').write_bytes(b'model')
            (source / 'meta').write_text('source key')
            (source / 'link').symlink_to('model')
            pilot.copy_cache(source, target)
            before = pilot.snapshot(target)
            self.assertEqual(before['content'], pilot.snapshot(source)['content'])
            self.assertNotEqual(before['identity']['.'], pilot.snapshot(source)['identity']['.'])
            (target / 'replacement').write_bytes(b'model')
            (target / 'replacement').replace(target / 'model')
            self.assertEqual(before['content'], pilot.snapshot(target)['content'])
            self.assertNotEqual(before['identity'], pilot.snapshot(target)['identity'])

    def test_external_symlink_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / 'src'
            source.mkdir()
            (source / 'link').symlink_to('../outside')
            with self.assertRaises(ValueError):
                pilot.snapshot(source)


if __name__ == '__main__':
    unittest.main()
