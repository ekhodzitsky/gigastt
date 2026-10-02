"""Meaningful integrity regressions for the independent comparison auditor."""
import unittest
import tempfile
from pathlib import Path
from audit_ready_asr import verify_pairing, read_ledger


class PairingTests(unittest.TestCase):
    def setUp(self):
        self.sample = dict(id='one', reference='word', sha256='abc', path='/audio',
                           language='kk', dataset='human', split='test')
        self.row = dict(self.sample, status='ok', hypothesis='word', error=None)

    def test_partial_prefix_allowed_but_reordering_rejected(self):
        second = dict(self.sample, id='two')
        verify_pairing([self.sample, second], [self.row], complete=False)
        with self.assertRaises(ValueError):
            verify_pairing([second, self.sample], [self.row], complete=False)

    def test_changed_reference_and_path_rejected(self):
        for key in ('reference', 'path', 'sha256'):
            with self.subTest(key=key), self.assertRaises(ValueError):
                verify_pairing([self.sample], [dict(self.row, **{key:'changed'})], complete=True)

    def test_failed_hypothesis_and_missing_final_row_rejected(self):
        with self.assertRaises(ValueError):
            verify_pairing([self.sample], [dict(self.row, status='failed')], complete=True)
        with self.assertRaises(ValueError):
            verify_pairing([self.sample], [], complete=True)

    def test_duplicate_ledger_id_rejected(self):
        with self.assertRaises(ValueError):
            verify_pairing([self.sample, dict(self.sample,id='two')], [self.row,self.row], complete=False)


class LedgerTests(unittest.TestCase):
    def test_torn_tail_is_only_allowed_for_explicit_partial_audit(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'rows.jsonl'
            path.write_bytes(b'{"id":"one"}\n{"id":')
            with self.assertRaises(ValueError):
                read_ledger(path, False)
            rows, torn = read_ledger(path, True)
            self.assertEqual(rows, [{'id': 'one'}])
            self.assertTrue(torn)


if __name__ == '__main__':
    unittest.main()
