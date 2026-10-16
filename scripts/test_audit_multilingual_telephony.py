#!/usr/bin/env python3
"""Synthetic paired error-accounting checks, independent of models and corpora."""
import copy
import unittest

from audit_multilingual_telephony import compare, verify_scores
from benchmark_multilingual_public import score


def row(identifier, reference, hypothesis, status='ok'):
    return dict(id=identifier, reference=reference, hypothesis=hypothesis, status=status,
                scores=score(reference, hypothesis))


class PairedAuditTests(unittest.TestCase):
    def test_digit_subsets_failures_and_alignment_deltas(self):
        baseline = [row('a', 'one two', 'one two'), row('b', 'hello 2', 'hello'),
                    row('c', 'қазақ тілі', '')]
        telephone = [row('a', 'one two', '', 'failed'), row('b', 'hello 2', 'hello 2'),
                     row('c', 'қазақ тілі', 'қазақ тілі')]
        verify_scores(baseline)
        verify_scores(telephone)
        full, digit_free, digit_bearing = compare(baseline, telephone)
        self.assertEqual((full['improved'], full['worsened'], full['unchanged']), (2, 1, 0))
        self.assertEqual((full['newly_empty'], full['recovered_empty']), (1, 1))
        self.assertEqual(full['delta']['word_errors'], -1)
        self.assertEqual(full['telephone']['failed_requests'], 1)
        self.assertEqual((digit_free['original']['n'], digit_bearing['original']['n']), (2, 1))
        self.assertEqual(digit_free['delta']['word_errors'], 0)
        self.assertEqual(digit_bearing['delta']['word_errors'], -1)

    def test_reject_corrupted_counts_and_failed_hypothesis(self):
        correct = row('a', 'one two', 'one')
        for key in ('word_errors', 'char_errors', 'substitutions', 'deletions', 'insertions'):
            corrupted = copy.deepcopy(correct)
            corrupted['scores'][key] += 1
            with self.assertRaises(ValueError):
                verify_scores([corrupted])
        with self.assertRaises(ValueError):
            verify_scores([row('a', 'one', 'one', 'failed')])


if __name__ == '__main__':
    unittest.main()
