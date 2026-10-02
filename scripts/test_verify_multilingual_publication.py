"""Guard against plausible but corrupted published benchmark snapshots."""
import copy
import unittest

from verify_multilingual_publication import totals, verify_result


class PublicationTests(unittest.TestCase):
    def setUp(self):
        self.sample = dict(id='one', reference='қазақ', sha256='frozen', language='kk', dataset='fixture', split='test')
        row = dict(self.sample, hypothesis='қазак', status='ok', duration_s=1,
                   scores=dict(reference_words=1, word_errors=1, substitutions=1, deletions=0,
                               insertions=0, reference_chars=5, char_errors=1))
        self.manifest = dict(samples=[self.sample])
        self.result = dict(completed=True, n=1, details=[row], summary=totals([row]))

    def test_turkic_letter_error_is_preserved(self):
        self.assertEqual(verify_result(self.result, self.manifest)['wer_pct'], 100)

    def test_forged_scores_even_with_matching_summary_are_rejected(self):
        bad = copy.deepcopy(self.result)
        bad['details'][0]['scores'].update(word_errors=0, substitutions=0, char_errors=0)
        bad['summary'] = totals(bad['details'])
        with self.assertRaises(ValueError):
            verify_result(bad, self.manifest)

    def test_missing_or_replaced_input_is_rejected(self):
        for mutation in ('missing', 'replaced'):
            bad = copy.deepcopy(self.result)
            if mutation == 'missing':
                bad.update(n=0, details=[])
            else:
                bad['details'][0]['reference'] = 'changed'
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                verify_result(bad, self.manifest)


if __name__ == '__main__':
    unittest.main()
