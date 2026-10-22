import importlib.util
from pathlib import Path
import unittest
spec = importlib.util.spec_from_file_location('heads', Path(__file__).with_name('punctuation-heads.py'))
heads = importlib.util.module_from_spec(spec)
spec.loader.exec_module(heads)

class AlignmentTests(unittest.TestCase):
    def test_deletion_does_not_shift_later_punctuation(self):
        result = heads.score_pair('Один лишний, два. Три!', 'Один два. Три!')
        self.assertEqual(result['punctuation'], {'tp': 2, 'fp': 0, 'fn': 1})
        self.assertEqual(result['errors'], 1)

    def test_conditional_metric_excludes_deleted_anchors(self):
        result = heads.score_pair('Один лишний, два. Три!', 'Один два. Три!')
        self.assertEqual(result['matched_punctuation'], {'tp': 2, 'fp': 0, 'fn': 0})

    def test_substitution_keeps_sentence_boundary(self):
        result = heads.score_pair('Привет, мир.', 'Привет, друг.')
        self.assertEqual(result['punctuation'], {'tp': 2, 'fp': 0, 'fn': 0})
        self.assertEqual(result['errors'], 1)

    def test_duplicate_punctuation_and_insertions_are_false_positives(self):
        result = heads.score_pair('Привет, мир.', 'Привет,, новый! мир.')
        self.assertEqual(result['punctuation'], {'tp': 2, 'fp': 2, 'fn': 0})

    def test_casing_survives_deleted_word(self):
        result = heads.score_pair('Это было. Москва здесь.', 'Это. Москва здесь.')
        self.assertEqual(result['casing'], {'tp': 2, 'fp': 0, 'fn': 0})

    def test_case_and_yo_are_not_lexical_errors(self):
        self.assertEqual(heads.score_pair('Ёлка здесь.', 'елка здесь.')['errors'], 0)

    def test_inserted_initial_quote_is_scored(self):
        self.assertEqual(heads.score_pair('Привет.', '«Привет.»')['punctuation'], {'tp': 1, 'fp': 2, 'fn': 0})

    def test_incomplete_or_duplicate_reports_are_rejected(self):
        cases = [{'id': 'a', 'reference': 'Привет.'}]
        for rows in [[], [{'id': 'b', 'text': 'Привет.'}],
                     [{'id': 'a', 'text': 'Привет.'}] * 2]:
            with self.assertRaises(ValueError):
                heads.score(cases, rows)

if __name__ == '__main__':
    unittest.main()
