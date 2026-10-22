import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location('punctuation_compare', Path(__file__).with_name('compare-punctuation.py'))
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class ScoringTests(unittest.TestCase):
    def test_inserted_punctuation_does_not_shift_later_matches(self):
        ref = module.events('Привет мир. Да!')
        hyp = module.events('Привет, мир. Да!')
        self.assertEqual(ref[0], hyp[0])
        self.assertEqual(module.score_counts(ref[1], hyp[1]), {'tp': 2, 'fp': 1, 'fn': 0})
        self.assertEqual(ref[2], hyp[2])

    def test_text_loss_is_not_silently_scored_as_good(self):
        result = module.score([{'id': 'a', 'reference': 'Привет мир!'}], {'rows': [{'id': 'a', 'text': 'Привет!'}]})
        self.assertEqual(result['lexical_failures'], ['a'])
        self.assertEqual(result['punctuation']['f1'], 0)

    def test_duplicate_or_missing_output_rejected(self):
        cases = [{'id': 'a', 'reference': 'Да.'}]
        for rows in [[], [{'id': 'a', 'text': 'Да.'}] * 2]:
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                module.score(cases, {'rows': rows})

    def test_normalization_preserves_numeric_decimal_and_hyphen(self):
        self.assertEqual(module.asr_text('Цена — 3.14, Иван-Петров!'), 'цена 3.14 иван-петров')


if __name__ == '__main__':
    unittest.main()
