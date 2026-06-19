"""Unit tests for benchmark/common.py WER normalization and manifest loading."""

from common import (
    compute_wer,
    compute_wer_naive,
    load_manifest,
    normalize_for_wer,
    normalize_for_wer_naive,
    word_edit_distance,
)


def _wer_info(reference: str, hypothesis: str) -> tuple[float, int, int, list[str], list[str]]:
    ref_words = normalize_for_wer(reference)
    hyp_words = normalize_for_wer(hypothesis)
    errors = word_edit_distance(ref_words, hyp_words)
    ref_count = len(ref_words)
    wer = (errors / ref_count * 100.0) if ref_count > 0 else 0.0
    return wer, errors, ref_count, ref_words, hyp_words


def test_phone_number_word_and_symbol_form():
    ref = (
        "положи деньги на номер мобильного счета плюс семь девятьсот девятнадцать "
        "триста тридцать пять восемьдесят девять тридцать один"
    )
    hyp = "Положи деньги на номер мобильного счёта +79193358931."
    wer, errors, ref_count, ref_words, hyp_words = _wer_info(ref, hyp)

    assert wer == 0.0
    assert errors == 0
    assert ref_words == hyp_words
    assert "79193358931" in ref_words
    assert "79193358931" in hyp_words
    # Non-digit words should still match.
    assert set(ref_words) == set(hyp_words)


def test_grouped_digits_and_minus():
    # The thousands scale must survive: 'минус тысяча девятьсот семьдесят два'
    # normalizes to 1972 (not the mathematically wrong 972), so it matches the
    # correct digit form '-1972' with zero edit distance (task 12).
    ref = "джой две тысячи двадцать минус тысяча девятьсот семьдесят два"
    hyp = "2020 -1972"
    wer, errors, ref_count, ref_words, hyp_words = _wer_info(ref, hyp)

    assert wer == 0.0
    assert errors == 0
    assert ref_words == hyp_words
    assert ref_words == ["2020", "1972"]


def test_minus_thousand_preserves_scale():
    # Regression (task 12): the removed _drop_bare_thousand_after_minus special
    # case used to swallow the thousands digit, aligning the spoken subtrahend
    # with the WRONG '-972'. The scale must be preserved and match '-1972'.
    assert normalize_for_wer("минус тысяча девятьсот семьдесят два") == ["1972"]
    assert normalize_for_wer("-1972") == ["1972"]


def test_date_and_currency_low_edit_distance():
    ref = (
        "сколько стоит двадцать одна американских долларов перевести в гуарани "
        "курс тринадцатое июня двадцатый год"
    )
    hyp = "Сколько стоит $21 перевести в гуарани, курс 13 июня 2020 года?"
    _, errors, _, ref_words, hyp_words = _wer_info(ref, hyp)

    assert "13" in ref_words
    assert "21" in ref_words
    assert "13" in hyp_words
    assert "2020" in hyp_words
    assert "21" in hyp_words
    assert errors <= 4


def test_both_sides_words_phone_number():
    ref = hyp = "ноль шестьсот шесть девятьсот семьдесят два двадцать один одиннадцать"
    wer, errors, _, ref_words, hyp_words = _wer_info(ref, hyp)

    assert wer == 0.0
    assert errors == 0
    assert ref_words == hyp_words
    # Accept any single merged digit token (the exact grouping depends on
    # how the speaker chunked the phone number).
    assert len(ref_words) == 1
    assert ref_words[0].isdigit()


def test_load_manifest_filters_empty_refs():
    result = load_manifest(dataset="golos_crowd_1k")
    assert isinstance(result, dict)
    assert result["skipped_empty_refs"] == 8
    samples = result["samples"]
    assert len(samples) == 1000 - 8
    for s in samples:
        assert s["reference"].strip() != ""


def test_compound_number_one_hundred_twenty_three():
    assert normalize_for_wer("сто двадцать три") == ["123"]


def test_compound_number_two_thousand_twenty():
    assert normalize_for_wer("две тысячи двадцать") == ["2020"]


def test_compound_number_two_thousand_twenty_one():
    assert normalize_for_wer("две тысячи двадцать один") == ["2021"]


def test_ordinal_twenty_first():
    assert normalize_for_wer("двадцать первый") == ["21"]


def test_ordinal_hundred_twenty_first():
    assert normalize_for_wer("сто двадцать первый") == ["121"]


def test_million_one():
    assert normalize_for_wer("один миллион") == ["1000000"]


def test_thousands_five():
    assert normalize_for_wer("пять тысяч") == ["5000"]


def test_thousands_one_female():
    assert normalize_for_wer("одна тысяча") == ["1000"]


def test_ordinal_thirteenth():
    assert normalize_for_wer("тринадцатое") == ["13"]


def test_digit_groups_merge_when_all_short():
    assert normalize_for_wer("+7 919 335 89 31") == ["79193358931"]


def test_digit_groups_do_not_merge_when_over_three():
    assert normalize_for_wer("2020 972") == ["2020", "972"]


def test_percent_and_ruble_signs_removed():
    ref = "пять процентов и сто рублей"
    hyp = "5% и 100₽"
    _, errors, _, ref_words, hyp_words = _wer_info(ref, hyp)
    assert errors == 0
    assert ref_words == hyp_words


def test_wer_zero_for_identical_transcriptions():
    wer, errors, ref_count, _, _ = _wer_info(
        "привет мир", "привет мир"
    )
    assert wer == 0.0
    assert errors == 0
    assert ref_count == 2


def test_accusative_thousand():
    assert normalize_for_wer("сбер переведи тысячу андрею") == [
        "сбер", "переведи", "1000", "андрею"
    ]


def test_ordinal_suffix_dropped():
    assert normalize_for_wer("36-я серия") == ["36", "серия"]


def test_latin_number_abbreviation_no():
    assert normalize_for_wer("No 755") == ["755"]


def test_percent_keeps_adjacent_numbers_separate():
    assert normalize_for_wer("15% 180") == ["15", "180"]


def test_chunked_thousands_merge():
    assert normalize_for_wer("3 000 ₽") == ["3000"]


# --- Verbatim ("naive") normalization: lowercase + ё→е + strip non-word
# characters + split, with NO words-to-digits ITN, digit merging, ordinal
# resolution, or anglicism mapping. The gap between naive and ITN WER isolates
# how much of the apparent error is writing convention vs acoustics. ---


def test_naive_lowercases_and_strips_punctuation():
    assert normalize_for_wer_naive("Привет, Мир!") == ["привет", "мир"]


def test_naive_folds_yo_to_ye():
    assert normalize_for_wer_naive("счёт") == ["счет"]


def test_naive_keeps_digits_and_words_apart_no_itn():
    # Naive does NOT convert number words to digits or strip currency/percent
    # words, so the digit and word forms stay incomparable.
    assert normalize_for_wer_naive("пять процентов") == ["пять", "процентов"]
    assert normalize_for_wer_naive("5%") == ["5"]


def test_naive_does_not_merge_digit_groups():
    # ITN merges short groups ("79193358931"); naive leaves them split.
    assert normalize_for_wer_naive("7 919 335") == ["7", "919", "335"]


def test_naive_does_not_map_anglicisms():
    assert normalize_for_wer_naive("youtube") == ["youtube"]


def test_naive_drops_chars_outside_ascii_cyrillic_class():
    # The strip class is exactly [a-zа-я0-9\s]; accented Latin and non-ASCII
    # digits are dropped (here "ï" is removed, leaving "nave"). This pins
    # cross-harness parity with the Rust benchmark's normalize_for_wer_naive,
    # which uses the same character class.
    assert normalize_for_wer_naive("naïve") == ["nave"]


def test_naive_wer_penalizes_convention_where_itn_forgives():
    # The ITN pipeline collapses "пять процентов" to the single token ["5"]
    # (пять→5, "процентов" dropped as an empty token) and "5%" to ["5"], so it
    # scores them as a perfect match. The verbatim pass keeps both words and the
    # digit apart, counting the convention difference as error. That gap is
    # exactly the "writing convention" share of the WER.
    ref = "пять процентов"
    hyp = "5%"
    itn_wer, itn_err, _ = compute_wer(ref, hyp)
    naive_wer, naive_err, naive_count = compute_wer_naive(ref, hyp)

    assert itn_err == 0
    assert itn_wer == 0.0
    assert naive_err > 0
    assert naive_wer > 0.0
    assert naive_count == 2


def test_naive_wer_equals_itn_on_plain_words():
    # With no digits, Latin tokens, or punctuation, the two passes agree.
    ref = hyp = "привет дорогой мир"
    itn_wer, itn_err, itn_count = compute_wer(ref, hyp)
    naive_wer, naive_err, naive_count = compute_wer_naive(ref, hyp)

    assert (itn_wer, itn_err, itn_count) == (0.0, 0, 3)
    assert (naive_wer, naive_err, naive_count) == (0.0, 0, 3)


def test_naive_wer_empty_reference_is_zero():
    wer, errors, ref_count = compute_wer_naive("", "что-то")
    assert wer == 0.0
    assert ref_count == 0
