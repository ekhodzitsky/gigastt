//! Inverse text normalization (ITN): Russian number-words → Arabic digits.
//!
//! The plain RNN-T recognition head ([`ModelVariant::Rnnt`](crate::model::ModelVariant::Rnnt))
//! spells numbers out as words (e.g. `"шестьдесят тысяч"`). This module turns
//! those number-word runs into digit tokens (`"60000"`) as an *optional*
//! post-processing pass. It runs **before** punctuation restoration so the
//! punctuator sees and cases the already-digitized text.
//!
//! [`apply_itn`] is the public entry point: whitespace-tokenize the input, run
//! the words→numbers state machine, merge adjacent short digit groups, then
//! rejoin with single spaces. Non-number tokens (and any punctuation attached
//! to them) pass through unchanged.
//!
//! This is a verbatim Rust port of the reference implementation in
//! `benchmark/common.py` (`_NUMBER_WORDS` / `_words_to_numbers` /
//! `_merge_digit_groups`), kept symmetric with the WER benchmark so the same
//! number normalization applies online and offline.

use std::collections::HashMap;
use std::sync::OnceLock;

const UNIT: i64 = 1;
const TEEN: i64 = 10;
const TEN: i64 = 10;
const HUNDRED: i64 = 100;
const THOUSAND: i64 = 1000;
const MILLION: i64 = 1_000_000;

/// Lazily-built `word -> (value, scale)` lookup, mirroring `_NUMBER_WORDS` in
/// the Python reference. Keys are lowercase so the lookup is case-insensitive
/// (it works on already-capitalized text too).
fn number_words() -> &'static HashMap<&'static str, (i64, i64)> {
    static TABLE: OnceLock<HashMap<&'static str, (i64, i64)>> = OnceLock::new();
    TABLE.get_or_init(|| {
        let mut m: HashMap<&'static str, (i64, i64)> = HashMap::new();
        let mut add = |value: i64, scale: i64, forms: &[&'static str]| {
            for &form in forms {
                m.insert(form, (value, scale));
            }
        };

        // Cardinals 0-9 with common case/gender forms.
        add(0, UNIT, &["ноль", "ноля", "нолю", "нолем", "нолём", "ноле"]);
        add(
            1,
            UNIT,
            &[
                "один",
                "одна",
                "одно",
                "одного",
                "одной",
                "одному",
                "одном",
                "одним",
            ],
        );
        add(2, UNIT, &["два", "две", "двух", "двум", "двумя"]);
        add(3, UNIT, &["три", "трех", "трёх", "трем", "трём", "тремя"]);
        add(
            4,
            UNIT,
            &[
                "четыре",
                "четырёх",
                "четырех",
                "четырём",
                "четырем",
                "четырьмя",
            ],
        );
        add(5, UNIT, &["пять", "пяти", "пятью"]);
        add(6, UNIT, &["шесть", "шести", "шестью"]);
        add(7, UNIT, &["семь", "семи", "семью"]);
        add(8, UNIT, &["восемь", "восьми", "восьмью"]);
        add(9, UNIT, &["девять", "девяти", "девятью"]);

        // Teens 10-19.
        add(10, TEEN, &["десять", "десяти", "десятью"]);
        add(11, TEEN, &["одиннадцать", "одиннадцати"]);
        add(12, TEEN, &["двенадцать", "двенадцати"]);
        add(13, TEEN, &["тринадцать", "тринадцати"]);
        add(14, TEEN, &["четырнадцать", "четырнадцати"]);
        add(15, TEEN, &["пятнадцать", "пятнадцати"]);
        add(16, TEEN, &["шестнадцать", "шестнадцати"]);
        add(17, TEEN, &["семнадцать", "семнадцати"]);
        add(18, TEEN, &["восемнадцать", "восемнадцати"]);
        add(19, TEEN, &["девятнадцать", "девятнадцати"]);

        // Tens 20-90.
        add(20, TEN, &["двадцать", "двадцати"]);
        add(30, TEN, &["тридцать", "тридцати"]);
        add(40, TEN, &["сорок", "сорока"]);
        add(50, TEN, &["пятьдесят", "пятидесяти"]);
        add(60, TEN, &["шестьдесят", "шестидесяти"]);
        add(70, TEN, &["семьдесят", "семидесяти"]);
        add(80, TEN, &["восемьдесят", "восьмидесяти"]);
        add(90, TEN, &["девяносто", "девяноста"]);

        // Hundreds 100-900.
        add(100, HUNDRED, &["сто", "ста"]);
        add(200, HUNDRED, &["двести", "двухсот"]);
        add(300, HUNDRED, &["триста", "трехсот", "трёхсот"]);
        add(400, HUNDRED, &["четыреста", "четырёхсот"]);
        add(500, HUNDRED, &["пятьсот", "пятисот"]);
        add(600, HUNDRED, &["шестьсот", "шестисот"]);
        add(700, HUNDRED, &["семьсот", "семисот"]);
        add(800, HUNDRED, &["восемьсот", "восьмисот"]);
        add(900, HUNDRED, &["девятьсот", "девятисот"]);

        // Scale words (all common case forms).
        add(
            1000,
            THOUSAND,
            &[
                "тысяча",
                "тысячи",
                "тысяч",
                "тысяче",
                "тысячу",
                "тысячей",
                "тысячам",
                "тысячами",
                "тысячах",
            ],
        );
        add(
            1_000_000,
            MILLION,
            &[
                "миллион",
                "миллиона",
                "миллионов",
                "миллиону",
                "миллионе",
                "миллионам",
                "миллионами",
                "миллионах",
            ],
        );

        m
    })
}

/// Convert Russian number-word sequences into Arabic digit tokens.
///
/// Compound numbers such as `"две тысячи двадцать"` become a single token
/// `"2020"`, while independent digit groups (e.g. phone-number chunks) are
/// emitted separately based on scale-order jumps. Verbatim port of the
/// reference `_words_to_numbers`.
fn words_to_numbers(tokens: &[String]) -> Vec<String> {
    let table = number_words();
    let mut result: Vec<String> = Vec::with_capacity(tokens.len());

    let mut current: i64 = 0;
    let mut running_total: i64 = 0;
    let mut prev_scale: i64 = 0;
    let mut in_number = false;

    // Local flush closure cannot borrow the mutable state we mutate in the
    // loop, so it is inlined as a macro-free helper via explicit blocks below.
    for token in tokens {
        // Case-insensitive lookup: the rnnt head can be lowercase, but ITN may
        // also run on already-capitalized text.
        let key = token.to_lowercase();
        match table.get(key.as_str()) {
            Some(&(value, scale)) => {
                in_number = true;
                if scale == THOUSAND || scale == MILLION {
                    if current == 0 {
                        current = 1;
                    }
                    running_total += current * scale;
                    current = 0;
                    prev_scale = scale;
                } else if current > 0 && scale >= prev_scale {
                    // Python: flush() then `current = value; in_number = True`.
                    // The flush emits `running_total + current` and resets the
                    // running total; `current`, `in_number`, and `prev_scale`
                    // are immediately re-set below, so only their final values
                    // matter here.
                    result.push((running_total + current).to_string());
                    running_total = 0;
                    current = value;
                    in_number = true;
                    prev_scale = scale;
                } else {
                    current += value;
                    prev_scale = scale;
                }
            }
            None => {
                // flush()
                let total = running_total + current;
                if in_number {
                    result.push(total.to_string());
                }
                current = 0;
                running_total = 0;
                prev_scale = 0;
                in_number = false;

                result.push(token.clone());
            }
        }
    }

    // Final flush().
    let total = running_total + current;
    if in_number {
        result.push(total.to_string());
    }

    result
}

/// True when `s` is non-empty and every char is an ASCII digit, mirroring
/// Python `str.isdigit()` for the digit tokens this pipeline produces.
fn is_digit_token(s: &str) -> bool {
    !s.is_empty() && s.bytes().all(|b| b.is_ascii_digit())
}

/// Merge adjacent digit tokens when every token in the run has length ≤ 3.
/// Verbatim port of the reference `_merge_digit_groups`.
fn merge_digit_groups(tokens: &[String]) -> Vec<String> {
    let mut result: Vec<String> = Vec::with_capacity(tokens.len());
    let n = tokens.len();
    let mut i = 0;
    while i < n {
        if is_digit_token(&tokens[i]) {
            let mut j = i;
            let mut group: Vec<&String> = Vec::new();
            while j < n && is_digit_token(&tokens[j]) {
                group.push(&tokens[j]);
                j += 1;
            }
            if !group.is_empty() && group.iter().all(|t| t.chars().count() <= 3) {
                result.push(group.iter().map(|t| t.as_str()).collect::<String>());
            } else {
                result.extend(group.iter().map(|t| (*t).clone()));
            }
            i = j;
        } else {
            result.push(tokens[i].clone());
            i += 1;
        }
    }
    result
}

/// Apply inverse text normalization: convert Russian number-words to digits.
///
/// Whitespace-tokenizes the input, runs the words→numbers state machine, merges
/// adjacent short digit groups, and rejoins with single spaces. Non-number
/// tokens (and any punctuation attached to them) pass through unchanged.
///
/// Examples: `"двадцать один"` → `"21"`, `"две тысячи двадцать"` → `"2020"`,
/// `"шестьдесят тысяч"` → `"60000"`, `"позвони на шестьдесят"` →
/// `"позвони на 60"`.
pub fn apply_itn(text: &str) -> String {
    let tokens: Vec<String> = text.split_whitespace().map(str::to_string).collect();
    let tokens = words_to_numbers(&tokens);
    let tokens = merge_digit_groups(&tokens);
    tokens.join(" ")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_apply_itn_tens_plus_unit() {
        assert_eq!(apply_itn("двадцать один"), "21");
    }

    #[test]
    fn test_apply_itn_compound_thousand() {
        assert_eq!(apply_itn("две тысячи двадцать"), "2020");
    }

    #[test]
    fn test_apply_itn_sixty_thousand() {
        assert_eq!(apply_itn("шестьдесят тысяч"), "60000");
    }

    #[test]
    fn test_apply_itn_hundred() {
        assert_eq!(apply_itn("сто"), "100");
    }

    #[test]
    fn test_apply_itn_plain_words_unchanged() {
        assert_eq!(apply_itn("привет как дела"), "привет как дела");
    }

    #[test]
    fn test_apply_itn_mixed_words_and_number() {
        assert_eq!(apply_itn("позвони на шестьдесят"), "позвони на 60");
    }

    #[test]
    fn test_apply_itn_case_insensitive() {
        // Capitalized number-words still convert (lowercased for lookup).
        assert_eq!(apply_itn("Двадцать один"), "21");
    }

    #[test]
    fn test_apply_itn_digit_group_merge() {
        // Adjacent short (<=3 char) digit groups merge: "сто двадцать три" is a
        // single compound (123), but two independent ≤3-digit runs join too.
        // "восемь девять пять" → 8 9 5 → all length 1 → merged "895".
        assert_eq!(apply_itn("восемь девять пять"), "895");
    }

    #[test]
    fn test_apply_itn_trailing_unit_folds_into_running_total() {
        // The state machine folds a trailing unit into the running total, so
        // "шестьдесят тысяч пять" → 60000 + 5 = "60005" as ONE token (verified
        // against the Python reference _words_to_numbers).
        assert_eq!(apply_itn("шестьдесят тысяч пять"), "60005");
    }

    #[test]
    fn test_apply_itn_long_digit_run_not_merged() {
        // A scale jump flushes into two digit tokens; the 4-char "1100" blocks
        // the run from merging, so they stay space-separated (verified against
        // the Python reference: ["1100", "200"]).
        assert_eq!(apply_itn("тысяча сто двести"), "1100 200");
    }

    #[test]
    fn test_apply_itn_phrase_with_trailing_words() {
        // The end-to-end pipeline example: number run digitizes, words pass through.
        assert_eq!(
            apply_itn("шестьдесят тысяч тенге сколько будет стоить"),
            "60000 тенге сколько будет стоить"
        );
    }

    #[test]
    fn test_apply_itn_empty_string() {
        assert_eq!(apply_itn(""), "");
    }

    #[test]
    fn test_apply_itn_punctuation_passes_through() {
        // Non-number token with attached punctuation is unchanged.
        assert_eq!(apply_itn("привет, мир"), "привет, мир");
    }
}
