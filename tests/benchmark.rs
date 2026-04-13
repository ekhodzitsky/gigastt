//! WER benchmark: transcribes Golos test fixtures and reports Word Error Rate.
//!
//! Outputs JSON to stdout for the autoresearch evaluator.
//! Harness is disabled (`harness = false` in Cargo.toml) so this runs as a binary.

use serde::Deserialize;
use std::path::{Path, PathBuf};

fn home_dir() -> Option<PathBuf> {
    #[cfg(unix)]
    { std::env::var_os("HOME").map(PathBuf::from) }
    #[cfg(windows)]
    { std::env::var_os("USERPROFILE").map(PathBuf::from) }
}

#[derive(Deserialize)]
struct Sample {
    filename: String,
    reference: String,
}

// --- Russian number-to-words tables ---

const ONES: &[&str] = &[
    "", "один", "два", "три", "четыре", "пять",
    "шесть", "семь", "восемь", "девять",
];

const TEENS: &[&str] = &[
    "десять", "одиннадцать", "двенадцать", "тринадцать", "четырнадцать",
    "пятнадцать", "шестнадцать", "семнадцать", "восемнадцать", "девятнадцать",
];

const TENS: &[&str] = &[
    "", "", "двадцать", "тридцать", "сорок", "пятьдесят",
    "шестьдесят", "семьдесят", "восемьдесят", "девяносто",
];

const HUNDREDS: &[&str] = &[
    "", "сто", "двести", "триста", "четыреста", "пятьсот",
    "шестьсот", "семьсот", "восемьсот", "девятьсот",
];

/// Convert a cardinal number (0–999_999) to Russian words.
/// Numbers above 999_999 are returned as digit strings.
fn number_to_words(n: u64) -> String {
    if n == 0 {
        return "ноль".to_string();
    }
    if n > 999_999 {
        return n.to_string();
    }

    let mut parts: Vec<&str> = Vec::new();
    let mut rem = n;

    // Thousands (1_000–999_000)
    if rem >= 1000 {
        let thousands = (rem / 1000) as usize;
        rem %= 1000;

        if thousands >= 100 {
            parts.push(HUNDREDS[thousands / 100]);
        }
        let t = thousands % 100;
        if t >= 20 {
            parts.push(TENS[t / 10]);
            match t % 10 {
                1 => parts.push("одна"),   // feminine for тысяча
                2 => parts.push("две"),    // feminine for тысяча
                o @ 3..=9 => parts.push(ONES[o]),
                _ => {}
            }
        } else if t >= 10 {
            parts.push(TEENS[t - 10]);
        } else if t > 0 {
            match t {
                1 => parts.push("одна"),
                2 => parts.push("две"),
                _ => parts.push(ONES[t]),
            }
        }

        let last_two = thousands % 100;
        let last_one = thousands % 10;
        if (11..=19).contains(&last_two) {
            parts.push("тысяч");
        } else {
            match last_one {
                1 => parts.push("тысяча"),
                2..=4 => parts.push("тысячи"),
                _ => parts.push("тысяч"),
            }
        }
    }

    // Hundreds + tens + ones (0–999)
    let r = rem as usize;
    if r >= 100 {
        parts.push(HUNDREDS[r / 100]);
    }
    let t = r % 100;
    if t >= 20 {
        parts.push(TENS[t / 10]);
        if !t.is_multiple_of(10) {
            parts.push(ONES[t % 10]);
        }
    } else if t >= 10 {
        parts.push(TEENS[t - 10]);
    } else if t > 0 {
        parts.push(ONES[t]);
    }

    parts.join(" ")
}

/// Try to convert a number to masculine ordinal form (-й suffix), 1–20 only.
fn try_ordinal_masculine(n: u64) -> Option<&'static str> {
    match n {
        1 => Some("первый"),
        2 => Some("второй"),
        3 => Some("третий"),
        4 => Some("четвертый"),
        5 => Some("пятый"),
        6 => Some("шестой"),
        7 => Some("седьмой"),
        8 => Some("восьмой"),
        9 => Some("девятый"),
        10 => Some("десятый"),
        11 => Some("одиннадцатый"),
        12 => Some("двенадцатый"),
        13 => Some("тринадцатый"),
        14 => Some("четырнадцатый"),
        15 => Some("пятнадцатый"),
        16 => Some("шестнадцатый"),
        17 => Some("семнадцатый"),
        18 => Some("восемнадцатый"),
        19 => Some("девятнадцатый"),
        20 => Some("двадцатый"),
        _ => None,
    }
}

/// Merge consecutive digit-only tokens when the second has exactly 3 digits
/// (Russian thousands separator: "60 000" → "60000").
fn merge_digit_groups(words: &[String]) -> Vec<String> {
    let mut result = Vec::new();
    let mut i = 0;
    while i < words.len() {
        if words[i].chars().all(|c| c.is_ascii_digit()) && !words[i].is_empty() {
            let mut merged = words[i].clone();
            while i + 1 < words.len()
                && words[i + 1].len() == 3
                && words[i + 1].chars().all(|c| c.is_ascii_digit())
            {
                i += 1;
                merged.push_str(&words[i]);
            }
            result.push(merged);
        } else {
            result.push(words[i].clone());
        }
        i += 1;
    }
    result
}

/// Resolve ordinal patterns: digit token followed by a single-char suffix like "й".
fn resolve_ordinals(words: &[String]) -> Vec<String> {
    let mut result = Vec::new();
    let mut i = 0;
    while i < words.len() {
        if i + 1 < words.len() && words[i + 1] == "й"
            && let Ok(n) = words[i].parse::<u64>()
            && let Some(ordinal) = try_ordinal_masculine(n)
        {
            result.push(ordinal.to_string());
            i += 2;
            continue;
        }
        result.push(words[i].clone());
        i += 1;
    }
    result
}

/// Convert remaining pure-digit tokens to Russian cardinal words.
fn convert_cardinal_numbers(words: &[String]) -> Vec<String> {
    let mut result = Vec::new();
    for w in words {
        if w.chars().all(|c| c.is_ascii_digit()) && !w.is_empty()
            && let Ok(n) = w.parse::<u64>()
        {
            for part in number_to_words(n).split_whitespace() {
                result.push(part.to_string());
            }
            continue;
        }
        result.push(w.clone());
    }
    result
}

/// Normalize text for WER comparison:
/// lowercase → ё→е → hyphens as spaces → strip punctuation → merge digit groups →
/// resolve ordinals → convert cardinal numbers → split into words.
fn normalize_for_wer(text: &str) -> Vec<String> {
    let text = text.to_lowercase();
    let text = text.replace('ё', "е");
    let text = text.replace('-', " ");

    let text: String = text
        .chars()
        .filter(|c| c.is_alphanumeric() || c.is_whitespace())
        .collect();

    let words: Vec<String> = text.split_whitespace().map(String::from).collect();
    let words = merge_digit_groups(&words);
    let words = resolve_ordinals(&words);
    convert_cardinal_numbers(&words)
}

/// Word-level edit distance (Levenshtein) between reference and hypothesis.
fn word_edit_distance(reference: &[String], hypothesis: &[String]) -> usize {
    let m = reference.len();
    let n = hypothesis.len();
    let mut prev: Vec<usize> = (0..=n).collect();
    let mut curr = vec![0usize; n + 1];

    for i in 1..=m {
        curr[0] = i;
        for j in 1..=n {
            if reference[i - 1] == hypothesis[j - 1] {
                curr[j] = prev[j - 1];
            } else {
                curr[j] = 1 + prev[j - 1].min(prev[j]).min(curr[j - 1]);
            }
        }
        std::mem::swap(&mut prev, &mut curr);
    }
    prev[n]
}

fn main() {
    let fixture_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures");
    let manifest_path = fixture_dir.join("manifest.json");

    // Check model availability — skip gracefully if missing
    let model_dir = home_dir()
        .map(|h| h.join(".gigastt").join("models"))
        .expect("HOME not set");

    if !model_dir.join("v3_e2e_rnnt_encoder.onnx").exists() {
        println!(
            r#"{{"pass": true, "score": null, "skipped": true, "reason": "model not found"}}"#
        );
        return;
    }

    let manifest: Vec<Sample> = serde_json::from_str(
        &std::fs::read_to_string(&manifest_path).expect("Failed to read manifest"),
    )
    .expect("Failed to parse manifest");

    let model_dir_str = model_dir.to_string_lossy();
    let engine =
        gigastt::inference::Engine::load(&model_dir_str).expect("Failed to load engine");

    let rt = tokio::runtime::Runtime::new().expect("Failed to create tokio runtime");
    let mut triplet = rt.block_on(engine.pool.checkout());

    let mut total_ref_words = 0usize;
    let mut total_errors = 0usize;
    let mut details = Vec::new();

    for sample in &manifest {
        let wav_path = fixture_dir.join(&sample.filename);
        let hypothesis = engine
            .transcribe_file(wav_path.to_str().unwrap(), &mut triplet)
            .expect("Transcription failed");

        let ref_words = normalize_for_wer(&sample.reference);
        let hyp_words = normalize_for_wer(&hypothesis.text);

        let errors = word_edit_distance(&ref_words, &hyp_words);
        let sample_wer = if ref_words.is_empty() {
            0.0
        } else {
            errors as f64 / ref_words.len() as f64 * 100.0
        };

        total_ref_words += ref_words.len();
        total_errors += errors;

        eprintln!(
            "  [WER {:5.1}%] {} | ref: \"{}\" | hyp: \"{}\"",
            sample_wer, sample.filename, sample.reference, hypothesis.text
        );

        details.push(serde_json::json!({
            "file": sample.filename,
            "reference": sample.reference,
            "hypothesis": hypothesis.text,
            "ref_norm": ref_words.join(" "),
            "hyp_norm": hyp_words.join(" "),
            "wer": (sample_wer * 10.0).round() / 10.0,
        }));
    }

    let wer = if total_ref_words > 0 {
        total_errors as f64 / total_ref_words as f64 * 100.0
    } else {
        0.0
    };
    let score = (100.0 - wer).max(0.0);
    let score_rounded = (score * 10.0).round() / 10.0;
    let wer_rounded = (wer * 10.0).round() / 10.0;

    eprintln!(
        "\n  WER: {:.1}% ({} errors / {} words)  Score: {:.1}",
        wer, total_errors, total_ref_words, score
    );

    let output = serde_json::json!({
        "pass": true,
        "score": score_rounded,
        "wer": wer_rounded,
        "total_words": total_ref_words,
        "total_errors": total_errors,
        "samples": manifest.len(),
        "details": details,
    });

    println!("{}", serde_json::to_string(&output).unwrap());
}
