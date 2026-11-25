//! WER benchmark: transcribes Golos test fixtures and reports Word Error Rate.
//!
//! Supports an external manifest in `~/.gigastt/benchmarks/golos_wav/manifest.json`.
//! Falls back to the bundled `tests/fixtures/manifest.json` when the external set
//! is missing.
//!
//! Environment variables:
//! - `GIGASTT_BENCHMARK_MAX_SAMPLES` — limit the number of samples (0 = unlimited).
//!
//! Corpus execution requires explicit `--run`. `--run --required` verifies the
//! pinned bundled corpus and current built-in RNNT INT8 model.
//! Outputs measured or explicit invalid/skipped JSON to stdout.
//! Harness is disabled (`harness = false` in Cargo.toml) so this runs as a binary.

#[path = "benchmark/admission.rs"]
mod admission;

use anyhow::{Context, Result, ensure};
use serde::Deserialize;
use std::path::{Path, PathBuf};

const MAX_WER: f64 = 15.0;
const PROGRESS_EVERY: usize = 50;

fn home_dir() -> Option<PathBuf> {
    #[cfg(unix)]
    {
        std::env::var_os("HOME").map(PathBuf::from)
    }
    #[cfg(windows)]
    {
        std::env::var_os("USERPROFILE").map(PathBuf::from)
    }
}

#[derive(Deserialize)]
struct Sample {
    filename: String,
    reference: String,
}

// --- Russian number-to-words tables ---

const ONES: &[&str] = &[
    "",
    "один",
    "два",
    "три",
    "четыре",
    "пять",
    "шесть",
    "семь",
    "восемь",
    "девять",
];

const TEENS: &[&str] = &[
    "десять",
    "одиннадцать",
    "двенадцать",
    "тринадцать",
    "четырнадцать",
    "пятнадцать",
    "шестнадцать",
    "семнадцать",
    "восемнадцать",
    "девятнадцать",
];

const TENS: &[&str] = &[
    "",
    "",
    "двадцать",
    "тридцать",
    "сорок",
    "пятьдесят",
    "шестьдесят",
    "семьдесят",
    "восемьдесят",
    "девяносто",
];

const HUNDREDS: &[&str] = &[
    "",
    "сто",
    "двести",
    "триста",
    "четыреста",
    "пятьсот",
    "шестьсот",
    "семьсот",
    "восемьсот",
    "девятьсот",
];

/// Convert a cardinal number (0–999_999) to Russian words.
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
                1 => parts.push("одна"),
                2 => parts.push("две"),
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
        if i + 1 < words.len()
            && words[i + 1] == "й"
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
        if w.chars().all(|c| c.is_ascii_digit())
            && !w.is_empty()
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

/// Transliterate common English brand names / loanwords to Russian.
fn translit_anglicisms(words: &[String]) -> Vec<String> {
    words
        .iter()
        .map(|w| {
            match w.as_str() {
                "synergy" => "синергия",
                "tv" => "тв",
                "pink" => "пинк",
                "sony" => "сони",
                "samsung" => "самсунг",
                "apple" => "эпл",
                "iphone" => "айфон",
                "google" => "гугл",
                "youtube" => "ютуб",
                "facebook" => "фейсбук",
                "instagram" => "инстаграм",
                "netflix" => "нетфликс",
                "spotify" => "спотифай",
                "whatsapp" => "ватсап",
                "telegram" => "телеграм",
                "vk" => "вк",
                "ok" => "ок",
                "aliexpress" => "алиэкспресс",
                _ => return w.clone(),
            }
            .to_string()
        })
        .collect()
}

/// Normalize text for WER comparison:
/// lowercase → ё→е → hyphens as spaces → strip punctuation → merge digit groups →
/// resolve ordinals → convert cardinal numbers → translit anglicisms → split into words.
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
    let words = convert_cardinal_numbers(&words);
    translit_anglicisms(&words)
}

/// Verbatim ("naive") normalization: lowercase, `ё`→`е`, keep only
/// alphanumeric/whitespace characters, split. Unlike `normalize_for_wer` it does
/// NOT convert dashes to spaces, merge digit groups, resolve ordinals, convert
/// cardinals to words, or transliterate anglicisms. Reporting WER over this pass
/// alongside the normalized WER isolates the writing-convention share of the
/// error (number style, punctuation, transliteration) from the acoustic share.
/// The definition mirrors the Python benchmark's `normalize_for_wer_naive`.
fn normalize_for_wer_naive(text: &str) -> Vec<String> {
    let text = text.to_lowercase();
    let text = text.replace('ё', "е");
    // Keep only the `[a-zа-я0-9]` + whitespace class — character-for-character
    // identical to the Python benchmark's `_NAIVE_STRIP_RE`. A broader
    // `is_alphanumeric` filter would keep accented Latin (`café`) and non-ASCII
    // digits that the Python ASCII-class pass drops, so the two harnesses'
    // verbatim WER would diverge on such transcripts; this keeps them equal.
    let text: String = text
        .chars()
        .filter(|c| {
            c.is_ascii_lowercase()
                || ('а'..='я').contains(c)
                || c.is_ascii_digit()
                || c.is_whitespace()
        })
        .collect();
    text.split_whitespace().map(String::from).collect()
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

/// Bootstrap 95% confidence interval for WER via resampling with replacement.
/// Uses a simple LCG RNG so no extra crate is required.
fn bootstrap_ci(per_sample: &[(usize, usize)], iterations: usize) -> (f64, f64) {
    let n = per_sample.len();
    if n == 0 {
        return (0.0, 0.0);
    }
    let mut rng: u64 = 123456789;
    let mut wers: Vec<f64> = Vec::with_capacity(iterations);
    for _ in 0..iterations {
        let mut total_ref = 0usize;
        let mut total_err = 0usize;
        for _ in 0..n {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            let idx = ((rng >> 32) as usize).wrapping_rem(n);
            total_ref += per_sample[idx].0;
            total_err += per_sample[idx].1;
        }
        let wer = if total_ref > 0 {
            total_err as f64 / total_ref as f64 * 100.0
        } else {
            0.0
        };
        wers.push(wer);
    }
    wers.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let lo = wers[(iterations * 25) / 1000];
    let hi = wers[(iterations * 975) / 1000];
    (lo, hi)
}

/// Measured quality metrics for one benchmark run. Absolute ceilings always
/// apply; relative baselines apply only when the matching field is populated
/// in `benchmark_baseline.json`.
#[derive(Clone, Copy, Debug)]
struct QualityMetrics {
    wer: f64,
    rtf: f64,
    rss_after_load_mb: u64,
    rss_after_decode_mb: u64,
    cold_start_ms: u64,
}

/// Absolute + relative ceilings loaded from the baseline file (with defaults).
#[derive(Clone, Copy, Debug)]
struct QualityCeilings {
    max_wer: f64,
    max_rtf: f64,
    max_rss_after_load_mb: u64,
    max_rss_after_decode_mb: u64,
    max_cold_start_ms: u64,
    wer_tolerance_pp: f64,
    rtf_tolerance_abs: f64,
    rss_tolerance_mb: u64,
    cold_start_tolerance_ms: u64,
}

impl Default for QualityCeilings {
    fn default() -> Self {
        // Generous absolute ceilings so GitHub Actions CPU runners still pass,
        // while still catching catastrophic regressions (wrong EP, FP32-only
        // load, OOM-adjacent RSS, hung cold start).
        Self {
            max_wer: MAX_WER,
            max_rtf: 5.0,
            max_rss_after_load_mb: 2048,
            max_rss_after_decode_mb: 2560,
            max_cold_start_ms: 180_000,
            wer_tolerance_pp: 1.5,
            rtf_tolerance_abs: 1.0,
            rss_tolerance_mb: 400,
            cold_start_tolerance_ms: 30_000,
        }
    }
}

fn ceilings_from_baseline(baseline: &serde_json::Value) -> QualityCeilings {
    let mut c = QualityCeilings::default();
    c.wer_tolerance_pp = baseline["tolerance_pp"]
        .as_f64()
        .unwrap_or(c.wer_tolerance_pp);
    if let Some(ceil) = baseline.get("ceilings") {
        c.max_wer = ceil["max_wer"].as_f64().unwrap_or(c.max_wer);
        c.max_rtf = ceil["max_rtf"].as_f64().unwrap_or(c.max_rtf);
        c.max_rss_after_load_mb = ceil["max_rss_after_load_mb"]
            .as_u64()
            .unwrap_or(c.max_rss_after_load_mb);
        c.max_rss_after_decode_mb = ceil["max_rss_after_decode_mb"]
            .as_u64()
            .unwrap_or(c.max_rss_after_decode_mb);
        c.max_cold_start_ms = ceil["max_cold_start_ms"]
            .as_u64()
            .unwrap_or(c.max_cold_start_ms);
    }
    if let Some(tol) = baseline.get("tolerances") {
        c.rtf_tolerance_abs = tol["rtf_abs"].as_f64().unwrap_or(c.rtf_tolerance_abs);
        c.rss_tolerance_mb = tol["rss_mb"].as_u64().unwrap_or(c.rss_tolerance_mb);
        c.cold_start_tolerance_ms = tol["cold_start_ms"]
            .as_u64()
            .unwrap_or(c.cold_start_tolerance_ms);
    }
    c
}

fn wer_failures(wer: f64, baseline: Option<f64>, ceilings: QualityCeilings) -> Vec<String> {
    let mut failures = Vec::new();
    if wer >= ceilings.max_wer {
        failures.push(format!(
            "WER {wer:.1}% exceeds absolute ceiling {max:.1}%",
            max = ceilings.max_wer
        ));
    }
    if let Some(base) = baseline {
        let delta = wer - base;
        if delta > ceilings.wer_tolerance_pp {
            failures.push(format!(
                "WER regressed {delta:+.1}pp vs baseline {base:.1}% (current {wer:.1}%, tolerance {tol:.1}pp)",
                tol = ceilings.wer_tolerance_pp
            ));
        }
    }
    failures
}

/// Pure multi-metric regression gate. Empty return = pass. Factored out of
/// `main` so the comparison (sign, boundary, null-skip) can be self-tested
/// model-free — the rest of the benchmark needs the ~225 MB INT8 model, this does
/// not.
fn gate_verdict_quality(
    m: QualityMetrics,
    baseline: &serde_json::Value,
    set_key: &str,
    ceilings: QualityCeilings,
) -> Vec<String> {
    let set = &baseline["sets"][set_key];
    let mut failures = wer_failures(m.wer, set["wer"].as_f64(), ceilings);

    // ---- absolute ceilings (always) ----
    if m.rtf > ceilings.max_rtf {
        failures.push(format!(
            "RTF {rtf:.3} exceeds absolute ceiling {max:.3}",
            rtf = m.rtf,
            max = ceilings.max_rtf
        ));
    }
    if m.rss_after_load_mb > ceilings.max_rss_after_load_mb {
        failures.push(format!(
            "RSS after load {rss} MB exceeds absolute ceiling {max} MB",
            rss = m.rss_after_load_mb,
            max = ceilings.max_rss_after_load_mb
        ));
    }
    if m.rss_after_decode_mb > ceilings.max_rss_after_decode_mb {
        failures.push(format!(
            "RSS after decode {rss} MB exceeds absolute ceiling {max} MB",
            rss = m.rss_after_decode_mb,
            max = ceilings.max_rss_after_decode_mb
        ));
    }
    if m.cold_start_ms > ceilings.max_cold_start_ms {
        failures.push(format!(
            "cold-start {ms} ms exceeds absolute ceiling {max} ms",
            ms = m.cold_start_ms,
            max = ceilings.max_cold_start_ms
        ));
    }

    // ---- relative baselines (only when committed) ----
    if let Some(base) = set["rtf"].as_f64() {
        let delta = m.rtf - base;
        if delta > ceilings.rtf_tolerance_abs {
            failures.push(format!(
                "RTF regressed {delta:+.3} vs baseline {base:.3} (current {rtf:.3}, tolerance {tol:.3})",
                rtf = m.rtf,
                tol = ceilings.rtf_tolerance_abs
            ));
        }
    }
    if let Some(base) = set["rss_after_load_mb"].as_u64() {
        let delta = m.rss_after_load_mb as i64 - base as i64;
        if delta > ceilings.rss_tolerance_mb as i64 {
            failures.push(format!(
                "RSS after load regressed {delta:+} MB vs baseline {base} MB (current {cur} MB, tolerance {tol} MB)",
                cur = m.rss_after_load_mb,
                tol = ceilings.rss_tolerance_mb
            ));
        }
    }
    if let Some(base) = set["cold_start_ms"].as_u64() {
        let delta = m.cold_start_ms as i64 - base as i64;
        if delta > ceilings.cold_start_tolerance_ms as i64 {
            failures.push(format!(
                "cold-start regressed {delta:+} ms vs baseline {base} ms (current {cur} ms, tolerance {tol} ms)",
                cur = m.cold_start_ms,
                tol = ceilings.cold_start_tolerance_ms
            ));
        }
    }

    failures
}

/// Back-compat wrapper used by the original WER-only self-tests.
fn gate_verdict(wer: f64, baseline_wer: Option<f64>, tolerance: f64, max_wer: f64) -> Vec<String> {
    let mut baseline = serde_json::json!({ "sets": { "t": {} } });
    if let Some(b) = baseline_wer {
        baseline["sets"]["t"]["wer"] = serde_json::json!(b);
    }
    let ceilings = QualityCeilings {
        max_wer,
        wer_tolerance_pp: tolerance,
        ..QualityCeilings::default()
    };
    let m = QualityMetrics {
        wer,
        rtf: 0.0,
        rss_after_load_mb: 0,
        rss_after_decode_mb: 0,
        cold_start_ms: 0,
    };
    gate_verdict_quality(m, &baseline, "t", ceilings)
}

/// Best-effort process RSS in mebibytes (Linux VmRSS / macOS `ps`).
fn current_rss_mb() -> u64 {
    #[cfg(target_os = "linux")]
    {
        if let Ok(status) = std::fs::read_to_string("/proc/self/status")
            && let Some(line) = status.lines().find(|l| l.starts_with("VmRSS:"))
        {
            let kb: u64 = line
                .split_whitespace()
                .nth(1)
                .and_then(|s| s.parse().ok())
                .unwrap_or(0);
            kb / 1024
        } else {
            0
        }
    }
    #[cfg(not(target_os = "linux"))]
    {
        if let Ok(output) = std::process::Command::new("ps")
            .args(["-o", "rss=", "-p", &std::process::id().to_string()])
            .output()
            && let Ok(rss_kb) = String::from_utf8_lossy(&output.stdout)
                .trim()
                .parse::<u64>()
        {
            rss_kb / 1024
        } else {
            0
        }
    }
}

fn wer_summary_row(wer: f64, baseline: Option<f64>, ceilings: QualityCeilings) -> String {
    let base = baseline
        .map(|b| format!("{b:.1}%"))
        .unwrap_or_else(|| "—".into());
    let verdict = if wer_failures(wer, baseline, ceilings).is_empty() {
        "ok"
    } else {
        "FAIL"
    };
    format!(
        "  | WER               | {base:>8} | {wer:5.1}% | ±{tol:.1}pp / <{max:.0}% | {verdict} |",
        tol = ceilings.wer_tolerance_pp,
        max = ceilings.max_wer
    )
}

/// Model-free assertions for `gate_verdict` + `read_baseline`, run via
/// `cargo test --test benchmark -- --self-test` (see CI). Guards against a
/// sign error / off-by-tolerance / broken null-skip in the gate shipping
/// silently, since the full benchmark is model-gated and skipped in PR CI.
fn run_gate_self_test() {
    // No baseline → relative gate skipped; only the absolute ceiling applies.
    assert!(
        gate_verdict(5.0, None, 1.5, 15.0).is_empty(),
        "no baseline, under ceiling → pass"
    );
    assert_eq!(
        gate_verdict(20.0, None, 1.5, 15.0).len(),
        1,
        "no baseline but over ceiling → fail"
    );
    // Relative-gate boundary: delta == tolerance passes; just over fails.
    assert!(
        gate_verdict(2.5, Some(1.0), 1.5, 15.0).is_empty(),
        "delta == tolerance must pass"
    );
    assert_eq!(
        gate_verdict(2.6, Some(1.0), 1.5, 15.0).len(),
        1,
        "delta just over tolerance must fail"
    );
    // An improvement (negative delta) always passes the relative gate.
    assert!(
        gate_verdict(0.5, Some(2.0), 1.5, 15.0).is_empty(),
        "improvement must pass"
    );
    // The absolute ceiling fires regardless of the baseline.
    assert!(
        gate_verdict(15.0, Some(14.9), 1.5, 15.0)
            .iter()
            .any(|f| f.contains("ceiling")),
        "wer == MAX_WER must fail on the absolute ceiling"
    );
    let ceilings = QualityCeilings {
        max_wer: 15.0,
        wer_tolerance_pp: 1.5,
        ..QualityCeilings::default()
    };
    for (wer, baseline, verdict) in [
        (15.0, Some(14.9), "FAIL"),
        (15.1, None, "FAIL"),
        (2.6, Some(1.0), "FAIL"),
        (2.5, Some(1.0), "ok"),
        (5.0, None, "ok"),
    ] {
        assert_eq!(
            gate_verdict(wer, baseline, ceilings.wer_tolerance_pp, ceilings.max_wer).is_empty(),
            verdict == "ok",
        );
        let row = wer_summary_row(wer, baseline, ceilings);
        assert!(row.ends_with(&format!("| {verdict} |")), "{row}");
    }
    admission::self_test();

    // normalize_for_wer_naive: verbatim rules only — lowercase, ё→е, strip
    // punctuation, NO words-to-digits ITN, digit merging, or anglicism mapping.
    assert_eq!(
        normalize_for_wer_naive("Привет, Мир!"),
        vec!["привет".to_string(), "мир".to_string()],
        "naive lowercases and strips punctuation"
    );
    assert_eq!(
        normalize_for_wer_naive("счёт"),
        vec!["счет".to_string()],
        "naive folds ё→е"
    );
    assert_eq!(
        normalize_for_wer_naive("пять процентов"),
        vec!["пять".to_string(), "процентов".to_string()],
        "naive does not strip percent/currency words"
    );
    assert_eq!(
        normalize_for_wer_naive("5%"),
        vec!["5".to_string()],
        "naive strips the percent sign but keeps the digit, with no ITN"
    );
    assert_eq!(
        normalize_for_wer_naive("7 919 335"),
        vec!["7".to_string(), "919".to_string(), "335".to_string()],
        "naive does not merge digit groups"
    );
    assert_eq!(
        normalize_for_wer_naive("naïve"),
        vec!["nave".to_string()],
        "naive drops accented Latin (outside [a-zа-я0-9]), matching the Python pass"
    );
    // The verbatim pass counts the digit↔word number convention as an error
    // where the normalized pass forgives it — that gap is the whole point.
    // (The Rust harness normalizes digits→words; the Python harness goes the
    // other way. Both forgive this pair, so the naive numbers stay comparable.)
    let ref_n = normalize_for_wer_naive("пять");
    let hyp_n = normalize_for_wer_naive("5");
    assert!(
        word_edit_distance(&ref_n, &hyp_n) > 0,
        "naive must penalize the digit/word difference the ITN pass forgives"
    );
    let ref_i = normalize_for_wer("пять");
    let hyp_i = normalize_for_wer("5");
    assert_eq!(
        word_edit_distance(&ref_i, &hyp_i),
        0,
        "the normalized (ITN) pass forgives the digit/word number convention"
    );

    // Multi-metric gate: absolute ceilings fire independently of relative baselines.
    let clean = QualityMetrics {
        wer: 1.0,
        rtf: 0.2,
        rss_after_load_mb: 300,
        rss_after_decode_mb: 400,
        cold_start_ms: 1000,
    };
    let empty = serde_json::json!({ "sets": { "bundled": {} } });
    assert!(
        gate_verdict_quality(clean, &empty, "bundled", QualityCeilings::default()).is_empty(),
        "clean metrics under absolute ceilings must pass"
    );

    let hot = QualityMetrics {
        wer: 1.0,
        rtf: 9.0, // over default max_rtf 5.0
        rss_after_load_mb: 300,
        rss_after_decode_mb: 400,
        cold_start_ms: 1000,
    };
    let rtf_fail = gate_verdict_quality(hot, &empty, "bundled", QualityCeilings::default());
    assert!(
        rtf_fail.iter().any(|f| f.contains("RTF")),
        "RTF absolute ceiling must fire: {rtf_fail:?}"
    );

    // Relative RTF regression with a committed baseline.
    let with_rtf = serde_json::json!({
        "sets": { "bundled": { "rtf": 0.15 } }
    });
    let regressed = QualityMetrics {
        wer: 1.0,
        rtf: 1.5, // +1.35 over baseline, > default rtf_tolerance_abs 1.0
        rss_after_load_mb: 300,
        rss_after_decode_mb: 400,
        cold_start_ms: 1000,
    };
    let rel = gate_verdict_quality(regressed, &with_rtf, "bundled", QualityCeilings::default());
    assert!(
        rel.iter().any(|f| f.contains("RTF regressed")),
        "relative RTF gate must fire: {rel:?}"
    );

    // RSS absolute ceiling.
    let fat = QualityMetrics {
        wer: 1.0,
        rtf: 0.2,
        rss_after_load_mb: 5000,
        rss_after_decode_mb: 5000,
        cold_start_ms: 1000,
    };
    let rss_fail = gate_verdict_quality(fat, &empty, "bundled", QualityCeilings::default());
    assert!(
        rss_fail.iter().any(|f| f.contains("RSS after load")),
        "RSS absolute ceiling must fire: {rss_fail:?}"
    );

    // ceilings_from_baseline reads committed ceilings.
    let custom = ceilings_from_baseline(&serde_json::json!({
        "tolerance_pp": 2.0,
        "ceilings": { "max_rtf": 1.5, "max_wer": 10.0 },
        "tolerances": { "rtf_abs": 0.25 }
    }));
    assert_eq!(custom.max_rtf, 1.5);
    assert_eq!(custom.max_wer, 10.0);
    assert_eq!(custom.wer_tolerance_pp, 2.0);
    assert_eq!(custom.rtf_tolerance_abs, 0.25);
}

fn main() {
    // nextest (and cargo test --list) invoke us with --list --format terse.
    let args: Vec<String> = std::env::args().collect();
    if args.len() > 1 && args[1] == "--list" {
        println!("benchmark: test");
        return;
    }
    // Manual CI runs this model-free self-test. Panics exit nonzero.
    if !args.iter().any(|a| a == "--run") || args.iter().any(|a| a == "--self-test") {
        if args.iter().any(|a| a == "--required") {
            eprintln!("--required needs --run");
            std::process::exit(1);
        }
        run_gate_self_test();
        println!("benchmark gate self-test: ok");
        return;
    }

    let required = args.iter().any(|a| a == "--required");
    if let Err(error) = run_benchmark(required) {
        println!(
            "{}",
            serde_json::json!({
                "pass": false, "score": null, "skipped": false, "status": "invalid",
                "reason": format!("{error:#}"),
            })
        );
        std::process::exit(1);
    }
}

fn run_benchmark(required: bool) -> Result<()> {
    let max_samples = std::env::var("GIGASTT_BENCHMARK_MAX_SAMPLES")
        .ok()
        .map(|s| s.parse::<usize>())
        .transpose()
        .context("invalid sample limit")?;
    let update_baseline = match std::env::var("GIGASTT_BENCHMARK_UPDATE_BASELINE")
        .ok()
        .as_deref()
    {
        None | Some("0") => false,
        Some("1") => true,
        _ => anyhow::bail!("invalid baseline update flag"),
    };
    ensure!(
        !required || (!update_baseline && max_samples.unwrap_or(0) == 0),
        "required mode forbids baseline changes and sample limits"
    );

    // Prefer external Golos benchmark set if available, unless CI forces the
    // hermetic bundled fixtures (stable sample count for RTF/RSS ceilings).
    let force_bundled = required || std::env::var_os("GIGASTT_BENCHMARK_FORCE_BUNDLED").is_some();
    let external_manifest = if force_bundled {
        None
    } else {
        home_dir()
            .map(|h| h.join(".gigastt/benchmarks/golos_wav/manifest.json"))
            .filter(|p| p.exists())
    };

    let using_bundled = external_manifest.is_none();
    let (manifest_path, fixture_dir) = if let Some(path) = external_manifest {
        let dir = path.parent().unwrap().to_path_buf();
        eprintln!("Using external benchmark set: {}", dir.display());
        (path, dir)
    } else {
        let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures");
        eprintln!("Using bundled fixtures: {}", dir.display());
        (dir.join("manifest.json"), dir)
    };

    let model_dir = home_dir()
        .map(|h| h.join(".gigastt").join("models"))
        .context("HOME not set")?;

    // Accept either head's encoder (rnnt default since v2.3, or e2e_rnnt; FP32 or
    // generated INT8). The engine auto-detects which variant to load.
    let has_model = [
        "v3_rnnt_encoder.onnx",
        "v3_rnnt_encoder_int8.onnx",
        "v3_e2e_rnnt_encoder.onnx",
        "v3_e2e_rnnt_encoder_int8.onnx",
    ]
    .iter()
    .any(|f| model_dir.join(f).exists());
    if !has_model {
        ensure!(!required, "model not found");
        println!(
            "{}",
            serde_json::json!({
                "pass": false, "score": null, "skipped": true, "status": "skipped",
                "reason": "model not found",
            })
        );
        return Ok(());
    }

    let baseline_path = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/benchmark_baseline.json");
    let set_key = if using_bundled { "bundled" } else { "external" };
    let mut baseline = admission::read_baseline(&baseline_path, set_key)?;
    let mut manifest: Vec<Sample> =
        serde_json::from_slice(&std::fs::read(&manifest_path).context("read manifest")?)
            .context("parse manifest")?;

    let full =
        max_samples.unwrap_or(0) == 0 || max_samples.is_some_and(|limit| limit >= manifest.len());
    ensure!(
        !update_baseline || full,
        "baseline updates require the full corpus"
    );
    if !full {
        manifest.truncate(max_samples.context("missing sample limit")?);
    }
    admission::validate_corpus(&manifest, &fixture_dir, &baseline, set_key, full)?;

    let model_dir_str = model_dir.to_string_lossy();
    // Cold-start: wall time of model verification (required mode) and
    // Engine::load (model mmap + session build + pool).
    // Measured before the decode loop so RTF is pure inference throughput.
    let cold_t0 = std::time::Instant::now();
    let engine = if required {
        admission::verify_required_model(&model_dir)?;
        // Match Engine::load's default pool, minimum and encoder thread settings.
        let pool_size = if cfg!(target_os = "android") { 1 } else { 2 };
        gigastt::inference::Engine::load_with_pools_threads_variant(
            &model_dir_str,
            Some(gigastt::model::ModelVariant::Rnnt),
            pool_size,
            pool_size,
            0,
            1,
        )
    } else {
        gigastt::inference::Engine::load(&model_dir_str)
    }
    .context("load engine")?;
    let cold_start_ms = cold_t0.elapsed().as_millis() as u64;
    let rss_after_load_mb = current_rss_mb();
    eprintln!("  cold-start: {cold_start_ms} ms | RSS after load: {rss_after_load_mb} MB");

    let rt = tokio::runtime::Runtime::new().context("create tokio runtime")?;
    let mut guard = rt
        .block_on(engine.pool.checkout())
        .context("checkout benchmark pool")?;

    let mut total_ref_words = 0usize;
    let mut total_errors = 0usize;
    let mut total_naive_ref_words = 0usize;
    let mut total_naive_errors = 0usize;
    let mut total_audio_s = 0.0_f64;
    let mut details = Vec::new();
    let mut per_sample: Vec<(usize, usize)> = Vec::with_capacity(manifest.len());
    let mut naive_per_sample: Vec<(usize, usize)> = Vec::with_capacity(manifest.len());

    let start_time = std::time::Instant::now();

    for (idx, sample) in manifest.iter().enumerate() {
        let wav_path = if Path::new(&sample.filename).is_absolute() {
            PathBuf::from(&sample.filename)
        } else {
            fixture_dir.join(&sample.filename)
        };

        let hypothesis = engine
            .transcribe_file(
                wav_path.to_str().context("non-UTF8 audio path")?,
                &mut guard,
            )
            .context("transcription failed")?;
        ensure!(
            hypothesis.duration_s.is_finite() && hypothesis.duration_s > 0.0,
            "invalid audio duration"
        );
        total_audio_s += hypothesis.duration_s;

        let ref_words = normalize_for_wer(&sample.reference);
        let hyp_words = normalize_for_wer(&hypothesis.text);

        let errors = word_edit_distance(&ref_words, &hyp_words);
        let sample_wer = if ref_words.is_empty() {
            0.0
        } else {
            errors as f64 / ref_words.len() as f64 * 100.0
        };

        let ref_words_naive = normalize_for_wer_naive(&sample.reference);
        let hyp_words_naive = normalize_for_wer_naive(&hypothesis.text);
        let naive_errors = word_edit_distance(&ref_words_naive, &hyp_words_naive);

        total_ref_words += ref_words.len();
        total_errors += errors;
        per_sample.push((ref_words.len(), errors));
        total_naive_ref_words += ref_words_naive.len();
        total_naive_errors += naive_errors;
        naive_per_sample.push((ref_words_naive.len(), naive_errors));

        if idx % PROGRESS_EVERY == 0 || idx + 1 == manifest.len() {
            let elapsed = start_time.elapsed().as_secs_f64();
            let rate = if idx > 0 { elapsed / idx as f64 } else { 0.0 };
            let remaining = rate * (manifest.len() - idx) as f64;
            eprintln!(
                "  [{}/{}] {:.1}s elapsed, ~{:.0}s remaining | [WER {:5.1}%] {}",
                idx + 1,
                manifest.len(),
                elapsed,
                remaining,
                sample_wer,
                sample.filename
            );
        }

        details.push(serde_json::json!({
            "file": sample.filename,
            "reference": sample.reference,
            "hypothesis": hypothesis.text,
            "ref_norm": ref_words.join(" "),
            "hyp_norm": hyp_words.join(" "),
            "wer": (sample_wer * 10.0).round() / 10.0,
        }));
    }

    ensure!(
        total_ref_words > 0 && total_audio_s.is_finite() && total_audio_s > 0.0,
        "no measured corpus evidence"
    );
    let wer = if total_ref_words > 0 {
        total_errors as f64 / total_ref_words as f64 * 100.0
    } else {
        0.0
    };
    let score = (100.0 - wer).max(0.0);
    let score_rounded = (score * 10.0).round() / 10.0;
    let wer_rounded = (wer * 10.0).round() / 10.0;

    // Bootstrap 95% confidence interval (resample with replacement).
    let (ci_lo, ci_hi) = bootstrap_ci(&per_sample, 1000);
    let ci_lo_r = (ci_lo * 10.0).round() / 10.0;
    let ci_hi_r = (ci_hi * 10.0).round() / 10.0;

    // Verbatim ("naive") WER: the same metric without words-to-digits ITN or
    // anglicism mapping. The gap (naive_delta = wer - naive_wer) is the
    // writing-convention share of the error. Reported for transparency only —
    // the regression gate below stays on the normalized WER.
    let naive_wer = if total_naive_ref_words > 0 {
        total_naive_errors as f64 / total_naive_ref_words as f64 * 100.0
    } else {
        0.0
    };
    let naive_wer_rounded = (naive_wer * 10.0).round() / 10.0;
    let (naive_ci_lo, naive_ci_hi) = bootstrap_ci(&naive_per_sample, 1000);
    let naive_ci_lo_r = (naive_ci_lo * 10.0).round() / 10.0;
    let naive_ci_hi_r = (naive_ci_hi * 10.0).round() / 10.0;
    let naive_delta_r = ((wer - naive_wer) * 10.0).round() / 10.0;

    let decode_wall_s = start_time.elapsed().as_secs_f64();
    let rtf = if total_audio_s > 0.0 {
        decode_wall_s / total_audio_s
    } else {
        0.0
    };
    let rtf_rounded = (rtf * 1000.0).round() / 1000.0;
    let rss_after_decode_mb = current_rss_mb();

    eprintln!(
        "\n  WER: {:.1}% ({} errors / {} words)  Score: {:.1}  Samples: {}",
        wer,
        total_errors,
        total_ref_words,
        score,
        manifest.len()
    );
    eprintln!("  95% CI: [{:.1}%, {:.1}%]", ci_lo_r, ci_hi_r);
    eprintln!(
        "  Verbatim (naive) WER: {:.1}% ({} errors / {} words)  Δ {:+.1}pp vs normalized",
        naive_wer_rounded, total_naive_errors, total_naive_ref_words, naive_delta_r
    );
    eprintln!(
        "  RTF: {:.3} (wall {:.2}s / audio {:.2}s) | RSS load/decode: {}/{} MB | cold-start: {} ms",
        rtf_rounded,
        decode_wall_s,
        total_audio_s,
        rss_after_load_mb,
        rss_after_decode_mb,
        cold_start_ms
    );

    // ---- Multi-metric regression gate --------------------------------------
    // Hard checks (non-zero exit so CI fails on regression):
    //   1. absolute ceilings for WER / RTF / RSS / cold-start (always)
    //   2. relative baselines per set when committed in benchmark_baseline.json
    // Populate relative fields with GIGASTT_BENCHMARK_UPDATE_BASELINE=1.
    if update_baseline {
        for (key, value) in [
            ("wer", serde_json::json!(wer_rounded)),
            ("rtf", serde_json::json!(rtf_rounded)),
            ("rss_after_load_mb", serde_json::json!(rss_after_load_mb)),
            (
                "rss_after_decode_mb",
                serde_json::json!(rss_after_decode_mb),
            ),
            ("cold_start_ms", serde_json::json!(cold_start_ms)),
        ] {
            baseline["sets"][set_key][key] = value;
        }
        std::fs::write(
            &baseline_path,
            serde_json::to_string_pretty(&baseline)? + "\n",
        )
        .context("write benchmark baseline")?;
        eprintln!("  Baseline updated: {}", baseline_path.display());
    }

    let ceilings = ceilings_from_baseline(&baseline);
    let metrics = QualityMetrics {
        wer,
        rtf,
        rss_after_load_mb,
        rss_after_decode_mb,
        cold_start_ms,
    };

    let baseline_wer = baseline["sets"][set_key]["wer"].as_f64();

    // Presentation table (pass/fail decision is pure gate_verdict_quality).
    eprintln!("\n  Quality gate [{set_key}]:");
    eprintln!("  | metric            | baseline | current | tol / ceiling | verdict |");
    eprintln!("  |-------------------|----------|---------|---------------|---------|");
    eprintln!("{}", wer_summary_row(wer, baseline_wer, ceilings));
    {
        let base = baseline["sets"][set_key]["rtf"].as_f64();
        let verdict = if rtf > ceilings.max_rtf {
            "FAIL"
        } else if let Some(b) = base {
            if rtf - b > ceilings.rtf_tolerance_abs {
                "FAIL"
            } else {
                "ok"
            }
        } else {
            "ok"
        };
        let base_s = base
            .map(|b| format!("{b:.3}"))
            .unwrap_or_else(|| "—".into());
        eprintln!(
            "  | RTF               | {base_s:>8} | {rtf:7.3} | ±{tol:.2} / <{max:.1} | {verdict} |",
            tol = ceilings.rtf_tolerance_abs,
            max = ceilings.max_rtf
        );
    }
    {
        let base = baseline["sets"][set_key]["rss_after_load_mb"].as_u64();
        let verdict = if rss_after_load_mb > ceilings.max_rss_after_load_mb {
            "FAIL"
        } else if let Some(b) = base {
            if rss_after_load_mb as i64 - b as i64 > ceilings.rss_tolerance_mb as i64 {
                "FAIL"
            } else {
                "ok"
            }
        } else {
            "ok"
        };
        let base_s = base.map(|b| format!("{b}")).unwrap_or_else(|| "—".into());
        eprintln!(
            "  | RSS after load   | {base_s:>6} MB | {rss_after_load_mb:5} MB | ±{tol} / <{max} | {verdict} |",
            tol = ceilings.rss_tolerance_mb,
            max = ceilings.max_rss_after_load_mb
        );
    }
    {
        let verdict = if rss_after_decode_mb > ceilings.max_rss_after_decode_mb {
            "FAIL"
        } else {
            "ok"
        };
        eprintln!(
            "  | RSS after decode |        — | {rss_after_decode_mb:5} MB |     / <{max} | {verdict} |",
            max = ceilings.max_rss_after_decode_mb
        );
    }
    {
        let base = baseline["sets"][set_key]["cold_start_ms"].as_u64();
        let verdict = if cold_start_ms > ceilings.max_cold_start_ms {
            "FAIL"
        } else if let Some(b) = base {
            if cold_start_ms as i64 - b as i64 > ceilings.cold_start_tolerance_ms as i64 {
                "FAIL"
            } else {
                "ok"
            }
        } else {
            "ok"
        };
        let base_s = base.map(|b| format!("{b}")).unwrap_or_else(|| "—".into());
        eprintln!(
            "  | cold-start        | {base_s:>6} ms | {cold_start_ms:5} ms | ±{tol} / <{max} | {verdict} |",
            tol = ceilings.cold_start_tolerance_ms,
            max = ceilings.max_cold_start_ms
        );
    }

    let failures = gate_verdict_quality(metrics, &baseline, set_key, ceilings);
    let passed = failures.is_empty();

    let output = serde_json::json!({
        "pass": passed,
        "status": "measured",
        "skipped": false,
        "score": score_rounded,
        "wer": wer_rounded,
        "ci_low": ci_lo_r,
        "ci_high": ci_hi_r,
        "naive_wer": naive_wer_rounded,
        "naive_ci_low": naive_ci_lo_r,
        "naive_ci_high": naive_ci_hi_r,
        "naive_total_errors": total_naive_errors,
        "naive_total_words": total_naive_ref_words,
        "naive_delta": naive_delta_r,
        "total_words": total_ref_words,
        "total_errors": total_errors,
        "samples": manifest.len(),
        "set": set_key,
        "baseline_wer": baseline_wer,
        "rtf": rtf_rounded,
        "audio_s": (total_audio_s * 100.0).round() / 100.0,
        "wall_s": (decode_wall_s * 100.0).round() / 100.0,
        "rss_after_load_mb": rss_after_load_mb,
        "rss_after_decode_mb": rss_after_decode_mb,
        "cold_start_ms": cold_start_ms,
        "details": details,
    });

    println!("{}", serde_json::to_string(&output).unwrap());

    if !passed {
        for f in &failures {
            eprintln!("  GATE FAILED: {f}");
        }
        std::process::exit(1);
    }
    Ok(())
}
