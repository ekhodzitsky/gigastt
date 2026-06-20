//! WER benchmark: transcribes Golos test fixtures and reports Word Error Rate.
//!
//! Supports an external manifest in `~/.gigastt/benchmarks/golos_wav/manifest.json`.
//! Falls back to the bundled `tests/fixtures/manifest.json` when the external set
//! is missing.
//!
//! Environment variables:
//! - `GIGASTT_BENCHMARK_MAX_SAMPLES` — limit the number of samples (0 = unlimited).
//!
//! Outputs JSON to stdout for the autoresearch evaluator.
//! Harness is disabled (`harness = false` in Cargo.toml) so this runs as a binary.

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

/// Load the committed WER baseline, or an empty default if it is missing/invalid.
fn read_baseline(path: &Path) -> serde_json::Value {
    std::fs::read_to_string(path)
        .ok()
        .and_then(|s| serde_json::from_str::<serde_json::Value>(&s).ok())
        .unwrap_or_else(|| serde_json::json!({ "tolerance_pp": 0.5, "sets": {} }))
}

/// Pure regression-gate decision: returns the list of failure messages (empty
/// = pass). Two checks — (1) relative: WER must not exceed the committed
/// baseline by more than `tolerance` (skipped when no baseline is committed);
/// (2) absolute: WER must stay below the `max_wer` ceiling. Factored out of
/// `main` so the comparison (sign, boundary, null-skip) can be self-tested
/// model-free — the rest of the benchmark needs the ~850 MB model, this does
/// not.
fn gate_verdict(wer: f64, baseline_wer: Option<f64>, tolerance: f64, max_wer: f64) -> Vec<String> {
    let mut failures = Vec::new();
    if let Some(base) = baseline_wer {
        let delta = wer - base;
        if delta > tolerance {
            failures.push(format!(
                "WER regressed {delta:+.1}pp vs baseline {base:.1}% (current {wer:.1}%, tolerance {tolerance:.1}pp)"
            ));
        }
    }
    if wer >= max_wer {
        failures.push(format!(
            "WER {wer:.1}% exceeds absolute ceiling {max_wer:.1}%"
        ));
    }
    failures
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
    // read_baseline: a missing file falls back to tolerance 0.5 / empty sets.
    let missing = read_baseline(Path::new("/nonexistent/benchmark_baseline.json"));
    assert_eq!(missing["tolerance_pp"].as_f64(), Some(0.5));
    assert!(missing["sets"]["bundled"]["wer"].as_f64().is_none());

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
}

fn main() {
    // nextest (and cargo test --list) invoke us with --list --format terse.
    let args: Vec<String> = std::env::args().collect();
    if args.len() > 1 && args[1] == "--list" {
        println!("benchmark: test");
        return;
    }
    // Model-free gate self-test (CI runs this on every PR; the full benchmark
    // is model-gated and skipped). Panics on failure → non-zero exit.
    if args.iter().any(|a| a == "--self-test") {
        run_gate_self_test();
        println!("benchmark gate self-test: ok");
        return;
    }

    let max_samples = std::env::var("GIGASTT_BENCHMARK_MAX_SAMPLES")
        .ok()
        .and_then(|s| s.parse::<usize>().ok());

    // Prefer external Golos benchmark set if available.
    let external_manifest = home_dir()
        .map(|h| h.join(".gigastt/benchmarks/golos_wav/manifest.json"))
        .filter(|p| p.exists());

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
        .expect("HOME not set");

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
        println!(
            r#"{{"pass": true, "score": null, "skipped": true, "reason": "model not found"}}"#
        );
        return;
    }

    let mut manifest: Vec<Sample> = serde_json::from_str(
        &std::fs::read_to_string(&manifest_path).expect("Failed to read manifest"),
    )
    .expect("Failed to parse manifest");

    if let Some(limit) = max_samples
        && limit > 0
        && manifest.len() > limit
    {
        manifest.truncate(limit);
    }

    let model_dir_str = model_dir.to_string_lossy();
    let engine = gigastt::inference::Engine::load(&model_dir_str).expect("Failed to load engine");

    let rt = tokio::runtime::Runtime::new().expect("Failed to create tokio runtime");
    let mut guard = rt
        .block_on(engine.pool.checkout())
        .expect("pool closed before benchmark started");

    let mut total_ref_words = 0usize;
    let mut total_errors = 0usize;
    let mut total_naive_ref_words = 0usize;
    let mut total_naive_errors = 0usize;
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
            .transcribe_file(wav_path.to_str().unwrap(), &mut guard)
            .expect("Transcription failed");

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

    // ---- Regression gate ---------------------------------------------------
    // Two hard checks (non-zero exit so CI actually fails on regression):
    //   1. relative — WER must not exceed the committed per-set baseline by more
    //      than `tolerance_pp`. This is the load-bearing gate; populate/refresh
    //      it with GIGASTT_BENCHMARK_UPDATE_BASELINE=1 on a machine with the
    //      model. An unpopulated baseline ("wer": null) skips this check.
    //   2. absolute — WER must stay below the MAX_WER ceiling (coarse backstop).
    let baseline_path = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/benchmark_baseline.json");
    let set_key = if using_bundled { "bundled" } else { "external" };

    if std::env::var_os("GIGASTT_BENCHMARK_UPDATE_BASELINE").is_some() {
        let mut baseline = read_baseline(&baseline_path);
        baseline["sets"][set_key] = serde_json::json!({
            "samples": manifest.len(),
            "wer": wer_rounded,
        });
        std::fs::write(
            &baseline_path,
            serde_json::to_string_pretty(&baseline).unwrap() + "\n",
        )
        .expect("Failed to write benchmark baseline");
        eprintln!(
            "  Baseline updated: set '{}' → WER {:.1}% ({})",
            set_key,
            wer_rounded,
            baseline_path.display()
        );
    }

    let baseline = read_baseline(&baseline_path);
    let tolerance = baseline["tolerance_pp"].as_f64().unwrap_or(0.5);
    let baseline_wer = baseline["sets"][set_key]["wer"].as_f64();

    // Presentation only: print the comparison table (or a no-baseline note).
    // The pass/fail decision is the pure `gate_verdict` below (self-tested).
    match baseline_wer {
        Some(base) => {
            let delta = wer - base;
            let verdict = if delta > tolerance { "FAIL" } else { "ok" };
            eprintln!("\n  Regression gate [{set_key}]:");
            eprintln!("  | set | baseline | current | Δ pp | tol pp | verdict |");
            eprintln!("  |-----|----------|---------|------|--------|---------|");
            eprintln!(
                "  | {set_key} | {base:.1}% | {wer:.1}% | {delta:+.1} | {tolerance:.1} | {verdict} |"
            );
        }
        None => {
            eprintln!(
                "\n  Regression gate [{set_key}]: no committed baseline — relative gate skipped. \
                 Run with GIGASTT_BENCHMARK_UPDATE_BASELINE=1 to populate {}.",
                baseline_path.display()
            );
        }
    }

    let failures = gate_verdict(wer, baseline_wer, tolerance, MAX_WER);
    let passed = failures.is_empty();

    let output = serde_json::json!({
        "pass": passed,
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
        "details": details,
    });

    println!("{}", serde_json::to_string(&output).unwrap());

    if !passed {
        for f in &failures {
            eprintln!("  GATE FAILED: {f}");
        }
        std::process::exit(1);
    }
}
