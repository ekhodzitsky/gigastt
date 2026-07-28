//! Corpus duration scan — which shipped benchmark manifests actually exercise
//! the chunked long-form path.
//!
//! A clip only reaches the chunked decode path when it is longer than
//! [`CHUNK_THRESHOLD_SAMPLES`] (30 s @ 16 kHz). This scan measures every clip in
//! every shipped manifest and reports, per manifest, how many cross that boundary.
//! It is model-free: it needs the benchmark corpora on disk but no ONNX model.
//!
//! Duration is read from the container header via [`probe_duration_file`], which
//! answers in microseconds for any WAV/FLAC that declares its frame count — the
//! whole benchmark set. Only a clip whose header declares no trustworthy duration
//! (`Ok(None)`) falls back to a full [`decode_audio_file`], the engine's own
//! ground truth. Reading the header instead of decoding every clip is the
//! difference between a ~50-minute single-threaded run and a few seconds, and it
//! answers the same question: seconds are sample-rate-independent, and the
//! engine's threshold is exactly `CHUNK_THRESHOLD_SAMPLES / 16 kHz = 30 s`.
//!
//! ```sh
//! cargo test -p gigastt-core --test corpus_duration_scan \
//!     corpus_over_threshold_scan -- --ignored --nocapture
//! ```
//!
//! Environment:
//! - `GIGASTT_SCAN_MANIFEST_DIR` — manifest directory (default `benchmark/manifests`).
//! - `GIGASTT_SCAN_MANIFESTS`    — comma-separated manifest stems to restrict to.
//!
//! ## Why this exists
//!
//! The claim "no shipped manifest holds a clip over 30 s" was made once off a scan
//! that silently skipped 7 of the 15 manifests and reported their absence as a
//! zero. It was wrong: those 7 manifests hold the clips that matter. So the one
//! invariant this file enforces structurally is that **nothing measurable is ever
//! dropped**:
//! - a manifest that cannot be read (absent audio, unparseable JSON, no `samples[]`)
//!   is reported as [`Corpus::Unreadable`] with the reason, never as a zero-count row;
//! - a `samples[]` entry with no usable `filename` is reported as *malformed*, never
//!   silently filtered out of the clip count;
//! - a clip that neither probe nor decode can measure is reported as *undecodable*,
//!   never counted as a clip under the threshold;
//! - the ignored scan asserts it measured at least one real clip, so a host without
//!   the corpora fails loudly instead of passing green having measured nothing.
//!
//! ## Keeping the fast path honest
//!
//! The header probe and a full decode should report the same duration; the report
//! shows, per manifest, how many clips each path answered so a reader can see which
//! rows are header-derived. A disagreement can only change a verdict right at the
//! 30 s boundary, where header rounding might flip over/under, so after the scan a
//! bounded handful of the clips nearest that boundary are decoded for real and
//! cross-checked — the report prints the largest observed duration delta and any
//! verdict that flipped.

use std::path::{Path, PathBuf};

use gigastt_core::inference::audio::{decode_audio_file, probe_duration_file};

/// Mirror of the crate-private `CHUNK_THRESHOLD_SAMPLES` in
/// `crates/gigastt-core/src/inference/engine.rs` (`16_000 * 30`). Integration
/// tests cannot see crate-private items; keep this in sync. The whole point of
/// the scan is to find clips that cross this exact boundary, so the value is
/// pinned by [`threshold_is_the_documented_thirty_seconds`].
const CHUNK_THRESHOLD_SAMPLES: usize = 16_000 * 30;

/// The engine decodes and resamples every clip to 16 kHz mono, so a decoded
/// buffer's length in samples divided by this is its duration in seconds.
const SAMPLE_RATE: usize = 16_000;

/// The chunked-path threshold expressed in seconds — the unit the header probe
/// answers in. Derived from the mirrored sample threshold so a decoded clip
/// (`buf.len() > CHUNK_THRESHOLD_SAMPLES`) and a probed clip
/// (`seconds > THRESHOLD_SECONDS`) apply the same 30 s boundary. Pinned to 30.0
/// by [`threshold_is_the_documented_thirty_seconds`].
const THRESHOLD_SECONDS: f64 = CHUNK_THRESHOLD_SAMPLES as f64 / SAMPLE_RATE as f64;

/// One clip to measure. The scan cares only about the resolved audio path; the
/// manifest's reference transcript is irrelevant to duration.
struct Sample {
    path: PathBuf,
}

/// A manifest after resolution. `Unreadable` is a first-class outcome, never a
/// silent gap: a manifest we could not turn into clips is reported with why.
enum Corpus {
    Ready {
        name: String,
        /// Entries with a usable `filename`, resolved against `audio_root`.
        samples: Vec<Sample>,
        /// Entries that were present in `samples[]` but had no usable `filename`.
        /// Reported, never dropped — an unusable entry is not a missing entry.
        malformed: Vec<String>,
    },
    Unreadable {
        name: String,
        why: String,
    },
}

fn manifest_dir() -> PathBuf {
    match std::env::var("GIGASTT_SCAN_MANIFEST_DIR") {
        Ok(d) => PathBuf::from(d),
        Err(_) => Path::new(env!("CARGO_MANIFEST_DIR")).join("../../benchmark/manifests"),
    }
}

fn env_filter() -> Option<Vec<String>> {
    std::env::var("GIGASTT_SCAN_MANIFESTS")
        .ok()
        .map(|s| s.split(',').map(|p| p.trim().to_string()).collect())
}

fn expand_tilde(p: &str) -> PathBuf {
    match p.strip_prefix("~/") {
        Some(rest) => match std::env::var_os("HOME") {
            Some(home) => PathBuf::from(home).join(rest),
            None => PathBuf::from(p),
        },
        None => PathBuf::from(p),
    }
}

fn load_corpora() -> Vec<Corpus> {
    load_corpora_in(&manifest_dir(), env_filter())
}

/// Read every `*.json` under `dir` and resolve it. Pure over its arguments (no
/// env access), so the self-tests can exercise it hermetically against a temp
/// directory. Every failure mode becomes a reported [`Corpus::Unreadable`]
/// rather than a dropped element.
fn load_corpora_in(dir: &Path, only: Option<Vec<String>>) -> Vec<Corpus> {
    let read = match std::fs::read_dir(dir) {
        Ok(rd) => rd,
        Err(e) => {
            return vec![Corpus::Unreadable {
                name: dir.display().to_string(),
                why: format!("manifest dir unreadable: {e}"),
            }];
        }
    };

    // Surface, rather than drop, both directory entries we cannot even stat and
    // manifests we can. Collect paths first so the report is deterministic.
    let mut corpora: Vec<Corpus> = Vec::new();
    let mut paths: Vec<PathBuf> = Vec::new();
    for entry in read {
        match entry {
            Ok(e) => paths.push(e.path()),
            Err(e) => corpora.push(Corpus::Unreadable {
                name: "<unreadable directory entry>".to_string(),
                why: format!("dir entry unreadable: {e}"),
            }),
        }
    }
    paths.retain(|p| p.extension().is_some_and(|x| x == "json"));
    paths.sort();

    for path in paths {
        let name = path
            .file_stem()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_else(|| path.display().to_string());
        if let Some(only) = &only
            && !only.contains(&name)
        {
            continue;
        }
        corpora.push(read_corpus(&path, name));
    }
    corpora
}

fn read_corpus(path: &Path, name: String) -> Corpus {
    let unreadable = |why: String| Corpus::Unreadable {
        name: name.clone(),
        why,
    };

    let raw = match std::fs::read_to_string(path) {
        Ok(r) => r,
        Err(e) => return unreadable(format!("read failed: {e}")),
    };
    let json: serde_json::Value = match serde_json::from_str(&raw) {
        Ok(j) => j,
        Err(e) => return unreadable(format!("parse failed: {e}")),
    };
    let root = match json.get("audio_root").and_then(|v| v.as_str()) {
        Some(r) => expand_tilde(r),
        None => return unreadable("manifest has no audio_root".to_string()),
    };
    if !root.is_dir() {
        return unreadable(format!(
            "audio_root absent on this host: {}",
            root.display()
        ));
    }

    let entries = match json.get("samples").and_then(|v| v.as_array()) {
        Some(a) if !a.is_empty() => a,
        Some(_) => return unreadable("manifest samples[] is empty".to_string()),
        None => return unreadable("manifest has no samples[] array".to_string()),
    };

    let mut samples = Vec::new();
    let mut malformed = Vec::new();
    for (i, s) in entries.iter().enumerate() {
        match s.get("filename").and_then(|f| f.as_str()) {
            Some(f) if !f.is_empty() => samples.push(Sample { path: root.join(f) }),
            _ => malformed.push(format!("entry #{i}: missing or empty \"filename\"")),
        }
    }
    Corpus::Ready {
        name,
        samples,
        malformed,
    }
}

/// Decode one clip to the engine's 16 kHz mono buffer. A decode failure is an
/// error to report, never a silent zero. Used for the fallback path and for the
/// boundary spot-check.
fn decode(path: &Path) -> Result<Vec<f32>, String> {
    decode_audio_file(&path.to_string_lossy()).map_err(|e| e.to_string())
}

/// How a clip's duration was established.
enum Measure {
    /// Answered from the container header. Carries the declared duration in
    /// seconds (frames / sample-rate — independent of any resampling).
    Probed(f64),
    /// The header declared no trustworthy duration, so the engine's own decode
    /// answered. Carries the 16 kHz mono sample count, kept as an integer so the
    /// over-threshold test stays the engine's exact `> CHUNK_THRESHOLD_SAMPLES`
    /// comparison rather than a float re-derivation.
    Decoded(usize),
    /// Neither path could measure the clip. Reported, never a silent zero.
    Unmeasurable(String),
}

/// Measure one clip: header probe first, full decode only when the header
/// declares no trustworthy duration (`Ok(None)`). A probe *error* means the
/// container cannot be parsed at all; `decode_audio_file` shares that same
/// container probe and would fail identically, so there is nothing to gain by
/// paying to reproduce the error — report it as unmeasurable instead.
fn measure(path: &Path) -> Measure {
    let p = path.to_string_lossy();
    match probe_duration_file(&p) {
        Ok(Some(seconds)) => Measure::Probed(seconds),
        Ok(None) => match decode_audio_file(&p) {
            Ok(buf) => Measure::Decoded(buf.len()),
            Err(e) => {
                Measure::Unmeasurable(format!("header gave no duration and decode failed: {e}"))
            }
        },
        Err(e) => Measure::Unmeasurable(format!(
            "container unprobeable (decode shares this probe): {e}"
        )),
    }
}

/// How many boundary-nearest probed clips to decode for real and cross-check
/// against the header. A handful, so the scan stays fast; a verdict can only flip
/// right at the threshold, so the nearest clips are the only ones worth checking.
const SPOTCHECK_CLIPS: usize = 24;

/// Scan every shipped manifest and report how many clips reach the chunked
/// long-form path. Model-free; needs the benchmark corpora on disk.
///
/// One row per manifest, split into how many clips the header probe answered
/// versus how many fell back to a full decode. A manifest whose audio is absent
/// or whose JSON will not parse prints `UNREADABLE` with the reason — it is never
/// a zero row. The final assertion fails if the whole scan measured nothing, so it
/// cannot pass green on a host that has no corpora.
#[test]
#[ignore = "requires the benchmark corpora under ~/.gigastt/benchmarks"]
fn corpus_over_threshold_scan() {
    let corpora = load_corpora();
    assert!(
        !corpora.is_empty(),
        "no manifests found in {}",
        manifest_dir().display()
    );

    println!(
        "\nchunked-path threshold: {} samples ({:.0}s @ {}kHz)\n",
        CHUNK_THRESHOLD_SAMPLES,
        THRESHOLD_SECONDS,
        SAMPLE_RATE / 1000,
    );
    println!(
        "{:<24} {:>6} {:>7} {:>8} {:>12} {:>10} {:>9} {:>8}",
        "manifest", "clips", "probed", "decoded", "undecodable", "malformed", "max_s", ">thresh",
    );
    println!("{}", "-".repeat(90));

    let mut total_over = 0usize;
    let mut total_probed = 0usize;
    let mut total_decoded = 0usize;
    let mut ready = 0usize;
    let mut unreadable: Vec<String> = Vec::new();
    let mut notes: Vec<String> = Vec::new();
    // Probed clips carried out of the loop so the boundary spot-check can pick the
    // ones nearest the threshold across the whole corpus set.
    let mut probed_clips: Vec<(String, PathBuf, f64)> = Vec::new();

    for c in &corpora {
        let (name, samples, malformed) = match c {
            Corpus::Unreadable { name, why } => {
                println!("{name:<24} UNREADABLE: {why}");
                unreadable.push(name.clone());
                continue;
            }
            Corpus::Ready {
                name,
                samples,
                malformed,
            } => (name, samples, malformed),
        };
        ready += 1;

        let (mut probed, mut decoded, mut over, mut max_s) = (0usize, 0usize, 0usize, 0.0f64);
        let mut bad: Vec<String> = Vec::new();
        for s in samples {
            match measure(&s.path) {
                Measure::Probed(seconds) => {
                    probed += 1;
                    max_s = max_s.max(seconds);
                    if seconds > THRESHOLD_SECONDS {
                        over += 1;
                    }
                    probed_clips.push((name.clone(), s.path.clone(), seconds));
                }
                Measure::Decoded(n) => {
                    decoded += 1;
                    max_s = max_s.max(n as f64 / SAMPLE_RATE as f64);
                    if n > CHUNK_THRESHOLD_SAMPLES {
                        over += 1;
                    }
                }
                Measure::Unmeasurable(e) => bad.push(format!("{}: {e}", s.path.display())),
            }
        }
        // clips == every entry the manifest declared. Nothing is unaccounted:
        // clips == probed + decoded + undecodable + malformed.
        let clips = samples.len() + malformed.len();
        total_over += over;
        total_probed += probed;
        total_decoded += decoded;

        println!(
            "{:<24} {:>6} {:>7} {:>8} {:>12} {:>10} {:>9.2} {:>8}",
            name,
            clips,
            probed,
            decoded,
            bad.len(),
            malformed.len(),
            max_s,
            over,
        );
        for b in bad.iter().take(5) {
            notes.push(format!(
                "[{name}] unmeasurable clip (NOT counted as short): {b}"
            ));
        }
        for m in malformed.iter().take(5) {
            notes.push(format!(
                "[{name}] malformed entry (NOT counted as a clip): {m}"
            ));
        }
    }

    println!();
    for n in &notes {
        println!("{n}");
    }

    // Cross-check the fast path against the ground truth on the clips where a
    // disagreement would actually matter: those nearest the 30 s boundary, where
    // header rounding could flip the over/under verdict. Bounded to a handful.
    spot_check_boundary(&mut probed_clips);

    let total_measured = total_probed + total_decoded;
    println!(
        "\n{ready} of {} manifest(s) readable; clips measured: {total_measured} \
         ({total_probed} by header probe, {total_decoded} by full decode); \
         over the chunked-path threshold: {total_over}",
        corpora.len(),
    );
    if !unreadable.is_empty() {
        println!("manifests NOT scanned (no data on this host): {unreadable:?}");
    }

    assert!(
        total_measured > 0,
        "scan measured nothing: {} manifest(s) present but none yielded a probe- or \
         decode-measurable clip (corpora absent on this host?). This scan must measure \
         real audio, not pass green having measured nothing.",
        corpora.len(),
    );
}

/// Decode the `SPOTCHECK_CLIPS` probed clips nearest the threshold and compare
/// each header duration against the real decode. Reports the largest observed
/// delta and any clip whose over/under verdict flips between the two paths — the
/// only place header rounding can change the headline count. `probed_clips` is
/// sorted in place by distance to the boundary.
fn spot_check_boundary(probed_clips: &mut [(String, PathBuf, f64)]) {
    probed_clips.sort_by(|a, b| {
        (a.2 - THRESHOLD_SECONDS)
            .abs()
            .total_cmp(&(b.2 - THRESHOLD_SECONDS).abs())
    });

    let mut checked = 0usize;
    let mut max_delta = 0.0f64;
    let mut flips: Vec<String> = Vec::new();
    let mut probe_only: Vec<String> = Vec::new();
    for (corpus, path, probe_s) in probed_clips.iter().take(SPOTCHECK_CLIPS) {
        match decode(path) {
            Ok(buf) => {
                checked += 1;
                let decode_s = buf.len() as f64 / SAMPLE_RATE as f64;
                max_delta = max_delta.max((decode_s - probe_s).abs());
                let probe_over = *probe_s > THRESHOLD_SECONDS;
                let decode_over = buf.len() > CHUNK_THRESHOLD_SAMPLES;
                if probe_over != decode_over {
                    flips.push(format!(
                        "[{corpus}] {} — probe {probe_s:.4}s (over={probe_over}) vs \
                         decode {decode_s:.4}s (over={decode_over})",
                        path.display(),
                    ));
                }
            }
            // The header parsed but the samples would not decode. Surprising and
            // worth surfacing, not swallowing.
            Err(e) => probe_only.push(format!(
                "[{corpus}] {} — header said {probe_s:.4}s but decode failed: {e}",
                path.display(),
            )),
        }
    }

    println!(
        "\nboundary spot-check: decoded {checked} of the probed clips nearest {THRESHOLD_SECONDS:.0}s"
    );
    if checked == 0 {
        println!("  no probe-answered clips to cross-check");
        return;
    }
    println!(
        "  largest |probe − decode| duration: {max_delta:.4}s; over/under verdict flips: {}",
        flips.len()
    );
    for f in &flips {
        println!("  FLIP {f}");
    }
    for f in &probe_only {
        println!("  PROBE-ONLY {f}");
    }
}

#[cfg(test)]
mod self_tests {
    use super::*;
    use std::fs;
    use tempfile::TempDir;

    /// Create `<tmp>/<name>/` and return its absolute path as a string, for use
    /// as a manifest `audio_root`. Manifests carry absolute (or `~/`-prefixed)
    /// roots, so the fixtures do too — a relative root would resolve against the
    /// test process CWD, not the temp dir.
    fn audio_root(tmp: &TempDir, name: &str) -> String {
        let p = tmp.path().join(name);
        fs::create_dir_all(&p).expect("create audio_root");
        p.to_string_lossy().into_owned()
    }

    fn write_manifest(tmp: &TempDir, stem: &str, body: &str) {
        fs::write(tmp.path().join(format!("{stem}.json")), body).expect("write manifest");
    }

    fn single(corpora: Vec<Corpus>) -> Corpus {
        assert_eq!(corpora.len(), 1, "expected exactly one manifest");
        corpora.into_iter().next().expect("one corpus")
    }

    #[test]
    fn ready_reports_every_entry_including_malformed_ones() {
        // Two usable entries and one entry with no filename. The malformed entry
        // must be reported, never filtered out of existence.
        let tmp = TempDir::new().expect("temp dir");
        let root = audio_root(&tmp, "audio");
        let body = serde_json::json!({
            "audio_root": root,
            "samples": [
                {"filename": "a.wav", "reference": "one"},
                {"reference": "no filename here"},
                {"filename": "b.wav"},
            ]
        })
        .to_string();
        write_manifest(&tmp, "mixed", &body);
        match single(load_corpora_in(tmp.path(), None)) {
            Corpus::Ready {
                samples, malformed, ..
            } => {
                assert_eq!(samples.len(), 2, "two usable filenames");
                assert_eq!(malformed.len(), 1, "the filename-less entry is reported");
                assert!(samples[0].path.ends_with("audio/a.wav"));
            }
            Corpus::Unreadable { why, .. } => panic!("should be readable, got: {why}"),
        }
    }

    #[test]
    fn absent_audio_root_is_unreadable_not_a_zero() {
        // audio_root names a directory that does not exist: the exact failure the
        // scan exists to catch. It must be Unreadable, not Ready with 0 clips.
        let tmp = TempDir::new().expect("temp dir");
        let missing = tmp.path().join("does_not_exist");
        let body = serde_json::json!({
            "audio_root": missing.to_string_lossy(),
            "samples": [{"filename": "a.wav"}],
        })
        .to_string();
        write_manifest(&tmp, "absent", &body);
        match single(load_corpora_in(tmp.path(), None)) {
            Corpus::Unreadable { why, .. } => assert!(
                why.contains("audio_root absent"),
                "reason should name the absent audio_root, got: {why}"
            ),
            Corpus::Ready { .. } => panic!("absent audio_root must not read as a zero-clip corpus"),
        }
    }

    #[test]
    fn unparseable_manifest_is_unreadable() {
        let tmp = TempDir::new().expect("temp dir");
        write_manifest(&tmp, "broken", "{ this is not json");
        match single(load_corpora_in(tmp.path(), None)) {
            Corpus::Unreadable { why, .. } => assert!(why.contains("parse failed"), "{why}"),
            Corpus::Ready { .. } => panic!("unparseable manifest must be Unreadable"),
        }
    }

    #[test]
    fn empty_samples_is_unreadable_not_a_zero() {
        let tmp = TempDir::new().expect("temp dir");
        let root = audio_root(&tmp, "audio");
        let body = serde_json::json!({"audio_root": root, "samples": []}).to_string();
        write_manifest(&tmp, "empty", &body);
        match single(load_corpora_in(tmp.path(), None)) {
            Corpus::Unreadable { why, .. } => assert!(why.contains("empty"), "{why}"),
            Corpus::Ready { .. } => panic!("empty samples[] must be Unreadable"),
        }
    }

    #[test]
    fn missing_manifest_dir_is_reported_not_silent() {
        let tmp = TempDir::new().expect("temp dir");
        let missing = tmp.path().join("no_such_subdir");
        match single(load_corpora_in(&missing, None)) {
            Corpus::Unreadable { why, .. } => {
                assert!(why.contains("manifest dir unreadable"), "{why}")
            }
            Corpus::Ready { .. } => panic!("a missing manifest dir must be reported"),
        }
    }

    #[test]
    fn manifest_filter_restricts_by_stem() {
        let tmp = TempDir::new().expect("temp dir");
        let root = audio_root(&tmp, "audio");
        for stem in ["keep", "drop"] {
            let body = serde_json::json!({"audio_root": root, "samples": [{"filename": "a.wav"}]})
                .to_string();
            write_manifest(&tmp, stem, &body);
        }
        let corpora = load_corpora_in(tmp.path(), Some(vec!["keep".to_string()]));
        assert_eq!(corpora.len(), 1);
        match &corpora[0] {
            Corpus::Ready { name, .. } => assert_eq!(name, "keep"),
            Corpus::Unreadable { why, .. } => panic!("{why}"),
        }
    }

    /// Pins the mirrored constant to the boundary the scan claims to measure, so
    /// a fat-fingered edit of the mirror is caught model-free.
    #[test]
    fn threshold_is_the_documented_thirty_seconds() {
        assert_eq!(CHUNK_THRESHOLD_SAMPLES, 30 * SAMPLE_RATE);
        assert_eq!(CHUNK_THRESHOLD_SAMPLES as f64 / SAMPLE_RATE as f64, 30.0);
    }
}
