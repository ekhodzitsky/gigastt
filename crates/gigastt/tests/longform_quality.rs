//! Long-form quality: what the overlapping-window stitch actually costs, and
//! how the encoder degrades with input length.
//!
//! # Why this file exists
//!
//! Everything in `docs/benchmarks.md` is measured on clips shorter than the
//! engine's 30 s single-pass threshold, so the chunked path — 24 s windows,
//! 2 s overlap, a time-cut stitch — has never been scored against a control
//! that isolates the stitch.
//!
//! The obvious control ("decode every clip on its own and sum the errors") is
//! wrong, and wrong by a lot. A concatenation of short unrelated clips loses
//! words *inside* a boundary-free 24 s span, with no seam anywhere near them:
//! the GigaAM encoder's accuracy falls off with input duration, and a per-clip
//! control charges that entire fall-off to the stitch. On a 25-minute
//! concatenation of ~2 s command clips that mis-attribution is worth tens of
//! WER points — a number about the fixture, not about the code.
//!
//! So this file changes two things:
//!
//! 1. **Corpus.** Acoustically continuous speech only: runs of consecutive
//!    segments of one audiobook chapter, one narrator, one channel, in their
//!    original order (see [`continuous_corpus`]). Never a shuffle of unrelated
//!    utterances.
//! 2. **Baseline.** The control decodes the *same* buffer in blocks sized to the
//!    24 s chunk window — held to `[18, 30]` s so their encoder-input length
//!    clusters near the window instead of scattering — cut only at segment
//!    boundaries so no word is ever split. An item whose utterance boundaries
//!    admit no in-band tiling is dropped from *both* paths rather than scored
//!    against a mismatched baseline. Both sides then see ~24 s of encoder input,
//!    and the difference between them is attributable to the stitch and only to
//!    the stitch.
//!
//! [`test_longform_stitch_cost`] reports chunked WER, segment baseline WER, their
//! delta, the deletion / substitution / insertion split behind each, and the
//! achieved baseline block-length distribution — so the residual encoder-length
//! confound is shown, not hidden. [`test_encoder_length_degradation_curve`]
//! records the effect that made the old per-clip baseline invalid, as a standing
//! measurement.
//!
//! # Measured on this machine
//!
//! 2026-07-28, Apple M1 Pro (macOS 25.1, arm64), `--release`, default CPU
//! execution provider, INT8 `rnnt` encoder (the head auto-detected on disk; each
//! test prints the head it actually loaded, so this line cannot silently drift
//! from the run). Corpus `~/.gigastt/benchmarks/ruls` (RuLS / OpenSLR 96): a
//! single LibriVox reading of Pushkin's «Поэмы» — one narrator, one book,
//! audiobook-read Russian *verse*, spread over 11 chapter sources. The stitch
//! cost below is the cost on clean single-speaker read speech and nothing else;
//! it does not carry to spontaneous, conversational, multi-speaker, far-field, or
//! noisy audio.
//!
//! WER here is verbatim / naive normalization (lowercase + `ё`→`е` +
//! `[a-zа-я0-9]`), so the absolute percentages are **not** comparable to the
//! ITN-normalized figures in `docs/benchmarks.md`. The stitch cost is a
//! *difference* between two decodes scored the same way, so it is
//! normalization-independent regardless.
//!
//! [`test_longform_stitch_cost`], 14 of 23 intake runs scored (9 dropped for
//! lacking an in-band baseline tiling), 21.5 min, 50 seams, 52 baseline blocks
//! (length 18.3 / 24.8 / 29.8 s min/median/max — clustered on the 24 s window):
//!
//! | | WER (naive-norm) | errors (del / sub / ins) | ref words |
//! |---|---|---|---|
//! | chunked path | 3.23% | 81 (8 / 70 / 3) | 2510 |
//! | 24 s-segment baseline | 3.35% | 84 (9 / 71 / 4) | 2510 |
//! | **stitch cost** | **−0.12 pp** | | |
//!
//! Controlling the baseline block length moved the headline off the +0.00 pp an
//! uncontrolled baseline reported, to −0.12 pp: length-matched, the chunked path
//! is marginally *better* than cutting the same buffer into 24 s blocks, so the
//! stitch still costs nothing — it does not lose to a clean-cut control. The two
//! paths remain different decodes (11 of 14 differ textually, the error mix
//! differs, per-item deltas scatter between −1.63 and +1.83 pp); they just cost
//! the same in total. One error is ~0.04 pp on this ref-word count, so the
//! headline is precise to about ±0.04 pp, and the ±1.8 pp per-item scatter is the
//! sampling noise a ceiling has to clear — hence the generous +2.0 pp default
//! `GIGASTT_LONGFORM_MAX_STITCH_PP`.
//!
//! [`test_encoder_length_degradation_curve`], 2 continuous runs pooled, one
//! encoder Run per point: one-pass WER 6.25% at 10 s → 2.21% at 30 s → 4.76% at
//! 59 s → 7.69% at 120 s, with word retention flat at ~98–102% throughout. Real
//! speech degrades with length, but by substituting, not by dropping words. The
//! golos concatenation over the same lengths collapses instead — retention 100%
//! at 9 s, 77% at 62 s, 9% at 92 s — which is the whole reason a per-clip control
//! on that fixture mis-attributes tens of points to the stitch.
//!
//! Re-measure before pinning a tighter `GIGASTT_LONGFORM_MAX_STITCH_PP` in CI:
//! the numbers are model-, encoder- (INT8 vs FP32) and EP-specific. On an ANE
//! build the chunked window is 30 s, not 24 s (see [`CHUNK_WINDOW_S`]); these
//! numbers are the CPU/ort path.
//!
//! # Running
//!
//! Both tests need the model (~850 MB) and the external RuLS benchmark corpus,
//! and both are `#[ignore]`d:
//!
//! ```sh
//! cargo test --release -p gigastt --test longform_quality -- --ignored --nocapture --test-threads=1
//! ```
//!
//! `--release` matters: the stitch test decodes the corpus twice.
//!
//! If the corpus is absent both tests **skip with a message naming exactly what
//! they wanted**. They never fall back to a different corpus: a number measured
//! on a different fixture is not the same measurement, and reporting it as if it
//! were is how the old baseline went wrong.
//!
//! # Environment variables
//!
//! - `GIGASTT_LONGFORM_MIN_ITEM_SECS` — shortest continuous run admitted into
//!   the corpus (default 60; must exceed the 30 s single-pass threshold so every
//!   item takes the chunked path).
//! - `GIGASTT_LONGFORM_MAX_ITEMS` — cap the item count for a fast smoke run
//!   (default: all).
//! - `GIGASTT_LONGFORM_MAX_STITCH_PP` — stitch-cost ceiling, in WER points: the
//!   gate fails when the cost exceeds it. Defaults to +2.0 pp (generous but real
//!   — above the worst single-item stitch, well below any genuine regression);
//!   set it lower to tighten, or very high to observe without gating.

mod common;

use gigastt::inference::audio::{decode_audio_file, encode_wav_pcm16};
use gigastt::inference::{Engine, SessionTriplet};
use serde::Deserialize;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

/// Sample rate of the whole pipeline; every decoded buffer is at this rate.
const SAMPLE_RATE: usize = 16000;

/// The engine's long-form chunk window (`chunk_window_samples` in
/// `crates/gigastt-core/src/inference/engine.rs`), mirrored here only to print
/// where seams land. The engine picks the true window at runtime from the loaded
/// encoder: 24 s on the ort/CPU path, 30 s on the ANE encoder. This mirror
/// follows the build's `ane` feature and is only a build-time upper bound — an
/// ANE binary can still fall back to the 24 s window at runtime (non-rnnt head,
/// missing ANE package). Every number recorded above was measured on the default
/// CPU path, where the window is 24 s. Nothing asserts on this constant, so a
/// mismatch only changes the printed seam estimate, never a pass/fail.
#[cfg(feature = "ane")]
const CHUNK_WINDOW_S: f64 = 30.0;
#[cfg(not(feature = "ane"))]
const CHUNK_WINDOW_S: f64 = 24.0;

/// Long-form overlap between adjacent chunks (`CHUNK_OVERLAP_SAMPLES`).
const CHUNK_OVERLAP_S: f64 = 2.0;

/// Inputs at or below this take one encoder Run over the whole buffer
/// (`CHUNK_THRESHOLD_SAMPLES`). Baseline blocks stay under it by construction,
/// so every baseline decode is a genuine single pass.
const SINGLE_PASS_MAX_S: f64 = 30.0;

/// Target length for a baseline block: the 24 s ort/CPU long-form window.
///
/// Held separately from [`CHUNK_WINDOW_S`] because a baseline block must stay a
/// single encoder pass (`<= SINGLE_PASS_MAX_S`), which leaves no room to target
/// the ANE path's 30 s window — so the single-pass baseline is a CPU/ort
/// comparison, matching the path these numbers were measured on. Both windows
/// coincide at 24 s on the default build.
const BASELINE_BLOCK_TARGET_S: f64 = 24.0;

/// Baseline blocks are held inside `[BASELINE_BLOCK_MIN_S, BASELINE_BLOCK_MAX_S]`
/// — the 24 s target ±6 s — so their encoder-input length clusters near the chunk
/// window instead of scattering from a few seconds up to the single-pass cap. A
/// shorter encoder input can decode differently, so an unconstrained tail block
/// would hand the baseline a length it never shares with the chunked path and
/// blur the stitch cost this file exists to isolate. The floor is the load-bearing
/// bound; the ceiling equals the single-pass threshold — a baseline block must
/// stay a single encoder pass — and is safe there because a block's length is an
/// exact integer sample count that the `EPS`-bounded planner never lets exceed
/// 30 s, with the `<= SINGLE_PASS_MAX_S` assert in the loop as a loud backstop.
/// Coarse-grained items (two utterances already over 30 s, no single one inside
/// the band) admit no tiling here and are dropped, not scored on a mismatched
/// baseline; the run reports how many.
const BASELINE_BLOCK_MIN_S: f64 = 18.0;
const BASELINE_BLOCK_MAX_S: f64 = 30.0;

/// Half-width of the window used to measure how quiet a baseline cut is.
const BOUNDARY_PROBE_S: f64 = 0.15;

// ---------------------------------------------------------------------------
// Corpus
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
struct Sample {
    filename: String,
    reference: String,
}

#[derive(Deserialize)]
struct Manifest {
    audio_root: String,
    samples: Vec<Sample>,
}

/// One segment of a continuous recording: its reference and its decoded 16 kHz
/// mono samples.
struct Segment {
    reference: String,
    samples: Vec<f32>,
}

impl Segment {
    fn seconds(&self) -> f64 {
        self.samples.len() as f64 / SAMPLE_RATE as f64
    }
}

/// A run of consecutive segments of one source (one chapter, one narrator, one
/// channel) in their original order — the unit this file scores.
struct Item {
    source: String,
    first: u32,
    last: u32,
    segments: Vec<Segment>,
}

impl Item {
    fn label(&self) -> String {
        format!("{} {:04}..{:04}", self.source, self.first, self.last)
    }

    fn samples(&self) -> Vec<f32> {
        let mut out = Vec::with_capacity(self.segments.iter().map(|s| s.samples.len()).sum());
        for seg in &self.segments {
            out.extend_from_slice(&seg.samples);
        }
        out
    }

    fn reference(&self) -> Vec<String> {
        self.segments
            .iter()
            .flat_map(|s| normalize_for_wer_naive(&s.reference))
            .collect()
    }

    fn seconds(&self) -> f64 {
        self.segments.iter().map(Segment::seconds).sum()
    }
}

/// Resolve a manifest's `audio_root`, expanding a leading `~`.
fn expand_home(raw: &str) -> Option<PathBuf> {
    match raw.strip_prefix("~/") {
        Some(rest) => common::home_dir().map(|h| h.join(rest)),
        None => Some(PathBuf::from(raw)),
    }
}

/// Split `name` into the source prefix and the trailing zero-padded segment
/// index, e.g. `poemi_02_pushkin_0104.wav` → (`poemi_02_pushkin`, 104).
fn split_indexed_name(name: &str) -> Option<(&str, u32)> {
    let stem = name.strip_suffix(".wav")?;
    let (prefix, index) = stem.rsplit_once('_')?;
    if index.is_empty() || !index.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    Some((prefix, index.parse().ok()?))
}

/// Sample rate, channel count and frame count of a WAV, read from the header
/// alone. Used to size candidate runs without decoding the whole corpus.
fn wav_header(path: &Path) -> Option<(u32, u16, usize)> {
    use std::io::Read;
    let mut head = [0u8; 4096];
    let mut file = std::fs::File::open(path).ok()?;
    let read = file.read(&mut head).ok()?;
    let head = &head[..read];
    if head.len() < 12 || &head[0..4] != b"RIFF" || &head[8..12] != b"WAVE" {
        return None;
    }
    let u16_at = |o: usize| u16::from_le_bytes([head[o], head[o + 1]]);
    let u32_at = |o: usize| u32::from_le_bytes([head[o], head[o + 1], head[o + 2], head[o + 3]]);

    let (mut rate, mut channels, mut bits) = (0u32, 0u16, 0u16);
    let mut pos = 12usize;
    while pos + 8 <= head.len() {
        let id = &head[pos..pos + 4];
        let size = u32_at(pos + 4) as usize;
        let body = pos + 8;
        if id == b"fmt " && body + 16 <= head.len() {
            channels = u16_at(body + 2);
            rate = u32_at(body + 4);
            bits = u16_at(body + 14);
        } else if id == b"data" {
            let bytes_per_frame = (bits as usize / 8) * channels.max(1) as usize;
            if bytes_per_frame == 0 || rate == 0 {
                return None;
            }
            return Some((rate, channels, size / bytes_per_frame));
        }
        pos = body + size + (size & 1);
    }
    None
}

/// Every maximal run of consecutive segment indices in the RuLS benchmark slice
/// that is at least `min_secs` long, decoded and returned in a deterministic
/// order.
///
/// RuLS (OpenSLR 96) is LibriVox-derived: `<book>_<chapter>_<reader>_<NNNN>.wav`,
/// where consecutive `NNNN` are consecutive utterances of one chapter read by
/// one narrator. Concatenating such a run reproduces a stretch of the original
/// recording — same voice, same room, same channel, continuous prose — which is
/// what makes the stitch the only thing that differs between the two decodes
/// this file compares.
///
/// The manifest is a seeded 1000-utterance slice, so index runs have gaps; only
/// gap-free runs are used, and each run is a separate item rather than being
/// spliced across a gap.
fn continuous_corpus(min_secs: f64) -> Result<Vec<Item>, String> {
    let manifest_path =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../benchmark/manifests/ruls.json");
    let raw = std::fs::read_to_string(&manifest_path)
        .map_err(|e| format!("read {}: {e}", manifest_path.display()))?;
    let manifest: Manifest = serde_json::from_str(&raw)
        .map_err(|e| format!("parse {}: {e}", manifest_path.display()))?;
    let root = expand_home(&manifest.audio_root)
        .ok_or_else(|| format!("cannot expand audio_root {}", manifest.audio_root))?;
    if !root.is_dir() {
        return Err(format!(
            "RuLS audio not found at {}. It is the corpus this gate is defined on \
             (consecutive utterances of one LibriVox chapter). Fetch it with \
             `python3 scripts/prepare_rulslib.py`; do not substitute another set.",
            root.display()
        ));
    }

    // Group by source prefix, keyed by segment index so runs come out sorted.
    let mut by_source: BTreeMap<&str, BTreeMap<u32, &Sample>> = BTreeMap::new();
    for sample in &manifest.samples {
        let name = Path::new(&sample.filename)
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or(&sample.filename);
        if let Some((prefix, index)) = split_indexed_name(name) {
            by_source.entry(prefix).or_default().insert(index, sample);
        }
    }

    // Maximal gap-free index runs, kept when long enough to reach the chunked path.
    let mut runs: Vec<(&str, Vec<(u32, &Sample)>)> = Vec::new();
    for (prefix, indexed) in &by_source {
        let mut current: Vec<(u32, &Sample)> = Vec::new();
        for (&index, &sample) in indexed {
            if current.last().is_some_and(|(prev, _)| *prev + 1 != index) {
                runs.push((prefix, std::mem::take(&mut current)));
            }
            current.push((index, sample));
        }
        if !current.is_empty() {
            runs.push((prefix, current));
        }
    }

    let mut items = Vec::new();
    let mut skipped_missing = 0usize;
    for (prefix, run) in runs {
        let paths: Vec<PathBuf> = run
            .iter()
            .map(|(_, s)| {
                let p = Path::new(&s.filename);
                if p.is_absolute() {
                    p.to_path_buf()
                } else {
                    root.join(p)
                }
            })
            .collect();
        if paths.iter().any(|p| !p.exists()) {
            skipped_missing += 1;
            continue;
        }
        let frames: Option<usize> = paths
            .iter()
            .map(|p| wav_header(p).map(|(_, _, frames)| frames))
            .sum();
        let Some(frames) = frames else { continue };
        // Header frame counts are at the file's own rate; the RuLS slice is
        // uniformly 16 kHz mono, and any outlier is filtered by the exact
        // duration check after decoding.
        if frames as f64 / (SAMPLE_RATE as f64) < min_secs {
            continue;
        }

        let mut segments = Vec::with_capacity(run.len());
        for (path, (_, sample)) in paths.iter().zip(&run) {
            let samples = decode_audio_file(&path.to_string_lossy())
                .map_err(|e| format!("decode {}: {e:#}", path.display()))?;
            segments.push(Segment {
                reference: sample.reference.clone(),
                samples,
            });
        }
        let item = Item {
            source: (*prefix).to_string(),
            first: run[0].0,
            last: run[run.len() - 1].0,
            segments,
        };
        if item.seconds() >= min_secs {
            items.push(item);
        }
    }

    if skipped_missing > 0 {
        eprintln!("  note: skipped {skipped_missing} run(s) with files missing on disk");
    }
    if items.is_empty() {
        return Err(format!(
            "no gap-free run of at least {min_secs:.0}s in {}; the slice is too sparse to score \
             the chunked path",
            root.display()
        ));
    }
    items.sort_by(|a, b| a.source.cmp(&b.source).then(a.first.cmp(&b.first)));
    Ok(items)
}

/// The golos command-clip set, concatenated in manifest order — the fixture the
/// invalid baseline was measured on. Kept for one purpose only: showing, in the
/// length curve, that it degrades with duration exactly like real continuous
/// speech does, which is why a per-clip control over-charges the stitch.
fn golos_concatenation(target_secs: f64) -> Result<Item, String> {
    let root = common::home_dir()
        .map(|h| h.join(".gigastt/benchmarks/golos_wav"))
        .ok_or_else(|| "cannot determine home directory".to_string())?;
    let manifest_path = root.join("manifest.json");
    let raw = std::fs::read_to_string(&manifest_path).map_err(|e| {
        format!(
            "read {}: {e}. The golos set is optional here; only the second half of \
             the length curve needs it.",
            manifest_path.display()
        )
    })?;
    let samples: Vec<Sample> = serde_json::from_str(&raw)
        .map_err(|e| format!("parse {}: {e}", manifest_path.display()))?;

    let mut segments: Vec<Segment> = Vec::new();
    let mut total = 0.0f64;
    for sample in &samples {
        if total >= target_secs {
            break;
        }
        let path = Path::new(&sample.filename);
        let path = if path.is_absolute() {
            path.to_path_buf()
        } else {
            root.join(path)
        };
        let decoded = decode_audio_file(&path.to_string_lossy())
            .map_err(|e| format!("decode {}: {e:#}", path.display()))?;
        total += decoded.len() as f64 / SAMPLE_RATE as f64;
        segments.push(Segment {
            reference: sample.reference.clone(),
            samples: decoded,
        });
    }
    if segments.is_empty() {
        return Err(format!("no usable clips under {}", root.display()));
    }
    Ok(Item {
        source: "golos concat".to_string(),
        first: 0,
        last: segments.len() as u32 - 1,
        segments,
    })
}

// ---------------------------------------------------------------------------
// WER
// ---------------------------------------------------------------------------

/// Verbatim normalization, character-for-character the benchmark harness's
/// `normalize_for_wer_naive`: lowercase, `ё`→`е`, keep only `[a-zа-я0-9]` plus
/// whitespace, split. Every number this file reports is a *difference* between
/// two decodes scored against the same reference with the same normalization,
/// so the writing-convention pass (ITN, digit merging, transliteration) cancels
/// out; leaving it out keeps this file free of the benchmark's number tables.
fn normalize_for_wer_naive(text: &str) -> Vec<String> {
    let text = text.to_lowercase();
    let text = text.replace('ё', "е");
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

/// Insertions, deletions and substitutions turning `reference` into
/// `hypothesis`, and their sum (the word error count). One Levenshtein
/// backtrace, so the split always adds up to the headline error count.
#[derive(Clone, Copy, Default)]
struct Errors {
    ins: usize,
    del: usize,
    sub: usize,
}

impl Errors {
    fn total(&self) -> usize {
        self.ins + self.del + self.sub
    }
}

impl std::ops::AddAssign for Errors {
    fn add_assign(&mut self, rhs: Self) {
        self.ins += rhs.ins;
        self.del += rhs.del;
        self.sub += rhs.sub;
    }
}

fn word_errors(reference: &[String], hypothesis: &[String]) -> Errors {
    let m = reference.len();
    let n = hypothesis.len();
    let mut d = vec![vec![0usize; n + 1]; m + 1];
    for (i, row) in d.iter_mut().enumerate() {
        row[0] = i;
    }
    for (j, cell) in d[0].iter_mut().enumerate() {
        *cell = j;
    }
    for i in 1..=m {
        for j in 1..=n {
            d[i][j] = if reference[i - 1] == hypothesis[j - 1] {
                d[i - 1][j - 1]
            } else {
                1 + d[i - 1][j - 1].min(d[i - 1][j]).min(d[i][j - 1])
            };
        }
    }
    let (mut i, mut j) = (m, n);
    let mut e = Errors::default();
    while i > 0 || j > 0 {
        if i > 0 && j > 0 && reference[i - 1] == hypothesis[j - 1] && d[i][j] == d[i - 1][j - 1] {
            i -= 1;
            j -= 1;
        } else if i > 0 && j > 0 && d[i][j] == d[i - 1][j - 1] + 1 {
            e.sub += 1;
            i -= 1;
            j -= 1;
        } else if j > 0 && d[i][j] == d[i][j - 1] + 1 {
            e.ins += 1;
            j -= 1;
        } else {
            e.del += 1;
            i -= 1;
        }
    }
    e
}

fn wer_pct(errors: usize, ref_words: usize) -> f64 {
    if ref_words == 0 {
        0.0
    } else {
        errors as f64 / ref_words as f64 * 100.0
    }
}

// ---------------------------------------------------------------------------
// Decoding helpers
// ---------------------------------------------------------------------------

fn load_engine() -> Engine {
    Engine::load(&common::model_dir()).expect("load engine")
}

/// Decode a buffer through the ordinary file entry point — the path a user
/// hits. Buffers over the 30 s threshold take the chunked branch; shorter ones
/// take the single-pass branch.
fn decode_via_file(engine: &Engine, samples: &[f32], triplet: &mut SessionTriplet) -> String {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("audio.wav");
    std::fs::write(&path, encode_wav_pcm16(samples, SAMPLE_RATE as u32)).expect("write wav");
    engine
        .transcribe_file(path.to_string_lossy().as_ref(), triplet)
        .expect("transcribe")
        .text
}

/// Decode a buffer with **exactly one encoder Run over the whole buffer**, at
/// any length.
///
/// The file entry point cannot do this above 30 s — that is precisely the
/// branch the length curve exists to justify — and the single-pass branch is
/// private. The streaming entry point can: `process_chunk` appends to the
/// session's window and re-runs the encoder over everything retained, so a
/// single oversized chunk is one Run over the whole buffer. The window cap only
/// takes effect *after* that decode, and it commits rather than discards, so no
/// words are lost.
///
/// This is a different decode configuration from the file path (streaming pad
/// floor, assembler-built text), so its absolute WER is not comparable with the
/// file path's. The curve compares it only against itself across lengths, and
/// prints the file path's number alongside wherever both are reachable.
fn decode_single_encoder_pass(
    engine: &Engine,
    samples: &[f32],
    triplet: &mut SessionTriplet,
) -> String {
    let mut state = engine.create_state(false);
    let mut parts: Vec<String> = engine
        .process_chunk(samples, &mut state, triplet)
        .expect("process_chunk")
        .into_iter()
        .filter(|seg| seg.is_final)
        .map(|seg| seg.text)
        .collect();
    if let Some(seg) = engine.finish_stream(&mut state, triplet) {
        parts.push(seg.text);
    }
    parts.join(" ")
}

// ---------------------------------------------------------------------------
// Segment baseline
// ---------------------------------------------------------------------------

/// Split `durations` into consecutive blocks whose lengths sit as close to
/// `target` as possible, with every block length inside `[min_len, max_len]`.
///
/// Exact (a small dynamic program over segment boundaries) rather than greedy,
/// because a greedy pack lands systematically *below* the target — and a shorter
/// encoder input decodes better, which would silently inflate the stitch cost
/// this baseline exists to isolate. The `min_len` floor is what stops a leftover
/// tail from decoding as an easy ten-second block; the `max_len` ceiling keeps
/// every block a single encoder pass.
///
/// Returns block boundaries as `[start, end)` index pairs, or `None` when no
/// tiling keeps every block inside the band — e.g. a single segment already
/// exceeds `max_len`, or a residual can never reach `min_len`. The caller drops
/// such an item rather than scoring it against a mismatched baseline.
fn plan_blocks(
    durations: &[f64],
    target: f64,
    min_len: f64,
    max_len: f64,
) -> Option<Vec<(usize, usize)>> {
    let n = durations.len();
    if n == 0 {
        return Some(Vec::new());
    }
    // Tiny slack so a block whose segments sum to exactly `min_len` / `max_len`
    // in exact arithmetic is not rejected by floating-point drift. At 1e-6 s it is
    // ~0.016 of a sample: a block whose true (integer-sample) length is one sample
    // over `max_len` still measures more than `EPS` past it and is rejected, so a
    // block can never round its way past the single-pass threshold.
    const EPS: f64 = 1e-6;
    let mut best = vec![f64::INFINITY; n + 1];
    let mut prev = vec![usize::MAX; n + 1];
    best[0] = 0.0;
    for end in 1..=n {
        let mut len = 0.0;
        for start in (0..end).rev() {
            len += durations[start];
            if len > max_len + EPS {
                break; // every smaller `start` only makes the block longer
            }
            if len + EPS < min_len {
                continue; // too short; a larger block (smaller `start`) may qualify
            }
            if best[start].is_finite() {
                let deviation = len - target;
                let cost = best[start] + deviation * deviation;
                if cost < best[end] {
                    best[end] = cost;
                    prev[end] = start;
                }
            }
        }
    }
    if !best[n].is_finite() {
        return None;
    }
    let mut blocks = Vec::new();
    let mut end = n;
    while end > 0 {
        let start = prev[end];
        blocks.push((start, end));
        end = start;
    }
    blocks.reverse();
    Some(blocks)
}

/// Root-mean-square level of `samples`.
fn rms(samples: &[f32]) -> f64 {
    if samples.is_empty() {
        return 0.0;
    }
    let sum: f64 = samples.iter().map(|s| (*s as f64) * (*s as f64)).sum();
    (sum / samples.len() as f64).sqrt()
}

/// How quiet each interior baseline cut is, as a fraction of the item's overall
/// level, measured over ±[`BOUNDARY_PROBE_S`] around the cut.
///
/// No word can be split by these cuts *by construction* — they fall on the
/// corpus's own utterance boundaries, and each side's reference is a whole
/// utterance. This measures the acoustic margin the corpus leaves around those
/// boundaries, which is what makes the comparison with a clock-driven chunk
/// boundary meaningful: values well under 1.0 mean the corpus pads its cuts with
/// silence, values near 1.0 mean it trims them tight.
fn boundary_levels(item: &Item, blocks: &[(usize, usize)]) -> Vec<f64> {
    let buffer = item.samples();
    let overall = rms(&buffer);
    if overall <= 0.0 {
        return Vec::new();
    }
    let mut offsets = Vec::new();
    let mut acc = 0usize;
    for seg in &item.segments {
        acc += seg.samples.len();
        offsets.push(acc);
    }
    let probe = (BOUNDARY_PROBE_S * SAMPLE_RATE as f64) as usize;
    blocks
        .iter()
        .skip(1)
        .map(|(start, _)| {
            let cut = offsets[start - 1];
            let lo = cut.saturating_sub(probe);
            let hi = (cut + probe).min(buffer.len());
            rms(&buffer[lo..hi]) / overall
        })
        .collect()
}

// ---------------------------------------------------------------------------
// 1. The gate: what the stitch costs
// ---------------------------------------------------------------------------

/// Score the chunked path against a control that differs from it *only* in the
/// stitch.
///
/// Both sides decode the same continuous buffer and are scored against the same
/// reference. The chunked side is one `transcribe_file` call over the whole
/// item, which takes the overlapping-window path. The baseline side decodes the
/// item in blocks sized like a chunk window (~24 s, never over the 30 s
/// single-pass threshold) cut only at segment boundaries, so every baseline
/// decode is a single encoder pass over a comparable input length and no word is
/// ever split by a cut.
///
/// What is left in the difference: the 2 s overlap, the time-cut stitch, and the
/// fact that a chunk boundary lands wherever the clock says rather than at a
/// pause. That is the honest stitch cost.
#[ignore]
#[test]
fn test_longform_stitch_cost() {
    let min_item_secs: f64 = std::env::var("GIGASTT_LONGFORM_MIN_ITEM_SECS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(60.0f64)
        .max(SINGLE_PASS_MAX_S + 1.0);

    let mut items = match continuous_corpus(min_item_secs) {
        Ok(items) => items,
        Err(why) => {
            eprintln!("SKIP test_longform_stitch_cost: {why}");
            return;
        }
    };
    if let Some(max) = std::env::var("GIGASTT_LONGFORM_MAX_ITEMS")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .filter(|n| *n > 0)
    {
        items.truncate(max);
    }

    // Corpus composition, self-reported so the provenance cannot drift from what
    // actually ran. RuLS (OpenSLR 96) is LibriVox-derived; this slice is one
    // author — Pushkin's «Поэмы» — read aloud, so every item is clean,
    // single-speaker, audiobook Russian *verse*. The stitch cost measured here is
    // the cost on that material and nothing else: it does not carry to
    // spontaneous, conversational, multi-speaker, far-field, or noisy audio.
    let sources: std::collections::BTreeSet<&str> =
        items.iter().map(|i| i.source.as_str()).collect();
    let intake_secs: f64 = items.iter().map(Item::seconds).sum();
    eprintln!(
        "  corpus: RuLS (Russian LibriVox / OpenSLR 96), audiobook-read Russian verse — \
         one narrator, one book (Pushkin «Поэмы»)"
    );
    eprintln!(
        "  intake: {} continuous run(s) across {} chapter source(s), {:.1} min total; clean \
         single-speaker read speech only — does NOT generalize to spontaneous, conversational, \
         multi-speaker, far-field, or noisy audio",
        items.len(),
        sources.len(),
        intake_secs / 60.0
    );

    let engine = load_engine();
    eprintln!(
        "  model head (loaded at runtime): {} encoder ({})",
        engine.variant().as_str(),
        if engine.is_int8() { "INT8" } else { "FP32" }
    );
    let mut guard = engine.pool.checkout_blocking().expect("pool checkout");

    let stride = CHUNK_WINDOW_S - CHUNK_OVERLAP_S;
    let mut ref_words = 0usize;
    let mut chunked_errors = Errors::default();
    let mut baseline_errors = Errors::default();
    let mut chunked_hyp_words = 0usize;
    let mut baseline_hyp_words = 0usize;
    let mut quietest = f64::INFINITY;
    let mut loudest_cut = 0.0f64;
    let mut cuts = 0usize;
    let mut identical = 0usize;
    let mut scored = 0usize;
    let mut dropped = 0usize;
    let mut seams = 0usize;
    let mut kept_secs = 0.0f64;
    let mut block_lengths: Vec<f64> = Vec::new();

    for item in &items {
        assert!(
            item.seconds() > SINGLE_PASS_MAX_S,
            "{} is {:.1}s — it would take the single-pass branch, not the chunked one",
            item.label(),
            item.seconds()
        );

        // Plan the word-safe baseline blocks first, so an item whose utterance
        // boundaries admit no tiling inside the length band is dropped from BOTH
        // paths — the two aggregates always cover exactly the same audio.
        let durations: Vec<f64> = item.segments.iter().map(Segment::seconds).collect();
        let Some(blocks) = plan_blocks(
            &durations,
            BASELINE_BLOCK_TARGET_S,
            BASELINE_BLOCK_MIN_S,
            BASELINE_BLOCK_MAX_S,
        ) else {
            dropped += 1;
            eprintln!(
                "  DROP {:<32} {:6.1}s — no word-safe baseline block in [{:.0}, {:.0}]s fits its \
                 utterance boundaries",
                item.label(),
                item.seconds(),
                BASELINE_BLOCK_MIN_S,
                BASELINE_BLOCK_MAX_S,
            );
            continue;
        };
        for (start, end) in &blocks {
            block_lengths.push(durations[*start..*end].iter().sum());
        }

        let reference = item.reference();
        let buffer = item.samples();

        // --- chunked path ---------------------------------------------------
        let chunked = normalize_for_wer_naive(&decode_via_file(&engine, &buffer, &mut guard));

        // --- segment baseline -----------------------------------------------
        for level in boundary_levels(item, &blocks) {
            quietest = quietest.min(level);
            loudest_cut = loudest_cut.max(level);
            cuts += 1;
        }
        let mut baseline_text = Vec::with_capacity(blocks.len());
        for (start, end) in &blocks {
            let block: Vec<f32> = item.segments[*start..*end]
                .iter()
                .flat_map(|s| s.samples.iter().copied())
                .collect();
            let block_secs = block.len() as f64 / SAMPLE_RATE as f64;
            assert!(
                block_secs <= SINGLE_PASS_MAX_S,
                "baseline block {block_secs:.1}s would take the chunked path"
            );
            baseline_text.push(decode_via_file(&engine, &block, &mut guard));
        }
        let baseline = normalize_for_wer_naive(&baseline_text.join(" "));

        let ce = word_errors(&reference, &chunked);
        let be = word_errors(&reference, &baseline);
        if chunked == baseline {
            identical += 1;
        }
        eprintln!(
            "  {:<34} {:6.1}s ref {:4}w | chunked {:5.2}% (d{} s{} i{}) | baseline {:5.2}% (d{} s{} i{}) | {:+.2} pp",
            item.label(),
            item.seconds(),
            reference.len(),
            wer_pct(ce.total(), reference.len()),
            ce.del,
            ce.sub,
            ce.ins,
            wer_pct(be.total(), reference.len()),
            be.del,
            be.sub,
            be.ins,
            wer_pct(ce.total(), reference.len()) - wer_pct(be.total(), reference.len()),
        );

        ref_words += reference.len();
        chunked_hyp_words += chunked.len();
        baseline_hyp_words += baseline.len();
        chunked_errors += ce;
        baseline_errors += be;
        scored += 1;
        kept_secs += item.seconds();
        seams += ((item.seconds() - CHUNK_WINDOW_S) / stride).ceil().max(0.0) as usize;
    }

    assert!(ref_words > 0, "empty reference");
    assert!(
        scored > 0,
        "every one of {} intake item(s) was dropped for lacking a word-safe baseline tiling",
        items.len()
    );
    let chunked_wer = wer_pct(chunked_errors.total(), ref_words);
    let baseline_wer = wer_pct(baseline_errors.total(), ref_words);
    let stitch_pp = chunked_wer - baseline_wer;

    eprintln!(
        "\n  chunked path      WER {chunked_wer:6.2}%  ({} err = {}d {}s {}i / {ref_words} ref words, {chunked_hyp_words} hyp words)",
        chunked_errors.total(),
        chunked_errors.del,
        chunked_errors.sub,
        chunked_errors.ins,
    );
    eprintln!(
        "  segment baseline  WER {baseline_wer:6.2}%  ({} err = {}d {}s {}i / {ref_words} ref words, {baseline_hyp_words} hyp words)",
        baseline_errors.total(),
        baseline_errors.del,
        baseline_errors.sub,
        baseline_errors.ins,
    );
    eprintln!("  stitch cost       {stitch_pp:+6.2} pp");
    eprintln!(
        "  identical output  {identical}/{scored} scored item(s) decoded word-for-word the same on both paths"
    );
    if cuts > 0 {
        eprintln!(
            "  baseline cuts     {cuts} at corpus utterance boundaries (no word split by \
             construction); acoustic margin vs item RMS: quietest {quietest:.3}, loudest {loudest_cut:.3}"
        );
    }

    eprintln!(
        "  scored / dropped  {scored} scored, {dropped} dropped of {} intake item(s)",
        items.len()
    );

    // The residual encoder-length confound, shown rather than hidden: how tightly
    // the baseline blocks actually clustered around the chunk window. The chunked
    // path always encodes ~24 s; the closer this spread sits to 24 s, the less a
    // shorter-input advantage can be hiding inside the baseline WER.
    block_lengths.sort_by(|a, b| a.partial_cmp(b).expect("no NaN block length"));
    let (bmin, bmed, bmax) = block_length_stats(&block_lengths);
    eprintln!(
        "  baseline blocks   {} block(s), length min {bmin:.1}s / median {bmed:.1}s / max {bmax:.1}s \
         (target {:.0}s, band [{:.0}, {:.0}]s)",
        block_lengths.len(),
        BASELINE_BLOCK_TARGET_S,
        BASELINE_BLOCK_MIN_S,
        BASELINE_BLOCK_MAX_S,
    );

    // WER below is VERBATIM / naive normalization (lowercase + `ё`→`е` +
    // `[a-zа-я0-9]`), NOT the ITN-normalized WER `docs/benchmarks.md` reports. The
    // stitch cost is a difference between two decodes scored the same way, so it is
    // normalization-independent; the absolute percentages are not, and must not be
    // compared against that document's figures.
    println!(
        "\n(WER columns are verbatim / naive-norm — not comparable to docs/benchmarks.md ITN-normalized WER.)"
    );
    println!(
        "\n| corpus | scored length | seams | chunked WER (naive-norm) | 24s-segment baseline WER (naive-norm) | stitch cost | baseline block s (min/med/max) |"
    );
    println!(
        "|--------|---------------|-------|--------------------------|---------------------------------------|-------------|--------------------------------|"
    );
    println!(
        "| RuLS Pushkin verse ×{scored} run(s) | {:.1} min | {seams} | {chunked_wer:.2}% | {baseline_wer:.2}% | {stitch_pp:+.2} pp | {bmin:.1}/{bmed:.1}/{bmax:.1} |",
        kept_secs / 60.0,
    );

    // A stitch that ate half the transcript would still score plausibly on a
    // deletion-heavy reference, so check the word count too.
    assert!(
        chunked_hyp_words as f64 >= baseline_hyp_words as f64 * 0.5,
        "chunked decode emitted {chunked_hyp_words} words vs {baseline_hyp_words} baseline — \
         the stitch lost most of the audio"
    );

    // Default gate: generous but real. The worst single-item stitch measured on
    // this corpus was +1.83 pp (per-item spread −1.63..+1.83 pp), so +2.0 pp clears
    // the sampling scatter of any one item while still catching a genuine
    // regression — a broken stitch drops words wholesale and blows past this by
    // tens of points. Override with GIGASTT_LONGFORM_MAX_STITCH_PP to tighten, or
    // set a large value to observe without gating.
    const DEFAULT_MAX_STITCH_PP: f64 = 2.0;
    let max_pp = std::env::var("GIGASTT_LONGFORM_MAX_STITCH_PP")
        .ok()
        .and_then(|s| s.parse::<f64>().ok())
        .unwrap_or(DEFAULT_MAX_STITCH_PP);
    assert!(
        stitch_pp <= max_pp,
        "stitch cost {stitch_pp:+.2} pp exceeds the ceiling {max_pp:.2} pp"
    );
}

/// Min, median and max of a sorted, non-empty slice of block lengths; zeros for
/// an empty slice (no scored item ever produces one, but the report must not
/// panic if a future filter leaves it empty).
fn block_length_stats(sorted: &[f64]) -> (f64, f64, f64) {
    if sorted.is_empty() {
        return (0.0, 0.0, 0.0);
    }
    let mid = sorted.len() / 2;
    let median = if sorted.len().is_multiple_of(2) {
        (sorted[mid - 1] + sorted[mid]) / 2.0
    } else {
        sorted[mid]
    };
    (sorted[0], median, sorted[sorted.len() - 1])
}

// ---------------------------------------------------------------------------
// 2. The standing measurement: encoder accuracy vs input length
// ---------------------------------------------------------------------------

/// Input lengths sampled by the curve, in seconds. Straddles both engine
/// constants: the 24 s chunk window and the 30 s single-pass threshold.
const CURVE_LENGTHS_S: [f64; 8] = [10.0, 20.0, 24.0, 30.0, 45.0, 60.0, 90.0, 120.0];

/// Word retention and WER as a function of how much audio one encoder Run sees.
///
/// This is the finding that invalidated the old per-clip baseline, so it is
/// recorded here as a measurement rather than a comment: accuracy falls off with
/// input duration on its own, seams or no seams. It is also the justification
/// for both constants — the encoder must not be handed the whole file (hence the
/// 30 s threshold) and the window must stay short (hence 24 s).
///
/// Run on two corpora, because the fall-off is not a fixture artifact:
/// continuous audiobook speech and the golos command-clip concatenation.
///
/// Prints a paste-ready markdown table for `docs/benchmarks.md`.
#[ignore]
#[test]
fn test_encoder_length_degradation_curve() {
    let longest = CURVE_LENGTHS_S.last().copied().unwrap_or(120.0);

    let real = match continuous_corpus(longest) {
        Ok(items) => items,
        Err(why) => {
            eprintln!("SKIP test_encoder_length_degradation_curve: {why}");
            return;
        }
    };
    if real.is_empty() {
        eprintln!(
            "SKIP test_encoder_length_degradation_curve: no continuous RuLS run reaches \
             {longest:.0}s, so the curve cannot be measured to its last point"
        );
        return;
    }
    eprintln!(
        "  real corpus: {} continuous run(s) reaching {longest:.0}s — {}",
        real.len(),
        real.iter().map(Item::label).collect::<Vec<_>>().join(", ")
    );

    let engine = load_engine();
    eprintln!(
        "  model head (loaded at runtime): {} encoder ({})",
        engine.variant().as_str(),
        if engine.is_int8() { "INT8" } else { "FP32" }
    );
    let mut guard = engine.pool.checkout_blocking().expect("pool checkout");

    let mut rows = Vec::new();
    rows.extend(curve_rows(&engine, &mut guard, "RuLS continuous", &real));

    match golos_concatenation(longest) {
        Ok(golos) => rows.extend(curve_rows(
            &engine,
            &mut guard,
            "golos concat",
            std::slice::from_ref(&golos),
        )),
        Err(why) => eprintln!("  note: golos half of the curve skipped: {why}"),
    }

    println!(
        "\n(WER columns are verbatim / naive-norm — not comparable to docs/benchmarks.md ITN-normalized WER.)"
    );
    println!(
        "\n| corpus | input length | ref words | one-pass hyp words | retention | one-pass WER (naive-norm) | file-path WER (naive-norm) |"
    );
    println!(
        "|--------|--------------|-----------|--------------------|-----------|---------------------------|----------------------------|"
    );
    for row in &rows {
        println!("{row}");
    }
}

/// Number of leading segments of `item` whose total duration lands closest to
/// `target`, and that duration.
///
/// Prefixes end on a segment boundary so the reference stays exact — no word is
/// half-spoken at the cut — which is why the achieved length is reported rather
/// than the requested one.
fn prefix_closest_to(item: &Item, target: f64) -> (usize, f64) {
    let (mut count, mut secs) = (0usize, 0.0f64);
    let (mut best_count, mut best_secs, mut best_gap) = (0usize, 0.0f64, f64::INFINITY);
    for segment in &item.segments {
        secs += segment.seconds();
        count += 1;
        let gap = (secs - target).abs();
        if gap < best_gap {
            best_gap = gap;
            best_count = count;
            best_secs = secs;
        }
        if secs > target {
            break;
        }
    }
    (best_count, best_secs)
}

/// Decode growing prefixes of every item and format one markdown row per length,
/// pooling errors across items so a single chapter's quirks do not carry the
/// curve.
fn curve_rows(
    engine: &Engine,
    triplet: &mut SessionTriplet,
    label: &str,
    items: &[Item],
) -> Vec<String> {
    let mut rows = Vec::new();
    let mut seen: Vec<Vec<usize>> = Vec::new();

    for target in CURVE_LENGTHS_S {
        let prefixes: Vec<(usize, f64)> = items
            .iter()
            .map(|item| prefix_closest_to(item, target))
            .collect();
        let counts: Vec<usize> = prefixes.iter().map(|(c, _)| *c).collect();
        if counts.iter().all(|c| *c == 0) || seen.contains(&counts) {
            continue;
        }
        seen.push(counts);

        let mut ref_words = 0usize;
        let mut one_pass_words = 0usize;
        let mut one_pass_errors = 0usize;
        let mut file_errors = 0usize;
        let mut total_secs = 0.0f64;
        let mut measured = 0usize;
        // The file entry point branches per item on that item's achieved prefix
        // length, so a row pooling items on both sides of the 30 s threshold is
        // genuinely mixed. Count the branch each decode actually took rather than
        // labelling the whole row from its mean length.
        let mut single_pass_decodes = 0usize;
        let mut chunked_decodes = 0usize;

        for (item, (count, secs)) in items.iter().zip(&prefixes) {
            if *count == 0 {
                continue;
            }
            let reference: Vec<String> = item.segments[..*count]
                .iter()
                .flat_map(|s| normalize_for_wer_naive(&s.reference))
                .collect();
            let buffer: Vec<f32> = item.segments[..*count]
                .iter()
                .flat_map(|s| s.samples.iter().copied())
                .collect();

            let one_pass =
                normalize_for_wer_naive(&decode_single_encoder_pass(engine, &buffer, triplet));
            let file_hyp = normalize_for_wer_naive(&decode_via_file(engine, &buffer, triplet));

            ref_words += reference.len();
            one_pass_words += one_pass.len();
            one_pass_errors += word_errors(&reference, &one_pass).total();
            file_errors += word_errors(&reference, &file_hyp).total();
            total_secs += secs;
            measured += 1;
            if *secs <= SINGLE_PASS_MAX_S {
                single_pass_decodes += 1;
            } else {
                chunked_decodes += 1;
            }
        }
        if measured == 0 || ref_words == 0 {
            continue;
        }

        let secs = total_secs / measured as f64;
        let one_pass_wer = wer_pct(one_pass_errors, ref_words);
        let file_wer = wer_pct(file_errors, ref_words);
        let retention = one_pass_words as f64 / ref_words as f64 * 100.0;
        // Label the file-path cell by the branch each pooled decode actually took,
        // so a row straddling the 30 s threshold reads as mixed instead of pinning
        // itself to whichever side the mean happened to fall on.
        let file_cell = match (single_pass_decodes, chunked_decodes) {
            (_, 0) => format!("{file_wer:.2}% (single pass)"),
            (0, _) => format!("{file_wer:.2}% (chunked)"),
            (s, c) => format!("{file_wer:.2}% (mixed: {s} single-pass / {c} chunked)"),
        };

        eprintln!(
            "  {label}: {secs:6.2}s ×{measured} ref {ref_words:4}w one-pass {one_pass_words:4}w \
             retention {retention:6.2}% WER {one_pass_wer:6.2}% | file {file_cell}"
        );
        rows.push(format!(
            "| {label} | {secs:.1} s | {ref_words} | {one_pass_words} | {retention:.1}% | {one_pass_wer:.2}% | {file_cell} |"
        ));
    }
    rows
}

// ---------------------------------------------------------------------------
// Unit tests for the harness itself (no model, no corpus)
// ---------------------------------------------------------------------------

#[test]
fn test_split_indexed_name_parses_ruls_layout() {
    assert_eq!(
        split_indexed_name("poemi_02_pushkin_0104.wav"),
        Some(("poemi_02_pushkin", 104))
    );
    assert_eq!(split_indexed_name("poemi_02_pushkin.wav"), None);
    assert_eq!(split_indexed_name("clip_abcd.wav"), None);
    assert_eq!(split_indexed_name("poemi_0001.flac"), None);
}

#[test]
fn test_plan_blocks_stays_under_cap_and_covers_everything() {
    let durations = vec![7.0; 20];
    let blocks = plan_blocks(&durations, 24.0, 18.0, 30.0).expect("plannable");
    assert_eq!(blocks.first().map(|b| b.0), Some(0));
    assert_eq!(blocks.last().map(|b| b.1), Some(20));
    for w in blocks.windows(2) {
        assert_eq!(w[0].1, w[1].0, "blocks must tile without gaps or overlap");
    }
    for (start, end) in &blocks {
        let len: f64 = durations[*start..*end].iter().sum();
        assert!(len <= 30.0, "block of {len}s exceeds the cap");
    }
}

#[test]
fn test_plan_blocks_targets_the_window_length() {
    // Greedy packing of 7s segments under a 24s target yields 21s blocks; the
    // dynamic program should prefer 28s ones, which sit closer to 24.
    let durations = vec![7.0; 8];
    let blocks = plan_blocks(&durations, 24.0, 18.0, 30.0).expect("plannable");
    let lengths: Vec<f64> = blocks
        .iter()
        .map(|(s, e)| durations[*s..*e].iter().sum())
        .collect();
    let worst = lengths
        .iter()
        .map(|l| (l - 24.0f64).abs())
        .fold(0.0f64, f64::max);
    assert!(worst <= 4.0, "blocks {lengths:?} stray far from the target");
}

#[test]
fn test_plan_blocks_enforces_the_min_floor() {
    // 10s segments: a block must be >= 18s (two segments) and <= 30s (three), so
    // the planner can never leave a lone 10s tail — the short-block confound the
    // floor exists to prevent. 70s tiles as 20 + 20 + 30 (all inside the band).
    let durations = vec![10.0; 7];
    let blocks = plan_blocks(&durations, 24.0, 18.0, 30.0).expect("plannable");
    assert_eq!(
        blocks.last().map(|b| b.1),
        Some(7),
        "must cover every segment"
    );
    for (start, end) in &blocks {
        let len: f64 = durations[*start..*end].iter().sum();
        assert!(
            (18.0..=30.0).contains(&len),
            "block of {len}s escaped the [18, 30]s band"
        );
    }
}

#[test]
fn test_plan_blocks_rejects_an_unsplittable_segment() {
    assert!(plan_blocks(&[45.0], 24.0, 18.0, 30.0).is_none());
}

#[test]
fn test_plan_blocks_rejects_when_a_residual_cannot_reach_the_floor() {
    // A lone 12s segment is below the 18s floor and cannot be merged with anything,
    // so no in-band tiling exists and the item must be dropped rather than scored.
    assert!(plan_blocks(&[24.0, 12.0], 24.0, 18.0, 30.0).is_none());
}

#[test]
fn test_prefix_closest_to_can_overshoot_the_target() {
    let item = Item {
        source: "synthetic".to_string(),
        first: 0,
        last: 3,
        segments: (0..4)
            .map(|_| Segment {
                reference: String::new(),
                samples: vec![0.0; SAMPLE_RATE * 7],
            })
            .collect(),
    };
    // 7s segments: the prefix nearest 24s is 3×7=21s, not the 2×7=14s that a
    // "largest prefix that fits" rule would pick.
    assert_eq!(prefix_closest_to(&item, 24.0), (3, 21.0));
    // Nearest 20s is still 21s — overshooting by 1s beats undershooting by 6s.
    assert_eq!(prefix_closest_to(&item, 20.0), (3, 21.0));
    // Beyond the item, the whole item is the closest prefix.
    assert_eq!(prefix_closest_to(&item, 120.0), (4, 28.0));
}

#[test]
fn test_word_errors_split_sums_to_the_edit_distance() {
    let reference = normalize_for_wer_naive("один два три четыре пять");
    let hypothesis = normalize_for_wer_naive("один три четыре пять шесть");
    let e = word_errors(&reference, &hypothesis);
    assert_eq!(e.del, 1, "«два» is missing");
    assert_eq!(e.ins, 1, "«шесть» is spurious");
    assert_eq!(e.sub, 0);
    assert_eq!(e.total(), 2);
}

#[test]
fn test_word_errors_identical_is_zero() {
    let words = normalize_for_wer_naive("совершенно одинаковый текст");
    assert_eq!(word_errors(&words, &words).total(), 0);
}
