//! Bit-identity probe for the file-decode path: prints one digest per input.
//!
//! Model-gated (`#[ignore]`, requires the GigaAM model (~850MB) at
//! `~/.gigastt/models`). It asserts nothing about the digest values — it is the
//! instrument for an A/B between two revisions, so a change to the decode path
//! can be shown to leave every transcript and every word timestamp untouched:
//!
//! ```sh
//! # revision A (e.g. the base branch)
//! git worktree add /tmp/base origin/main
//! cp crates/gigastt-core/tests/longform_digest.rs \
//!    /tmp/base/crates/gigastt-core/tests/
//! (cd /tmp/base && cargo test -p gigastt-core --test longform_digest \
//!    -- --ignored --nocapture) | tee /tmp/base.txt
//!
//! # revision B (the change under review)
//! cargo test -p gigastt-core --test longform_digest -- --ignored --nocapture \
//!   | tee /tmp/head.txt
//!
//! diff /tmp/base.txt /tmp/head.txt
//! ```
//!
//! Give each side its own `CARGO_TARGET_DIR`. Cargo's freshness check is
//! mtime-based, so two trees sharing one target directory can silently re-run
//! the binary built from the *other* tree — which yields a perfect match for
//! entirely the wrong reason, in the one probe whose whole job is to detect a
//! silent no-op. If you must share a target dir, check provenance before
//! believing the result: `tail -2 target/release/deps/longform_digest-*.d`
//! should name the `CARGO_MANIFEST_DIR` you think you measured.
//!
//! The digest covers the transcript text, the word count, and every word's text
//! plus the **bit patterns** of its start/end timestamps, so a timestamp that
//! moves by a single ULP changes it.
//!
//! No audio is committed for this: the inputs are the tracked `golos_*.wav`
//! fixtures, used both on their own (each is well under the 30 s single-pass
//! ceiling) and tiled into one long buffer whose prefixes straddle that ceiling
//! and the long-form window/stride boundaries.

use gigastt_core::inference::Engine;
use gigastt_core::inference::audio::{decode_audio_file, encode_wav_pcm16};
use gigastt_core::inference::{TranscribeResult, WordInfo};
use gigastt_core::model::default_model_dir;

const FIXTURE_DIR: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../gigastt/tests/fixtures/golos_"
);

/// Clips digested on their own. Each is a few seconds, so they pin the
/// single-pass branch.
const SOLO: [&str; 5] = ["00", "01", "02", "03", "04"];

/// Clips concatenated — repeating from the start once exhausted — into the
/// buffer the long-form prefixes are cut from.
const POOL: [&str; 10] = ["00", "01", "02", "03", "04", "05", "06", "07", "08", "09"];

/// Prefix lengths of the tiled pool, in samples @16 kHz: 29.0 s and 29.9 s
/// (single-pass), 30.0 s (exactly the ceiling, still single-pass), then 30.1 /
/// 35 / 44.0 / 44.01 / 90 s — the last five chunked, and 704_000 / 704_160 sit
/// on and just past twice the 22 s stride.
const PREFIXES: [usize; 8] = [
    464_000, 478_400, 480_000, 481_600, 560_000, 704_000, 704_160, 1_440_000,
];

/// Samples at or below this take the single-pass branch (30 s @16 kHz). Spelled
/// out rather than imported: the point of the probe is to notice if the branch
/// an input takes ever moves.
const SINGLE_PASS_MAX: usize = 480_000;

const FNV_OFFSET: u64 = 0xcbf5_1f19_4761_9ec5;
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

fn feed(h: &mut u64, bytes: &[u8]) {
    for b in bytes {
        *h ^= u64::from(*b);
        *h = h.wrapping_mul(FNV_PRIME);
    }
}

/// FNV-1a over the transcript and every word's text + timestamp bit patterns.
fn digest(text: &str, words: &[WordInfo]) -> u64 {
    let mut h = FNV_OFFSET;
    feed(&mut h, text.as_bytes());
    feed(&mut h, &words.len().to_le_bytes());
    for w in words {
        feed(&mut h, w.word.as_bytes());
        feed(&mut h, &w.start.to_bits().to_le_bytes());
        feed(&mut h, &w.end.to_bits().to_le_bytes());
    }
    h
}

fn fixture(id: &str) -> String {
    format!("{FIXTURE_DIR}{id}.wav")
}

fn row(label: &str, samples: usize, res: &TranscribeResult) -> u64 {
    // A silently broken engine would emit empty transcripts, whose digests match
    // across revisions for the wrong reason. Refuse to report that as evidence.
    assert!(
        !res.text.trim().is_empty() && !res.words.is_empty(),
        "{label}: empty transcript ({} samples) — the probe has nothing to compare",
        samples
    );
    let d = digest(&res.text, &res.words);
    let branch = if samples <= SINGLE_PASS_MAX {
        "single-pass"
    } else {
        "chunked"
    };
    println!(
        "{label:<14} {samples:>9} {:>8.2}  {branch:<11} {:>5}  {d:016x}",
        samples as f64 / 16000.0,
        res.words.len(),
    );
    d
}

#[test]
#[ignore = "requires the GigaAM model (~850MB) at ~/.gigastt/models"]
fn longform_transcript_digests() {
    let model_dir = default_model_dir();
    let engine = Engine::load(&model_dir).expect("load engine");
    let mut triplet = engine.pool.checkout_blocking().expect("checkout triplet");

    println!("model dir: {model_dir}");
    println!(
        "{:<14} {:>9} {:>8}  {:<11} {:>5}  digest",
        "input", "samples", "seconds", "branch", "words"
    );

    let mut all = FNV_OFFSET;

    for id in SOLO {
        let path = fixture(id);
        let samples = decode_audio_file(&path).expect("decode fixture");
        let res = engine
            .transcribe_file(&path, &mut triplet)
            .expect("transcribe fixture");
        feed(
            &mut all,
            &row(&format!("golos_{id}"), samples.len(), &res).to_le_bytes(),
        );
    }

    // Tile the pool until it covers the longest prefix, so every prefix is a cut
    // of one deterministic buffer.
    let longest = PREFIXES.iter().copied().max().unwrap_or(0);
    let clips: Vec<Vec<f32>> = POOL
        .iter()
        .map(|id| decode_audio_file(&fixture(id)).expect("decode fixture"))
        .collect();
    let mut pool: Vec<f32> = Vec::with_capacity(longest);
    while pool.len() < longest {
        for clip in &clips {
            pool.extend_from_slice(clip);
        }
    }

    for len in PREFIXES {
        let wav = encode_wav_pcm16(&pool[..len], 16000);
        let res = engine
            .transcribe_bytes(&wav, &mut triplet)
            .expect("transcribe prefix");
        feed(
            &mut all,
            &row(&format!("pool[..{len}]"), len, &res).to_le_bytes(),
        );
    }

    println!("combined: {all:016x}");
}
