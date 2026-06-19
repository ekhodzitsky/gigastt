//! Integration test: streaming transcription quality must match batch.
//!
//! Model-gated (`#[ignore]`, requires ~850MB GigaAM model at `~/.gigastt/models`).
//! Run with: `cargo test -p gigastt-core --test streaming_quality -- --ignored --nocapture`.
//!
//! Regression guard for the streaming-recognition-quality bug:
//! the streaming path used to feed the offline Conformer encoder isolated per-chunk
//! windows with no left context, collapsing a full phrase to a single token («И»).
//! This test streams `golos_00.wav` through `Engine::process_chunk` in 100 ms chunks
//! and asserts the committed streaming transcript is close to the batch transcript.

use std::collections::HashSet;

use gigastt_core::inference::Engine;
use gigastt_core::inference::audio::decode_audio_file;
use gigastt_core::model::default_model_dir;

/// Normalize a transcript into a set of lowercased alphanumeric word tokens
/// (drops punctuation like `—`, `?`), so the comparison is robust to spacing
/// and punctuation differences between the batch and streaming paths.
fn norm_words(s: &str) -> HashSet<String> {
    s.split_whitespace()
        .map(|w| {
            w.chars()
                .filter(|c| c.is_alphanumeric())
                .collect::<String>()
                .to_lowercase()
        })
        .filter(|w| !w.is_empty())
        .collect()
}

#[test]
#[ignore = "requires the GigaAM model (~850MB) at ~/.gigastt/models"]
fn streaming_transcript_matches_batch() {
    let model_dir = default_model_dir();
    let fixture = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../gigastt/tests/fixtures/golos_00.wav"
    );

    let engine = Engine::load(&model_dir).expect("load engine");
    let mut triplet = engine.pool.checkout_blocking().expect("checkout triplet");

    // Batch reference (the path that is known to work correctly).
    let batch_text = engine
        .transcribe_file(fixture, &mut triplet)
        .expect("batch transcribe")
        .text;

    // Stream the same clip in 100 ms (1600-sample @ 16 kHz) chunks and collect
    // the committed transcript: every finalized segment plus the closing flush.
    let samples = decode_audio_file(fixture).expect("decode fixture");
    let mut state = engine.create_state(false);
    let mut committed: Vec<String> = Vec::new();
    for chunk in samples.chunks(1600) {
        let segments = engine
            .process_chunk(chunk, &mut state, &mut triplet)
            .expect("process_chunk");
        for seg in segments {
            if seg.is_final && !seg.text.trim().is_empty() {
                committed.push(seg.text);
            }
        }
    }
    if let Some(seg) = engine.finish_stream(&mut state, &mut triplet)
        && !seg.text.trim().is_empty()
    {
        committed.push(seg.text);
    }
    let streaming_text = committed.join(" ").trim().to_string();

    eprintln!("batch:     {batch_text:?}");
    eprintln!("streaming: {streaming_text:?}");

    let batch_w = norm_words(&batch_text);
    let stream_w = norm_words(&streaming_text);
    let shared = batch_w.intersection(&stream_w).count();
    let overlap = if batch_w.is_empty() {
        0.0
    } else {
        shared as f64 / batch_w.len() as f64
    };

    assert!(
        stream_w.len() >= 4,
        "streaming produced too few words ({}): {streaming_text:?} (batch: {batch_text:?})",
        stream_w.len()
    );
    assert!(
        overlap >= 0.5,
        "streaming transcript diverges from batch: word-overlap {overlap:.2} (< 0.50)\n  \
         streaming: {streaming_text:?}\n  batch:     {batch_text:?}"
    );
}

/// Audio longer than the 5 s streaming window must keep transcribing across the
/// window slide (left-context carry + dedup), not collapse or stall. Feeds three
/// concatenated copies of the fixture (~12 s) so the window cap forces slides.
#[test]
#[ignore = "requires the GigaAM model (~850MB) at ~/.gigastt/models"]
fn streaming_long_audio_slides_window() {
    let model_dir = default_model_dir();
    let fixture = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../gigastt/tests/fixtures/golos_00.wav"
    );

    let engine = Engine::load(&model_dir).expect("load engine");
    let mut triplet = engine.pool.checkout_blocking().expect("checkout triplet");

    let one = decode_audio_file(fixture).expect("decode fixture");
    let mut samples = Vec::new();
    for _ in 0..3 {
        samples.extend_from_slice(&one); // ~12 s > 5 s window → forces slides
    }

    let mut state = engine.create_state(false);
    let mut committed: Vec<String> = Vec::new();
    for chunk in samples.chunks(1600) {
        for seg in engine
            .process_chunk(chunk, &mut state, &mut triplet)
            .expect("process_chunk")
        {
            if seg.is_final && !seg.text.trim().is_empty() {
                committed.push(seg.text);
            }
        }
    }
    if let Some(seg) = engine.finish_stream(&mut state, &mut triplet)
        && !seg.text.trim().is_empty()
    {
        committed.push(seg.text);
    }
    let streaming_text = committed.join(" ").trim().to_string();
    eprintln!("long streaming: {streaming_text:?}");

    let total_words = streaming_text.split_whitespace().count();
    let unique = norm_words(&streaming_text);
    // Across ~12 s of speech the slide path must keep producing content, not
    // collapse to a single token or stall after the first window.
    assert!(
        total_words >= 8,
        "long-audio streaming produced too few words ({total_words}): {streaming_text:?}"
    );
    assert!(
        unique.contains("сколько") && unique.contains("стоить"),
        "long-audio streaming lost content words across slides: {streaming_text:?}"
    );
}

/// Streaming word timestamps must track real elapsed time, not be inflated by
/// the encoder subsampling factor (a mel-vs-encoder frame unit
/// mismatch used to multiply every post-first-chunk `start`/`end` by ~4×). The
/// inflation only appears once the window slides (a non-zero frame offset), so
/// this feeds several seconds of audio to force slides, then asserts no word
/// lands far beyond the audio's real duration. Fixed structurally (the offset
/// is now derived from slid-off samples); this is the regression guard.
#[test]
#[ignore = "requires the GigaAM model (~850MB) at ~/.gigastt/models"]
fn streaming_word_timestamps_not_inflated() {
    let model_dir = default_model_dir();
    let fixture = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../gigastt/tests/fixtures/golos_00.wav"
    );

    let engine = Engine::load(&model_dir).expect("load engine");
    let mut triplet = engine.pool.checkout_blocking().expect("checkout triplet");

    let one = decode_audio_file(fixture).expect("decode fixture");
    let mut samples = Vec::new();
    for _ in 0..3 {
        samples.extend_from_slice(&one); // ~12 s > 5 s window → forces slides
    }
    let audio_dur_s = samples.len() as f64 / 16000.0;

    let mut state = engine.create_state(false);
    let mut max_end_s = 0.0_f64;
    let mut word_count = 0usize;
    let mut record = |segments: Vec<gigastt_core::inference::TranscriptSegment>| {
        for seg in segments {
            if seg.is_final {
                for w in seg.words {
                    max_end_s = max_end_s.max(w.end);
                    word_count += 1;
                }
            }
        }
    };
    for chunk in samples.chunks(1600) {
        record(
            engine
                .process_chunk(chunk, &mut state, &mut triplet)
                .expect("process_chunk"),
        );
    }
    if let Some(seg) = engine.finish_stream(&mut state, &mut triplet) {
        record(vec![seg]);
    }

    eprintln!("audio_dur={audio_dur_s:.2}s  max_word_end={max_end_s:.2}s  words={word_count}");
    assert!(
        word_count >= 5,
        "expected several timestamped words across the stream, got {word_count}"
    );
    // A ~4× inflation would push post-slide words to tens of seconds on this
    // ~12 s clip; a 1.5× tolerance catches it while allowing frame rounding.
    assert!(
        max_end_s <= audio_dur_s * 1.5,
        "word end {max_end_s:.2}s far exceeds audio duration {audio_dur_s:.2}s \
         — streaming timestamp inflation regressed"
    );
}
