//! Integration test: streaming transcription quality must match batch.
//!
//! Model-gated (`#[ignore]`, requires ~225 MB INT8 GigaAM model at `~/.gigastt/models`).
//! Run with: `cargo test -p gigastt-core --test streaming_quality -- --ignored --nocapture`.
//!
//! Regression guard for the streaming-recognition-quality bug:
//! the streaming path used to feed the offline Conformer encoder isolated per-chunk
//! windows with no left context, collapsing a full phrase to a single token («И»).
//! This test streams `golos_00.wav` through `Engine::process_chunk` in 100 ms chunks
//! and asserts the committed streaming transcript is close to the batch transcript.

#![cfg(feature = "file-decode")]

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

/// Ordered, lowercased alphanumeric word sequence — unlike [`norm_words`]
/// (a set), this preserves order and duplicates, which is what a WER-style
/// comparison needs (a set-overlap check cannot see transpositions, dropped
/// duplicates, or repeated words).
fn word_seq(s: &str) -> Vec<String> {
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

/// Levenshtein edit distance between two word sequences (ordered).
fn word_edit_distance(a: &[String], b: &[String]) -> usize {
    let mut prev: Vec<usize> = (0..=b.len()).collect();
    let mut cur = vec![0usize; b.len() + 1];
    for (i, wa) in a.iter().enumerate() {
        cur[0] = i + 1;
        for (j, wb) in b.iter().enumerate() {
            let sub = prev[j] + usize::from(wa != wb);
            cur[j + 1] = (prev[j + 1] + 1).min(cur[j] + 1).min(sub);
        }
        std::mem::swap(&mut prev, &mut cur);
    }
    prev[b.len()]
}

/// Corpus WER: total edits over total reference words (not a mean of per-clip
/// WERs, which would over-weight short clips).
fn corpus_wer(pairs: &[(Vec<String>, Vec<String>)]) -> f64 {
    let edits: usize = pairs.iter().map(|(r, h)| word_edit_distance(r, h)).sum();
    let words: usize = pairs.iter().map(|(r, _)| r.len()).sum();
    if words == 0 {
        0.0
    } else {
        edits as f64 / words as f64
    }
}

/// Stream one clip in 100 ms chunks and return the committed transcript
/// (final segments in order + the closing flush), mirroring how clients
/// concatenate `/v1/ws` output.
fn stream_clip(engine: &Engine, fixture: &str) -> String {
    let mut triplet = engine.pool.checkout_blocking().expect("checkout triplet");
    let samples = decode_audio_file(fixture).expect("decode fixture");
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
    committed.join(" ").trim().to_string()
}

#[test]
#[ignore = "requires the GigaAM model (~225 MB INT8) at ~/.gigastt/models"]
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

/// Audio longer than the streaming window (2.5 s by default) must keep
/// transcribing across the window slide (left-context carry + dedup), not
/// collapse or stall. Feeds three concatenated copies of the fixture (~12 s)
/// so the window cap forces several slides.
#[test]
#[ignore = "requires the GigaAM model (~225 MB INT8) at ~/.gigastt/models"]
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
        samples.extend_from_slice(&one); // ~12 s > 2.5 s window → forces slides
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
#[ignore = "requires the GigaAM model (~225 MB INT8) at ~/.gigastt/models"]
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
        samples.extend_from_slice(&one); // ~12 s > 2.5 s window → forces slides
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

/// Sequence-aware stream-vs-file guard over several labelled fixtures
/// (replaces set-overlap as the quality signal: order and duplicates count).
///
/// Streams the first 10 Golos fixtures (100 ms chunks) under three configs —
/// 2.5 s window with stable-prefix off (legacy mode), raised 7.5 s window
/// (legacy mode), and 2.5 s with stable-prefix commits (the default) — and
/// asserts against the manifest references:
///
/// - legacy default window: corpus streaming WER stays under 15% absolute (a
///   real regression like the historical single-token collapse lands far above
///   that),
/// - stable-prefix at the default window: must not cost accuracy — corpus WER
///   no worse than the plain 2.5 s pass + 2 pp, and
/// - raised window: corpus streaming WER stays within +5 pp of corpus batch
///   WER — the configurable window must close the long-phrase stream-vs-file
///   gap (`--stream-max-window-secs`).
///
/// Calibration (2026-08-30, INT8 `rnnt`, first 10 Golos fixtures, 47 ref
/// words): batch 0.000; streaming 0.106 at 2.5 s and 0.021 at 7.5 s — the
/// default-window gap is real on these short clips, so parity is asserted at
/// the raised window, not at the default.
#[test]
#[ignore = "requires the GigaAM model (~225 MB INT8) at ~/.gigastt/models"]
fn streaming_wer_tracks_file_on_labelled_fixtures() {
    let model_dir = default_model_dir();
    let manifest_path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../gigastt/tests/fixtures/manifest.json"
    );
    let manifest: Vec<serde_json::Value> =
        serde_json::from_str(&std::fs::read_to_string(manifest_path).expect("read manifest"))
            .expect("parse manifest");

    // Stable-prefix is the engine default; the baseline and wide runs pin the
    // legacy mode off so the gate keeps measuring both behaviours.
    let engine = Engine::load(&model_dir)
        .expect("load engine")
        .with_stream_stable_prefix(false);
    let wide_engine = Engine::load(&model_dir)
        .expect("load engine")
        .with_stream_max_window_secs(7.5)
        .with_stream_stable_prefix(false);
    let stable_engine = Engine::load(&model_dir)
        .expect("load engine")
        .with_stream_stable_prefix(true);
    let mut batch_pairs: Vec<(Vec<String>, Vec<String>)> = Vec::new();
    let mut stream_pairs: Vec<(Vec<String>, Vec<String>)> = Vec::new();
    let mut wide_pairs: Vec<(Vec<String>, Vec<String>)> = Vec::new();
    let mut stable_pairs: Vec<(Vec<String>, Vec<String>)> = Vec::new();

    for entry in manifest.iter().take(10) {
        let filename = entry["filename"].as_str().expect("manifest filename");
        let reference = entry["reference"].as_str().expect("manifest reference");
        let fixture = format!(
            "{}/../gigastt/tests/fixtures/{filename}",
            env!("CARGO_MANIFEST_DIR")
        );

        let mut triplet = engine.pool.checkout_blocking().expect("checkout triplet");
        let batch_text = engine
            .transcribe_file(&fixture, &mut triplet)
            .expect("batch transcribe")
            .text;
        let streaming_text = stream_clip(&engine, &fixture);
        let wide_text = stream_clip(&wide_engine, &fixture);
        let stable_text = stream_clip(&stable_engine, &fixture);

        let reference = word_seq(reference);
        let batch_edits = word_edit_distance(&reference, &word_seq(&batch_text));
        let stream_edits = word_edit_distance(&reference, &word_seq(&streaming_text));
        let wide_edits = word_edit_distance(&reference, &word_seq(&wide_text));
        let stable_edits = word_edit_distance(&reference, &word_seq(&stable_text));
        eprintln!(
            "{filename}: ref={} words, batch={batch_edits}, 2.5s={stream_edits}, 7.5s={wide_edits}, 2.5s+stable={stable_edits}\n  \
             batch:           {batch_text:?}\n  streaming 2.5s:  {streaming_text:?}\n  streaming 7.5s:  {wide_text:?}\n  stream 2.5s+sp:  {stable_text:?}",
            reference.len()
        );
        batch_pairs.push((reference.clone(), word_seq(&batch_text)));
        stream_pairs.push((reference.clone(), word_seq(&streaming_text)));
        wide_pairs.push((reference.clone(), word_seq(&wide_text)));
        stable_pairs.push((reference, word_seq(&stable_text)));
    }

    let batch_wer = corpus_wer(&batch_pairs);
    let stream_wer = corpus_wer(&stream_pairs);
    let wide_wer = corpus_wer(&wide_pairs);
    let stable_wer = corpus_wer(&stable_pairs);
    eprintln!(
        "corpus WER: batch={batch_wer:.4} 2.5s={stream_wer:.4} 7.5s={wide_wer:.4} 2.5s+stable={stable_wer:.4}"
    );

    assert!(
        stream_wer < 0.15,
        "streaming corpus WER {stream_wer:.4} regressed past 15% at the default window"
    );
    assert!(
        stable_wer <= stream_wer + 0.02,
        "stable-prefix WER {stable_wer:.4} worse than plain {stream_wer:.4} at the default window"
    );
    assert!(
        wide_wer <= batch_wer + 0.05,
        "raised-window streaming WER {wide_wer:.4} diverges from batch {batch_wer:.4} by more than 5 pp \
         — --stream-max-window-secs no longer closes the stream-vs-file gap"
    );
}
