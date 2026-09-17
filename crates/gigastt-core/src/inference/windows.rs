//! Streaming and long-form window geometry (samples @16 kHz).
//!
//! Free of ONNX sessions so the constants and backend-aware window selection
//! are unit-testable without a loaded model. Called from [`super::engine::Engine`]
//! for streaming slide/stride and file-transcription chunking.

use super::audio::WindowSpec;

/// Default max streaming encoder window before sliding (samples @16kHz, 2.5s).
/// Configurable at serve time via `--stream-max-window-secs` (see
/// [`stream_max_window_samples`]); the engine stores the resolved value.
/// Re-decoding the whole window each stride gives the offline Conformer left
/// context; this cap bounds the per-stride encoder cost. With the 1.5s retained
/// left context and the 0.8s stride, a 2.5s window keeps the steady-state
/// re-encode overlap near ~3x (vs ~6.25x at a 5s window) — roughly half the
/// streaming encoder work. Short utterances stay on par with batch (ordered-WER
/// `streaming_quality` tests); phrases longer than the window can degrade
/// (stream-vs-file gap measured in docs/benchmarks.md) — raising the window is
/// the mitigation, at a linear encoder-cost increase per stride.
///
/// Hitting the cap **commits a stable prefix** and slides; it does **not** emit
/// a speech-final `final` (that would mean "utterance complete" to assistants).
pub(crate) const STREAM_MAX_WINDOW_SAMPLES: usize = 16000 * 5 / 2;
/// Bounds for the configurable streaming window (seconds). The floor keeps the
/// window larger than the retained left context plus one decode stride (a
/// smaller cap would slide almost immediately and re-commit degenerate tails);
/// the ceiling matches the Conformer's useful-context limit used for file
/// chunking (see `CHUNK_THRESHOLD_SAMPLES`).
pub(crate) const MIN_STREAM_WINDOW_SECS: f64 = 2.4;
pub(crate) const MAX_STREAM_WINDOW_SECS: f64 = 30.0;

/// Resolve a user-requested streaming window length (seconds) to samples
/// @16kHz, clamped to [`MIN_STREAM_WINDOW_SECS`]..=[`MAX_STREAM_WINDOW_SECS`].
/// Non-finite input falls back to the default. Pure so the clamping policy is
/// unit-testable without a loaded model.
pub(crate) fn stream_max_window_samples(secs: f64) -> usize {
    if !secs.is_finite() {
        return STREAM_MAX_WINDOW_SAMPLES;
    }
    let clamped = secs.clamp(MIN_STREAM_WINDOW_SECS, MAX_STREAM_WINDOW_SECS);
    (clamped * 16000.0).round() as usize
}
/// Left-context audio retained across a streaming finalize/slide (samples @16kHz,
/// ~1.5s) so the next window keeps acoustic context instead of restarting cold.
pub(crate) const STREAM_LEFT_CONTEXT_SAMPLES: usize = 16000 * 3 / 2;
/// Decode stride: re-run the encoder only after this much NEW audio has
/// accumulated (samples @16kHz, 0.8s) instead of on every ~100ms chunk.
/// Re-decoding the window is the dominant streaming cost, so the stride keeps
/// the engine real-time; `finish_stream` decodes the sub-stride remainder at EOF.
pub(crate) const STREAM_DECODE_STRIDE_SAMPLES: usize = 16000 * 4 / 5;
/// Commit horizon for stable-prefix slides (seconds): words decoded from the
/// last stretch of the window are not committed even when consecutive
/// hypotheses agree on them — near the buffer edge a word may still be
/// mid-formation, decoded from incomplete audio (the edge truncates it, and
/// two consecutive truncated decodes agree with each other). 1.0 s covers the
/// observed edge-truncation window.
pub(crate) const STREAM_COMMIT_HORIZON_SECS: f64 = 1.0;
/// Consecutive cap hits with zero hypothesis agreement after which the whole
/// live tail is committed anyway, so a pathological stream cannot grow the
/// retained buffer (and its per-chunk encoder cost) without bound.
pub(crate) const STREAM_CAP_STREAK_MAX: usize = 3;

/// File-transcription chunking threshold (samples @16kHz, 30s). Inputs at or
/// below this length take the single-pass path unchanged; longer inputs are
/// split into overlapping windows so the encoder's peak activation memory is
/// bounded by the chunk size, not the file length. Different encoder context
/// and independent decodes can change recognition near a boundary; see
/// `docs/longform-stitching.md` for the measured scope and limitations.
/// (A higher single-pass ceiling for CTC was tried for
/// stretch RTF on ~40s clips; measured wall time was worse than 24s windows —
/// larger activation tensors thrash CPU caches — so both head families share
/// this 30s ceiling.)
pub(crate) const CHUNK_THRESHOLD_SAMPLES: usize = 16000 * 30;
/// Long-form decode window on ort / CoreML-EP / CUDA (samples @16kHz, 24s).
/// Bounds per-chunk encoder activation memory; the ANE path uses a longer
/// window via [`chunk_window_samples`].
pub(crate) const CHUNK_WINDOW_SAMPLES_ORT: usize = 16000 * 24;
/// Long-form decode window on the ANE encoder (samples @16kHz, 30s). Full chunks
/// fill ANE bucket 3000 at ~99.97% (vs ~80% fill at 24s), recovering pad-up
/// waste. Peak activation is free on-device; ort keeps the shorter window.
pub(crate) const CHUNK_WINDOW_SAMPLES_ANE: usize = 16000 * 30;
/// Overlap retained between consecutive long-form windows (samples @16kHz, 2s),
/// so a word straddling a seam is decoded fully in at least one chunk. The
/// stitch step de-dups words in the overlap region (see [`super::token_format::stitch_chunk_words`]).
pub(crate) const CHUNK_OVERLAP_SAMPLES: usize = 16000 * 2;

/// Select the long-form chunk window length for the active encoder backend.
///
/// ANE uses 30s so each full chunk nearly fills bucket 3000; every other
/// backend keeps 24s to bound peak encoder activation memory on CPU/EP paths.
/// Pure so the selection is unit-tested without a loaded model.
pub(crate) fn chunk_window_samples(ane_encoder: bool) -> usize {
    if ane_encoder {
        CHUNK_WINDOW_SAMPLES_ANE
    } else {
        CHUNK_WINDOW_SAMPLES_ORT
    }
}

/// Long-form window geometry for the active encoder backend: the single-pass
/// ceiling, the backend's window length, and the fixed inter-window overlap.
/// Free-standing (like [`chunk_window_samples`]) so the geometry is unit-tested
/// without a loaded model. `ctc` is accepted for call-site uniformity (CTC and
/// RNN-T share the same 30s ceiling after measurement).
pub(crate) fn window_spec(ane_encoder: bool, _ctc: bool) -> WindowSpec {
    WindowSpec::new(
        CHUNK_THRESHOLD_SAMPLES,
        chunk_window_samples(ane_encoder),
        CHUNK_OVERLAP_SAMPLES,
    )
}

#[cfg(test)]
mod tests {
    use super::super::{ENCODER_SUBSAMPLING, HOP_LENGTH};
    use super::*;

    #[test]
    #[allow(clippy::assertions_on_constants)] // intentional compile-time sanity check on the chunk constants
    fn test_chunk_constants_sane() {
        // Window > overlap (positive stride) and threshold ≥ window so the
        // single-pass path covers everything up to one full window.
        assert!(CHUNK_WINDOW_SAMPLES_ORT > CHUNK_OVERLAP_SAMPLES);
        assert!(CHUNK_WINDOW_SAMPLES_ANE > CHUNK_OVERLAP_SAMPLES);
        assert!(CHUNK_THRESHOLD_SAMPLES >= CHUNK_WINDOW_SAMPLES_ORT);
        assert!(CHUNK_THRESHOLD_SAMPLES >= CHUNK_WINDOW_SAMPLES_ANE);
    }

    #[test]
    fn test_window_spec_reproduces_legacy_chunk_geometry() {
        // The long-form loop used to derive its own window/stride and read the
        // overlap straight off `CHUNK_OVERLAP_SAMPLES`; it now takes all three
        // from the spec. Any divergence here moves every seam, so pin it.
        let frame_samples = HOP_LENGTH * ENCODER_SUBSAMPLING;
        for ane in [false, true] {
            let window = chunk_window_samples(ane);
            let legacy_stride = ((window - CHUNK_OVERLAP_SAMPLES) / frame_samples) * frame_samples;
            for ctc in [false, true] {
                let spec = window_spec(ane, ctc);
                assert_eq!(spec.window(), window, "window (ane={ane}, ctc={ctc})");
                assert_eq!(
                    spec.stride(),
                    legacy_stride,
                    "stride (ane={ane}, ctc={ctc})"
                );
                assert_eq!(
                    spec.overlap(),
                    CHUNK_OVERLAP_SAMPLES,
                    "overlap (ane={ane}, ctc={ctc})"
                );
                assert!(spec.is_single_pass(CHUNK_THRESHOLD_SAMPLES));
                assert!(!spec.is_single_pass(CHUNK_THRESHOLD_SAMPLES + 1));
            }
        }
    }

    #[test]
    fn test_stream_max_window_samples_clamps() {
        // Default token resolves to the legacy 2.5 s constant.
        assert_eq!(stream_max_window_samples(2.5), STREAM_MAX_WINDOW_SAMPLES);
        // Floor: must exceed left context (1.5 s) + one stride (0.8 s).
        assert_eq!(
            stream_max_window_samples(0.5),
            (MIN_STREAM_WINDOW_SECS * 16000.0) as usize
        );
        // Ceiling: Conformer useful-context limit.
        assert_eq!(
            stream_max_window_samples(120.0),
            (MAX_STREAM_WINDOW_SECS * 16000.0) as usize
        );
        // In-range values pass through.
        assert_eq!(stream_max_window_samples(7.5), 16000 * 15 / 2);
    }

    #[test]
    fn test_chunk_window_samples_backend_aware() {
        // ort / non-ANE: 24s keeps peak encoder activation bounded.
        assert_eq!(chunk_window_samples(false), 16000 * 24);
        // ANE: 30s fills bucket 3000 at ~99.97%.
        assert_eq!(chunk_window_samples(true), 16000 * 30);
        assert!(chunk_window_samples(true) > chunk_window_samples(false));
    }
}
