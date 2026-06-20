//! Voice activity detection (VAD) via the Silero v5 ONNX model.
//!
//! Used for two optional, opt-in features on top of the recognition engine:
//!
//! 1. **File silence skipping** — [`SileroVad::speech_regions`] returns the
//!    speech spans of a clip so the engine can decode only those, skipping long
//!    pauses. Speedup is proportional to the silence fraction.
//! 2. **Streaming endpointing** — [`VadEndpointer`] tracks trailing silence
//!    across streamed chunks and signals when an utterance has ended, finalizing
//!    a segment sooner / more reliably than the decoder's blank-run heuristic.
//!
//! The model is loaded through the same `ort` runtime the recognition engine
//! already uses (no extra dependency, no second ONNX Runtime). The Silero v5
//! graph (opset 16, conv + LSTM) takes a fixed 512-sample window at 16 kHz plus
//! a recurrent state tensor `[2, 1, 128]`, and returns a speech probability in
//! `[0, 1]` together with the next state.
//!
//! All of the segmentation / endpointing decision logic is split into pure
//! functions ([`regions_from_probs`], [`Hangover`]) so it can be unit-tested on
//! synthetic probability sequences without loading the model.

use std::path::Path;

use anyhow::{Context, Result};
use ort::session::Session;
use ort::value::TensorRef;
use parking_lot::Mutex;

/// `ort` errors are not `Send`/`Sync`; stringify them so they cross `?` into
/// `anyhow` like everywhere else in the crate.
fn ort_err(e: impl std::fmt::Display) -> anyhow::Error {
    anyhow::anyhow!("{e}")
}

/// Silero VAD ONNX filename on disk. Single source of truth shared with the
/// model-download path in [`crate::model`].
pub const VAD_MODEL_FILE: &str = "silero_vad.onnx";

/// Sample rate the engine (and Silero) operate at.
pub const VAD_SAMPLE_RATE: i64 = 16000;

/// Fixed Silero v5 window at 16 kHz (~32 ms). The model only accepts this size.
pub const VAD_FRAME_SAMPLES: usize = 512;

/// Length of the Silero recurrent state tensor (`[2, 1, 128]` flattened).
const VAD_STATE_LEN: usize = 2 * 128;

/// Tunable thresholds for turning a per-frame speech-probability sequence into
/// speech spans (file path) and endpoint decisions (streaming).
#[derive(Debug, Clone, Copy)]
pub struct VadConfig {
    /// Speech-probability threshold in `[0, 1]`; frames at or above are speech.
    pub threshold: f32,
    /// Minimum trailing silence before a speech region is closed / an utterance
    /// is considered ended (endpointing).
    pub min_silence_ms: u32,
    /// Speech runs shorter than this are dropped as noise (file path only).
    pub min_speech_ms: u32,
    /// Padding added on each side of a kept speech region so onsets/offsets are
    /// not clipped (file path only).
    pub speech_pad_ms: u32,
}

impl Default for VadConfig {
    fn default() -> Self {
        // Silero's own defaults, lightly adapted: 0.5 threshold, ~500 ms of
        // silence to close a turn, 250 ms minimum speech, 100 ms pad.
        Self {
            threshold: 0.5,
            min_silence_ms: 500,
            min_speech_ms: 250,
            speech_pad_ms: 100,
        }
    }
}

impl VadConfig {
    fn ms_to_samples(ms: u32) -> usize {
        (VAD_SAMPLE_RATE as usize * ms as usize) / 1000
    }
}

/// Silero v5 VAD model wrapped around the shared `ort` runtime.
///
/// The ONNX session is behind a [`Mutex`] because VAD runs off the hot decode
/// loop (either once per file or once per streamed chunk) and is not worth
/// pooling. The recurrent state is owned by the caller (per stream / per call),
/// never by this struct, so a single `SileroVad` can serve many concurrent
/// streams.
pub struct SileroVad {
    session: Mutex<Session>,
}

impl SileroVad {
    /// Load the Silero VAD ONNX model from `model_path`.
    ///
    /// # Errors
    ///
    /// Returns an error if the file is missing or `ort` fails to build the
    /// session. The caller treats an error as "VAD unavailable" and proceeds
    /// without it — VAD is strictly optional.
    pub fn load(model_path: &Path) -> Result<Self> {
        let session = Session::builder()
            .map_err(ort_err)?
            .commit_from_file(model_path)
            .map_err(ort_err)
            .with_context(|| format!("Failed to load VAD model {}", model_path.display()))?;
        tracing::info!("VAD model loaded from {}", model_path.display());
        Ok(Self {
            session: Mutex::new(session),
        })
    }

    /// Run one fixed 512-sample window through the model, advancing `state`
    /// (the `[2, 1, 128]` recurrent tensor, flattened to [`VAD_STATE_LEN`]).
    /// Returns the speech probability in `[0, 1]`.
    ///
    /// `frame` shorter than [`VAD_FRAME_SAMPLES`] is zero-padded; longer is
    /// truncated, matching Silero's own contract.
    fn run_frame(&self, frame: &[f32], state: &mut [f32; VAD_STATE_LEN]) -> Result<f32> {
        let mut input = [0.0f32; VAD_FRAME_SAMPLES];
        let n = frame.len().min(VAD_FRAME_SAMPLES);
        input[..n].copy_from_slice(&frame[..n]);

        let input_t = TensorRef::from_array_view(([1_usize, VAD_FRAME_SAMPLES], input.as_slice()))?;
        let state_t = TensorRef::from_array_view(([2_usize, 1, 128], state.as_slice()))?;
        let sr_t = TensorRef::from_array_view(([1_usize], [VAD_SAMPLE_RATE].as_slice()))?;

        let prob = {
            let mut session = self.session.lock();
            let outputs = session
                .run(ort::inputs![
                    "input" => input_t,
                    "state" => state_t,
                    "sr" => sr_t,
                ])
                .context("VAD model inference failed")?;

            // Copy the next state out before the borrow ends. A length other
            // than VAD_STATE_LEN means the model's recurrent-state contract
            // changed; surface it rather than silently running later frames on
            // stale state (the non-blocking caller then disables VAD).
            let (_, new_state) = outputs["stateN"]
                .try_extract_tensor::<f32>()
                .context("failed to extract VAD state")?;
            if new_state.len() != VAD_STATE_LEN {
                anyhow::bail!(
                    "unexpected VAD state length {} (expected {VAD_STATE_LEN})",
                    new_state.len()
                );
            }
            state.copy_from_slice(new_state);

            let (_, prob) = outputs["output"]
                .try_extract_tensor::<f32>()
                .context("failed to extract VAD probability")?;
            prob.first().copied().unwrap_or(0.0)
        };
        Ok(prob)
    }

    /// Speech probability for every non-overlapping 512-sample window of
    /// `samples` (the trailing partial window, if any, is included zero-padded).
    pub fn frame_probs(&self, samples: &[f32]) -> Result<Vec<f32>> {
        let mut state = [0.0f32; VAD_STATE_LEN];
        let mut probs = Vec::with_capacity(samples.len() / VAD_FRAME_SAMPLES + 1);
        let mut i = 0;
        while i < samples.len() {
            let end = (i + VAD_FRAME_SAMPLES).min(samples.len());
            probs.push(self.run_frame(&samples[i..end], &mut state)?);
            i = end;
        }
        Ok(probs)
    }

    /// Detect the speech spans of `samples` as `[start, end)` sample ranges
    /// (inclusive start, exclusive end) on the original timeline.
    ///
    /// Empty when no frame clears `cfg.threshold`.
    pub fn speech_regions(&self, samples: &[f32], cfg: &VadConfig) -> Result<Vec<(usize, usize)>> {
        let probs = self.frame_probs(samples)?;
        Ok(regions_from_probs(
            &probs,
            VAD_FRAME_SAMPLES,
            samples.len(),
            cfg,
        ))
    }
}

/// Turn a per-frame speech-probability sequence into merged `[start, end)`
/// speech-sample spans. Pure (no model) so it is unit-testable on synthetic
/// probabilities.
///
/// `frame_samples` is the samples-per-probability stride ([`VAD_FRAME_SAMPLES`]
/// in production); `total_samples` clamps the final span to the real signal
/// length. Applies, in order: threshold, min-silence merge (gaps shorter than
/// `min_silence_ms` do not split a region), min-speech drop, and symmetric
/// `speech_pad_ms` padding (clamped to `[0, total_samples]`, then re-merged if
/// padding makes neighbours overlap).
pub fn regions_from_probs(
    probs: &[f32],
    frame_samples: usize,
    total_samples: usize,
    cfg: &VadConfig,
) -> Vec<(usize, usize)> {
    if probs.is_empty() || total_samples == 0 {
        return Vec::new();
    }

    let min_silence = VadConfig::ms_to_samples(cfg.min_silence_ms);
    let min_speech = VadConfig::ms_to_samples(cfg.min_speech_ms);
    let pad = VadConfig::ms_to_samples(cfg.speech_pad_ms);

    // 1. Raw speech runs from the thresholded probabilities.
    let mut regions: Vec<(usize, usize)> = Vec::new();
    let mut run_start: Option<usize> = None;
    for (i, &p) in probs.iter().enumerate() {
        let speech = p >= cfg.threshold;
        if speech && run_start.is_none() {
            run_start = Some(i * frame_samples);
        } else if !speech && let Some(s) = run_start.take() {
            regions.push((s, i * frame_samples));
        }
    }
    if let Some(s) = run_start.take() {
        regions.push((s, total_samples));
    }
    if regions.is_empty() {
        return regions;
    }

    // 2. Merge regions separated by a silence gap shorter than min_silence.
    let mut merged: Vec<(usize, usize)> = Vec::with_capacity(regions.len());
    for (s, e) in regions {
        match merged.last_mut() {
            Some(last) if s.saturating_sub(last.1) < min_silence => last.1 = e,
            _ => merged.push((s, e)),
        }
    }

    // 3. Drop regions shorter than min_speech (measured before padding).
    merged.retain(|(s, e)| e - s >= min_speech);
    if merged.is_empty() {
        return merged;
    }

    // 4. Pad each side, clamp to the signal, then re-merge any overlaps the
    //    padding introduced.
    let mut padded: Vec<(usize, usize)> = Vec::with_capacity(merged.len());
    for (s, e) in merged {
        let ps = s.saturating_sub(pad);
        let pe = (e + pad).min(total_samples);
        match padded.last_mut() {
            Some(last) if ps <= last.1 => last.1 = last.1.max(pe),
            _ => padded.push((ps, pe)),
        }
    }
    padded
}

/// Map a timestamp on the compressed (silence-removed) timeline back to the
/// original timeline, given the kept speech `regions` (original `[start, end)`
/// sample ranges, in order) and `sample_rate`. Pure — unit-tested directly.
///
/// File transcription with VAD decodes a buffer formed by concatenating the
/// speech regions, so decoded word timestamps are in compressed time; this
/// undoes that compression. A time at or past the end of all regions clamps to
/// the last region's end (guards rounding past the final frame).
pub fn remap_compressed_seconds(
    t_compressed_s: f64,
    regions: &[(usize, usize)],
    sample_rate: f64,
) -> f64 {
    if regions.is_empty() {
        return t_compressed_s;
    }
    let target = (t_compressed_s * sample_rate).max(0.0);
    let mut acc = 0.0f64; // compressed-sample offset at the current region's start
    for &(s, e) in regions {
        let len = (e - s) as f64;
        if target <= acc + len {
            let into = (target - acc).max(0.0);
            return (s as f64 + into) / sample_rate;
        }
        acc += len;
    }
    let &(_, end) = regions.last().expect("non-empty checked above");
    end as f64 / sample_rate
}

/// Streaming endpoint detector: feeds streamed audio through the VAD in fixed
/// frames, tracks trailing silence, and reports when an utterance has ended
/// (≥ `min_silence_ms` of silence *after* speech was seen).
///
/// Owns its recurrent state and a small leftover buffer so callers can push
/// arbitrary chunk sizes. The threshold/silence logic is exercised directly in
/// tests via [`Hangover`].
pub struct VadEndpointer {
    state: [f32; VAD_STATE_LEN],
    leftover: Vec<f32>,
    hangover: Hangover,
}

impl VadEndpointer {
    /// New endpointer for the given config.
    pub fn new(cfg: &VadConfig) -> Self {
        Self {
            state: [0.0f32; VAD_STATE_LEN],
            leftover: Vec::with_capacity(VAD_FRAME_SAMPLES),
            hangover: Hangover::new(cfg),
        }
    }

    /// Feed a chunk of 16 kHz samples. Returns `true` exactly once per utterance
    /// when trailing silence first crosses `min_silence_ms` after speech — the
    /// caller should finalize the current segment. Resets internally so the next
    /// speech run can trigger again.
    ///
    /// On model inference failure the chunk is treated as non-endpointing (logged
    /// by the caller) so streaming is never blocked by VAD.
    pub fn push(&mut self, vad: &SileroVad, samples: &[f32]) -> Result<bool> {
        self.leftover.extend_from_slice(samples);
        let mut endpoint = false;
        let mut off = 0;
        while off + VAD_FRAME_SAMPLES <= self.leftover.len() {
            let prob = vad.run_frame(
                &self.leftover[off..off + VAD_FRAME_SAMPLES],
                &mut self.state,
            )?;
            off += VAD_FRAME_SAMPLES;
            if self.hangover.update(prob, VAD_FRAME_SAMPLES) {
                endpoint = true;
            }
        }
        // Retain only the unprocessed tail.
        if off > 0 {
            self.leftover.drain(..off);
        }
        Ok(endpoint)
    }
}

/// Pure trailing-silence state machine shared by the streaming endpointer.
///
/// `update` is fed one frame's probability at a time and returns `true` on the
/// single frame where trailing silence first reaches `min_silence_ms` after
/// speech has been observed. After firing it disarms until speech resumes, so
/// one utterance yields exactly one endpoint.
#[derive(Debug)]
pub struct Hangover {
    threshold: f32,
    min_silence_samples: usize,
    seen_speech: bool,
    trailing_silence: usize,
    armed: bool,
}

impl Hangover {
    fn new(cfg: &VadConfig) -> Self {
        Self {
            threshold: cfg.threshold,
            min_silence_samples: VadConfig::ms_to_samples(cfg.min_silence_ms),
            seen_speech: false,
            trailing_silence: 0,
            armed: false,
        }
    }

    /// Advance by one frame of `frame_samples` samples with speech probability
    /// `prob`. Returns `true` on the endpoint-crossing frame. Thresholds are
    /// fixed at construction ([`Hangover::new`]).
    fn update(&mut self, prob: f32, frame_samples: usize) -> bool {
        if prob >= self.threshold {
            self.seen_speech = true;
            self.armed = true;
            self.trailing_silence = 0;
            return false;
        }
        if !self.seen_speech {
            return false;
        }
        self.trailing_silence += frame_samples;
        if self.armed && self.trailing_silence >= self.min_silence_samples {
            self.armed = false; // fire once until speech resumes
            return true;
        }
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg(
        threshold: f32,
        min_silence_ms: u32,
        min_speech_ms: u32,
        speech_pad_ms: u32,
    ) -> VadConfig {
        VadConfig {
            threshold,
            min_silence_ms,
            min_speech_ms,
            speech_pad_ms,
        }
    }

    #[test]
    fn test_ms_to_samples_16khz() {
        assert_eq!(VadConfig::ms_to_samples(1000), 16000);
        assert_eq!(VadConfig::ms_to_samples(500), 8000);
        assert_eq!(VadConfig::ms_to_samples(0), 0);
    }

    #[test]
    fn test_regions_empty_probs_is_empty() {
        let c = VadConfig::default();
        assert!(regions_from_probs(&[], 512, 0, &c).is_empty());
        assert!(regions_from_probs(&[0.9, 0.9], 512, 0, &c).is_empty());
    }

    #[test]
    fn test_regions_all_silence_is_empty() {
        let c = cfg(0.5, 0, 0, 0);
        let probs = vec![0.1f32; 10];
        assert!(regions_from_probs(&probs, 512, 10 * 512, &c).is_empty());
    }

    #[test]
    fn test_regions_single_block_no_pad_no_mins() {
        let c = cfg(0.5, 0, 0, 0);
        // frames: silence, speech, speech, silence
        let probs = [0.1, 0.9, 0.9, 0.1];
        let r = regions_from_probs(&probs, 100, 400, &c);
        assert_eq!(r, vec![(100, 300)]);
    }

    #[test]
    fn test_regions_trailing_speech_clamps_to_total() {
        let c = cfg(0.5, 0, 0, 0);
        let probs = [0.1, 0.9, 0.9];
        // last speech run never closes → clamp to total_samples (not 3*100).
        let r = regions_from_probs(&probs, 100, 250, &c);
        assert_eq!(r, vec![(100, 250)]);
    }

    #[test]
    fn test_regions_min_silence_merges_short_gap() {
        // gap of one 100-sample frame = 100 samples; min_silence 1000 samples
        // (≈ wide) so the two speech blocks merge into one.
        let c = cfg(0.5, /*min_silence_ms*/ 100, 0, 0); // 100ms = 1600 samples
        let probs = [0.9, 0.1, 0.9];
        let r = regions_from_probs(&probs, 100, 300, &c);
        assert_eq!(r, vec![(0, 300)]);
    }

    #[test]
    fn test_regions_long_gap_keeps_two_regions() {
        // min_silence small (0) so any gap splits.
        let c = cfg(0.5, 0, 0, 0);
        let probs = [0.9, 0.1, 0.1, 0.9];
        let r = regions_from_probs(&probs, 100, 400, &c);
        assert_eq!(r, vec![(0, 100), (300, 400)]);
    }

    #[test]
    fn test_regions_min_speech_drops_short_blip() {
        // One 100-sample speech frame, min_speech 1600 samples → dropped.
        let c = cfg(0.5, 0, /*min_speech_ms*/ 100, 0);
        let probs = [0.1, 0.9, 0.1];
        assert!(regions_from_probs(&probs, 100, 300, &c).is_empty());
    }

    #[test]
    fn test_regions_padding_extends_and_clamps() {
        let c = cfg(0.5, 0, 0, /*speech_pad_ms*/ 10); // 10ms = 160 samples
        let probs = [0.1, 0.9, 0.1];
        // raw region (100, 200); pad ±160 → (0 clamped, 360).
        let r = regions_from_probs(&probs, 100, 1000, &c);
        assert_eq!(r, vec![(0, 360)]);
    }

    #[test]
    fn test_regions_padding_merges_overlapping_neighbours() {
        let c = cfg(0.5, 0, 0, 50); // 50ms = 800 samples pad
        // raw regions (0,100) and (300,400) — the trailing silence frame closes
        // the second run at 400; pad ±800 makes them overlap → merge to (0,1200).
        let probs = [0.9, 0.1, 0.1, 0.9, 0.1];
        let r = regions_from_probs(&probs, 100, 2000, &c);
        assert_eq!(r, vec![(0, 1200)]);
    }

    #[test]
    fn test_hangover_fires_once_after_min_silence() {
        let c = cfg(0.5, /*min_silence_ms*/ 100, 0, 0); // 1600 samples = ~3.125 frames @512
        let mut h = Hangover::new(&c);
        // speech
        assert!(!h.update(0.9, 512));
        // silence accumulates: need >=1600 samples → 4 frames (2048) to cross.
        assert!(!h.update(0.1, 512)); // 512
        assert!(!h.update(0.1, 512)); // 1024
        assert!(!h.update(0.1, 512)); // 1536
        assert!(h.update(0.1, 512)); // 2048 >= 1600 → fire
        // does not fire again on continued silence
        assert!(!h.update(0.1, 512));
    }

    #[test]
    fn test_hangover_no_fire_before_any_speech() {
        let c = cfg(0.5, 0, 0, 0);
        let mut h = Hangover::new(&c);
        // leading silence must never fire (no speech seen yet).
        for _ in 0..10 {
            assert!(!h.update(0.1, 512));
        }
    }

    #[test]
    fn test_hangover_rearms_for_next_utterance() {
        let c = cfg(0.5, 50, 0, 0); // 800 samples → 2 frames @512 (1024) to cross
        let mut h = Hangover::new(&c);
        h.update(0.9, 512); // speech
        assert!(!h.update(0.1, 512)); // 512
        assert!(h.update(0.1, 512)); // 1024 >= 800 → fire #1
        // new speech re-arms
        assert!(!h.update(0.9, 512));
        assert!(!h.update(0.1, 512)); // 512
        assert!(h.update(0.1, 512)); // 1024 → fire #2
    }

    #[test]
    fn test_remap_no_regions_is_identity() {
        assert_eq!(remap_compressed_seconds(1.5, &[], 16000.0), 1.5);
    }

    #[test]
    fn test_remap_single_region_offsets_by_start() {
        // One region [16000, 32000) = original [1.0s, 2.0s). Compressed time 0
        // maps to 1.0s; compressed 0.5s maps to 1.5s.
        let regions = [(16000usize, 32000usize)];
        assert_eq!(remap_compressed_seconds(0.0, &regions, 16000.0), 1.0);
        assert_eq!(remap_compressed_seconds(0.5, &regions, 16000.0), 1.5);
    }

    #[test]
    fn test_remap_second_region_skips_silence_gap() {
        // Regions: [0, 16000) then [48000, 64000) — a 2 s silence gap was cut.
        // Compressed timeline: [0,1s) then [1s,2s). A compressed time of 1.5s
        // falls in the second region 0.5s in → original 48000/16000 + 0.5 = 3.5s.
        let regions = [(0usize, 16000usize), (48000usize, 64000usize)];
        assert_eq!(remap_compressed_seconds(0.5, &regions, 16000.0), 0.5);
        assert_eq!(remap_compressed_seconds(1.5, &regions, 16000.0), 3.5);
    }

    #[test]
    fn test_remap_past_end_clamps_to_last_region_end() {
        let regions = [(0usize, 16000usize), (48000usize, 64000usize)];
        // Compressed 10s is well past total speech (2s) → clamp to 64000/16000 = 4.0s.
        assert_eq!(remap_compressed_seconds(10.0, &regions, 16000.0), 4.0);
    }

    /// Model-gated: exercises the real Silero ONNX session through `ort` to
    /// confirm the I/O plumbing (scalar `sr`, `[2,1,128]` recurrent state).
    /// Run with the model present at `~/.gigastt/models/vad/silero_vad.onnx`:
    /// `cargo test -p gigastt-core --lib vad::tests::test_silero -- --ignored`.
    #[test]
    #[ignore = "requires the Silero VAD model at ~/.gigastt/models/vad/silero_vad.onnx"]
    fn test_silero_silence_low_prob_and_runs() {
        let home = std::env::var("HOME").expect("HOME");
        let path = std::path::PathBuf::from(home).join(".gigastt/models/vad/silero_vad.onnx");
        // The Silero VAD model is a separate, optional download (not part of the
        // GigaAM model cache). Skip gracefully when it is absent so the
        // `--ignored` coverage run doesn't fail where only GigaAM is present.
        if !path.exists() {
            eprintln!("skipping {}: Silero VAD model not present", path.display());
            return;
        }
        let vad = SileroVad::load(&path).expect("load silero");

        // 1 s of pure silence → several frames, all low probability.
        let silence = vec![0.0f32; 16000];
        let probs = vad.frame_probs(&silence).expect("frame_probs");
        assert!(!probs.is_empty(), "expected at least one frame");
        for p in &probs {
            assert!((0.0..=1.0).contains(p), "prob {p} out of range");
        }
        let max_silence = probs.iter().cloned().fold(0.0f32, f32::max);
        assert!(
            max_silence < 0.5,
            "silence should be below threshold, got {max_silence}"
        );

        // A loud 200 Hz tone is not speech either, but it must run cleanly and
        // stay in range (the point is to exercise the session, not classify).
        let tone: Vec<f32> = (0..16000)
            .map(|i| 0.5 * (2.0 * std::f32::consts::PI * 200.0 * i as f32 / 16000.0).sin())
            .collect();
        let probs2 = vad.frame_probs(&tone).expect("frame_probs tone");
        for p in &probs2 {
            assert!((0.0..=1.0).contains(p), "tone prob {p} out of range");
        }

        // No speech anywhere → no regions.
        assert!(
            vad.speech_regions(&silence, &VadConfig::default())
                .expect("regions")
                .is_empty()
        );
    }

    fn silero_model_path() -> std::path::PathBuf {
        let home = std::env::var("HOME").expect("HOME");
        std::path::PathBuf::from(home).join(".gigastt/models/vad/silero_vad.onnx")
    }

    /// Model-gated: drives [`VadEndpointer::push`] with sub-frame chunks to
    /// exercise the leftover-buffer accumulation + drain across `push` calls
    /// (the model is required because `push` runs every full frame through the
    /// real Silero session). Verifies the chunk-accumulation mechanics, not
    /// classification: chunks that individually fall short of one 512-sample
    /// frame must not error and must not endpoint (no frame processed yet); once
    /// a full frame's worth of samples accumulates, the frame is consumed and
    /// the remainder retained for the next push.
    #[test]
    #[ignore = "requires the Silero VAD model at ~/.gigastt/models/vad/silero_vad.onnx"]
    fn test_endpointer_buffers_subframe_chunks_across_pushes() {
        let path = silero_model_path();
        if !path.exists() {
            eprintln!("skipping {}: Silero VAD model not present", path.display());
            return;
        }
        let vad = SileroVad::load(&path).expect("load silero");
        let c = VadConfig::default();
        let mut ep = VadEndpointer::new(&c);

        // Two sub-frame silence chunks that together fall short of one frame:
        // no frame is processed, so no endpoint.
        let part = vec![0.0f32; 200];
        assert!(!ep.push(&vad, &part).expect("push part 1"));
        assert!(!ep.push(&vad, &part).expect("push part 2")); // 400 < 512 buffered

        // A third chunk crosses the frame boundary (600 buffered) → exactly one
        // full frame is consumed and the remainder retained; still no endpoint
        // on silence alone.
        let rest = vec![0.0f32; 200];
        assert!(!ep.push(&vad, &rest).expect("push part 3")); // 600 buffered, 1 frame
    }

    /// Model-gated: a single large silence chunk processes many frames in one
    /// `push` (the inner accumulation loop) and must never endpoint before any
    /// speech is seen; a following empty push processes no frames and stays
    /// non-endpointing.
    #[test]
    #[ignore = "requires the Silero VAD model at ~/.gigastt/models/vad/silero_vad.onnx"]
    fn test_endpointer_no_endpoint_on_leading_silence() {
        let path = silero_model_path();
        if !path.exists() {
            eprintln!("skipping {}: Silero VAD model not present", path.display());
            return;
        }
        let vad = SileroVad::load(&path).expect("load silero");
        let c = VadConfig::default();
        let mut ep = VadEndpointer::new(&c);

        // 1 s of silence = ~31 frames in a single push; leading silence (no
        // speech yet) must never report an endpoint.
        let silence = vec![0.0f32; 16000];
        assert!(
            !ep.push(&vad, &silence).expect("push silence"),
            "leading silence must not endpoint"
        );
        // A follow-up empty push processes no frames and stays non-endpointing.
        assert!(!ep.push(&vad, &[]).expect("push empty"));
    }
}
