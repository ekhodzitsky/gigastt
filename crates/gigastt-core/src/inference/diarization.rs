//! Speaker diarization via polyvoice, with WeSpeaker inference owned by gigastt.
//!
//! Both offline and streaming pipelines use the same CPU ONNX Runtime embedder.
//! The frontend retains WeSpeaker's rank-3, 80-bin fbank + CMVN contract;
//! polyvoice supplies VAD and clustering, without choosing the production backend.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use parking_lot::Mutex;
use polyvoice::pipeline::{LegacyPipeline, LegacyPipelineError};
use polyvoice::streaming::StreamingPipeline;
use polyvoice::{
    ClusterConfig, DiarizationConfig as DiaConfig, Embedder, EmbedderError, EnergyVad,
    FbankOnnxExtractor, VadConfig,
};

use super::DiarizationOutcome;

mod speaker;
use speaker::SpeakerEmbedder;
type LoadedSpeakerEncoder = Arc<SpeakerEmbedder>;

/// WeSpeaker ResNet34 embedding dimension.
pub(crate) const SPEAKER_EMBEDDING_DIM: usize = 256;
/// CPU session pool size shared across concurrent diarization sessions.
const SPEAKER_POOL_SIZE: usize = 4;

/// Legacy polyvoice encoder handle retained for Rust source compatibility.
/// Engine-owned speaker inference uses gigastt's CPU runtime instead.
pub type SpeakerEncoder = Arc<FbankOnnxExtractor>;

/// Per-session streaming diarization state.
pub type StreamingDiarizationState = StreamingPipeline<EnergyVad, SharedExtractor>;

/// Adapter that lets a single shared CPU speaker encoder back the
/// per-session [`StreamingPipeline`]s, which take ownership of their extractor.
/// The session pool inside the extractor is shared across sessions via `Arc`.
pub struct SharedExtractor(LoadedSpeakerEncoder);

impl Embedder for SharedExtractor {
    fn dim(&self) -> usize {
        self.0.dim()
    }

    fn embed(&self, samples: &[f32]) -> Result<Vec<f32>, EmbedderError> {
        #[cfg(test)]
        let _probe = crate::sidecar_probe::Probe::new("speaker_embedding_total");
        self.0.embed(samples)
    }
}

/// Load the fixed WeSpeaker frontend and CPU runtime session pool.
pub(crate) fn load_speaker_encoder(
    model_path: &Path,
    pool_size: usize,
) -> anyhow::Result<SpeakerEmbedder> {
    SpeakerEmbedder::load(model_path, pool_size)
}

/// Lazy WeSpeaker handle: path probed at engine boot, encoder loaded on
/// first diarization request so unused speaker files do not inflate ready RSS.
///
/// Load is attempted once. Success or permanent failure is cached so concurrent
/// diarization requests do not race multiple session opens, and a corrupt
/// model does not re-spam warnings on every request.
pub struct LazySpeakerEncoder {
    path: PathBuf,
    slot: Mutex<SpeakerLoadSlot>,
}

enum SpeakerLoadSlot {
    /// File present; ONNX session not yet opened.
    Pending,
    Ready(LoadedSpeakerEncoder),
    /// Load was attempted and failed; do not retry until engine reload.
    Failed,
}

impl LazySpeakerEncoder {
    #[cfg(test)]
    pub(crate) fn is_pending(&self) -> bool {
        matches!(*self.slot.lock(), SpeakerLoadSlot::Pending)
    }

    /// True when the speaker encoder is resident.
    #[cfg(test)]
    pub(crate) fn is_loaded(&self) -> bool {
        matches!(*self.slot.lock(), SpeakerLoadSlot::Ready(_))
    }

    /// Path that will be (or was) used for the ONNX load.
    #[cfg(test)]
    pub(crate) fn path(&self) -> &Path {
        &self.path
    }

    /// Return a shared encoder, loading on first call.
    ///
    /// Returns `None` when load fails (already logged). Subsequent calls after a
    /// failure also return `None` without re-attempting.
    pub fn get_or_load(&self) -> Option<LoadedSpeakerEncoder> {
        let mut slot = self.slot.lock();
        match &*slot {
            SpeakerLoadSlot::Ready(enc) => return Some(Arc::clone(enc)),
            SpeakerLoadSlot::Failed => return None,
            SpeakerLoadSlot::Pending => {}
        }
        match load_speaker_encoder(&self.path, SPEAKER_POOL_SIZE) {
            Ok(enc) => {
                tracing::info!("Speaker encoder loaded (diarization available)");
                let enc = Arc::new(enc);
                *slot = SpeakerLoadSlot::Ready(Arc::clone(&enc));
                Some(enc)
            }
            Err(e) => {
                tracing::warn!("Speaker encoder not loaded, diarization unavailable: {e:#}");
                *slot = SpeakerLoadSlot::Failed;
                None
            }
        }
    }
}

/// Probe for `model_dir/wespeaker_resnet34.onnx` without opening a session.
///
/// Returns `None` when the file is missing (diarization unavailable). Presence
/// alone is enough to advertise diarization capability; the session is opened
/// later via [`LazySpeakerEncoder::get_or_load`].
pub fn probe_speaker_encoder(model_dir: &Path) -> Option<LazySpeakerEncoder> {
    let path = model_dir.join("wespeaker_resnet34.onnx");
    if !path.exists() {
        tracing::warn!("wespeaker_resnet34.onnx not found, diarization unavailable");
        return None;
    }
    tracing::info!(
        "Speaker encoder present at {} (lazy load on first diarization request)",
        path.display()
    );
    Some(LazySpeakerEncoder {
        path,
        slot: Mutex::new(SpeakerLoadSlot::Pending),
    })
}

/// Open a per-session streaming diarization pipeline sharing `encoder`.
pub fn open_streaming(encoder: &LoadedSpeakerEncoder) -> Option<StreamingDiarizationState> {
    let config = DiaConfig {
        cluster: ClusterConfig {
            threshold: 0.5,
            ..ClusterConfig::default()
        },
        ..DiaConfig::default()
    };
    let vad_config = VadConfig::default();
    let vad = EnergyVad::new(-40.0, 16000, vad_config.frame_size);
    let extractor = SharedExtractor(Arc::clone(encoder));
    match StreamingPipeline::new(vad, extractor, config, vad_config) {
        Ok(pipeline) => Some(pipeline),
        Err(e) => {
            tracing::warn!("Failed to initialize streaming diarization: {e:#}");
            None
        }
    }
}

/// Feed PCM samples into the streaming pipeline; log and ignore feed errors.
pub fn feed_chunk(state: &mut StreamingDiarizationState, samples: &[f32]) {
    if let Err(e) = state.feed(samples) {
        tracing::warn!("Diarization feed failed: {e:#}");
    }
}

/// Speaker index of the most recent turn, if any.
pub fn last_turn_speaker(state: &StreamingDiarizationState) -> Option<u32> {
    state.turns().last().map(|t| t.speaker.0)
}

/// One offline diarization turn (seconds + speaker index).
#[derive(Debug, Clone)]
pub struct LabeledTurn {
    pub start: f64,
    pub end: f64,
    pub speaker: u32,
}

/// Run offline diarization over `samples` (16 kHz mono f32).
///
/// `Ok(turns)` on success (empty input yields `Ok(vec![])`). On failure the
/// error is *classified* so the caller can tell the client why no speakers were
/// produced instead of degrading silently:
/// - [`DiarizationOutcome::DurationCeiling`] when the clusterer refuses the
///   buffer for exceeding its single-pass duration limit — carrying the real
///   input and ceiling seconds polyvoice reported — and
/// - [`DiarizationOutcome::Failed`] for any other pipeline error (still logged).
pub fn run_offline(
    encoder: &LoadedSpeakerEncoder,
    samples: &[f32],
) -> Result<Vec<LabeledTurn>, DiarizationOutcome> {
    #[cfg(test)]
    let _probe = crate::sidecar_probe::Probe::new("diarization_total");
    let config = DiaConfig::default();
    let vad_config = VadConfig::default();
    let pipeline = LegacyPipeline::new(config, vad_config);
    let mut vad = EnergyVad::new(-40.0, 16000, vad_config.frame_size);
    match pipeline.run(samples, encoder.as_ref(), &mut vad) {
        Ok(dia_result) => Ok(dia_result
            .turns
            .into_iter()
            .map(|t| LabeledTurn {
                start: t.time.start,
                end: t.time.end,
                speaker: t.speaker.0,
            })
            .collect()),
        Err(e) => Err(classify_offline_error(e)),
    }
}

/// Internal cancellation stays separate from capability/decode outcomes.
pub(crate) enum OfflineRunError {
    Cancelled,
    Declined(DiarizationOutcome),
}

struct CancellableEmbedder<'a, E> {
    inner: &'a E,
    abort: &'a (dyn Fn() -> bool + Sync),
}

impl<E: Embedder> Embedder for CancellableEmbedder<'_, E> {
    fn dim(&self) -> usize {
        self.inner.dim()
    }
    fn embed(&self, samples: &[f32]) -> Result<Vec<f32>, EmbedderError> {
        if (self.abort)() {
            return Err(EmbedderError::InferenceFailed {
                detail: "cancelled".into(),
            });
        }
        let result = self.inner.embed(samples);
        if (self.abort)() {
            return Err(EmbedderError::InferenceFailed {
                detail: "cancelled".into(),
            });
        }
        result
    }
}

struct CancellableVad<'a, V> {
    inner: V,
    abort: &'a (dyn Fn() -> bool + Sync),
}

impl<V: polyvoice::VoiceActivityDetector> polyvoice::VoiceActivityDetector
    for CancellableVad<'_, V>
{
    fn reset(&mut self) {
        self.inner.reset();
    }
    fn sample_rate(&self) -> u32 {
        self.inner.sample_rate()
    }
    fn process(&mut self, samples: &[f32]) -> Result<Vec<f32>, polyvoice::VadError> {
        if (self.abort)() {
            return Err(polyvoice::VadError::Model("cancelled".into()));
        }
        let result = self.inner.process(samples);
        if (self.abort)() {
            return Err(polyvoice::VadError::Model("cancelled".into()));
        }
        result
    }
}

/// The existing pipeline still owns clustering. Active model calls and the
/// clustering call finish synchronously; checks prevent starting later stages.
pub(crate) fn run_offline_with_abort(
    encoder: &LoadedSpeakerEncoder,
    samples: &[f32],
    abort: Option<&(dyn Fn() -> bool + Sync)>,
) -> Result<Vec<LabeledTurn>, OfflineRunError> {
    let Some(abort) = abort else {
        return run_offline(encoder, samples).map_err(OfflineRunError::Declined);
    };
    if abort() {
        return Err(OfflineRunError::Cancelled);
    }
    let vad_config = VadConfig::default();
    let pipeline = LegacyPipeline::new(DiaConfig::default(), vad_config);
    let mut vad = CancellableVad {
        inner: EnergyVad::new(-40.0, 16000, vad_config.frame_size),
        abort,
    };
    let extractor = CancellableEmbedder {
        inner: encoder.as_ref(),
        abort,
    };
    let result = pipeline.run(samples, &extractor, &mut vad);
    if abort() {
        return Err(OfflineRunError::Cancelled);
    }
    result
        .map(|result| {
            result
                .turns
                .into_iter()
                .map(|turn| LabeledTurn {
                    start: turn.time.start,
                    end: turn.time.end,
                    speaker: turn.speaker.0,
                })
                .collect()
        })
        .map_err(|error| OfflineRunError::Declined(classify_offline_error(error)))
}

/// Map a polyvoice [`LegacyPipelineError`] to the client-facing [`DiarizationOutcome`].
///
/// The duration ceiling is surfaced with the real numbers polyvoice reported;
/// every other failure is logged and collapsed to
/// [`DiarizationOutcome::Failed`].
fn classify_offline_error(e: LegacyPipelineError) -> DiarizationOutcome {
    match e {
        LegacyPipelineError::AudioTooLong {
            actual_secs,
            max_secs,
        } => DiarizationOutcome::DurationCeiling {
            input_secs: actual_secs as f64,
            ceiling_secs: max_secs as f64,
        },
        other => {
            tracing::warn!("Offline diarization failed: {other:#}");
            DiarizationOutcome::Failed
        }
    }
}

#[cfg(test)]
thread_local! {
    static TURN_CONTAINMENT_CHECKS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

fn turn_contains(turn: &LabeledTurn, midpoint: f64) -> bool {
    #[cfg(test)]
    TURN_CONTAINMENT_CHECKS.with(|count| count.set(count.get() + 1));
    turn.start <= midpoint && turn.end >= midpoint
}

/// Assign `speaker` on each word by midpoint-in-turn lookup.
/// Ordered, non-overlapping turns and ordered finite midpoints use a linear scan.
/// Other inputs retain the original first-matching-turn behavior.
pub fn assign_speakers_by_midpoint(turns: &[LabeledTurn], words: &mut [super::WordInfo]) {
    if turns.is_empty() || words.is_empty() {
        return;
    }
    let ordered_turns = turns
        .iter()
        .all(|t| t.start.is_finite() && t.end.is_finite() && t.start <= t.end)
        && turns.windows(2).all(|pair| pair[0].end <= pair[1].start);
    let ordered_words = words
        .iter()
        .try_fold(f64::NEG_INFINITY, |previous, word| {
            let midpoint = (word.start + word.end) / 2.0;
            (midpoint.is_finite() && midpoint >= previous).then_some(midpoint)
        })
        .is_some();
    if ordered_turns && ordered_words {
        let mut cursor = 0;
        for word in words {
            let mid = (word.start + word.end) / 2.0;
            // Strict inequality preserves the earlier turn at a shared endpoint.
            while cursor < turns.len() && turns[cursor].end < mid {
                cursor += 1;
            }
            if let Some(turn) = turns.get(cursor).filter(|turn| turn_contains(turn, mid)) {
                word.speaker = Some(turn.speaker);
            }
        }
        return;
    }
    // Keep original first-match semantics for overlaps, unsorted or nonfinite data.
    for word in words {
        let mid = (word.start + word.end) / 2.0;
        if let Some(turn) = turns.iter().find(|t| turn_contains(t, mid)) {
            word.speaker = Some(turn.speaker);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // A duration-ceiling refusal must surface the *real* numbers polyvoice
    // reported (not a re-derived guess) so the client sees both the input and
    // the ceiling. This is the sole case that carries numbers.
    #[test]
    fn test_classify_offline_error_duration_ceiling_carries_numbers() {
        let outcome = classify_offline_error(LegacyPipelineError::AudioTooLong {
            actual_secs: 5400.0,
            max_secs: 3600.0,
        });
        assert_eq!(
            outcome,
            DiarizationOutcome::DurationCeiling {
                input_secs: 5400.0,
                ceiling_secs: 3600.0,
            }
        );
    }

    // Every other pipeline error collapses to `Failed` (logged, no numbers).
    #[test]
    fn test_classify_offline_error_other_is_failed() {
        assert_eq!(
            classify_offline_error(LegacyPipelineError::UnsupportedSampleRate {
                expected: 16_000,
                actual: 8_000,
            }),
            DiarizationOutcome::Failed
        );
    }

    // Missing models fail without panicking; the real-model oracle separately
    // verifies rank-3 fbank input and CPU inference behavior.
    #[test]
    fn test_load_speaker_encoder_missing_model_errors() {
        let missing = Path::new("/nonexistent/gigastt-test/wespeaker_resnet34.onnx");
        let result = load_speaker_encoder(missing, 1);
        assert!(
            result.is_err(),
            "a missing WeSpeaker model must surface as Err, not panic or Ok"
        );
    }

    #[test]
    fn test_probe_speaker_encoder_absent_returns_none() {
        let dir = tempfile::tempdir().expect("tempdir");
        assert!(
            probe_speaker_encoder(dir.path()).is_none(),
            "missing wespeaker file must not advertise diarization"
        );
    }

    /// Presence of the speaker file only probes — no encoder session until
    /// `get_or_load`. A zero-byte placeholder is enough to exercise the probe
    /// without shipping the real WeSpeaker weights in unit tests.
    #[test]
    fn test_probe_speaker_encoder_defers_onnx_load() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("wespeaker_resnet34.onnx");
        std::fs::write(&path, b"").expect("write placeholder");

        let lazy = probe_speaker_encoder(dir.path()).expect("file present → probe succeeds");
        assert_eq!(lazy.path(), path.as_path());
        assert!(
            !lazy.is_loaded(),
            "probe must not open a speaker-encoder session at boot"
        );

        // Corrupt/empty speaker model fails once and stays failed (no retry storm).
        assert!(lazy.get_or_load().is_none());
        assert!(!lazy.is_loaded());
        assert!(
            lazy.get_or_load().is_none(),
            "failed load must be sticky until engine reload"
        );
    }

    #[test]
    #[ignore = "requires the WeSpeaker diarization model"]
    fn test_speaker_encoder_accepts_waveform_audio() {
        let model_path =
            Path::new(&crate::model::default_model_dir()).join("wespeaker_resnet34.onnx");
        let encoder = load_speaker_encoder(&model_path, 1).expect("speaker encoder should load");
        let samples: Vec<f32> = (0..24_000)
            .map(|i| {
                let phase = std::f32::consts::TAU * 220.0 * i as f32 / 16_000.0;
                0.1 * phase.sin()
            })
            .collect();

        let embedding = encoder
            .embed(&samples)
            .expect("waveform must be converted to rank-3 fbank features");

        assert_eq!(embedding.len(), SPEAKER_EMBEDDING_DIM);
        assert!(embedding.iter().all(|value| value.is_finite()));
    }

    #[test]
    #[ignore = "requires the WeSpeaker diarization model"]
    fn test_lazy_speaker_encoder_loads_on_demand() {
        let model_dir = crate::model::default_model_dir();
        let lazy = probe_speaker_encoder(Path::new(&model_dir))
            .expect("WeSpeaker model should be present");
        assert!(!lazy.is_loaded());
        let enc = lazy.get_or_load().expect("first get_or_load should load");
        assert!(lazy.is_loaded());
        // Second call returns the same Arc (shared session pool).
        let enc2 = lazy.get_or_load().expect("second get_or_load");
        assert!(Arc::ptr_eq(&enc, &enc2));
    }
}

#[cfg(test)]
mod assignment_tests {
    use super::*;
    use crate::inference::WordInfo;

    fn meeting(size: usize) -> (Vec<LabeledTurn>, Vec<WordInfo>) {
        let turns = (0..size)
            .map(|i| LabeledTurn {
                start: i as f64,
                end: i as f64 + 1.0,
                speaker: (i % 4) as u32,
            })
            .collect();
        let words = (0..size)
            .map(|i| WordInfo::new("word", i as f64 + 0.1, i as f64 + 0.9, 0.8, None))
            .collect();
        (turns, words)
    }

    #[test]
    fn test_ordered_speaker_assignment_avoids_quadratic_turn_search() {
        let (turns, mut words) = meeting(1000);
        TURN_CONTAINMENT_CHECKS.with(|count| count.set(0));
        assign_speakers_by_midpoint(&turns, &mut words);
        assert!(
            words
                .iter()
                .enumerate()
                .all(|(i, w)| w.speaker == Some((i % 4) as u32))
        );
        let comparisons = TURN_CONTAINMENT_CHECKS.with(|count| count.get());
        assert!(
            comparisons <= 2 * (turns.len() + words.len()),
            "{comparisons} turn containment checks"
        );
    }
    fn legacy(turns: &[LabeledTurn], words: &mut [WordInfo]) {
        for word in words {
            let mid = (word.start + word.end) / 2.0;
            if let Some(turn) = turns.iter().find(|t| t.start <= mid && t.end >= mid) {
                word.speaker = Some(turn.speaker);
            }
        }
    }

    #[test]
    fn test_speaker_assignment_preserves_boundaries_gaps_overlaps_and_unsorted_inputs() {
        let cases = [
            vec![],
            vec![(0.0, 1.0), (1.0, 2.0), (3.0, 4.0)],
            vec![(0.0, 3.0), (1.0, 2.0)],
            vec![(3.0, 4.0), (0.0, 2.0)],
            vec![(0.0, 0.0), (0.0, 1.0), (1.0, 1.0)],
            vec![(f64::NAN, 2.0), (0.0, f64::INFINITY)],
            vec![(2.0, 1.0), (0.0, 3.0)],
        ];
        for intervals in cases {
            let turns: Vec<_> = intervals
                .into_iter()
                .enumerate()
                .map(|(i, (start, end))| LabeledTurn {
                    start,
                    end,
                    speaker: i as u32,
                })
                .collect();
            for reverse in [false, true] {
                for nonfinite in [false, true] {
                    let mut mids = vec![-1.0, 0.0, 0.5, 1.0, 2.0, 2.5, 3.0, 4.0, 5.0];
                    if nonfinite {
                        mids.extend([f64::NAN, f64::INFINITY, f64::NEG_INFINITY]);
                    }
                    if reverse {
                        mids.reverse();
                    }
                    let mut actual: Vec<_> = mids
                        .into_iter()
                        .map(|m| WordInfo::new("word", m, m, 0.8, Some(99)))
                        .collect();
                    let mut expected = actual.clone();
                    legacy(&turns, &mut expected);
                    assign_speakers_by_midpoint(&turns, &mut actual);
                    assert_eq!(
                        actual.iter().map(|w| w.speaker).collect::<Vec<_>>(),
                        expected.iter().map(|w| w.speaker).collect::<Vec<_>>()
                    );
                }
            }
            assign_speakers_by_midpoint(&turns, &mut []);
        }
    }

    #[test]
    #[ignore = "synthetic assignment-only timing; run explicitly with --nocapture"]
    fn benchmark_speaker_assignment_scaling() {
        // The candidate includes the test-only containment counter; these
        // debug timings are conservative and exclude embedding/clustering.
        for size in [100, 1000, 5000] {
            let (turns, template) = meeting(size);
            let mut old = Vec::new();
            let mut new = Vec::new();
            for round in 0..6 {
                for optimized in if round % 2 == 0 {
                    [false, true]
                } else {
                    [true, false]
                } {
                    let mut words = template.clone();
                    let start = std::time::Instant::now();
                    if optimized {
                        assign_speakers_by_midpoint(&turns, &mut words);
                    } else {
                        legacy(&turns, &mut words);
                    }
                    let nanos = start.elapsed().as_nanos();
                    std::hint::black_box(words);
                    if optimized {
                        new.push(nanos);
                    } else {
                        old.push(nanos);
                    }
                }
            }
            old.sort_unstable();
            new.sort_unstable();
            eprintln!(
                "words={size} turns={size} legacy_median_ns={} ordered_median_ns={}",
                old[old.len() / 2],
                new[new.len() / 2]
            );
        }
    }
}

#[cfg(test)]
mod cancellation_tests {
    use super::*;
    use polyvoice::VoiceActivityDetector;
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

    struct CountingEmbedder<'a> {
        calls: &'a AtomicUsize,
        cancelled: &'a AtomicBool,
    }
    impl Embedder for CountingEmbedder<'_> {
        fn dim(&self) -> usize {
            2
        }
        fn embed(&self, _: &[f32]) -> Result<Vec<f32>, EmbedderError> {
            self.calls.fetch_add(1, Ordering::Relaxed);
            self.cancelled.store(true, Ordering::Relaxed);
            Ok(vec![1.0, 0.0])
        }
    }
    #[test]
    fn test_diarization_embed_cancellation_stops_before_next_window() {
        let calls = AtomicUsize::new(0);
        let cancelled = AtomicBool::new(false);
        let inner = CountingEmbedder {
            calls: &calls,
            cancelled: &cancelled,
        };
        let abort = || cancelled.load(Ordering::Relaxed);
        let guarded = CancellableEmbedder {
            inner: &inner,
            abort: &abort,
        };
        let config = VadConfig::default();
        let pipeline = LegacyPipeline::new(DiaConfig::default(), config);
        let samples = vec![0.5; 16000 * 8];
        let mut vad = EnergyVad::new(-40.0, 16000, config.frame_size);
        assert!(pipeline.run(&samples, &guarded, &mut vad).is_err());
        assert!(guarded.embed(&samples).is_err());
        assert_eq!(calls.load(Ordering::Relaxed), 1);

        // Without cancellation the unchanged pipeline reaches later windows.
        calls.store(0, Ordering::Relaxed);
        let expected = pipeline.run(&samples, &inner, &mut vad).unwrap();
        assert!(calls.load(Ordering::Relaxed) > 1);
        let guarded = CancellableEmbedder {
            inner: &inner,
            abort: &|| false,
        };
        let mut vad = CancellableVad {
            inner: EnergyVad::new(-40.0, 16000, config.frame_size),
            abort: &|| false,
        };
        let actual = pipeline.run(&samples, &guarded, &mut vad).unwrap();
        assert_eq!(actual.turns.len(), expected.turns.len());
        for (actual, expected) in actual.turns.iter().zip(&expected.turns) {
            assert_eq!(actual.time.start, expected.time.start);
            assert_eq!(actual.time.end, expected.time.end);
            assert_eq!(actual.speaker, expected.speaker);
        }
    }

    #[test]
    fn test_diarization_vad_cancellation_prevents_processing() {
        let mut guarded = CancellableVad {
            inner: EnergyVad::new(-40.0, 16000, 512),
            abort: &|| true,
        };
        // Empty input normally succeeds, so the error proves the boundary check.
        assert!(guarded.process(&[]).is_err());
    }
}

#[cfg(all(test, feature = "file-decode"))]
mod model_tests;
