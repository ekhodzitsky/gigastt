//! Speaker diarization via [polyvoice] (feature-gated).
//!
//! Both pipelines take the v1.0 [`polyvoice::Embedder`] contract: the offline
//! [`polyvoice::pipeline::LegacyPipeline`] and the per-session
//! [`polyvoice::streaming::StreamingPipeline`] are generic over `E: Embedder`,
//! and [`FbankOnnxExtractor`] implements it directly (tract, polyvoice 0.21).
//! The legacy `EmbeddingExtractor` / `EmbeddingError` surface this module used
//! to contain is soft-deprecated upstream and is no longer referenced here.
//!
//! The WeSpeaker model (`wespeaker_resnet34.onnx`) expects rank-3 fbank input;
//! keep [`load_speaker_encoder`] on the `FbankOnnxExtractor` constructor, not
//! the old rank-2 waveform `OnnxEmbeddingExtractor`.

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

/// WeSpeaker ResNet34 embedding dimension.
pub(crate) const SPEAKER_EMBEDDING_DIM: usize = 256;
/// Tract session pool size shared across concurrent diarization sessions.
const SPEAKER_POOL_SIZE: usize = 4;

/// Shared WeSpeaker encoder handle (`Arc` over the fbank extractor).
pub type SpeakerEncoder = Arc<FbankOnnxExtractor>;

/// Per-session streaming diarization state.
pub type StreamingDiarizationState = StreamingPipeline<EnergyVad, SharedExtractor>;

/// Adapter that lets a single shared [`FbankOnnxExtractor`] back the
/// per-session [`StreamingPipeline`]s, which take ownership of their extractor.
/// The session pool inside the extractor is shared across sessions via `Arc`.
pub struct SharedExtractor(Arc<FbankOnnxExtractor>);

impl Embedder for SharedExtractor {
    fn dim(&self) -> usize {
        self.0.dim()
    }

    fn embed(&self, samples: &[f32]) -> Result<Vec<f32>, EmbedderError> {
        self.0.embed(samples)
    }
}

/// Load a WeSpeaker ResNet34 encoder from `model_path`.
///
/// Uses the fbank constructor (rank-3 input). A missing/corrupt path returns
/// `Err` — never panics.
///
/// Tract always runs on CPU (`ExecutionProvider::auto()` is also `Cpu` since
/// polyvoice 0.21 dropped ort). Named EPs are accepted by the constructor and
/// ignored.
pub(crate) fn load_speaker_encoder(
    model_path: &Path,
    pool_size: usize,
) -> anyhow::Result<FbankOnnxExtractor> {
    Ok(FbankOnnxExtractor::new(
        model_path,
        SPEAKER_EMBEDDING_DIM,
        pool_size,
        polyvoice::onnx::ExecutionProvider::Cpu,
    )?)
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
    Ready(SpeakerEncoder),
    /// Load was attempted and failed; do not retry until engine reload.
    Failed,
}

impl LazySpeakerEncoder {
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
    pub fn get_or_load(&self) -> Option<SpeakerEncoder> {
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
pub fn open_streaming(encoder: &SpeakerEncoder) -> Option<StreamingDiarizationState> {
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
    encoder: &SpeakerEncoder,
    samples: &[f32],
) -> Result<Vec<LabeledTurn>, DiarizationOutcome> {
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

/// Assign `speaker` on each word by midpoint-in-turn lookup.
pub fn assign_speakers_by_midpoint(turns: &[LabeledTurn], words: &mut [super::WordInfo]) {
    for word in words {
        let mid = (word.start + word.end) / 2.0;
        if let Some(turn) = turns.iter().find(|t| t.start <= mid && t.end >= mid) {
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

    // Model-free guard for the WeSpeaker rank-3 fbank fix: `load_speaker_encoder`
    // must stay wired to polyvoice's `FbankOnnxExtractor` (rank-3 fbank input) via
    // its 3-arg constructor, NOT the old rank-2 raw-waveform
    // `OnnxEmbeddingExtractor` (4-arg) that caused the `Got: 2 Expected: 3`
    // failure. The extractor reads the ONNX model at construction, so a
    // nonexistent path returns Err (never panics/Ok).
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
