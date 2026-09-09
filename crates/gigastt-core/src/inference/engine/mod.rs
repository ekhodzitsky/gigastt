//! ONNX Runtime inference engine for GigaAM v3.

use anyhow::Context;
use std::path::{Path, PathBuf};

use crate::error::GigasttError;
use crate::model::ModelVariant;
#[allow(unused_imports)]
use crate::runtime::factory::RuntimeFactory;
use crate::runtime::production_factory_variant_with_cache;
use crate::runtime::tensor::{Shape, TensorDataView};

use super::audio;
use super::audio::{PcmWindow, PcmWindows, SliceWindows, WindowSpec};
use super::bias;
use super::ctc;
use super::decode;
use super::load_files::{
    ResolvedModelFiles, encoder_model_path, load_triplets_runtime, resolve_variant_required,
};
use super::pool::{Pool, PoolError, PoolGuard, SessionPool, SessionTriplet};
use super::sizing;
use super::state::{
    DecoderState, EndpointMode, EndpointReason, FeatureExtractor, StreamingState,
    TranscriptAssembler, TranscriptSegment, WordInfo, aggregate_confidence,
};
use super::token_format::{TokenFormatter, overlap_mid_seconds, stitch_chunk_words};
use super::tokenizer::Tokenizer;
use super::types::{
    DEFAULT_HOTWORDS_BOOST, DiarizationOutcome, HotwordError, HotwordOverride,
    MAX_HOTWORD_PHRASE_CHARS, MAX_HOTWORDS_PER_REQUEST, OverrideError, TranscribeOverrides,
    TranscribeRequest, TranscribeResult, TranscribeSource, merge_channel_results,
};
use super::windows::{
    STREAM_CAP_STREAK_MAX, STREAM_COMMIT_HORIZON_SECS, STREAM_DECODE_STRIDE_SAMPLES,
    STREAM_LEFT_CONTEXT_SAMPLES, STREAM_MAX_WINDOW_SAMPLES, window_spec,
};
use super::{ENCODER_SUBSAMPLING, HOP_LENGTH, N_FFT, N_MELS, SECONDS_PER_FRAME, now_timestamp};

#[cfg(feature = "diarization")]
use super::diarization::{self, LazySpeakerEncoder};

/// Cooperative-run hooks threaded through the decode call chain.
///
/// Both fields are `None` on the historical path, in which case every decode
/// function behaves byte-for-byte as before — the hooks are only ever consulted
/// through `if let Some(_)` guards, adding no work when absent. `abort` is
/// polled at window boundaries so a cancelled run releases its pooled session
/// within one window; `on_progress` receives the cumulative count of processed
/// 16 kHz samples after each long-form window so a server watchdog can reset its
/// no-progress deadline and drive a real progress bar.
#[derive(Clone, Copy, Default)]
pub(crate) struct DecodeControls<'a> {
    pub(crate) abort: Option<&'a dyn Fn() -> bool>,
    pub(crate) on_progress: Option<&'a dyn Fn(u64)>,
}

impl DecodeControls<'_> {
    /// True once the caller has requested cancellation.
    #[inline]
    pub(crate) fn aborted(&self) -> bool {
        self.abort.is_some_and(|a| a())
    }

    /// Report cumulative processed 16 kHz samples, if a sink is attached.
    #[inline]
    pub(crate) fn report(&self, processed_16k_samples: u64) {
        if let Some(on_progress) = self.on_progress {
            on_progress(processed_16k_samples);
        }
    }

    /// Drop the progress sink but keep the abort hook. Used for the
    /// `channels=split` path, where each channel restarts the sample clock and a
    /// shared monotonic progress counter would go backwards.
    #[inline]
    pub(crate) fn abort_only(&self) -> Self {
        Self {
            abort: self.abort,
            on_progress: None,
        }
    }
}

/// Default number of session triplets in the pool.
///
/// Each pooled triplet still materializes encoder weights; ORT's shared
/// `PrepackedWeights` container (enabled on the CPU production factory) shares
/// prepacked kernel buffers across sessions when the EP supports it — raw
/// initializer tensors are not guaranteed to collapse to 1× (remeasure the
/// pool delta after ORT upgrades). An extra INT8 slot costs on the order of
/// tens of megabytes resident because the encoder is memory-mapped and
/// shared; the default pool stays at 2. Raise `--pool-size` when higher
/// concurrency is needed.
#[cfg(target_os = "android")]
const DEFAULT_POOL_SIZE: usize = 1;
#[cfg(not(target_os = "android"))]
const DEFAULT_POOL_SIZE: usize = 2;

/// ONNX Runtime inference engine for GigaAM v3 (`rnnt` head by default).
///
/// Thread-safe: inference sessions live in a [`SessionPool`] so `Engine` can be
/// shared across connections via `Arc<Engine>`. The pool size acts as the
/// concurrency limit — no separate semaphore needed. Typical usage:
///
/// ```ignore
/// let engine = Engine::load("~/.gigastt/models")?;
/// let mut guard = engine.pool.checkout().await?;
/// let result = engine.transcribe_file("audio.wav", &mut guard)?;
/// // result.text; guard is returned to the pool on drop
/// ```
///
/// For streaming recognition, use [`create_state`](Engine::create_state) +
/// [`process_chunk`](Engine::process_chunk) + [`flush_state`](Engine::flush_state).
pub struct Engine {
    /// Pool of session triplets for interactive inference (WebSocket + SSE
    /// streaming). REST file transcription uses [`Engine::batch_pool`] when it
    /// is set, so a long batch job can't starve real-time streaming.
    pub pool: SessionPool,
    /// Optional dedicated pool for batch REST file transcription, split off
    /// from `pool` at load time. `None` means REST shares the interactive pool.
    pub batch_pool: Option<SessionPool>,
    tokenizer: Tokenizer,
    features: FeatureExtractor,
    /// Recognition head detected on disk at load time. Drives the default
    /// punctuation policy (`auto`): on for [`ModelVariant::Rnnt`] (bare output),
    /// off for [`ModelVariant::E2eRnnt`] (already punctuated).
    variant: ModelVariant,
    /// Optional punctuation / casing restorer applied to file-transcription
    /// output and to finalized streaming segments. `None` = pass-through (the
    /// default, and the only behaviour when no punct model is installed).
    /// Attached via [`Engine::with_punctuator`].
    punctuator: Option<crate::punctuation::Punctuator>,
    /// Whether to run inverse text normalization (Russian number-words →
    /// digits) on file-transcription output and finalized streaming segments,
    /// *before* the punctuation pass. Off by default; toggled via
    /// [`Engine::with_itn`].
    itn: bool,
    /// Optional contextual hotword biaser applied inside the greedy RNN-T decode
    /// loop (shallow fusion). `None` = no biasing (the default), and the decode
    /// path is then byte-for-byte identical to the un-biased engine. Attached
    /// via [`Engine::with_biaser`]. Shared across the session pool by reference.
    biaser: Option<bias::Biaser>,
    /// Optional Silero VAD. When set, file transcription skips silent regions
    /// (decoding only detected speech) and streaming endpointing is owned by
    /// VAD-detected trailing silence (the decoder's blank-run heuristic is
    /// ignored, so `vad_config.min_silence_ms` fully controls finalization).
    /// `None` = no VAD: the file path decodes the whole buffer and streaming
    /// endpointing is byte-for-byte unchanged. Attached via [`Engine::with_vad`].
    vad: Option<crate::vad::SileroVad>,
    /// Thresholds for the VAD (speech threshold, min silence/speech, padding).
    /// Ignored when `vad` is `None`.
    vad_config: crate::vad::VadConfig,
    /// Default streaming utterance-end policy for new sessions. Overridable
    /// per session via WS `configure.endpoint_mode`.
    endpoint_mode: EndpointMode,
    /// Max retained streaming encoder window (samples @16kHz) before the window
    /// commits a stable prefix and slides. Defaults to
    /// [`STREAM_MAX_WINDOW_SAMPLES`] (2.5 s); overridden at serve time via
    /// [`Engine::with_stream_max_window_secs`]. Longer windows improve WER on
    /// long phrases at a linear per-stride encoder-cost increase.
    stream_max_window_samples: usize,
    /// Whether the INT8 quantized encoder is in use.
    int8: bool,
    /// Stable-prefix commit at the streaming window cap: when true, only the
    /// prefix two consecutive window hypotheses agree on (minus a commit
    /// horizon at the window edge) enters the stable prefix; the rest stays
    /// revisable. When false (default), the cap commits the whole live tail.
    stream_stable_prefix: bool,
    /// True when pooled encoder sessions run on the ANE fixed-shape pad-up path.
    /// Selects the 30s long-form chunk window (vs 24s for ort). Derived from
    /// the loaded encoder session at boot, not from compile-time features alone,
    /// so non-rnnt heads on an `ane`-feature binary still use the ort window.
    ane_encoder: bool,
    /// Max session triplets one file decode may hold for overlapping-window
    /// parallelism. `1` (default) is the historical serial loop. Values `> 1`
    /// `try_checkout` idle extra slots from the batch pool; if none are free
    /// the file stays serial. Never waits for a slot (that would deadlock two
    /// long files each holding one triplet).
    file_window_concurrency: usize,
    /// Lazy speaker encoder for diarization (`None` if model file is absent).
    ///
    /// Boot only probes for `wespeaker_resnet34.onnx`; the ONNX session is
    /// opened on the first diarization request so unused speaker files do not
    /// inflate ready RSS (~+39 MiB when loaded). Shared across sessions via
    /// the `Arc` inside the loaded encoder.
    #[cfg(feature = "diarization")]
    speaker_encoder: Option<LazySpeakerEncoder>,
}

mod api;
mod channels;
mod config;
mod file_stream;
mod infer;
mod load;
mod stream;
#[cfg(test)]
mod tests;
mod transcribe;
mod words;
