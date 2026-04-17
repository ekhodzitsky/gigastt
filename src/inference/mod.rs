//! ONNX Runtime inference engine for GigaAM v3 e2e_rnnt.
//!
//! Loads encoder, decoder, and joiner ONNX models and runs the RNN-T streaming decode loop.

pub mod audio;
mod decode;
mod features;
mod tokenizer;

#[cfg(feature = "diarization")]
pub mod diarization;

#[cfg(all(feature = "coreml", feature = "cuda"))]
compile_error!("Features `coreml` and `cuda` are mutually exclusive. Choose one.");

use anyhow::Context;
#[cfg(any(feature = "coreml", feature = "cuda"))]
use ort::ep;
use ort::session::Session;
use ort::value::TensorRef;
use serde::Serialize;
use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};

use crate::error::GigasttError;

use features::MelSpectrogram;
use tokenizer::Tokenizer;

/// Number of mel frequency bins used for spectrogram features.
pub const N_MELS: usize = 64;
/// FFT window size in samples (320 samples = 20ms at 16kHz).
pub const N_FFT: usize = 320;
/// Hop length between consecutive FFT frames in samples (160 samples = 10ms at 16kHz).
pub const HOP_LENGTH: usize = 160;
/// Hidden dimension of the RNN-T prediction (decoder) network.
pub const PRED_HIDDEN: usize = 320;

fn ort_err(e: impl std::fmt::Display) -> anyhow::Error {
    anyhow::anyhow!("{e}")
}

pub(crate) fn now_timestamp() -> f64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs_f64()
}

/// Seconds per encoder frame (HOP_LENGTH * 4 / 16000 = 0.04s).
const SECONDS_PER_FRAME: f64 = (HOP_LENGTH as f64) * 4.0 / 16000.0;

/// Default number of session triplets in the pool.
const DEFAULT_POOL_SIZE: usize = 4;

/// A set of ONNX sessions for one inference pipeline (encoder + decoder + joiner).
///
/// Moved out of the pool on checkout and returned on checkin.
/// Each triplet is independent and can run inference concurrently with others.
pub struct SessionTriplet {
    pub(crate) encoder: Session,
    pub(crate) decoder: Session,
    pub(crate) joiner: Session,
}

/// Pool of pre-loaded [`SessionTriplet`]s backed by an mpsc channel.
///
/// Checkout = `recv` from channel, checkin = `send` back.
/// The pool size acts as the concurrency limit — no separate semaphore needed.
pub struct SessionPool {
    sender: tokio::sync::mpsc::Sender<SessionTriplet>,
    receiver: tokio::sync::Mutex<tokio::sync::mpsc::Receiver<SessionTriplet>>,
    total: usize,
    available: AtomicUsize,
}

impl SessionPool {
    /// Create a pool pre-filled with the given triplets.
    pub fn new(triplets: Vec<SessionTriplet>) -> Self {
        let total = triplets.len();
        let (sender, receiver) = tokio::sync::mpsc::channel(total);
        for triplet in triplets {
            sender
                .try_send(triplet)
                .expect("channel capacity matches triplet count");
        }
        Self {
            sender,
            receiver: tokio::sync::Mutex::new(receiver),
            total,
            available: AtomicUsize::new(total),
        }
    }

    /// Checkout a triplet from the pool. Blocks if none available.
    pub async fn checkout(&self) -> SessionTriplet {
        let triplet = self
            .receiver
            .lock()
            .await
            .recv()
            .await
            .expect("Pool sender dropped — this is a bug");
        self.available.fetch_sub(1, Ordering::Relaxed);
        triplet
    }

    /// Return a triplet to the pool (async).
    pub async fn checkin(&self, triplet: SessionTriplet) {
        self.sender
            .send(triplet)
            .await
            .expect("Pool receiver dropped — this is a bug");
        self.available.fetch_add(1, Ordering::Relaxed);
    }

    /// Return a triplet to the pool (blocking, for use inside `spawn_blocking`).
    pub fn blocking_checkin(&self, triplet: SessionTriplet) {
        self.sender
            .blocking_send(triplet)
            .expect("Pool receiver dropped — this is a bug");
        self.available.fetch_add(1, Ordering::Relaxed);
    }

    /// Total number of triplets in the pool.
    pub fn total(&self) -> usize {
        self.total
    }

    /// Number of currently available (not checked-out) triplets.
    pub fn available(&self) -> usize {
        self.available.load(Ordering::Relaxed)
    }
}

/// Decoder LSTM hidden state persisted across streaming chunks.
///
/// Created via [`DecoderState::new`] or obtained from [`StreamingState::decoder`].
/// Holds the RNN-T prediction network state between decode steps.
#[non_exhaustive]
pub struct DecoderState {
    /// LSTM hidden state vector (length [`PRED_HIDDEN`]).
    pub h: Vec<f32>,
    /// LSTM cell state vector (length [`PRED_HIDDEN`]).
    pub c: Vec<f32>,
    /// Previously emitted token ID (initialized to `blank_id`).
    pub prev_token: i64,
    /// Count of consecutive blank frames (used for endpointing).
    pub consecutive_blanks: usize,
}

impl DecoderState {
    /// Create a new decoder state initialized to zeros with the given blank token ID.
    pub fn new(blank_id: usize) -> Self {
        Self {
            h: vec![0.0; PRED_HIDDEN],
            c: vec![0.0; PRED_HIDDEN],
            prev_token: blank_id as i64,
            consecutive_blanks: 0,
        }
    }
}

/// A recognized word with timing and confidence metadata.
///
/// Produced by the RNN-T decoder during [`Engine::process_chunk`] or [`Engine::transcribe_file`].
/// Timestamps are in seconds relative to the start of the audio stream.
#[derive(Debug, Clone, Serialize)]
#[non_exhaustive]
pub struct WordInfo {
    /// The recognized word text (BPE tokens joined, `▁` stripped).
    pub word: String,
    /// Start time in seconds from the beginning of the audio stream.
    pub start: f64,
    /// End time in seconds from the beginning of the audio stream.
    pub end: f64,
    /// Softmax confidence score (0.0–1.0), averaged over constituent BPE tokens.
    pub confidence: f32,
    /// Speaker label from diarization (zero-based index). Omitted if diarization is disabled.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub speaker: Option<u32>,
}

/// Per-connection diarization state accumulating audio and speaker assignments.
#[cfg(feature = "diarization")]
pub struct DiarizationStreamState {
    /// Raw 16 kHz f32 samples accumulated since the last embedding extraction.
    pub audio_buffer: Vec<f32>,
    /// Online speaker cluster tracking centroids across the session.
    pub cluster: diarization::SpeakerCluster,
    /// Speaker ID assigned to the most recent segment.
    pub current_speaker: Option<u32>,
}

/// Per-connection streaming state that persists across audio chunks.
///
/// Created via [`Engine::create_state`]. Holds the decoder LSTM state, an audio
/// sample buffer for incomplete frames, and accumulated transcript text/words.
/// Pass this to [`Engine::process_chunk`] for each incoming audio chunk and
/// [`Engine::flush_state`] when the stream ends.
#[non_exhaustive]
pub struct StreamingState {
    /// Decoder LSTM hidden state (persisted across chunks).
    pub decoder: DecoderState,
    /// Leftover audio samples that didn't fill a complete frame.
    pub audio_buffer: Vec<f32>,
    /// Accumulated transcript text across chunks (reset on endpointing).
    pub accumulated_text: String,
    /// Accumulated words with timestamps (reset on endpointing).
    pub accumulated_words: Vec<WordInfo>,
    /// Total encoder frames processed so far (for absolute timestamp offset).
    pub total_frames: usize,
    /// Diarization state (present only when diarization is enabled).
    #[cfg(feature = "diarization")]
    pub diarization_state: Option<DiarizationStreamState>,
}

/// ONNX Runtime inference engine for GigaAM v3 e2e_rnnt.
///
/// Thread-safe: inference sessions live in a [`SessionPool`] so `Engine` can be
/// shared across connections via `Arc<Engine>`. The pool size acts as the
/// concurrency limit — no separate semaphore needed. Typical usage:
///
/// ```ignore
/// let engine = Engine::load("~/.gigastt/models")?;
/// let mut triplet = engine.pool.checkout().await;
/// let text = engine.transcribe_file("audio.wav", &mut triplet)?;
/// engine.pool.checkin(triplet).await;
/// ```
///
/// For streaming recognition, use [`create_state`](Engine::create_state) +
/// [`process_chunk`](Engine::process_chunk) + [`flush_state`](Engine::flush_state).
pub struct Engine {
    /// Pool of ONNX session triplets for concurrent inference.
    pub pool: SessionPool,
    tokenizer: Tokenizer,
    mel: MelSpectrogram,
    /// Whether the INT8 quantized encoder is in use.
    int8: bool,
    /// Speaker encoder for diarization (None if model file is absent).
    #[cfg(feature = "diarization")]
    pub speaker_encoder: Option<diarization::SpeakerEncoder>,
}

impl Engine {
    /// Whether the INT8 quantized encoder is loaded.
    pub fn is_int8(&self) -> bool {
        self.int8
    }

    /// Load ONNX models from the given directory and create an inference engine.
    ///
    /// Creates a pool of [`DEFAULT_POOL_SIZE`] session triplets for concurrent inference.
    /// Expects files: `v3_e2e_rnnt_encoder.onnx` (or `_int8.onnx`), `v3_e2e_rnnt_decoder.onnx`,
    /// `v3_e2e_rnnt_joint.onnx`, and `v3_e2e_rnnt_vocab.txt`.
    ///
    /// # Errors
    ///
    /// Returns [`GigasttError::ModelLoad`] if model files are missing or ONNX session creation fails.
    pub fn load(model_dir: &str) -> Result<Self, GigasttError> {
        Self::load_with_pool_size(model_dir, DEFAULT_POOL_SIZE)
    }

    /// Load ONNX models with a custom pool size.
    pub fn load_with_pool_size(model_dir: &str, pool_size: usize) -> Result<Self, GigasttError> {
        let dir = Path::new(model_dir);
        if !dir.join("v3_e2e_rnnt_encoder.onnx").exists() {
            return Err(GigasttError::ModelLoad(format!(
                "v3_e2e_rnnt_encoder.onnx not found in {model_dir}"
            )));
        }
        Self::load_inner(dir, model_dir, pool_size)
            .map_err(|e| GigasttError::ModelLoad(format!("{e:#}")))
    }

    /// Load a single set of encoder/decoder/joiner ONNX sessions from disk.
    fn load_sessions(
        dir: &Path,
        prepacked: &ort::session::builder::PrepackedWeights,
    ) -> anyhow::Result<(Session, Session, Session)> {
        let encoder_path = if dir.join("v3_e2e_rnnt_encoder_int8.onnx").exists() {
            dir.join("v3_e2e_rnnt_encoder_int8.onnx")
        } else {
            dir.join("v3_e2e_rnnt_encoder.onnx")
        };

        #[cfg(feature = "coreml")]
        let (encoder, decoder, joiner) = {
            // CoreML has its own cache (`coreml_cache/`) for compiled subgraphs.
            // We do NOT call `.with_optimized_model_path(...)` here: CoreML EP
            // replaces part of the graph with compiled nodes that cannot be
            // re-serialized as ONNX, and ORT errors out with
            // `Unable to serialize model as it contains compiled nodes`
            // on macOS 14+. The CoreML cache below is sufficient.
            let cache_dir = dir.join("coreml_cache");
            let coreml_ep = ep::CoreML::default()
                .with_compute_units(ep::coreml::ComputeUnits::CPUAndNeuralEngine)
                .with_specialization_strategy(ep::coreml::SpecializationStrategy::FastPrediction)
                .with_model_cache_dir(cache_dir.to_string_lossy())
                .build();

            let encoder = Session::builder()
                .map_err(ort_err)?
                .with_prepacked_weights(prepacked)
                .map_err(ort_err)?
                .with_execution_providers([coreml_ep.clone()])
                .map_err(ort_err)?
                .commit_from_file(&encoder_path)
                .map_err(ort_err)?;
            let decoder = Session::builder()
                .map_err(ort_err)?
                .with_prepacked_weights(prepacked)
                .map_err(ort_err)?
                .with_execution_providers([coreml_ep.clone()])
                .map_err(ort_err)?
                .commit_from_file(dir.join("v3_e2e_rnnt_decoder.onnx"))
                .map_err(ort_err)?;
            let joiner = Session::builder()
                .map_err(ort_err)?
                .with_prepacked_weights(prepacked)
                .map_err(ort_err)?
                .with_execution_providers([coreml_ep])
                .map_err(ort_err)?
                .commit_from_file(dir.join("v3_e2e_rnnt_joint.onnx"))
                .map_err(ort_err)?;
            (encoder, decoder, joiner)
        };

        #[cfg(feature = "cuda")]
        let (encoder, decoder, joiner) = {
            // CUDA EP compiles subgraphs that cannot be re-serialized as ONNX,
            // so we do NOT call `.with_optimized_model_path(...)` here — same
            // reason as the CoreML block. ORT's CUDA EP keeps its own caches
            // internally.
            let cuda_ep = ep::CUDA::default().build();

            let encoder = Session::builder()
                .map_err(ort_err)?
                .with_prepacked_weights(prepacked)
                .map_err(ort_err)?
                .with_execution_providers([cuda_ep.clone()])
                .map_err(ort_err)?
                .commit_from_file(&encoder_path)
                .map_err(ort_err)?;
            let decoder = Session::builder()
                .map_err(ort_err)?
                .with_prepacked_weights(prepacked)
                .map_err(ort_err)?
                .with_execution_providers([cuda_ep.clone()])
                .map_err(ort_err)?
                .commit_from_file(dir.join("v3_e2e_rnnt_decoder.onnx"))
                .map_err(ort_err)?;
            let joiner = Session::builder()
                .map_err(ort_err)?
                .with_prepacked_weights(prepacked)
                .map_err(ort_err)?
                .with_execution_providers([cuda_ep])
                .map_err(ort_err)?
                .commit_from_file(dir.join("v3_e2e_rnnt_joint.onnx"))
                .map_err(ort_err)?;
            (encoder, decoder, joiner)
        };

        #[cfg(not(any(feature = "coreml", feature = "cuda")))]
        let (encoder, decoder, joiner) = {
            let cache_dir = dir.join("optimized_cache");
            std::fs::create_dir_all(&cache_dir).ok();
            let encoder = Session::builder()
                .map_err(ort_err)?
                .with_prepacked_weights(prepacked)
                .map_err(ort_err)?
                .with_optimized_model_path(cache_dir.join("encoder_optimized.onnx"))
                .map_err(ort_err)?
                .commit_from_file(&encoder_path)
                .map_err(ort_err)?;
            let decoder = Session::builder()
                .map_err(ort_err)?
                .with_prepacked_weights(prepacked)
                .map_err(ort_err)?
                .commit_from_file(dir.join("v3_e2e_rnnt_decoder.onnx"))
                .map_err(ort_err)?;
            let joiner = Session::builder()
                .map_err(ort_err)?
                .with_prepacked_weights(prepacked)
                .map_err(ort_err)?
                .commit_from_file(dir.join("v3_e2e_rnnt_joint.onnx"))
                .map_err(ort_err)?;
            (encoder, decoder, joiner)
        };

        Ok((encoder, decoder, joiner))
    }

    fn load_inner(dir: &Path, model_dir: &str, pool_size: usize) -> anyhow::Result<Self> {
        let is_int8 = dir.join("v3_e2e_rnnt_encoder_int8.onnx").exists();
        if is_int8 {
            tracing::info!("Using INT8 quantized encoder");
        }

        tracing::info!("Loading ONNX models from {model_dir} (pool_size={pool_size})...");

        #[cfg(feature = "coreml")]
        tracing::info!("Using CoreML execution provider (Neural Engine + CPU)");
        #[cfg(feature = "cuda")]
        tracing::info!("Using CUDA execution provider (falls back to CPU if unavailable)");
        #[cfg(not(any(feature = "coreml", feature = "cuda")))]
        tracing::info!("Using CPU execution provider");

        // Shared prepacked weights container (Arc-based, thread-safe)
        let prepacked = ort::session::builder::PrepackedWeights::new();

        let triplets: Vec<SessionTriplet> = std::thread::scope(|s| {
            let handles: Vec<_> = (0..pool_size)
                .map(|i| {
                    let pp = &prepacked;
                    s.spawn(move || {
                        tracing::info!(
                            "Loading session triplet {}/{pool_size} (shared weights)",
                            i + 1
                        );
                        let (encoder, decoder, joiner) = Self::load_sessions(dir, pp)?;
                        Ok(SessionTriplet {
                            encoder,
                            decoder,
                            joiner,
                        })
                    })
                })
                .collect();
            handles
                .into_iter()
                .map(|h| h.join().expect("Thread panicked during model loading"))
                .collect::<anyhow::Result<Vec<_>>>()
        })?;

        let tokenizer = Tokenizer::load(&dir.join("v3_e2e_rnnt_vocab.txt"))?;
        let mel = MelSpectrogram::new();

        tracing::info!(
            "Models loaded (vocab_size={}, pool_size={pool_size})",
            tokenizer.vocab_size()
        );

        #[cfg(feature = "diarization")]
        let speaker_encoder = match diarization::SpeakerEncoder::load(dir) {
            Ok(enc) => {
                tracing::info!("Speaker encoder loaded (diarization available)");
                Some(enc)
            }
            Err(e) => {
                tracing::warn!("Speaker encoder not loaded, diarization unavailable: {e:#}");
                None
            }
        };

        Ok(Self {
            pool: SessionPool::new(triplets),
            tokenizer,
            mel,
            int8: is_int8,
            #[cfg(feature = "diarization")]
            speaker_encoder,
        })
    }

    /// Return `true` if a speaker encoder is loaded and diarization is available.
    #[cfg(feature = "diarization")]
    pub fn has_speaker_encoder(&self) -> bool {
        self.speaker_encoder.is_some()
    }

    /// Create a fresh streaming state for a new connection.
    ///
    /// Pass `diarization_enabled = true` to activate speaker diarization for
    /// this session (requires the `diarization` feature and a loaded speaker
    /// encoder; silently ignored otherwise).
    pub fn create_state(
        &self,
        #[cfg(feature = "diarization")] diarization_enabled: bool,
    ) -> StreamingState {
        #[cfg(feature = "diarization")]
        let diarization_state = if diarization_enabled && self.speaker_encoder.is_some() {
            Some(DiarizationStreamState {
                audio_buffer: Vec::new(),
                cluster: diarization::SpeakerCluster::new(),
                current_speaker: None,
            })
        } else {
            None
        };

        StreamingState {
            decoder: DecoderState::new(self.tokenizer.blank_id()),
            audio_buffer: Vec::new(),
            accumulated_text: String::new(),
            accumulated_words: Vec::new(),
            total_frames: 0,
            #[cfg(feature = "diarization")]
            diarization_state,
        }
    }

    /// Process a chunk of 16kHz f32 audio samples and return any new transcript segments.
    ///
    /// Returns [`TranscriptSegment`] with `is_final == false` during speech (Partial),
    /// and `is_final == true` on endpointing (~600ms silence detected).
    /// Streaming state (LSTM hidden/cell, leftover audio, accumulated text) is maintained in `state`.
    ///
    /// # Errors
    ///
    /// Returns [`GigasttError::Inference`] if the ONNX runtime fails.
    pub fn process_chunk(
        &self,
        samples: &[f32],
        state: &mut StreamingState,
        triplet: &mut SessionTriplet,
    ) -> Result<Vec<TranscriptSegment>, GigasttError> {
        if samples.is_empty() {
            return Ok(vec![]);
        }

        // Keep a copy of the 16kHz samples for diarization before the buffer
        // logic potentially pads/realigns them. Skip allocation when diarization
        // is not active for this session.
        #[cfg(feature = "diarization")]
        let samples_16k_copy = if state.diarization_state.is_some() {
            Some(samples.to_vec())
        } else {
            None
        };

        let samples = match audio::prepare_audio_buffer(samples, &mut state.audio_buffer) {
            Some(s) => s,
            None => return Ok(vec![]),
        };
        let samples = &samples[..];

        let mel_start = std::time::Instant::now();
        let (features, num_frames) = self.mel.compute(samples);
        tracing::debug!(
            elapsed_us = mel_start.elapsed().as_micros() as u64,
            "mel_compute"
        );
        if num_frames == 0 {
            return Ok(vec![]);
        }

        #[cfg_attr(not(feature = "diarization"), allow(unused_mut))]
        let (mut new_words, endpoint) = self
            .run_inference(
                triplet,
                &features,
                num_frames,
                &mut state.decoder,
                state.total_frames,
            )
            .map_err(|e| GigasttError::Inference(format!("{e:#}")))?;
        state.total_frames += num_frames;

        // --- Diarization: accumulate audio, extract embeddings, assign speakers ---
        #[cfg(feature = "diarization")]
        if let (Some(dia), Some(copy), Some(enc)) = (
            state.diarization_state.as_mut(),
            samples_16k_copy.as_ref(),
            self.speaker_encoder.as_ref(),
        ) {
            dia.audio_buffer.extend_from_slice(copy);

            while dia.audio_buffer.len() >= diarization::SEGMENT_SAMPLES {
                let segment: Vec<f32> = dia
                    .audio_buffer
                    .drain(..diarization::SEGMENT_SAMPLES)
                    .collect();
                match enc.extract_embedding(&segment) {
                    Ok(embedding) => {
                        let speaker = dia.cluster.assign(&embedding);
                        dia.current_speaker = Some(speaker);
                    }
                    Err(e) => {
                        tracing::warn!("Embedding extraction failed: {e:#}");
                    }
                }
            }

            // Annotate all words in this chunk with current speaker
            if let Some(speaker_id) = dia.current_speaker {
                for w in &mut new_words {
                    w.speaker = Some(speaker_id);
                }
            }
        }

        if new_words.is_empty() && !endpoint {
            return Ok(vec![]);
        }

        // Accumulate new words
        for w in &new_words {
            if !state.accumulated_text.is_empty() {
                state.accumulated_text.push(' ');
            }
            state.accumulated_text.push_str(&w.word);
        }
        state.accumulated_words.extend(new_words);

        let text = state.accumulated_text.clone();
        let words = state.accumulated_words.clone();
        let ts = now_timestamp();

        if endpoint {
            // Endpoint detected: emit Final and reset accumulation
            state.accumulated_text.clear();
            state.accumulated_words.clear();
            state.decoder.consecutive_blanks = 0;
            Ok(vec![TranscriptSegment {
                text,
                words,
                is_final: true,
                timestamp: ts,
            }])
        } else {
            // Ongoing speech: emit Partial
            Ok(vec![TranscriptSegment {
                text,
                words,
                is_final: false,
                timestamp: ts,
            }])
        }
    }

    /// Flush accumulated text as a Final segment (called on Stop/Close).
    pub fn flush_state(&self, state: &mut StreamingState) -> Option<TranscriptSegment> {
        if state.accumulated_text.is_empty() {
            return None;
        }
        let seg = TranscriptSegment {
            text: std::mem::take(&mut state.accumulated_text),
            words: std::mem::take(&mut state.accumulated_words),
            is_final: true,
            timestamp: now_timestamp(),
        };
        Some(seg)
    }

    /// Transcribe an audio file to text (supports WAV, MP3, M4A/AAC, OGG, FLAC).
    ///
    /// Decodes the file to mono 16kHz, runs the full encoder+decoder pipeline,
    /// and returns the recognized text with word-level details and duration.
    ///
    /// # Errors
    ///
    /// Returns [`GigasttError::InvalidAudio`] if the file cannot be decoded, or
    /// [`GigasttError::Inference`] if the ONNX runtime fails.
    pub fn transcribe_file(
        &self,
        path: &str,
        triplet: &mut SessionTriplet,
    ) -> Result<TranscribeResult, GigasttError> {
        let float_samples = audio::decode_audio_file(path)
            .map_err(|e| GigasttError::InvalidAudio(format!("{e:#}")))?;
        let duration_s = float_samples.len() as f64 / 16000.0;

        let (features, num_frames) = self.mel.compute(&float_samples);
        tracing::info!("Extracted {} mel frames", num_frames);

        let mut decoder_state = DecoderState::new(self.tokenizer.blank_id());
        let (words, _endpoint) = self
            .run_inference(triplet, &features, num_frames, &mut decoder_state, 0)
            .map_err(|e| GigasttError::Inference(format!("{e:#}")))?;
        let text: String = words
            .iter()
            .map(|w| w.word.as_str())
            .collect::<Vec<_>>()
            .join(" ");
        Ok(TranscribeResult {
            text,
            words,
            duration_s,
        })
    }

    /// Transcribe audio from raw bytes in memory (no temp file needed).
    pub fn transcribe_bytes(
        &self,
        data: &[u8],
        triplet: &mut SessionTriplet,
    ) -> Result<TranscribeResult, GigasttError> {
        let float_samples = audio::decode_audio_bytes(data)
            .map_err(|e| GigasttError::InvalidAudio(format!("{e:#}")))?;
        let duration_s = float_samples.len() as f64 / 16000.0;

        let (features, num_frames) = self.mel.compute(&float_samples);
        tracing::info!("Extracted {} mel frames", num_frames);

        let mut decoder_state = DecoderState::new(self.tokenizer.blank_id());
        let (words, _endpoint) = self
            .run_inference(triplet, &features, num_frames, &mut decoder_state, 0)
            .map_err(|e| GigasttError::Inference(format!("{e:#}")))?;
        let text: String = words
            .iter()
            .map(|w| w.word.as_str())
            .collect::<Vec<_>>()
            .join(" ");
        Ok(TranscribeResult {
            text,
            words,
            duration_s,
        })
    }

    fn run_inference(
        &self,
        triplet: &mut SessionTriplet,
        features: &[f32],
        num_frames: usize,
        decoder_state: &mut DecoderState,
        frame_offset: usize,
    ) -> anyhow::Result<(Vec<WordInfo>, bool)> {
        // Encoder input: audio_signal [1, 64, num_frames], length [1]
        let signal_tensor = TensorRef::from_array_view(([1_usize, N_MELS, num_frames], features))?;
        let length_data = [num_frames as i64];
        let length_tensor = TensorRef::from_array_view(([1_usize], length_data.as_slice()))?;

        let enc_start = std::time::Instant::now();
        let encoder_outputs = triplet
            .encoder
            .run(ort::inputs![signal_tensor, length_tensor])
            .context("Encoder inference failed")?;
        tracing::info!(
            elapsed_ms = enc_start.elapsed().as_millis() as u64,
            "encoder_inference"
        );

        let (_enc_shape, enc_data) = encoder_outputs[0]
            .try_extract_tensor::<f32>()
            .context("Failed to extract encoder output")?;
        let (_len_shape, len_data) = encoder_outputs[1]
            .try_extract_tensor::<i32>()
            .context("Failed to extract encoder length")?;

        let enc_len = usize::try_from(len_data[0]).context("Negative encoder length")?;

        tracing::debug!("Encoder output: {} frames", enc_len);

        // Copy encoder data so we can release the encoder output borrow
        let enc_data_owned: Vec<f32> = enc_data.to_vec();
        drop(encoder_outputs);

        // RNN-T greedy decode
        let dec_start = std::time::Instant::now();
        let result = decode::greedy_decode(
            &mut triplet.decoder,
            &mut triplet.joiner,
            &enc_data_owned,
            enc_len,
            self.tokenizer.blank_id(),
            decoder_state,
        )?;
        tracing::info!(
            elapsed_ms = dec_start.elapsed().as_millis() as u64,
            "greedy_decode"
        );

        // Convert token infos to words with timestamps
        let words = self.tokens_to_words(&result.tokens, frame_offset);

        let preview: String = words
            .iter()
            .take(10)
            .map(|w| w.word.as_str())
            .collect::<Vec<_>>()
            .join(" ");
        let ellipsis = if words.len() > 10 { "..." } else { "" };
        tracing::info!(
            "Decoded {} tokens → \"{preview}{ellipsis}\"",
            result.tokens.len()
        );

        Ok((words, result.endpoint_detected))
    }

    /// Convert decoded tokens into words with timestamps and confidence.
    fn tokens_to_words(&self, tokens: &[decode::TokenInfo], frame_offset: usize) -> Vec<WordInfo> {
        if tokens.is_empty() {
            return Vec::new();
        }

        let token_ids: Vec<usize> = tokens.iter().map(|t| t.token_id).collect();
        let raw_text = self.tokenizer.decode(&token_ids);
        if raw_text.is_empty() {
            return Vec::new();
        }

        // Group tokens by words (BPE ▁ marks word boundaries)
        let mut words = Vec::new();
        let mut current_word = String::new();
        let mut word_start_frame: Option<usize> = None;
        let mut word_end_frame: usize = 0;
        let mut word_confidences: Vec<f32> = Vec::new();

        for token in tokens {
            let token_text = self.tokenizer.token_text(token.token_id);
            let is_word_boundary = token_text.starts_with('\u{2581}');

            if is_word_boundary && !current_word.is_empty() {
                // Emit previous word
                let avg_conf: f32 = if word_confidences.is_empty() {
                    1.0
                } else {
                    word_confidences.iter().sum::<f32>() / word_confidences.len() as f32
                };
                words.push(WordInfo {
                    word: current_word.clone(),
                    start: (word_start_frame.unwrap_or(0) + frame_offset) as f64
                        * SECONDS_PER_FRAME,
                    end: (word_end_frame + frame_offset) as f64 * SECONDS_PER_FRAME,
                    confidence: avg_conf,
                    speaker: None,
                });
                current_word.clear();
                word_confidences.clear();
                word_start_frame = None;
            }

            let clean = token_text.replace('\u{2581}', "");
            if !clean.is_empty() {
                current_word.push_str(&clean);
                if word_start_frame.is_none() {
                    word_start_frame = Some(token.frame_index);
                }
                word_end_frame = token.frame_index;
                word_confidences.push(token.confidence);
            }
        }

        // Emit last word
        if !current_word.is_empty() {
            let avg_conf: f32 = if word_confidences.is_empty() {
                1.0
            } else {
                word_confidences.iter().sum::<f32>() / word_confidences.len() as f32
            };
            words.push(WordInfo {
                word: current_word,
                start: (word_start_frame.unwrap_or(0) + frame_offset) as f64 * SECONDS_PER_FRAME,
                end: (word_end_frame + frame_offset) as f64 * SECONDS_PER_FRAME,
                confidence: avg_conf,
                speaker: None,
            });
        }

        words
    }
}

/// Result of file transcription, including word-level details.
#[derive(Debug, Clone, Serialize)]
pub struct TranscribeResult {
    pub text: String,
    pub words: Vec<WordInfo>,
    pub duration_s: f64,
}

/// A transcript segment emitted by the inference engine.
///
/// Partial segments (`is_final == false`) represent interim results that may change.
/// Final segments (`is_final == true`) represent completed utterances after endpointing.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct TranscriptSegment {
    /// Recognized text for this segment.
    pub text: String,
    /// Individual words with timing and confidence metadata.
    pub words: Vec<WordInfo>,
    /// Whether this segment is final (utterance complete) or partial (interim).
    pub is_final: bool,
    /// Unix timestamp (seconds since epoch) when this segment was produced.
    pub timestamp: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_decoder_state_new_zeros() {
        let blank_id = 1024;
        let state = DecoderState::new(blank_id);
        assert!(state.h.iter().all(|&v| v == 0.0));
        assert!(state.c.iter().all(|&v| v == 0.0));
        assert_eq!(state.prev_token, blank_id as i64);
    }

    #[test]
    fn test_decoder_state_dimensions() {
        let state = DecoderState::new(1024);
        assert_eq!(state.h.len(), PRED_HIDDEN);
        assert_eq!(state.c.len(), PRED_HIDDEN);
    }

    #[test]
    fn test_decoder_state_custom_blank_id() {
        let state = DecoderState::new(42);
        assert_eq!(state.prev_token, 42);
    }
}
