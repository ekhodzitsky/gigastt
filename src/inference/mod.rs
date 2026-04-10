//! ONNX Runtime inference engine for GigaAM v3 e2e_rnnt.
//!
//! Loads encoder, decoder, and joiner ONNX models and runs the RNN-T streaming decode loop.

mod decode;
mod features;
mod tokenizer;
pub mod audio;

use anyhow::{Context, Result};
use ort::ep;
use ort::session::Session;
use ort::value::TensorRef;
use std::path::Path;
use std::sync::Mutex;

use features::MelSpectrogram;
use tokenizer::Tokenizer;

pub const N_MELS: usize = 64;
pub const N_FFT: usize = 320;
pub const HOP_LENGTH: usize = 160;
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

/// Decoder LSTM state persisted across chunks.
pub struct DecoderState {
    pub h: Vec<f32>,
    pub c: Vec<f32>,
    pub prev_token: i64,
}

impl DecoderState {
    pub fn new(blank_id: usize) -> Self {
        Self {
            h: vec![0.0; PRED_HIDDEN],
            c: vec![0.0; PRED_HIDDEN],
            prev_token: blank_id as i64,
        }
    }
}

/// Per-connection streaming state.
pub struct StreamingState {
    pub decoder: DecoderState,
    pub audio_buffer: Vec<f32>,
}

pub struct Engine {
    encoder: Mutex<Session>,
    decoder: Mutex<Session>,
    joiner: Mutex<Session>,
    tokenizer: Tokenizer,
    mel: MelSpectrogram,
}

impl Engine {
    pub fn load(model_dir: &str) -> Result<Self> {
        let dir = Path::new(model_dir);
        anyhow::ensure!(
            dir.join("v3_e2e_rnnt_encoder.onnx").exists(),
            "v3_e2e_rnnt_encoder.onnx not found in {model_dir}"
        );

        // Prefer INT8 quantized encoder if available (~4x smaller, ~43% faster)
        let encoder_path = if dir.join("v3_e2e_rnnt_encoder_int8.onnx").exists() {
            tracing::info!("Using INT8 quantized encoder");
            dir.join("v3_e2e_rnnt_encoder_int8.onnx")
        } else {
            dir.join("v3_e2e_rnnt_encoder.onnx")
        };

        tracing::info!("Loading ONNX models from {model_dir}...");

        // CoreML EP: Neural Engine + model cache
        // Note: MLProgram format is incompatible with GigaAM v3 models, using default NeuralNetwork
        let cache_dir = dir.join("coreml_cache");
        let coreml_ep = ep::CoreML::default()
            .with_compute_units(ep::coreml::ComputeUnits::CPUAndNeuralEngine)
            .with_specialization_strategy(ep::coreml::SpecializationStrategy::FastPrediction)
            .with_model_cache_dir(cache_dir.to_string_lossy())
            .build();

        let encoder = Session::builder()
            .map_err(ort_err)?
            .with_execution_providers([coreml_ep.clone()])
            .map_err(ort_err)?
            .commit_from_file(&encoder_path)
            .map_err(ort_err)?;

        let decoder = Session::builder()
            .map_err(ort_err)?
            .with_execution_providers([coreml_ep.clone()])
            .map_err(ort_err)?
            .commit_from_file(dir.join("v3_e2e_rnnt_decoder.onnx"))
            .map_err(ort_err)?;

        let joiner = Session::builder()
            .map_err(ort_err)?
            .with_execution_providers([coreml_ep])
            .map_err(ort_err)?
            .commit_from_file(dir.join("v3_e2e_rnnt_joint.onnx"))
            .map_err(ort_err)?;

        let tokenizer = Tokenizer::load(&dir.join("v3_e2e_rnnt_vocab.txt"))?;
        let mel = MelSpectrogram::new();

        tracing::info!("Models loaded (vocab_size={})", tokenizer.vocab_size());

        Ok(Self {
            encoder: Mutex::new(encoder),
            decoder: Mutex::new(decoder),
            joiner: Mutex::new(joiner),
            tokenizer,
            mel,
        })
    }

    /// Create a fresh streaming state for a new connection.
    pub fn create_state(&self) -> StreamingState {
        StreamingState {
            decoder: DecoderState::new(self.tokenizer.blank_id()),
            audio_buffer: Vec::new(),
        }
    }

    /// Process a chunk of f32 audio samples and return any new transcript segments.
    ///
    /// Streaming state (LSTM hidden/cell, leftover audio) is maintained in `state`.
    pub fn process_chunk(
        &self,
        samples: &[f32],
        state: &mut StreamingState,
    ) -> Result<Vec<TranscriptSegment>> {
        if samples.is_empty() {
            return Ok(vec![]);
        }

        let samples = match audio::prepare_audio_buffer(samples, &mut state.audio_buffer) {
            Some(s) => s,
            None => return Ok(vec![]),
        };
        let samples = &samples[..];

        let (features, num_frames) = self.mel.compute(samples);
        if num_frames == 0 {
            return Ok(vec![]);
        }

        let text = self.run_inference(&features, num_frames, &mut state.decoder)?;

        if text.is_empty() {
            return Ok(vec![]);
        }

        Ok(vec![TranscriptSegment {
            text,
            is_final: true,
            timestamp: now_timestamp(),
        }])
    }

    /// Transcribe an audio file (supports WAV, MP3, M4A/AAC, OGG, FLAC).
    pub fn transcribe_file(&self, path: &str) -> Result<String> {
        let float_samples = audio::decode_audio_file(path)?;

        let (features, num_frames) = self.mel.compute(&float_samples);
        tracing::info!("Extracted {} mel frames", num_frames);

        let mut decoder_state = DecoderState::new(self.tokenizer.blank_id());
        self.run_inference(&features, num_frames, &mut decoder_state)
    }

    fn run_inference(
        &self,
        features: &[f32],
        num_frames: usize,
        decoder_state: &mut DecoderState,
    ) -> Result<String> {
        // Encoder input: audio_signal [1, 64, num_frames], length [1]
        let signal_tensor =
            TensorRef::from_array_view(([1_usize, N_MELS, num_frames], features))?;
        let length_data = [num_frames as i64];
        let length_tensor =
            TensorRef::from_array_view(([1_usize], length_data.as_slice()))?;

        let mut encoder = self.encoder.lock().unwrap_or_else(|e| {
            tracing::warn!("Encoder mutex was poisoned, recovering");
            e.into_inner()
        });
        let encoder_outputs = encoder
            .run(ort::inputs![signal_tensor, length_tensor])
            .context("Encoder inference failed")?;

        let (_enc_shape, enc_data) = encoder_outputs[0]
            .try_extract_tensor::<f32>()
            .context("Failed to extract encoder output")?;
        let (_len_shape, len_data) = encoder_outputs[1]
            .try_extract_tensor::<i32>()
            .context("Failed to extract encoder length")?;

        let enc_len = usize::try_from(len_data[0]).context("Negative encoder length")?;

        tracing::debug!("Encoder output: {} frames", enc_len);

        // Need to copy encoder data since we need to drop encoder_outputs before locking decoder
        let enc_data_owned: Vec<f32> = enc_data.to_vec();
        drop(encoder_outputs);
        drop(encoder);

        // RNN-T greedy decode
        let mut decoder = self.decoder.lock().unwrap_or_else(|e| {
            tracing::warn!("Decoder mutex was poisoned, recovering");
            e.into_inner()
        });
        let mut joiner = self.joiner.lock().unwrap_or_else(|e| {
            tracing::warn!("Joiner mutex was poisoned, recovering");
            e.into_inner()
        });

        let token_ids = decode::greedy_decode(
            &mut decoder,
            &mut joiner,
            &enc_data_owned,
            enc_len,
            self.tokenizer.blank_id(),
            decoder_state,
        )?;

        let text = self.tokenizer.decode(&token_ids);
        let preview: String = text.chars().take(80).collect();
        let ellipsis = if text.len() > 80 { "..." } else { "" };
        tracing::info!("Decoded {} tokens → \"{preview}{ellipsis}\"", token_ids.len());

        Ok(text)
    }
}

#[derive(Debug, Clone)]
pub struct TranscriptSegment {
    pub text: String,
    pub is_final: bool,
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
