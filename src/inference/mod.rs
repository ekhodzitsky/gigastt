//! ONNX Runtime inference engine for GigaAM v3 e2e_rnnt.
//!
//! Loads encoder, decoder, and joiner ONNX models and runs the RNN-T streaming decode loop.

use anyhow::{Context, Result};
use std::path::Path;

pub struct Engine {
    _model_dir: String,
    // TODO: ort::Session for encoder, decoder, joiner
    // TODO: tokenizer (tokens.txt char-level mapping)
}

impl Engine {
    pub fn load(model_dir: &str) -> Result<Self> {
        let dir = Path::new(model_dir);
        anyhow::ensure!(dir.join("encoder.onnx").exists(), "encoder.onnx not found in {model_dir}");

        tracing::info!("Loading ONNX models from {model_dir}...");

        // TODO: Initialize ort::Session for each model component
        // let encoder = ort::Session::builder()?.with_model_from_file(dir.join("encoder.onnx"))?;
        // let decoder = ort::Session::builder()?.with_model_from_file(dir.join("decoder.onnx"))?;
        // let joiner = ort::Session::builder()?.with_model_from_file(dir.join("joiner.onnx"))?;

        tracing::info!("Models loaded");

        Ok(Self {
            _model_dir: model_dir.to_string(),
        })
    }

    /// Process a chunk of PCM16 audio and return any new transcript segments.
    pub fn process_chunk(&self, _pcm16: &[i16]) -> Result<Vec<TranscriptSegment>> {
        // TODO: RNN-T streaming decode loop
        // 1. Convert PCM16 to float32, extract features
        // 2. Run encoder on features
        // 3. Run joiner(encoder_out, decoder_state) in a loop
        // 4. Emit tokens until blank
        // 5. Return partial/final segments
        Ok(vec![])
    }

    /// Transcribe an entire WAV file (offline mode).
    pub fn transcribe_file(&self, path: &str) -> Result<String> {
        let reader = hound::WavReader::open(path).context("Failed to open WAV file")?;
        let spec = reader.spec();
        anyhow::ensure!(spec.channels == 1, "Expected mono audio, got {} channels", spec.channels);
        anyhow::ensure!(spec.bits_per_sample == 16, "Expected 16-bit audio");

        let samples: Vec<i16> = reader.into_samples::<i16>().filter_map(|s| s.ok()).collect();
        tracing::info!("Read {} samples at {} Hz", samples.len(), spec.sample_rate);

        let segments = self.process_chunk(&samples)?;
        Ok(segments.iter().map(|s| s.text.as_str()).collect::<Vec<_>>().join(" "))
    }
}

#[derive(Debug, Clone)]
pub struct TranscriptSegment {
    pub text: String,
    pub is_final: bool,
    pub timestamp: f64,
}
