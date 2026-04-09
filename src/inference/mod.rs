//! ONNX Runtime inference engine for GigaAM v3 e2e_rnnt.
//!
//! Loads encoder, decoder, and joiner ONNX models and runs the RNN-T streaming decode loop.

mod decode;
mod features;
mod tokenizer;

use anyhow::{Context, Result};
use ort::session::Session;
use ort::value::TensorRef;
use std::path::Path;
use std::sync::Mutex;

use features::MelSpectrogram;
use tokenizer::Tokenizer;

/// Decoder LSTM state persisted across chunks.
pub struct DecoderState {
    pub h: Vec<f32>,
    pub c: Vec<f32>,
    pub prev_token: i64,
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

        tracing::info!("Loading ONNX models from {model_dir}...");

        let encoder = Session::builder()
            .map_err(|e| anyhow::anyhow!("{e}"))?
            .commit_from_file(dir.join("v3_e2e_rnnt_encoder.onnx"))
            .map_err(|e| anyhow::anyhow!("{e}"))?;

        let decoder = Session::builder()
            .map_err(|e| anyhow::anyhow!("{e}"))?
            .commit_from_file(dir.join("v3_e2e_rnnt_decoder.onnx"))
            .map_err(|e| anyhow::anyhow!("{e}"))?;

        let joiner = Session::builder()
            .map_err(|e| anyhow::anyhow!("{e}"))?
            .commit_from_file(dir.join("v3_e2e_rnnt_joint.onnx"))
            .map_err(|e| anyhow::anyhow!("{e}"))?;

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
            decoder: DecoderState {
                h: vec![0.0; 320],
                c: vec![0.0; 320],
                prev_token: self.tokenizer.blank_id() as i64,
            },
            audio_buffer: Vec::new(),
        }
    }

    /// Process a chunk of PCM16 audio and return any new transcript segments.
    ///
    /// Streaming state (LSTM hidden/cell, leftover audio) is maintained in `state`.
    pub fn process_chunk(
        &self,
        pcm16: &[i16],
        state: &mut StreamingState,
    ) -> Result<Vec<TranscriptSegment>> {
        if pcm16.is_empty() {
            return Ok(vec![]);
        }

        // Convert to f32 and prepend leftover samples from the previous chunk
        let new_samples: Vec<f32> = pcm16.iter().map(|&s| s as f32 / 32768.0).collect();
        let mut samples = std::mem::take(&mut state.audio_buffer);
        samples.extend_from_slice(&new_samples);

        // Save leftover samples that don't fill a complete hop (160 samples)
        let hop_length = 160;
        let n_fft = 320;
        let usable = if samples.len() >= n_fft {
            // Number of complete frames: (len - n_fft) / hop + 1
            // Samples consumed: (num_frames - 1) * hop + n_fft
            let num_frames = (samples.len() - n_fft) / hop_length + 1;
            (num_frames - 1) * hop_length + n_fft
        } else {
            0
        };

        if usable == 0 {
            // Not enough samples even for one frame — buffer everything
            state.audio_buffer = samples;
            return Ok(vec![]);
        }

        // Keep leftover for next chunk
        state.audio_buffer = samples[usable..].to_vec();
        let samples = &samples[..usable];

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
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs_f64(),
        }])
    }

    /// Transcribe an entire WAV file (offline mode).
    pub fn transcribe_file(&self, path: &str) -> Result<String> {
        let reader = hound::WavReader::open(path).context("Failed to open WAV file")?;
        let spec = reader.spec();
        anyhow::ensure!(
            spec.channels == 1,
            "Expected mono audio, got {} channels",
            spec.channels
        );
        anyhow::ensure!(spec.bits_per_sample == 16, "Expected 16-bit audio");

        if spec.sample_rate != 16000 {
            anyhow::bail!(
                "Expected 16kHz audio, got {} Hz. Please resample first.",
                spec.sample_rate
            );
        }

        let samples: Vec<i16> = reader
            .into_samples::<i16>()
            .collect::<std::result::Result<Vec<_>, _>>()
            .context("Corrupted samples in WAV file")?;
        tracing::info!(
            "Read {} samples at {} Hz ({:.1}s)",
            samples.len(),
            spec.sample_rate,
            samples.len() as f64 / spec.sample_rate as f64
        );

        let float_samples: Vec<f32> = samples.iter().map(|&s| s as f32 / 32768.0).collect();
        let (features, num_frames) = self.mel.compute(&float_samples);
        tracing::info!("Extracted {} mel frames", num_frames);

        let mut decoder_state = DecoderState {
            h: vec![0.0; 320],
            c: vec![0.0; 320],
            prev_token: self.tokenizer.blank_id() as i64,
        };
        self.run_inference(&features, num_frames, &mut decoder_state)
    }

    fn run_inference(
        &self,
        features: &[f32],
        num_frames: usize,
        decoder_state: &mut DecoderState,
    ) -> Result<String> {
        let n_mels = 64;

        // Encoder input: audio_signal [1, 64, num_frames], length [1]
        let signal_tensor =
            TensorRef::from_array_view(([1_usize, n_mels, num_frames], features))?;
        let length_data = [num_frames as i64];
        let length_tensor =
            TensorRef::from_array_view(([1_usize], length_data.as_slice()))?;

        let mut encoder = self.encoder.lock().unwrap_or_else(|e| e.into_inner());
        let encoder_outputs = encoder
            .run(ort::inputs![signal_tensor, length_tensor])
            .context("Encoder inference failed")?;

        let (_enc_shape, enc_data) = encoder_outputs[0]
            .try_extract_tensor::<f32>()
            .context("Failed to extract encoder output")?;
        let (_len_shape, len_data) = encoder_outputs[1]
            .try_extract_tensor::<i32>()
            .context("Failed to extract encoder length")?;

        let enc_len = len_data[0] as usize;

        tracing::debug!("Encoder output: {} frames", enc_len);

        // Need to copy encoder data since we need to drop encoder_outputs before locking decoder
        let enc_data_owned: Vec<f32> = enc_data.to_vec();
        drop(encoder_outputs);
        drop(encoder);

        // RNN-T greedy decode
        let mut decoder = self.decoder.lock().unwrap_or_else(|e| e.into_inner());
        let mut joiner = self.joiner.lock().unwrap_or_else(|e| e.into_inner());

        let token_ids = decode::greedy_decode(
            &mut decoder,
            &mut joiner,
            &enc_data_owned,
            enc_len,
            self.tokenizer.blank_id(),
            decoder_state,
        )?;

        let text = self.tokenizer.decode(&token_ids);
        tracing::info!("Decoded {} tokens → \"{}\"", token_ids.len(), text);

        Ok(text)
    }
}

#[derive(Debug, Clone)]
pub struct TranscriptSegment {
    pub text: String,
    pub is_final: bool,
    pub timestamp: f64,
}
