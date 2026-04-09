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

pub const N_MELS: usize = 64;
pub const N_FFT: usize = 320;
pub const HOP_LENGTH: usize = 160;
pub const PRED_HIDDEN: usize = 320;

fn ort_err(e: impl std::fmt::Display) -> anyhow::Error {
    anyhow::anyhow!("{e}")
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

        tracing::info!("Loading ONNX models from {model_dir}...");

        let encoder = Session::builder()
            .map_err(ort_err)?
            .commit_from_file(dir.join("v3_e2e_rnnt_encoder.onnx"))
            .map_err(ort_err)?;

        let decoder = Session::builder()
            .map_err(ort_err)?
            .commit_from_file(dir.join("v3_e2e_rnnt_decoder.onnx"))
            .map_err(ort_err)?;

        let joiner = Session::builder()
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

        let samples = match prepare_audio_buffer(&new_samples, &mut state.audio_buffer) {
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
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs_f64(),
        }])
    }

    /// Transcribe an audio file (supports WAV, MP3, M4A/AAC, OGG, FLAC).
    pub fn transcribe_file(&self, path: &str) -> Result<String> {
        let float_samples = Self::decode_audio_file(path)?;

        let (features, num_frames) = self.mel.compute(&float_samples);
        tracing::info!("Extracted {} mel frames", num_frames);

        let mut decoder_state = DecoderState::new(self.tokenizer.blank_id());
        self.run_inference(&features, num_frames, &mut decoder_state)
    }

    /// Decode any supported audio file to mono f32 samples at 16kHz.
    fn decode_audio_file(path: &str) -> Result<Vec<f32>> {
        use symphonia::core::audio::SampleBuffer;
        use symphonia::core::codecs::DecoderOptions;
        use symphonia::core::formats::FormatOptions;
        use symphonia::core::io::MediaSourceStream;
        use symphonia::core::meta::MetadataOptions;
        use symphonia::core::probe::Hint;

        let file = std::fs::File::open(path)
            .with_context(|| format!("Failed to open audio file: {path}"))?;
        let mss = MediaSourceStream::new(Box::new(file), Default::default());

        let mut hint = Hint::new();
        if let Some(ext) = std::path::Path::new(path).extension().and_then(|e| e.to_str()) {
            hint.with_extension(ext);
        }

        let probed = symphonia::default::get_probe()
            .format(&hint, mss, &FormatOptions::default(), &MetadataOptions::default())
            .context("Unsupported audio format")?;

        let mut format = probed.format;

        let track = format
            .default_track()
            .context("No audio track found")?;
        let track_id = track.id;
        let sample_rate = track
            .codec_params
            .sample_rate
            .context("Unknown sample rate")?;
        let channels = track
            .codec_params
            .channels
            .map(|c| c.count())
            .unwrap_or(1);

        tracing::info!("Audio: {sample_rate}Hz, {channels}ch, format={}",
            std::path::Path::new(path).extension().unwrap_or_default().to_string_lossy());

        let mut decoder = symphonia::default::get_codecs()
            .make(&track.codec_params, &DecoderOptions::default())
            .context("Unsupported audio codec")?;

        let mut all_samples: Vec<f32> = Vec::new();

        loop {
            let packet = match format.next_packet() {
                Ok(p) => p,
                Err(symphonia::core::errors::Error::IoError(ref e))
                    if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
                Err(e) => return Err(anyhow::anyhow!("Error reading packet: {e}")),
            };

            if packet.track_id() != track_id {
                continue;
            }

            let decoded = decoder.decode(&packet).context("Decode error")?;
            let spec = *decoded.spec();
            let num_frames = decoded.frames();

            let mut sample_buf = SampleBuffer::<f32>::new(num_frames as u64, spec);
            sample_buf.copy_interleaved_ref(decoded);
            let samples = sample_buf.samples();

            // Mix to mono if multi-channel
            if spec.channels.count() > 1 {
                let ch = spec.channels.count();
                for frame in 0..num_frames {
                    let mut sum = 0.0_f32;
                    for c in 0..ch {
                        sum += samples[frame * ch + c];
                    }
                    all_samples.push(sum / ch as f32);
                }
            } else {
                all_samples.extend_from_slice(samples);
            }
        }

        tracing::info!(
            "Decoded {} samples at {}Hz ({:.1}s)",
            all_samples.len(),
            sample_rate,
            all_samples.len() as f64 / sample_rate as f64
        );

        // Resample to 16kHz if needed
        if sample_rate != 16000 {
            all_samples = Self::resample(&all_samples, sample_rate, 16000);
            tracing::info!("Resampled to 16kHz: {} samples", all_samples.len());
        }

        Ok(all_samples)
    }

    /// Simple linear interpolation resampler.
    pub(crate) fn resample(samples: &[f32], from_rate: u32, to_rate: u32) -> Vec<f32> {
        if samples.is_empty() || from_rate == 0 || to_rate == 0 {
            return Vec::new();
        }
        let ratio = from_rate as f64 / to_rate as f64;
        let out_len = (samples.len() as f64 / ratio) as usize;
        let mut output = Vec::with_capacity(out_len);

        for i in 0..out_len {
            let src_pos = i as f64 * ratio;
            let idx = src_pos as usize;
            let frac = src_pos - idx as f64;

            let sample = if idx + 1 < samples.len() {
                samples[idx] as f64 * (1.0 - frac) + samples[idx + 1] as f64 * frac
            } else {
                samples[idx.min(samples.len() - 1)] as f64
            };
            output.push(sample as f32);
        }

        output
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

const MAX_BUFFER_SAMPLES: usize = 16000 * 5; // 5 seconds at 16kHz

/// Prepare audio buffer for processing: merge new samples with leftover,
/// truncate if too long, split into usable samples and new leftover.
///
/// Returns `Some(usable_samples)` if enough data for at least one frame,
/// `None` if all data was buffered for the next call.
/// Updates `buffer` in-place with leftover samples.
pub(crate) fn prepare_audio_buffer(
    new_samples: &[f32],
    buffer: &mut Vec<f32>,
) -> Option<Vec<f32>> {
    let mut all_samples = std::mem::take(buffer);
    all_samples.extend_from_slice(new_samples);

    if all_samples.len() > MAX_BUFFER_SAMPLES {
        tracing::warn!("Audio buffer exceeded 5s limit, truncating");
        all_samples = all_samples[all_samples.len() - MAX_BUFFER_SAMPLES..].to_vec();
    }

    let hop_length = HOP_LENGTH;
    let n_fft = N_FFT;
    let usable = if all_samples.len() >= n_fft {
        let num_frames = (all_samples.len() - n_fft) / hop_length + 1;
        (num_frames - 1) * hop_length + n_fft
    } else {
        0
    };

    if usable == 0 {
        *buffer = all_samples;
        return None;
    }

    *buffer = all_samples[usable..].to_vec();
    Some(all_samples[..usable].to_vec())
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- DecoderState tests ---

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

    // --- resample tests ---

    #[test]
    fn test_resample_downsample_length() {
        let input: Vec<f32> = (0..4800).map(|i| (i as f32).sin()).collect();
        let output = Engine::resample(&input, 48000, 16000);
        // 48000→16000 is 3:1, so output ≈ input/3
        assert_eq!(output.len(), 1600);
    }

    #[test]
    fn test_resample_upsample_length() {
        let input: Vec<f32> = (0..800).map(|i| (i as f32).sin()).collect();
        let output = Engine::resample(&input, 8000, 16000);
        assert_eq!(output.len(), 1600);
    }

    #[test]
    fn test_resample_preserves_dc() {
        // Constant signal should remain constant after resampling
        let input = vec![0.5_f32; 4800];
        let output = Engine::resample(&input, 48000, 16000);
        for &sample in &output {
            assert!((sample - 0.5).abs() < 1e-5, "DC signal not preserved: {sample}");
        }
    }

    #[test]
    fn test_resample_empty() {
        let output = Engine::resample(&[], 48000, 16000);
        assert!(output.is_empty());
    }

    #[test]
    fn test_resample_zero_rate_returns_empty() {
        let input = vec![1.0, 2.0, 3.0];
        assert!(Engine::resample(&input, 0, 16000).is_empty());
        assert!(Engine::resample(&input, 16000, 0).is_empty());
    }

    #[test]
    fn test_resample_same_rate() {
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let output = Engine::resample(&input, 16000, 16000);
        assert_eq!(output.len(), input.len());
        for (a, b) in input.iter().zip(output.iter()) {
            assert!((a - b).abs() < 1e-5);
        }
    }

    // --- prepare_audio_buffer tests ---

    #[test]
    fn test_buffer_short_input_returns_none() {
        // Less than N_FFT (320) samples → buffer everything
        let new_samples = vec![0.0; 100];
        let mut buffer = Vec::new();
        let result = prepare_audio_buffer(&new_samples, &mut buffer);
        assert!(result.is_none());
        assert_eq!(buffer.len(), 100);
    }

    #[test]
    fn test_buffer_exact_frame() {
        // Exactly N_FFT (320) samples → one frame, no leftover
        let new_samples = vec![1.0; N_FFT];
        let mut buffer = Vec::new();
        let result = prepare_audio_buffer(&new_samples, &mut buffer);
        assert!(result.is_some());
        assert_eq!(result.unwrap().len(), N_FFT);
        assert!(buffer.is_empty());
    }

    #[test]
    fn test_buffer_leftover_correct() {
        // N_FFT + 50 samples → one frame usable, 50 leftover
        let new_samples = vec![1.0; N_FFT + 50];
        let mut buffer = Vec::new();
        let result = prepare_audio_buffer(&new_samples, &mut buffer);
        assert!(result.is_some());
        let usable = result.unwrap();
        assert_eq!(usable.len(), N_FFT); // one frame
        assert_eq!(buffer.len(), 50);
    }

    #[test]
    fn test_buffer_accumulates_across_calls() {
        let mut buffer = Vec::new();
        // First call: 200 samples (< 320) → buffered
        let result = prepare_audio_buffer(&vec![1.0; 200], &mut buffer);
        assert!(result.is_none());
        assert_eq!(buffer.len(), 200);

        // Second call: 200 more → total 400, enough for 1 frame (320), leftover 80
        let result = prepare_audio_buffer(&vec![2.0; 200], &mut buffer);
        assert!(result.is_some());
        let usable = result.unwrap();
        assert_eq!(usable.len(), 320);
        assert_eq!(buffer.len(), 80);
    }

    #[test]
    fn test_buffer_truncation_at_5s() {
        // More than 80000 samples (5s at 16kHz) → truncate to last 80000
        let mut buffer = vec![0.0; 90000];
        let new_samples = vec![1.0; 1000];
        let result = prepare_audio_buffer(&new_samples, &mut buffer);
        // Total was 91000, truncated to 80000, then split into usable + leftover
        assert!(result.is_some());
        let usable = result.unwrap();
        assert!(usable.len() + buffer.len() <= MAX_BUFFER_SAMPLES);
    }

    #[test]
    fn test_buffer_multi_frame() {
        // N_FFT + HOP_LENGTH = 480 → 2 frames, no leftover
        let new_samples = vec![1.0; N_FFT + HOP_LENGTH];
        let mut buffer = Vec::new();
        let result = prepare_audio_buffer(&new_samples, &mut buffer);
        assert!(result.is_some());
        // 2 frames: usable = (2-1)*160 + 320 = 480
        assert_eq!(result.unwrap().len(), N_FFT + HOP_LENGTH);
        assert!(buffer.is_empty());
    }
}
