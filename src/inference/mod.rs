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
        let mut all_samples = std::mem::take(&mut state.audio_buffer);
        all_samples.extend_from_slice(&new_samples);

        const MAX_BUFFER_SAMPLES: usize = 16000 * 5; // 5 seconds at 16kHz
        if all_samples.len() > MAX_BUFFER_SAMPLES {
            tracing::warn!("Audio buffer exceeded 5s limit, truncating");
            all_samples = all_samples[all_samples.len() - MAX_BUFFER_SAMPLES..].to_vec();
        }

        let samples = all_samples;

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
    fn resample(samples: &[f32], from_rate: u32, to_rate: u32) -> Vec<f32> {
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
