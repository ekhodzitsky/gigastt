//! Dump decoded 16 kHz PCM and log-mel features for matched-input diagnostics.
//! Usage: cargo run -p gigastt-core --example inspect_audio -- INPUT OUTPUT_PREFIX

use anyhow::{Context, Result};
use gigastt_core::inference::{FeatureExtractor, N_MELS, audio};
use std::io::{BufWriter, Write};

fn write_floats(path: &str, values: &[f32]) -> Result<()> {
    let mut file = BufWriter::new(std::fs::File::create(path)?);
    for value in values {
        file.write_all(&value.to_le_bytes())?;
    }
    file.flush()?;
    Ok(())
}

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let input = args.next().context("missing input audio path")?;
    let prefix = args.next().context("missing output prefix")?;
    let samples = audio::decode_audio_file(&input)?;
    let (features, frames) = FeatureExtractor::new().compute(&samples);
    write_floats(&format!("{prefix}.pcm.f32"), &samples)?;
    write_floats(&format!("{prefix}.mel.f32"), &features)?;
    std::fs::write(
        format!("{prefix}.json"),
        serde_json::to_vec_pretty(&serde_json::json!({
            "sample_rate": 16000,
            "samples": samples.len(),
            "mel_shape": [N_MELS, frames],
            "encoding": "little-endian float32",
        }))?,
    )?;
    Ok(())
}
