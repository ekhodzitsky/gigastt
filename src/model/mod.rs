//! Model download and management.
//!
//! Downloads GigaAM v3 e2e_rnnt ONNX files from HuggingFace to `~/.gigastt/models/`.

use anyhow::{Context, Result};
use futures_util::StreamExt;
use sha2::{Digest, Sha256};
use std::path::Path;
use tokio::io::AsyncWriteExt;

/// Simple download progress reporter (no external deps).
struct DownloadProgress {
    total: u64,
    current: u64,
    last_percent: u8,
}

impl DownloadProgress {
    fn new(total: u64) -> Self {
        Self { total, current: 0, last_percent: 0 }
    }

    fn update(&mut self, bytes: u64) {
        self.current += bytes;
        let percent = if self.total > 0 { (self.current * 100 / self.total) as u8 } else { 0 };
        if percent != self.last_percent {
            self.last_percent = percent;
            eprint!("\rDownloading... {percent}% ({:.1}MB / {:.1}MB)",
                self.current as f64 / 1_048_576.0, self.total as f64 / 1_048_576.0);
        }
    }

    fn finish(&self) {
        eprintln!("\rDownload complete ({:.1}MB)                    ", self.current as f64 / 1_048_576.0);
    }
}

const HF_REPO: &str = "istupakov/gigaam-v3-onnx";
const MODEL_FILES: &[&str] = &["v3_e2e_rnnt_encoder.onnx", "v3_e2e_rnnt_decoder.onnx", "v3_e2e_rnnt_joint.onnx", "v3_e2e_rnnt_vocab.txt"];

/// SHA-256 checksums for model integrity verification.
/// Set to None to skip verification (first download).
const MODEL_CHECKSUMS: &[(&str, Option<&str>)] = &[
    ("v3_e2e_rnnt_encoder.onnx", None),
    ("v3_e2e_rnnt_decoder.onnx", None),
    ("v3_e2e_rnnt_joint.onnx", None),
    ("v3_e2e_rnnt_vocab.txt", None),
];

#[cfg(feature = "diarization")]
const SPEAKER_HF_REPO: &str = "onnx-community/wespeaker-voxceleb-resnet34-LM";
#[cfg(feature = "diarization")]
pub const SPEAKER_MODEL_FILE: &str = "wespeaker_resnet34.onnx";

fn home_dir() -> Option<std::path::PathBuf> {
    #[cfg(unix)]
    { std::env::var_os("HOME").map(std::path::PathBuf::from) }
    #[cfg(windows)]
    { std::env::var_os("USERPROFILE").map(std::path::PathBuf::from) }
}

/// Return the default model directory path (`~/.gigastt/models/`).
///
/// Falls back to `.gigastt/models` if the home directory cannot be determined.
pub fn default_model_dir() -> String {
    home_dir()
        .map(|h| h.join(".gigastt").join("models").to_string_lossy().into_owned())
        .unwrap_or_else(|| ".gigastt/models".into())
}

/// Ensure model files exist in `model_dir`, downloading from HuggingFace if missing.
///
/// Downloads encoder, decoder, joiner ONNX models and vocabulary from
/// the `istupakov/gigaam-v3-onnx` repository. Shows progress bars during download.
pub async fn ensure_model(model_dir: &str) -> Result<()> {
    let dir = Path::new(model_dir);

    if model_files_exist(dir) {
        tracing::info!("Model found at {model_dir}");
        return Ok(());
    }

    tracing::info!("Model not found, downloading from HuggingFace...");
    std::fs::create_dir_all(dir).context("Failed to create model directory")?;

    for file in MODEL_FILES {
        download_file(file, dir).await?;
    }

    tracing::info!("Model download complete");
    Ok(())
}

/// Ensure the speaker diarization model exists in `model_dir`, downloading from HuggingFace if missing.
///
/// Downloads `wespeaker_resnet34.onnx` from `onnx-community/wespeaker-voxceleb-resnet34-LM`.
#[cfg(feature = "diarization")]
pub async fn ensure_speaker_model(model_dir: &str) -> Result<()> {
    let dir = Path::new(model_dir);
    let dest = dir.join(SPEAKER_MODEL_FILE);

    if dest.exists() {
        tracing::info!("Speaker model found at {}", dest.display());
        return Ok(());
    }

    tracing::info!("Speaker model not found, downloading from HuggingFace...");
    std::fs::create_dir_all(dir).context("Failed to create model directory")?;

    let hf_path = "onnx/model.onnx";
    let url = format!("https://huggingface.co/{SPEAKER_HF_REPO}/resolve/main/{hf_path}");

    let response = reqwest::get(&url).await.context("HTTP request failed")?;
    let status = response.status();
    if !status.is_success() {
        anyhow::bail!("Download failed for {SPEAKER_MODEL_FILE}: HTTP {status}");
    }
    let total_size = response.content_length().unwrap_or(0);

    let mut pb = DownloadProgress::new(total_size);

    let mut file = tokio::fs::File::create(&dest)
        .await
        .context("Failed to create speaker model file")?;
    let mut stream = response.bytes_stream();

    let mut downloaded: u64 = 0;
    while let Some(chunk) = stream.next().await {
        let chunk = chunk.context("Download stream error")?;
        file.write_all(&chunk).await.context("Failed to write chunk")?;
        downloaded += chunk.len() as u64;
        pb.update(chunk.len() as u64);
    }

    file.flush().await?;
    pb.finish();
    tracing::info!("Saved {} ({downloaded} bytes)", SPEAKER_MODEL_FILE);

    Ok(())
}

fn model_files_exist(dir: &Path) -> bool {
    MODEL_FILES.iter().all(|f| dir.join(f).exists())
}

async fn download_file(filename: &str, dir: &Path) -> Result<()> {
    let url = format!(
        "https://huggingface.co/{HF_REPO}/resolve/main/{filename}"
    );
    let dest = dir.join(filename);

    tracing::info!("Downloading {filename}...");

    let response = reqwest::get(&url).await.context("HTTP request failed")?;
    let status = response.status();
    if !status.is_success() {
        anyhow::bail!("Download failed for {filename}: HTTP {status}");
    }
    let total_size = response.content_length().unwrap_or(0);

    let mut progress = DownloadProgress::new(total_size);

    let mut file = tokio::fs::File::create(&dest)
        .await
        .context("Failed to create model file")?;
    let mut stream = response.bytes_stream();

    let mut downloaded: u64 = 0;
    while let Some(chunk) = stream.next().await {
        let chunk = chunk.context("Download stream error")?;
        file.write_all(&chunk).await.context("Failed to write chunk")?;
        downloaded += chunk.len() as u64;
        progress.update(chunk.len() as u64);
    }

    file.flush().await?;
    progress.finish();
    tracing::info!("Saved {filename} ({downloaded} bytes)");

    // Verify SHA-256 if checksum is known
    if let Some(expected) = MODEL_CHECKSUMS.iter()
        .find(|(name, _)| *name == filename)
        .and_then(|(_, hash)| *hash)
    {
        let data = tokio::fs::read(&dest).await.context("Failed to read downloaded file for verification")?;
        let mut hasher = Sha256::new();
        hasher.update(&data);
        let actual = format!("{:x}", hasher.finalize());
        if actual != expected {
            tokio::fs::remove_file(&dest).await.ok();
            anyhow::bail!("SHA-256 mismatch for {filename}: expected {expected}, got {actual}");
        }
        tracing::info!("SHA-256 verified: {filename}");
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_home_dir_returns_some() {
        // On any CI or developer machine HOME / USERPROFILE should be set.
        assert!(home_dir().is_some(), "home_dir() must return Some on this platform");
    }

    #[test]
    fn test_default_model_dir_contains_gigastt() {
        let dir = default_model_dir();
        assert!(
            dir.contains(".gigastt"),
            "default_model_dir() should contain \".gigastt\", got: {dir}"
        );
    }

    #[test]
    fn test_download_progress_basic() {
        let mut progress = DownloadProgress::new(1_000_000);
        // Should not panic on normal update.
        progress.update(500_000);
        assert_eq!(progress.current, 500_000);
        assert_eq!(progress.last_percent, 50);
        progress.finish();
    }

    #[test]
    fn test_download_progress_zero_total() {
        let mut progress = DownloadProgress::new(0);
        // Must not divide by zero.
        progress.update(100);
        assert_eq!(progress.last_percent, 0);
        progress.finish();
    }
}
