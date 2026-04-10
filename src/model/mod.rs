//! Model download and management.
//!
//! Downloads GigaAM v3 e2e_rnnt ONNX files from HuggingFace to `~/.gigastt/models/`.

use anyhow::{Context, Result};
use futures_util::StreamExt;
use indicatif::{ProgressBar, ProgressStyle};
use std::path::Path;
use tokio::io::AsyncWriteExt;

const HF_REPO: &str = "istupakov/gigaam-v3-onnx";
const MODEL_FILES: &[&str] = &["v3_e2e_rnnt_encoder.onnx", "v3_e2e_rnnt_decoder.onnx", "v3_e2e_rnnt_joint.onnx", "v3_e2e_rnnt_vocab.txt"];

/// Return the default model directory path (`~/.gigastt/models/`).
///
/// Falls back to `.gigastt/models` if the home directory cannot be determined.
pub fn default_model_dir() -> String {
    dirs::home_dir()
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
    let total_size = response.content_length().unwrap_or(0);

    let pb = ProgressBar::new(total_size);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("{spinner:.green} [{bar:40.cyan/blue}] {bytes}/{total_bytes} ({eta})")
            .expect("valid template")
            .progress_chars("#>-"),
    );

    let mut file = tokio::fs::File::create(&dest)
        .await
        .context("Failed to create model file")?;
    let mut stream = response.bytes_stream();

    let mut downloaded: u64 = 0;
    while let Some(chunk) = stream.next().await {
        let chunk = chunk.context("Download stream error")?;
        file.write_all(&chunk).await.context("Failed to write chunk")?;
        downloaded += chunk.len() as u64;
        pb.set_position(downloaded);
    }

    file.flush().await?;
    pb.finish_with_message("done");
    tracing::info!("Saved {filename} ({downloaded} bytes)");

    Ok(())
}
