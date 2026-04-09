//! Model download and management.
//!
//! Downloads GigaAM v3 e2e_rnnt ONNX files from HuggingFace to `~/.gigastt/models/`.

use anyhow::{Context, Result};
use indicatif::{ProgressBar, ProgressStyle};
use std::path::Path;

const HF_REPO: &str = "istupakov/gigaam-v3-onnx";
const MODEL_FILES: &[&str] = &["encoder.onnx", "decoder.onnx", "joiner.onnx", "tokens.txt"];

pub fn default_model_dir() -> String {
    dirs::home_dir()
        .map(|h| h.join(".gigastt").join("models").to_string_lossy().into_owned())
        .unwrap_or_else(|| ".gigastt/models".into())
}

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
        "https://huggingface.co/{HF_REPO}/resolve/main/v3_e2e_rnnt/{filename}"
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

    let bytes = response.bytes().await.context("Failed to read response body")?;
    pb.set_position(bytes.len() as u64);
    pb.finish_with_message("done");

    std::fs::write(&dest, &bytes).context("Failed to write model file")?;
    tracing::info!("Saved {filename} ({} bytes)", bytes.len());

    Ok(())
}
