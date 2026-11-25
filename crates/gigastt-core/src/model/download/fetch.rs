//! Streaming SHA-256 fetch with atomic rename.

use anyhow::{Context, Result};
use futures_util::StreamExt;
use std::path::Path;
use tokio::io::AsyncWriteExt;

use crate::sha256::{Sha256, hex_lower};

use super::super::progress::{DownloadProgress, ProgressEvent, ProgressSink};
use super::super::variant::ModelVariant;

/// Generate a unique `.partial` path so concurrent processes never write
/// to the same staging file. Uses PID and nanosecond timestamp.
#[cfg(feature = "net")]
pub(crate) fn partial_path_unique(final_path: &Path) -> std::path::PathBuf {
    let stamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    let mut s: std::ffi::OsString = final_path.as_os_str().to_owned();
    s.push(format!(".partial.{}.{}", std::process::id(), stamp));
    std::path::PathBuf::from(s)
}

/// RAII guard that removes a staged `.partial` file when the download path
/// returns early (stream error, write failure, finalize failure). Armed when
/// the staging path is chosen and disarmed only after the partial has been
/// renamed into its final location, so a flaky network cannot accumulate
/// orphaned `.partial` files.
#[cfg(feature = "net")]
pub(crate) struct PartialFileGuard(Option<std::path::PathBuf>);

#[cfg(feature = "net")]
impl PartialFileGuard {
    pub(crate) fn new(path: std::path::PathBuf) -> Self {
        Self(Some(path))
    }

    /// Disarm after the partial was renamed into place: the file no longer
    /// exists at the staging path, so there is nothing left to clean up.
    pub(crate) fn disarm(mut self) {
        self.0 = None;
    }
}

#[cfg(feature = "net")]
impl Drop for PartialFileGuard {
    fn drop(&mut self) {
        if let Some(path) = self.0.take() {
            // Best-effort: a missing file (already cleaned up, or never
            // created because the request failed before staging) is fine,
            // and a removal failure must not mask the original error.
            let _ = std::fs::remove_file(path);
        }
    }
}

/// Compute SHA-256 for a file synchronously, returning the lowercase hex digest.
#[cfg(feature = "net")]
pub(crate) fn sha256_file(path: &Path) -> Result<String> {
    let data = std::fs::read(path)
        .with_context(|| format!("Failed to read file for verification: {}", path.display()))?;
    let mut hasher = Sha256::new();
    hasher.update(&data);
    Ok(hex_lower(&hasher.finalize()))
}

/// Verify a staged `.partial` file against `expected_sha256` (when provided)
/// and atomically rename it into `final_path`. On mismatch the partial is
/// removed so a corrupt artefact cannot be mistaken for a good download on
/// restart. On success the partial no longer exists and `final_path` is the
/// only visible artefact. Separated from the network path so the filesystem
/// contract can be unit-tested without a mock HTTP server.
#[cfg(feature = "net")]
pub(crate) fn finalize_download(
    partial_path: &Path,
    final_path: &Path,
    expected_sha256: Option<&str>,
    label: &str,
) -> Result<()> {
    if let Some(expected) = expected_sha256 {
        let actual = sha256_file(partial_path)?;
        if actual != expected {
            // Remove the corrupt partial so a retry starts clean and so a
            // restart cannot promote the partial to final via race.
            let _ = std::fs::remove_file(partial_path);
            anyhow::bail!("SHA-256 mismatch for {label}: expected {expected}, got {actual}");
        }
        tracing::info!("SHA-256 verified: {label}");
    }

    std::fs::rename(partial_path, final_path).with_context(|| {
        format!(
            "Failed to rename {} -> {}",
            partial_path.display(),
            final_path.display()
        )
    })?;
    Ok(())
}

#[cfg(feature = "net")]
pub(crate) async fn download_file(variant: ModelVariant, filename: &str, dir: &Path) -> Result<()> {
    let url = format!(
        "https://huggingface.co/{}/resolve/main/{filename}",
        variant.hf_repo()
    );
    let final_dest = dir.join(filename);
    let expected = variant.checksum(filename);
    stream_to_partial_then_finalize(&url, &final_dest, expected, filename).await
}

static OFFLINE_MODE: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Disable subsequent model downloads for this process, without changing its
/// environment. Call before starting model acquisition. This setting is
/// one-way: the environment variable cannot override an explicit opt-out.
pub fn enable_offline_mode() {
    OFFLINE_MODE.store(true, std::sync::atomic::Ordering::Relaxed);
}

/// Whether air-gapped mode is active: explicit opt-out or `GIGASTT_OFFLINE` set to anything but
/// `""` / `"0"`. In this mode every would-be network fetch fails fast with an
/// instruction naming the missing file instead of a connect timeout.
#[cfg(feature = "net")]
fn offline_mode() -> bool {
    OFFLINE_MODE.load(std::sync::atomic::Ordering::Relaxed)
        || std::env::var_os("GIGASTT_OFFLINE").is_some_and(|v| !v.is_empty() && v != "0")
}

/// Hard ceiling on a single model-file download. The largest shipped INT8
/// encoder (`ml_ctc_large`) is ~592 MB; 2 GiB leaves room for ANE packages
/// without letting a redirected host fill the disk.
#[cfg(feature = "net")]
pub(crate) const MAX_DOWNLOAD_BYTES: u64 = 2 * 1024 * 1024 * 1024;

/// Return an error once `so_far + extra` would exceed [`MAX_DOWNLOAD_BYTES`].
#[cfg(feature = "net")]
pub(crate) fn reject_if_over_download_cap(so_far: u64, extra: u64) -> Result<()> {
    if so_far.saturating_add(extra) > MAX_DOWNLOAD_BYTES {
        anyhow::bail!(
            "download exceeded size cap of {MAX_DOWNLOAD_BYTES} bytes \
             ({so_far} already written, +{extra})"
        );
    }
    Ok(())
}

/// Streaming download with SHA-256 verification and atomic rename.
///
/// Stages the response into `<final_dest>.partial`, verifies the hash (when
/// `expected_sha256` is provided), and atomically renames the partial into
/// the final path. On checksum mismatch or crash the final path is never
/// observable, so a retry starts from a clean slate.
///
/// Shared by [`ensure_model`] (per-file download loop) and
/// [`ensure_speaker_model`] (single-file diarization download) so the
/// TOCTOU + progress + retry semantics match bit-for-bit.
#[cfg(feature = "net")]
pub(crate) async fn stream_to_partial_then_finalize(
    url: &str,
    final_dest: &Path,
    expected_sha256: Option<&str>,
    label: &str,
) -> Result<()> {
    stream_to_partial_then_finalize_with_sink(
        url,
        final_dest,
        expected_sha256,
        label,
        &ProgressSink::global(),
    )
    .await
}

/// Sink-parameterized core of [`stream_to_partial_then_finalize`] so tests
/// can capture the emitted [`ProgressEvent`]s instead of parsing process
/// stdout.
#[cfg(feature = "net")]
pub(crate) async fn stream_to_partial_then_finalize_with_sink(
    url: &str,
    final_dest: &Path,
    expected_sha256: Option<&str>,
    label: &str,
    sink: &ProgressSink,
) -> Result<()> {
    // Air-gapped guard: every network fetch in this crate funnels through this
    // function, so one check here covers the model download, the prequantized
    // bundle, ANE packages, and the speaker / punctuation / VAD models. Checked
    // per call (not cached) so the `--offline` CLI flag, which sets the env var
    // at startup, and an externally exported variable behave identically.
    if offline_mode() {
        anyhow::bail!(
            "offline mode (GIGASTT_OFFLINE=1): refusing to download {label}; \
             place the file at {} manually (see docs/deployment.md, \
             \"Air-gapped / offline installation\")",
            final_dest.display()
        );
    }

    let partial = partial_path_unique(final_dest);
    // Remove the staged partial on any early return below; disarmed after the
    // successful rename at the end of this function.
    let cleanup = PartialFileGuard::new(partial.clone());

    tracing::info!("Downloading {label}...");

    // Configured client: bound the connect/TLS handshake and per-read stalls, and
    // cap redirects. NOT a whole-request timeout (a legitimate ~225 MB INT8 download can
    // take minutes) and NO host pinning (HF LFS 302-redirects to a CloudFront host).
    let client = reqwest::Client::builder()
        .connect_timeout(std::time::Duration::from_secs(30))
        .read_timeout(std::time::Duration::from_secs(300))
        .redirect(reqwest::redirect::Policy::limited(5))
        .build()
        .context("Failed to build HTTP client")?;
    let response = client
        .get(url)
        .send()
        .await
        .context("HTTP request failed")?;
    let status = response.status();
    if !status.is_success() {
        anyhow::bail!("Download failed for {label}: HTTP {status}");
    }
    let total_size = response.content_length().unwrap_or(0);
    if total_size > 0 {
        reject_if_over_download_cap(0, total_size)?;
    }

    let mut progress = DownloadProgress::new(total_size);

    let mut file = tokio::fs::File::create(&partial)
        .await
        .context("Failed to create partial model file")?;
    let mut stream = response.bytes_stream();

    let mut downloaded: u64 = 0;
    while let Some(chunk) = stream.next().await {
        let chunk = chunk.context("Download stream error")?;
        reject_if_over_download_cap(downloaded, chunk.len() as u64)?;
        file.write_all(&chunk)
            .await
            .context("Failed to write chunk")?;
        downloaded += chunk.len() as u64;
        progress.update(chunk.len() as u64, sink, label);
    }

    file.flush().await?;
    drop(file);
    progress.finish(sink, label);
    tracing::info!("Wrote partial {} ({downloaded} bytes)", partial.display());

    if expected_sha256.is_some() {
        sink.event(&ProgressEvent::Verify {
            file: label.to_string(),
        });
    }
    finalize_download(&partial, final_dest, expected_sha256, label)?;
    cleanup.disarm();
    tracing::info!("Saved {label}");

    Ok(())
}
