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
        Self {
            total,
            current: 0,
            last_percent: 0,
        }
    }

    fn update(&mut self, bytes: u64) {
        self.current += bytes;
        let percent = (self.current * 100)
            .checked_div(self.total)
            .map(|p| p as u8)
            .unwrap_or(0);
        if percent != self.last_percent {
            self.last_percent = percent;
            eprint!(
                "\rDownloading... {percent}% ({:.1}MB / {:.1}MB)",
                self.current as f64 / 1_048_576.0,
                self.total as f64 / 1_048_576.0
            );
        }
    }

    fn finish(&self) {
        eprintln!(
            "\rDownload complete ({:.1}MB)                    ",
            self.current as f64 / 1_048_576.0
        );
    }
}

const HF_REPO: &str = "istupakov/gigaam-v3-onnx";
const MODEL_FILES: &[&str] = &[
    "v3_e2e_rnnt_encoder.onnx",
    "v3_e2e_rnnt_decoder.onnx",
    "v3_e2e_rnnt_joint.onnx",
    "v3_e2e_rnnt_vocab.txt",
];

/// SHA-256 checksums for model integrity verification.
const MODEL_CHECKSUMS: &[(&str, Option<&str>)] = &[
    (
        "v3_e2e_rnnt_encoder.onnx",
        Some("cd60b3764a832e8560ae6d3ad0b10adc1a42ffae412b9476f25620aae4f4a508"),
    ),
    (
        "v3_e2e_rnnt_decoder.onnx",
        Some("7b0a16d67fd2cb37061decc93c69e364a9ab27afee3c57495d55b1c974cf7231"),
    ),
    (
        "v3_e2e_rnnt_joint.onnx",
        Some("602ff7017a93311aad34df1437c8d7f49911353c13d6eae7a6ee7b041339465c"),
    ),
    (
        "v3_e2e_rnnt_vocab.txt",
        Some("39abae20e692998290c574e606f11a9edef2902a1995463fcff63d1490cf22b7"),
    ),
];

#[cfg(feature = "diarization")]
const SPEAKER_HF_REPO: &str = "onnx-community/wespeaker-voxceleb-resnet34-LM";
#[cfg(feature = "diarization")]
pub const SPEAKER_MODEL_FILE: &str = "wespeaker_resnet34.onnx";

/// SHA-256 of the upstream speaker-diarization model (`onnx/model.onnx` at
/// `onnx-community/wespeaker-voxceleb-resnet34-LM`, 26 535 549 bytes).
/// Verified against the canonical HuggingFace copy on 2026-04-20; if the
/// upstream model is ever rotated, update this constant alongside the
/// SPEAKER_MODEL_FILE bump.
#[cfg(feature = "diarization")]
const SPEAKER_MODEL_SHA256: &str =
    "3955447b0499dc9e0a4541a895df08b03c69098eba4e56c02b5603e9f7f4fcbb";

fn home_dir() -> Option<std::path::PathBuf> {
    #[cfg(unix)]
    {
        std::env::var_os("HOME").map(std::path::PathBuf::from)
    }
    #[cfg(windows)]
    {
        std::env::var_os("USERPROFILE").map(std::path::PathBuf::from)
    }
}

/// Return the default model directory path (`~/.gigastt/models/`).
///
/// Falls back to `.gigastt/models` if the home directory cannot be determined.
pub fn default_model_dir() -> String {
    home_dir()
        .map(|h| {
            h.join(".gigastt")
                .join("models")
                .to_string_lossy()
                .into_owned()
        })
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
/// Downloads `wespeaker_resnet34.onnx` from `onnx-community/wespeaker-voxceleb-resnet34-LM`
/// into `<model_dir>/wespeaker_resnet34.onnx.partial`, verifies its SHA-256 against
/// `SPEAKER_MODEL_SHA256`, and atomically renames it into place. On checksum mismatch or
/// crash the final path is never observable, so a subsequent `ensure_speaker_model` call
/// will re-download from scratch rather than loading a tampered model (V1-02).
#[cfg(feature = "diarization")]
pub async fn ensure_speaker_model(model_dir: &str) -> Result<()> {
    let dir = Path::new(model_dir);
    let final_dest = dir.join(SPEAKER_MODEL_FILE);

    if final_dest.exists() {
        tracing::info!("Speaker model found at {}", final_dest.display());
        return Ok(());
    }

    tracing::info!("Speaker model not found, downloading from HuggingFace...");
    std::fs::create_dir_all(dir).context("Failed to create model directory")?;

    let partial = partial_path(&final_dest);
    if partial.exists() {
        let _ = tokio::fs::remove_file(&partial).await;
    }

    let hf_path = "onnx/model.onnx";
    let url = format!("https://huggingface.co/{SPEAKER_HF_REPO}/resolve/main/{hf_path}");

    let response = reqwest::get(&url).await.context("HTTP request failed")?;
    let status = response.status();
    if !status.is_success() {
        anyhow::bail!("Download failed for {SPEAKER_MODEL_FILE}: HTTP {status}");
    }
    let total_size = response.content_length().unwrap_or(0);

    let mut pb = DownloadProgress::new(total_size);

    let mut file = tokio::fs::File::create(&partial)
        .await
        .context("Failed to create partial speaker model file")?;
    let mut stream = response.bytes_stream();

    let mut downloaded: u64 = 0;
    while let Some(chunk) = stream.next().await {
        let chunk = chunk.context("Download stream error")?;
        file.write_all(&chunk)
            .await
            .context("Failed to write chunk")?;
        downloaded += chunk.len() as u64;
        pb.update(chunk.len() as u64);
    }

    file.flush().await?;
    drop(file);
    pb.finish();
    tracing::info!(
        "Wrote partial {} ({downloaded} bytes)",
        partial.display()
    );

    finalize_download(
        &partial,
        &final_dest,
        Some(SPEAKER_MODEL_SHA256),
        SPEAKER_MODEL_FILE,
    )?;
    tracing::info!("Saved {}", SPEAKER_MODEL_FILE);

    Ok(())
}

fn model_files_exist(dir: &Path) -> bool {
    MODEL_FILES.iter().all(|f| dir.join(f).exists())
}

/// Append `.partial` to a path; used as the staging location for downloads
/// so a crash between the write and the SHA verification never leaves a
/// half-written file under the final name that `model_files_exist` accepts.
fn partial_path(final_path: &Path) -> std::path::PathBuf {
    let mut s: std::ffi::OsString = final_path.as_os_str().to_owned();
    s.push(".partial");
    std::path::PathBuf::from(s)
}

/// Compute SHA-256 for a file synchronously, returning the lowercase hex digest.
fn sha256_file(path: &Path) -> Result<String> {
    let data = std::fs::read(path)
        .with_context(|| format!("Failed to read file for verification: {}", path.display()))?;
    let mut hasher = Sha256::new();
    hasher.update(&data);
    Ok(format!("{:x}", hasher.finalize()))
}

/// Verify a staged `.partial` file against `expected_sha256` (when provided)
/// and atomically rename it into `final_path`. On mismatch the partial is
/// removed so a corrupt artefact cannot be mistaken for a good download on
/// restart. On success the partial no longer exists and `final_path` is the
/// only visible artefact. Separated from the network path so the filesystem
/// contract can be unit-tested without a mock HTTP server.
fn finalize_download(
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

async fn download_file(filename: &str, dir: &Path) -> Result<()> {
    let url = format!("https://huggingface.co/{HF_REPO}/resolve/main/{filename}");
    let final_dest = dir.join(filename);
    let partial = partial_path(&final_dest);

    // If a stale partial from a previous crashed run exists, drop it before
    // writing the new one so we never concatenate chunks into an old prefix.
    if partial.exists() {
        let _ = tokio::fs::remove_file(&partial).await;
    }

    tracing::info!("Downloading {filename}...");

    let response = reqwest::get(&url).await.context("HTTP request failed")?;
    let status = response.status();
    if !status.is_success() {
        anyhow::bail!("Download failed for {filename}: HTTP {status}");
    }
    let total_size = response.content_length().unwrap_or(0);

    let mut progress = DownloadProgress::new(total_size);

    let mut file = tokio::fs::File::create(&partial)
        .await
        .context("Failed to create partial model file")?;
    let mut stream = response.bytes_stream();

    let mut downloaded: u64 = 0;
    while let Some(chunk) = stream.next().await {
        let chunk = chunk.context("Download stream error")?;
        file.write_all(&chunk)
            .await
            .context("Failed to write chunk")?;
        downloaded += chunk.len() as u64;
        progress.update(chunk.len() as u64);
    }

    file.flush().await?;
    drop(file);
    progress.finish();
    tracing::info!("Wrote partial {} ({downloaded} bytes)", partial.display());

    // Verify SHA-256 on the partial file (TOCTOU-safe: never rename an
    // unverified file into the final path). Missing checksum falls through
    // to a plain atomic rename.
    let expected = MODEL_CHECKSUMS
        .iter()
        .find(|(name, _)| *name == filename)
        .and_then(|(_, hash)| *hash);
    finalize_download(&partial, &final_dest, expected, filename)?;
    tracing::info!("Saved {filename}");

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn test_home_dir_returns_some() {
        // On any CI or developer machine HOME / USERPROFILE should be set.
        assert!(
            home_dir().is_some(),
            "home_dir() must return Some on this platform"
        );
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

    /// Compute the SHA-256 of a byte slice as a lowercase hex digest.
    fn sha256_hex(bytes: &[u8]) -> String {
        let mut hasher = Sha256::new();
        hasher.update(bytes);
        format!("{:x}", hasher.finalize())
    }

    /// V1-01: Helper to stage a `.partial` file with arbitrary bytes, mimicking
    /// the state of a fully streamed download prior to verification.
    fn stage_partial(final_path: &Path, bytes: &[u8]) -> std::path::PathBuf {
        let partial = partial_path(final_path);
        let mut f = std::fs::File::create(&partial).expect("create partial");
        f.write_all(bytes).expect("write partial");
        f.sync_all().expect("sync partial");
        partial
    }

    #[test]
    fn test_partial_path_appends_suffix() {
        let p = partial_path(Path::new("/tmp/gigastt/encoder.onnx"));
        assert_eq!(
            p,
            std::path::PathBuf::from("/tmp/gigastt/encoder.onnx.partial"),
        );
    }

    /// V1-01: on the success path, `.partial` disappears and the final path
    /// appears in a single atomic step.
    #[test]
    fn test_download_writes_partial_then_renames() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let final_path = tmp.path().join("encoder.onnx");
        let payload = b"fake encoder weights";
        let expected = sha256_hex(payload);

        let partial = stage_partial(&final_path, payload);
        assert!(partial.exists(), "precondition: partial is present");
        assert!(!final_path.exists(), "precondition: final is absent");

        finalize_download(&partial, &final_path, Some(&expected), "encoder.onnx")
            .expect("finalize should succeed");

        assert!(
            !partial.exists(),
            "partial must be gone after atomic rename"
        );
        assert!(
            final_path.exists(),
            "final path must exist after atomic rename"
        );
        assert_eq!(std::fs::read(&final_path).unwrap(), payload);
    }

    /// V1-01: if the process dies between the network write and the
    /// SHA verification / rename, `model_files_exist` must NOT see the
    /// file under its final name. We simulate the crash by staging a
    /// `.partial` and never calling `finalize_download`.
    #[test]
    fn test_download_crash_before_rename_leaves_no_final_file() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let final_path = tmp.path().join("encoder.onnx");
        let partial = stage_partial(&final_path, b"half-written junk");

        assert!(partial.exists(), "partial must exist to simulate crash");
        assert!(
            !final_path.exists(),
            "crash before rename must never leave the final artefact visible"
        );

        // All four MODEL_FILES stay missing from this tempdir, so
        // model_files_exist must refuse to short-circuit the download path.
        assert!(
            !model_files_exist(tmp.path()),
            "model_files_exist must not accept a staged partial"
        );
    }

    /// V1-01: SHA mismatch removes the partial and leaves the final path
    /// empty, so a retry starts from a clean slate.
    #[test]
    fn test_download_rejects_sha256_mismatch() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let final_path = tmp.path().join("decoder.onnx");
        let payload = b"real bytes";
        // Intentionally wrong expected hash (hash of different bytes).
        let wrong_expected = sha256_hex(b"different bytes");

        let partial = stage_partial(&final_path, payload);

        let err = finalize_download(&partial, &final_path, Some(&wrong_expected), "decoder.onnx")
            .expect_err("mismatch must error");
        let msg = format!("{err}");
        assert!(
            msg.contains("SHA-256 mismatch"),
            "unexpected error: {msg}"
        );

        assert!(
            !partial.exists(),
            "partial must be removed on SHA mismatch"
        );
        assert!(
            !final_path.exists(),
            "final must never appear on SHA mismatch"
        );
    }

    /// V1-01: success path with no checksum available still renames
    /// atomically (partial gone, final present, bytes preserved).
    #[test]
    fn test_download_atomic_on_success_without_checksum() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let final_path = tmp.path().join("vocab.txt");
        let payload = b"token0\ntoken1\n";

        let partial = stage_partial(&final_path, payload);

        finalize_download(&partial, &final_path, None, "vocab.txt")
            .expect("no-checksum finalize should succeed");

        assert!(!partial.exists(), "partial must be gone after rename");
        assert!(final_path.exists(), "final path must exist");
        assert_eq!(std::fs::read(&final_path).unwrap(), payload);
    }

    /// V1-01: sha256_file matches the in-memory hash of the same bytes.
    #[test]
    fn test_sha256_file_matches_in_memory_hash() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let p = tmp.path().join("blob");
        let payload = b"gigastt-model-bytes";
        std::fs::write(&p, payload).unwrap();

        let got = sha256_file(&p).expect("sha256_file");
        let want = sha256_hex(payload);
        assert_eq!(got, want);
    }

    /// V1-02: `SPEAKER_MODEL_SHA256` is a 64-char lowercase hex digest
    /// matching the SHA-256 of the upstream `onnx/model.onnx` blob
    /// (no accidental truncation / placeholder at compile time).
    #[cfg(feature = "diarization")]
    #[test]
    fn test_speaker_model_sha256_shape() {
        assert_eq!(
            SPEAKER_MODEL_SHA256.len(),
            64,
            "SPEAKER_MODEL_SHA256 must be a 64-char hex digest"
        );
        assert!(
            SPEAKER_MODEL_SHA256
                .chars()
                .all(|c| c.is_ascii_digit() || ('a'..='f').contains(&c)),
            "SPEAKER_MODEL_SHA256 must be lowercase hex; got: {SPEAKER_MODEL_SHA256}"
        );
    }

    /// V1-02: mismatching bytes against `SPEAKER_MODEL_SHA256` must delete
    /// the partial and refuse to promote it — exercises the full
    /// speaker-model finalize contract without touching the network.
    #[cfg(feature = "diarization")]
    #[test]
    fn test_speaker_model_rejects_sha256_mismatch() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let final_path = tmp.path().join(SPEAKER_MODEL_FILE);
        // Definitely not the real speaker-model bytes.
        let partial = stage_partial(&final_path, b"not the real wespeaker weights");

        let err = finalize_download(
            &partial,
            &final_path,
            Some(SPEAKER_MODEL_SHA256),
            SPEAKER_MODEL_FILE,
        )
        .expect_err("speaker mismatch must error");
        assert!(
            format!("{err}").contains("SHA-256 mismatch"),
            "unexpected error: {err}"
        );

        assert!(
            !partial.exists(),
            "partial speaker model must be removed on mismatch"
        );
        assert!(
            !final_path.exists(),
            "final speaker model must never appear on mismatch"
        );
    }

    /// V1-02: when the partial bytes DO hash to `SPEAKER_MODEL_SHA256`, the
    /// finalize path promotes them atomically. Network-free: we forge a
    /// "matching" partial by precomputing the hash of an arbitrary payload
    /// and passing it as the expected value.
    #[cfg(feature = "diarization")]
    #[test]
    fn test_speaker_model_partial_promoted_on_match() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let final_path = tmp.path().join(SPEAKER_MODEL_FILE);
        let payload = b"wespeaker-surrogate";
        let expected = sha256_hex(payload);

        let partial = stage_partial(&final_path, payload);

        finalize_download(&partial, &final_path, Some(&expected), SPEAKER_MODEL_FILE)
            .expect("matching partial must promote");

        assert!(!partial.exists());
        assert!(final_path.exists());
        assert_eq!(std::fs::read(&final_path).unwrap(), payload);
    }
}
