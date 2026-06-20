//! Model download and management.
//!
//! Downloads GigaAM v3 RNN-T ONNX files from HuggingFace to `~/.gigastt/models/`.
//! Two recognition heads are selectable via [`ModelVariant`]: the plain `rnnt`
//! head (default — lower WER, bare lowercase output) and the `e2e_rnnt` head
//! (punctuation / casing / ITN baked in).

use anyhow::{Context, Result};
use futures_util::StreamExt;
use sha2::{Digest, Sha256};
use std::path::Path;
use tokio::io::AsyncWriteExt;

#[cfg(unix)]
use std::os::fd::AsRawFd;

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

/// HuggingFace repo hosting the optional RUPunct punctuation model (MIT).
const PUNCT_HF_REPO: &str = "ekhodzitsky/rupunct-small-onnx";

/// Direct URL for the optional Silero v5 VAD model (MIT), pinned to a release
/// tag. SHA-256 below guards integrity regardless of the host.
const VAD_MODEL_URL: &str =
    "https://github.com/snakers4/silero-vad/raw/v5.1.2/src/silero_vad/data/silero_vad.onnx";

/// SHA-256 of the pinned Silero v5.1.2 `silero_vad.onnx` (verified 2026-06-19).
const VAD_MODEL_SHA256: &str = "2623a2953f6ff3d2c1e61740c6cdb7168133479b267dfef114a4a3cc5bdd788f";

/// The three files the punctuation pass needs, with their pinned SHA-256
/// checksums. Filenames mirror the `PUNCT_*` constants in [`crate::punctuation`].
/// Verified against the canonical HuggingFace copies on 2026-06-19.
const PUNCT_FILES: &[(&str, &str)] = &[
    (
        crate::punctuation::PUNCT_MODEL_FILE,
        "b105da023474d98aa13ba18953ae67b04b17bd0595034bc06030c17536893933",
    ),
    (
        crate::punctuation::PUNCT_TOKENIZER_FILE,
        "7ca617388c2092a3a84272025c52bbf3c6db0aee225c0351186295c0b5d3ddc6",
    ),
    (
        crate::punctuation::PUNCT_CONFIG_FILE,
        "6924a8cf41ec2bd3a3aa73a387ae0ccd0aed253ec7cac4d2f53c7d27440891eb",
    ),
];

/// Selectable GigaAM v3 recognition head.
///
/// Both heads ship in the same HuggingFace repo (`HF_REPO`) and share the
/// inference pipeline; they differ only in their ONNX files and vocabulary.
///
/// - [`ModelVariant::Rnnt`] (default): plain RNN-T head. Lower WER on the
///   golos_crowd_1k set (3.29% vs 9.65%) but emits bare lowercase Russian with
///   no punctuation / casing / ITN. Uses a 34-token character vocabulary.
/// - [`ModelVariant::E2eRnnt`]: end-to-end head with punctuation, casing, and
///   inverse text normalization baked in. Uses a 1025-token BPE vocabulary.
///
/// Real upstream filenames are kept on disk (no canonical-prefix rename), and
/// the engine auto-detects the variant from the encoder file present in the
/// model directory, so on-disk layout fully determines which head runs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ModelVariant {
    /// Plain RNN-T head (default). Bare lowercase output, lower WER.
    #[default]
    Rnnt,
    /// End-to-end RNN-T head with punctuation / casing / ITN.
    E2eRnnt,
}

impl ModelVariant {
    /// Basename of the FP32 encoder ONNX file for this variant.
    pub fn encoder_file(self) -> &'static str {
        match self {
            ModelVariant::Rnnt => "v3_rnnt_encoder.onnx",
            ModelVariant::E2eRnnt => "v3_e2e_rnnt_encoder.onnx",
        }
    }

    /// Basename of the locally-generated INT8 quantized encoder ONNX file.
    pub fn encoder_int8_file(self) -> &'static str {
        match self {
            ModelVariant::Rnnt => "v3_rnnt_encoder_int8.onnx",
            ModelVariant::E2eRnnt => "v3_e2e_rnnt_encoder_int8.onnx",
        }
    }

    /// Basename of the decoder ONNX file for this variant.
    pub fn decoder_file(self) -> &'static str {
        match self {
            ModelVariant::Rnnt => "v3_rnnt_decoder.onnx",
            ModelVariant::E2eRnnt => "v3_e2e_rnnt_decoder.onnx",
        }
    }

    /// Basename of the joiner ONNX file for this variant.
    pub fn joint_file(self) -> &'static str {
        match self {
            ModelVariant::Rnnt => "v3_rnnt_joint.onnx",
            ModelVariant::E2eRnnt => "v3_e2e_rnnt_joint.onnx",
        }
    }

    /// Basename of the vocabulary file for this variant.
    ///
    /// Note the asymmetry: the plain `rnnt` head's vocab is `v3_vocab.txt`
    /// (NOT `v3_rnnt_vocab.txt`), while `e2e_rnnt` uses `v3_e2e_rnnt_vocab.txt`.
    pub fn vocab_file(self) -> &'static str {
        match self {
            ModelVariant::Rnnt => "v3_vocab.txt",
            ModelVariant::E2eRnnt => "v3_e2e_rnnt_vocab.txt",
        }
    }

    /// Files downloaded from HuggingFace for this variant (encoder, decoder,
    /// joiner, vocab). The INT8 encoder is generated locally, not downloaded.
    pub fn download_files(self) -> [&'static str; 4] {
        [
            self.encoder_file(),
            self.decoder_file(),
            self.joint_file(),
            self.vocab_file(),
        ]
    }

    /// Pinned SHA-256 checksum for a downloaded file, or `None` when no checksum
    /// is pinned for it. Verified against the canonical HuggingFace copies.
    pub fn checksum(self, filename: &str) -> Option<&'static str> {
        let table = match self {
            ModelVariant::Rnnt => RNNT_CHECKSUMS,
            ModelVariant::E2eRnnt => E2E_RNNT_CHECKSUMS,
        };
        table
            .iter()
            .find(|(name, _)| *name == filename)
            .and_then(|(_, hash)| *hash)
    }

    /// Detect which variant's files are present in `dir` by probing for the
    /// encoder file (FP32 or generated INT8). Returns `None` when neither
    /// variant's encoder is present. `Rnnt` takes precedence when (anomalously)
    /// both encoders coexist, mirroring the engine's default.
    pub fn detect_in_dir(dir: &Path) -> Option<Self> {
        [ModelVariant::Rnnt, ModelVariant::E2eRnnt]
            .into_iter()
            .find(|&variant| {
                dir.join(variant.encoder_file()).exists()
                    || dir.join(variant.encoder_int8_file()).exists()
            })
    }
}

impl std::str::FromStr for ModelVariant {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.trim().to_ascii_lowercase().as_str() {
            "rnnt" => Ok(ModelVariant::Rnnt),
            "e2e_rnnt" | "e2e-rnnt" => Ok(ModelVariant::E2eRnnt),
            other => Err(format!(
                "unknown model variant '{other}' (expected 'rnnt' or 'e2e_rnnt')"
            )),
        }
    }
}

/// SHA-256 checksums for the plain `rnnt` head's downloaded files.
/// Computed from the canonical HuggingFace copies at `HF_REPO` on 2026-06-19.
const RNNT_CHECKSUMS: &[(&str, Option<&str>)] = &[
    (
        "v3_rnnt_encoder.onnx",
        Some("7ae7509c3f1128369564df0b00e2ee4950adf539de2392ac5c800a5bc04c7132"),
    ),
    (
        "v3_rnnt_decoder.onnx",
        Some("443c3b7bd42b453611618135d6b1e7d9467e5dd97c8a68501da4aa355750c0da"),
    ),
    (
        "v3_rnnt_joint.onnx",
        Some("fd1d02f45c2ad3d6b67cc149811ad794ab4b020ed49a0a9e2790a8619d1cddd8"),
    ),
    (
        "v3_vocab.txt",
        Some("a9143c30844d3c0bee3e9e927e4084774eb1b9eeaafc473b2c4521e4911a7c07"),
    ),
];

/// SHA-256 checksums for the `e2e_rnnt` head's downloaded files.
const E2E_RNNT_CHECKSUMS: &[(&str, Option<&str>)] = &[
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

/// Return the default punctuation-model directory (`~/.gigastt/models/punct/`),
/// a sibling of [`default_model_dir`].
///
/// Holds the optional RUPunct ONNX punctuation/casing restorer used to
/// post-process the plain `rnnt` head's bare lowercase output. The artifact
/// auto-downloads from `ekhodzitsky/rupunct-small-onnx` via
/// [`ensure_punct_model`] when the punct pass is enabled (see
/// [`crate::punctuation`]); a download failure simply disables the punct pass.
pub fn default_punct_model_dir() -> String {
    home_dir()
        .map(|h| {
            h.join(".gigastt")
                .join("models")
                .join("punct")
                .to_string_lossy()
                .into_owned()
        })
        .unwrap_or_else(|| ".gigastt/models/punct".into())
}

/// Return the default VAD-model directory (`~/.gigastt/models/vad/`), a sibling
/// of [`default_model_dir`].
///
/// Holds the optional Silero v5 ONNX voice-activity detector used for file
/// silence skipping and streaming endpointing. The artifact auto-downloads via
/// [`ensure_vad_model`] when VAD is enabled (see [`crate::vad`]); a download
/// failure simply disables VAD.
pub fn default_vad_model_dir() -> String {
    home_dir()
        .map(|h| {
            h.join(".gigastt")
                .join("models")
                .join("vad")
                .to_string_lossy()
                .into_owned()
        })
        .unwrap_or_else(|| ".gigastt/models/vad".into())
}

/// Acquire an advisory exclusive lock on a file inside `dir` so that only
/// one process downloads models at a time. The lock is released when the
/// returned file is dropped.
#[cfg(unix)]
fn acquire_download_lock(dir: &Path) -> Result<std::fs::File> {
    let lock_path = dir.join(".download.lock");
    let file = std::fs::OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(true)
        .open(&lock_path)
        .context("Failed to create download lock file")?;
    let fd = file.as_raw_fd();
    // SAFETY: `fd` is valid because it comes from `as_raw_fd()` on an owned
    // `File` that outlives this call. `flock` is an advisory lock; the file
    // remains owned by `file` and is closed (releasing the lock) when this
    // function's caller drops the returned `File`.
    let ret = unsafe { libc::flock(fd, libc::LOCK_EX) };
    if ret != 0 {
        anyhow::bail!("Failed to acquire download lock (another process is downloading)");
    }
    Ok(file)
}

/// Decision returned by [`resolve_variant`].
#[derive(Debug, PartialEq, Eq)]
pub enum VariantAction {
    /// Use the variant already present on disk — no download needed.
    Use(ModelVariant),
    /// Download (or re-download) the specified variant.
    Download(ModelVariant),
}

/// Pure decision function: given an optional user-requested variant and the
/// variant already fully present on disk, return what `ensure_model` should do.
///
/// Precedence rules:
/// - **Explicit request + matching install** → `Use` (no-op).
/// - **Explicit request + different/no install** → `Download` the requested variant.
/// - **No request + existing install** → `Use` that install (never clobber it).
/// - **No request + empty dir** → `Download` the default (`Rnnt`).
pub fn resolve_variant(
    requested: Option<ModelVariant>,
    existing: Option<ModelVariant>,
) -> VariantAction {
    match (requested, existing) {
        (Some(req), Some(ex)) if req == ex => VariantAction::Use(req),
        (Some(req), _) => VariantAction::Download(req),
        (None, Some(ex)) => VariantAction::Use(ex),
        (None, None) => VariantAction::Download(ModelVariant::default()),
    }
}

/// Ensure a model is present in `model_dir`, auto-detecting the installed
/// variant and downloading the default (`Rnnt`) only when the directory holds
/// no usable model. Equivalent to `ensure_model_variant(None, model_dir)` with
/// the resolved variant discarded. Preserves the pre-variant public signature.
pub async fn ensure_model(model_dir: &str) -> Result<()> {
    ensure_model_variant(None, model_dir).await?;
    Ok(())
}

/// Ensure an appropriate model variant's files exist in `model_dir`,
/// downloading from HuggingFace if missing.
///
/// When `requested` is `Some(v)`, the function enforces that variant `v` is
/// present, downloading it if it isn't (or if the dir holds a different variant).
///
/// When `requested` is `None`, the function respects whatever is already
/// installed: if any variant's complete file set is in `model_dir`, it is used
/// as-is and **no network request is made**. Only when the directory is empty
/// (no usable model found) does it fall back to downloading the default
/// (`Rnnt`).
///
/// Returns the variant that is now ready in `model_dir`.
pub async fn ensure_model_variant(
    requested: Option<ModelVariant>,
    model_dir: &str,
) -> Result<ModelVariant> {
    let dir = Path::new(model_dir);

    // Determine the variant that is fully present on disk (all download files
    // exist). `detect_in_dir` only checks for the encoder, so we filter to
    // variants whose complete download set is present.
    let existing = ModelVariant::detect_in_dir(dir).filter(|&v| is_model_present(v, dir));

    let variant = match resolve_variant(requested, existing) {
        VariantAction::Use(v) => {
            tracing::info!("Using existing {v:?} model at {model_dir}");
            return Ok(v);
        }
        VariantAction::Download(v) => v,
    };

    if let Some(other) = existing
        && other != variant
    {
        tracing::warn!(
            "Model directory {model_dir} holds {other:?} files but {variant:?} was \
             requested; downloading the {variant:?} set (variants are never mixed)"
        );
    }

    // Create the directory before acquiring the lock so the lock file can be
    // created inside it.
    std::fs::create_dir_all(dir).context("Failed to create model directory")?;

    #[cfg(unix)]
    let _lock = acquire_download_lock(dir)?;

    // Double-check after acquiring the lock in case another process finished
    // the download while we were waiting.
    if is_model_present(variant, dir) {
        tracing::info!("Model ({variant:?}) found at {model_dir} after lock acquisition");
        return Ok(variant);
    }

    tracing::info!("Model ({variant:?}) not found, downloading from HuggingFace...");

    for file in variant.download_files() {
        download_file(variant, file, dir).await?;
    }

    tracing::info!("Model download complete");
    Ok(variant)
}

/// Ensure the speaker diarization model exists in `model_dir`, downloading from HuggingFace if missing.
///
/// Downloads `wespeaker_resnet34.onnx` from `onnx-community/wespeaker-voxceleb-resnet34-LM`
/// into `<model_dir>/wespeaker_resnet34.onnx.partial`, verifies its SHA-256 against
/// `SPEAKER_MODEL_SHA256`, and atomically renames it into place. On checksum mismatch or
/// crash the final path is never observable, so a subsequent `ensure_speaker_model` call
/// will re-download from scratch rather than loading a tampered model.
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

    let url = format!("https://huggingface.co/{SPEAKER_HF_REPO}/resolve/main/onnx/model.onnx");
    stream_to_partial_then_finalize(
        &url,
        &final_dest,
        Some(SPEAKER_MODEL_SHA256),
        SPEAKER_MODEL_FILE,
    )
    .await
}

/// Ensure the optional punctuation model exists in `punct_model_dir`,
/// downloading any missing files from the `ekhodzitsky/rupunct-small-onnx`
/// HuggingFace repo (public, MIT).
///
/// Downloads the three files the punctuation pass needs
/// (`rupunct_small_int8.onnx`, `tokenizer.json`, `config.json`) — only those
/// not already present — using the same streaming-download + atomic-rename +
/// SHA-256 infra as the main model download. Files already on disk are left
/// untouched, so a second call is a no-op (no re-download).
///
/// The pass is strictly optional: callers treat a download error as
/// "punctuation unavailable" and proceed with bare text.
pub async fn ensure_punct_model(punct_model_dir: &str) -> Result<()> {
    let dir = Path::new(punct_model_dir);

    if PUNCT_FILES.iter().all(|(file, _)| dir.join(file).exists()) {
        tracing::info!("Punctuation model found at {punct_model_dir}");
        return Ok(());
    }

    tracing::info!("Punctuation model not found, downloading from HuggingFace...");
    std::fs::create_dir_all(dir).context("Failed to create punctuation model directory")?;

    #[cfg(unix)]
    let _lock = acquire_download_lock(dir)?;

    for (file, sha256) in PUNCT_FILES {
        let final_dest = dir.join(file);
        if final_dest.exists() {
            continue;
        }
        let url = format!("https://huggingface.co/{PUNCT_HF_REPO}/resolve/main/{file}");
        stream_to_partial_then_finalize(&url, &final_dest, Some(sha256), file).await?;
    }

    tracing::info!("Punctuation model download complete");
    Ok(())
}

/// Ensure the optional Silero VAD model exists in `vad_model_dir`, downloading
/// it from the pinned Silero release (MIT) if missing.
///
/// Uses the same streaming-download + atomic-rename + SHA-256 infra as the main
/// model download. A file already on disk is left untouched (no re-download).
///
/// VAD is strictly optional: callers treat a download error as "VAD
/// unavailable" and proceed without silence skipping / VAD endpointing.
pub async fn ensure_vad_model(vad_model_dir: &str) -> Result<()> {
    let dir = Path::new(vad_model_dir);
    let final_dest = dir.join(crate::vad::VAD_MODEL_FILE);

    if final_dest.exists() {
        tracing::info!("VAD model found at {}", final_dest.display());
        return Ok(());
    }

    tracing::info!("VAD model not found, downloading from {VAD_MODEL_URL}...");
    std::fs::create_dir_all(dir).context("Failed to create VAD model directory")?;

    #[cfg(unix)]
    let _lock = acquire_download_lock(dir)?;

    // Another process may have finished while we waited for the lock.
    if final_dest.exists() {
        return Ok(());
    }

    stream_to_partial_then_finalize(
        VAD_MODEL_URL,
        &final_dest,
        Some(VAD_MODEL_SHA256),
        crate::vad::VAD_MODEL_FILE,
    )
    .await?;

    tracing::info!("VAD model download complete");
    Ok(())
}

/// True when every downloaded file for `variant` is present in `dir`.
///
/// Checks the *downloaded* set (FP32 encoder, decoder, joiner, vocab); the
/// locally-generated INT8 encoder is not required for presence.
pub fn is_model_present(variant: ModelVariant, dir: &Path) -> bool {
    variant
        .download_files()
        .iter()
        .all(|f| dir.join(f).exists())
}

/// Append `.partial` to a path; retained for tests that assert the legacy
/// staging name. Production download path uses `partial_path_unique`.
#[cfg(test)]
fn partial_path(final_path: &Path) -> std::path::PathBuf {
    let mut s: std::ffi::OsString = final_path.as_os_str().to_owned();
    s.push(".partial");
    std::path::PathBuf::from(s)
}

/// Generate a unique `.partial` path so concurrent processes never write
/// to the same staging file. Uses PID and nanosecond timestamp.
fn partial_path_unique(final_path: &Path) -> std::path::PathBuf {
    let stamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    let mut s: std::ffi::OsString = final_path.as_os_str().to_owned();
    s.push(format!(".partial.{}.{}", std::process::id(), stamp));
    std::path::PathBuf::from(s)
}

/// Compute SHA-256 for a file synchronously, returning the lowercase hex digest.
fn sha256_file(path: &Path) -> Result<String> {
    let data = std::fs::read(path)
        .with_context(|| format!("Failed to read file for verification: {}", path.display()))?;
    let mut hasher = Sha256::new();
    hasher.update(&data);
    Ok(hex::encode(hasher.finalize()))
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

async fn download_file(variant: ModelVariant, filename: &str, dir: &Path) -> Result<()> {
    let url = format!("https://huggingface.co/{HF_REPO}/resolve/main/{filename}");
    let final_dest = dir.join(filename);
    let expected = variant.checksum(filename);
    stream_to_partial_then_finalize(&url, &final_dest, expected, filename).await
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
async fn stream_to_partial_then_finalize(
    url: &str,
    final_dest: &Path,
    expected_sha256: Option<&str>,
    label: &str,
) -> Result<()> {
    let partial = partial_path_unique(final_dest);

    tracing::info!("Downloading {label}...");

    // Configured client: bound the connect/TLS handshake and per-read stalls, and
    // cap redirects. NOT a whole-request timeout (a legitimate ~850 MB download can
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

    finalize_download(&partial, final_dest, expected_sha256, label)?;
    tracing::info!("Saved {label}");

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
        hex::encode(hasher.finalize())
    }

    /// Helper to stage a `.partial` file with arbitrary bytes, mimicking
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

    /// On the success path, `.partial` disappears and the final path
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

    /// If the process dies between the network write and the
    /// SHA verification / rename, `is_model_present` must NOT see the
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

        // No variant's files exist in this tempdir, so is_model_present must
        // refuse to short-circuit the download path.
        assert!(
            !is_model_present(ModelVariant::Rnnt, tmp.path()),
            "is_model_present must not accept a staged partial"
        );
        assert!(
            !is_model_present(ModelVariant::E2eRnnt, tmp.path()),
            "is_model_present must not accept a staged partial"
        );
    }

    /// SHA mismatch removes the partial and leaves the final path
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
        assert!(msg.contains("SHA-256 mismatch"), "unexpected error: {msg}");

        assert!(!partial.exists(), "partial must be removed on SHA mismatch");
        assert!(
            !final_path.exists(),
            "final must never appear on SHA mismatch"
        );
    }

    /// Success path with no checksum available still renames
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

    /// sha256_file matches the in-memory hash of the same bytes.
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

    /// `SPEAKER_MODEL_SHA256` is a 64-char lowercase hex digest
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

    /// Mismatching bytes against `SPEAKER_MODEL_SHA256` must delete
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

    /// When the partial bytes DO hash to `SPEAKER_MODEL_SHA256`, the
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

    #[test]
    fn test_partial_path_unique_contains_pid_and_timestamp() {
        let p = partial_path_unique(Path::new("/tmp/final.onnx"));
        let s = p.to_string_lossy();
        assert!(s.contains(".partial."));
        assert!(s.contains(&std::process::id().to_string()));
    }

    #[cfg(unix)]
    #[test]
    fn test_acquire_download_lock_creates_lock_file() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let lock = acquire_download_lock(tmp.path()).expect("acquire lock");
        assert!(tmp.path().join(".download.lock").exists());
        drop(lock);
    }

    #[tokio::test]
    async fn test_stream_to_partial_then_finalize_success() {
        let server = wiremock::MockServer::start().await;
        let payload = b"fake model bytes";
        wiremock::Mock::given(wiremock::matchers::method("GET"))
            .and(wiremock::matchers::path("/model.onnx"))
            .respond_with(
                wiremock::ResponseTemplate::new(200)
                    .set_body_bytes(payload.as_slice())
                    .insert_header("content-length", payload.len().to_string()),
            )
            .mount(&server)
            .await;

        let tmp = tempfile::tempdir().expect("tempdir");
        let final_path = tmp.path().join("model.onnx");
        let url = format!("{}/model.onnx", server.uri());

        stream_to_partial_then_finalize(&url, &final_path, None, "model.onnx")
            .await
            .expect("download should succeed");

        assert!(final_path.exists());
        assert_eq!(std::fs::read(&final_path).unwrap(), payload);
    }

    #[tokio::test]
    async fn test_stream_to_partial_then_finalize_http_error() {
        let server = wiremock::MockServer::start().await;
        wiremock::Mock::given(wiremock::matchers::method("GET"))
            .and(wiremock::matchers::path("/missing.onnx"))
            .respond_with(wiremock::ResponseTemplate::new(404))
            .mount(&server)
            .await;

        let tmp = tempfile::tempdir().expect("tempdir");
        let final_path = tmp.path().join("missing.onnx");
        let url = format!("{}/missing.onnx", server.uri());

        let err = stream_to_partial_then_finalize(&url, &final_path, None, "missing.onnx")
            .await
            .expect_err("404 should fail");
        let msg = format!("{err}");
        assert!(msg.contains("404"), "error should mention 404: {msg}");
    }

    #[tokio::test]
    async fn test_stream_to_partial_then_finalize_checksum_mismatch() {
        let server = wiremock::MockServer::start().await;
        let payload = b"wrong bytes";
        wiremock::Mock::given(wiremock::matchers::method("GET"))
            .and(wiremock::matchers::path("/model.onnx"))
            .respond_with(wiremock::ResponseTemplate::new(200).set_body_bytes(payload.as_slice()))
            .mount(&server)
            .await;

        let tmp = tempfile::tempdir().expect("tempdir");
        let final_path = tmp.path().join("model.onnx");
        let url = format!("{}/model.onnx", server.uri());
        let wrong_hash = sha256_hex(b"different bytes");

        let err =
            stream_to_partial_then_finalize(&url, &final_path, Some(&wrong_hash), "model.onnx")
                .await
                .expect_err("checksum mismatch should fail");
        let msg = format!("{err}");
        assert!(
            msg.contains("SHA-256 mismatch"),
            "error should mention mismatch: {msg}"
        );
    }

    #[test]
    fn test_punct_files_checksums_are_pinned() {
        // Three files, each with a 64-char lowercase hex digest — no truncation
        // or placeholder slipping into a release.
        assert_eq!(PUNCT_FILES.len(), 3);
        for (file, sum) in PUNCT_FILES {
            assert_eq!(
                sum.len(),
                64,
                "{file} punct checksum must be 64 hex chars, got: {sum}"
            );
            assert!(
                sum.chars()
                    .all(|c| c.is_ascii_digit() || ('a'..='f').contains(&c)),
                "{file} punct checksum must be lowercase hex, got: {sum}"
            );
        }
    }

    /// `ensure_punct_model` short-circuits (no network, no `.partial`) when all
    /// three files are already present.
    #[tokio::test]
    async fn test_ensure_punct_model_present_no_download() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let dir = tmp.path();
        for (file, _) in PUNCT_FILES {
            std::fs::write(dir.join(file), b"stub").unwrap();
        }

        ensure_punct_model(dir.to_str().unwrap())
            .await
            .expect("present model must short-circuit");

        let partials: Vec<_> = std::fs::read_dir(dir)
            .unwrap()
            .filter_map(|e| e.ok())
            .filter(|e| e.file_name().to_string_lossy().contains(".partial"))
            .collect();
        assert!(partials.is_empty(), "no .partial files: {partials:?}");
        for (file, _) in PUNCT_FILES {
            assert_eq!(std::fs::read(dir.join(file)).unwrap(), b"stub");
        }
    }

    #[test]
    fn test_model_variant_default_is_rnnt() {
        assert_eq!(ModelVariant::default(), ModelVariant::Rnnt);
    }

    #[test]
    fn test_model_variant_rnnt_file_mapping() {
        let v = ModelVariant::Rnnt;
        assert_eq!(v.encoder_file(), "v3_rnnt_encoder.onnx");
        assert_eq!(v.encoder_int8_file(), "v3_rnnt_encoder_int8.onnx");
        assert_eq!(v.decoder_file(), "v3_rnnt_decoder.onnx");
        assert_eq!(v.joint_file(), "v3_rnnt_joint.onnx");
        // The rnnt vocab name is asymmetric: v3_vocab.txt, NOT v3_rnnt_vocab.txt.
        assert_eq!(v.vocab_file(), "v3_vocab.txt");
        assert_eq!(
            v.download_files(),
            [
                "v3_rnnt_encoder.onnx",
                "v3_rnnt_decoder.onnx",
                "v3_rnnt_joint.onnx",
                "v3_vocab.txt",
            ]
        );
    }

    #[test]
    fn test_model_variant_e2e_rnnt_file_mapping() {
        let v = ModelVariant::E2eRnnt;
        assert_eq!(v.encoder_file(), "v3_e2e_rnnt_encoder.onnx");
        assert_eq!(v.encoder_int8_file(), "v3_e2e_rnnt_encoder_int8.onnx");
        assert_eq!(v.decoder_file(), "v3_e2e_rnnt_decoder.onnx");
        assert_eq!(v.joint_file(), "v3_e2e_rnnt_joint.onnx");
        assert_eq!(v.vocab_file(), "v3_e2e_rnnt_vocab.txt");
        assert_eq!(
            v.download_files(),
            [
                "v3_e2e_rnnt_encoder.onnx",
                "v3_e2e_rnnt_decoder.onnx",
                "v3_e2e_rnnt_joint.onnx",
                "v3_e2e_rnnt_vocab.txt",
            ]
        );
    }

    #[test]
    fn test_model_variant_from_str() {
        use std::str::FromStr;
        assert_eq!(ModelVariant::from_str("rnnt").unwrap(), ModelVariant::Rnnt);
        assert_eq!(
            ModelVariant::from_str("e2e_rnnt").unwrap(),
            ModelVariant::E2eRnnt
        );
        assert_eq!(
            ModelVariant::from_str("E2E-RNNT").unwrap(),
            ModelVariant::E2eRnnt
        );
        assert_eq!(
            ModelVariant::from_str(" RNNT ").unwrap(),
            ModelVariant::Rnnt
        );
        assert!(ModelVariant::from_str("whisper").is_err());
    }

    #[test]
    fn test_model_variant_checksums_are_pinned() {
        // Every downloaded file for both variants has a pinned 64-char hex
        // checksum — security parity, no placeholder slipping into a release.
        for variant in [ModelVariant::Rnnt, ModelVariant::E2eRnnt] {
            for file in variant.download_files() {
                let sum = variant
                    .checksum(file)
                    .unwrap_or_else(|| panic!("{variant:?} {file} must have a pinned checksum"));
                assert_eq!(
                    sum.len(),
                    64,
                    "{variant:?} {file} checksum must be 64 hex chars, got: {sum}"
                );
                assert!(
                    sum.chars()
                        .all(|c| c.is_ascii_digit() || ('a'..='f').contains(&c)),
                    "{variant:?} {file} checksum must be lowercase hex, got: {sum}"
                );
            }
        }
    }

    #[test]
    fn test_detect_in_dir_rnnt_by_fp32_encoder() {
        let tmp = tempfile::tempdir().expect("tempdir");
        std::fs::write(tmp.path().join("v3_rnnt_encoder.onnx"), b"fp32").unwrap();
        assert_eq!(
            ModelVariant::detect_in_dir(tmp.path()),
            Some(ModelVariant::Rnnt)
        );
    }

    #[test]
    fn test_detect_in_dir_rnnt_by_int8_encoder() {
        let tmp = tempfile::tempdir().expect("tempdir");
        std::fs::write(tmp.path().join("v3_rnnt_encoder_int8.onnx"), b"int8").unwrap();
        assert_eq!(
            ModelVariant::detect_in_dir(tmp.path()),
            Some(ModelVariant::Rnnt)
        );
    }

    #[test]
    fn test_detect_in_dir_e2e_by_fp32_encoder() {
        let tmp = tempfile::tempdir().expect("tempdir");
        std::fs::write(tmp.path().join("v3_e2e_rnnt_encoder.onnx"), b"fp32").unwrap();
        assert_eq!(
            ModelVariant::detect_in_dir(tmp.path()),
            Some(ModelVariant::E2eRnnt)
        );
    }

    #[test]
    fn test_detect_in_dir_e2e_by_int8_encoder() {
        let tmp = tempfile::tempdir().expect("tempdir");
        std::fs::write(tmp.path().join("v3_e2e_rnnt_encoder_int8.onnx"), b"int8").unwrap();
        assert_eq!(
            ModelVariant::detect_in_dir(tmp.path()),
            Some(ModelVariant::E2eRnnt)
        );
    }

    #[test]
    fn test_detect_in_dir_none_when_empty() {
        let tmp = tempfile::tempdir().expect("tempdir");
        assert_eq!(ModelVariant::detect_in_dir(tmp.path()), None);
    }

    #[test]
    fn test_is_model_present_per_variant() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let dir = tmp.path();
        // Stage a full rnnt download set.
        for f in ModelVariant::Rnnt.download_files() {
            std::fs::write(dir.join(f), b"x").unwrap();
        }
        assert!(
            is_model_present(ModelVariant::Rnnt, dir),
            "rnnt set is complete"
        );
        assert!(
            !is_model_present(ModelVariant::E2eRnnt, dir),
            "e2e set is absent — must not be reported present"
        );
    }

    #[test]
    fn test_is_model_present_false_when_one_file_missing() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let dir = tmp.path();
        // Stage all but the vocab.
        for f in [
            ModelVariant::Rnnt.encoder_file(),
            ModelVariant::Rnnt.decoder_file(),
            ModelVariant::Rnnt.joint_file(),
        ] {
            std::fs::write(dir.join(f), b"x").unwrap();
        }
        assert!(
            !is_model_present(ModelVariant::Rnnt, dir),
            "a missing vocab must make the set incomplete"
        );
    }

    // ── resolve_variant decision table ──────────────────────────────────────

    #[test]
    fn test_resolve_variant_none_empty_dir_downloads_default() {
        // None requested + no existing → download Rnnt (the default)
        assert_eq!(
            resolve_variant(None, None),
            VariantAction::Download(ModelVariant::Rnnt),
        );
    }

    #[test]
    fn test_resolve_variant_none_e2e_present_uses_e2e() {
        // None requested + E2eRnnt already installed → use it, no download
        assert_eq!(
            resolve_variant(None, Some(ModelVariant::E2eRnnt)),
            VariantAction::Use(ModelVariant::E2eRnnt),
        );
    }

    #[test]
    fn test_resolve_variant_none_rnnt_present_uses_rnnt() {
        // None requested + Rnnt already installed → use it, no download
        assert_eq!(
            resolve_variant(None, Some(ModelVariant::Rnnt)),
            VariantAction::Use(ModelVariant::Rnnt),
        );
    }

    #[test]
    fn test_resolve_variant_some_rnnt_rnnt_present_uses_rnnt() {
        // Explicit Rnnt + Rnnt installed → no download needed
        assert_eq!(
            resolve_variant(Some(ModelVariant::Rnnt), Some(ModelVariant::Rnnt)),
            VariantAction::Use(ModelVariant::Rnnt),
        );
    }

    #[test]
    fn test_resolve_variant_some_e2e_rnnt_present_downloads_e2e() {
        // Explicit E2eRnnt + Rnnt installed → must switch, so download E2eRnnt
        assert_eq!(
            resolve_variant(Some(ModelVariant::E2eRnnt), Some(ModelVariant::Rnnt)),
            VariantAction::Download(ModelVariant::E2eRnnt),
        );
    }

    #[test]
    fn test_resolve_variant_some_e2e_empty_downloads_e2e() {
        // Explicit E2eRnnt + nothing installed → download E2eRnnt
        assert_eq!(
            resolve_variant(Some(ModelVariant::E2eRnnt), None),
            VariantAction::Download(ModelVariant::E2eRnnt),
        );
    }

    #[test]
    fn test_resolve_variant_some_rnnt_e2e_present_downloads_rnnt() {
        // Explicit Rnnt + E2eRnnt installed → must switch, download Rnnt
        assert_eq!(
            resolve_variant(Some(ModelVariant::Rnnt), Some(ModelVariant::E2eRnnt)),
            VariantAction::Download(ModelVariant::Rnnt),
        );
    }

    /// Verify that `ensure_model(None, dir)` with a complete E2eRnnt install
    /// does NOT create any `.partial` files (no download triggered).
    #[tokio::test]
    async fn test_ensure_model_none_respects_existing_e2e_install() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let dir = tmp.path();

        // Stage a full E2eRnnt download set (stub bytes, no real ONNX needed).
        for f in ModelVariant::E2eRnnt.download_files() {
            std::fs::write(dir.join(f), b"stub").unwrap();
        }

        // ensure_model_variant with None must return E2eRnnt without downloading.
        let variant = ensure_model_variant(None, dir.to_str().unwrap())
            .await
            .expect("ensure_model_variant should succeed");

        assert_eq!(
            variant,
            ModelVariant::E2eRnnt,
            "must use the installed E2eRnnt"
        );

        // No .partial files must have been created.
        let partials: Vec<_> = std::fs::read_dir(dir)
            .unwrap()
            .filter_map(|e| e.ok())
            .filter(|e| e.file_name().to_string_lossy().contains(".partial"))
            .collect();
        assert!(
            partials.is_empty(),
            "no .partial files must exist: {partials:?}"
        );

        // Files must be untouched (still stub bytes).
        for f in ModelVariant::E2eRnnt.download_files() {
            assert_eq!(
                std::fs::read(dir.join(f)).unwrap(),
                b"stub",
                "{f} must be unchanged"
            );
        }
    }

    /// The legacy public `ensure_model(dir)` wrapper delegates to
    /// `ensure_model_variant(None, dir)`: with a complete install already on
    /// disk it must succeed without touching the network (no `.partial` files).
    #[tokio::test]
    async fn test_ensure_model_wrapper_uses_existing_install() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let dir = tmp.path();

        // Stage a complete default (Rnnt) download set.
        for f in ModelVariant::Rnnt.download_files() {
            std::fs::write(dir.join(f), b"stub").unwrap();
        }

        ensure_model(dir.to_str().unwrap())
            .await
            .expect("ensure_model must succeed against an existing install");

        let partials: Vec<_> = std::fs::read_dir(dir)
            .unwrap()
            .filter_map(|e| e.ok())
            .filter(|e| e.file_name().to_string_lossy().contains(".partial"))
            .collect();
        assert!(
            partials.is_empty(),
            "ensure_model must not download when the set is present: {partials:?}"
        );
    }

    /// `ensure_model_variant(Some(Rnnt), dir)` against a matching install is the
    /// `VariantAction::Use` branch: returns Rnnt with no download.
    #[tokio::test]
    async fn test_ensure_model_variant_explicit_match_uses_existing() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let dir = tmp.path();

        for f in ModelVariant::Rnnt.download_files() {
            std::fs::write(dir.join(f), b"stub").unwrap();
        }

        let variant = ensure_model_variant(Some(ModelVariant::Rnnt), dir.to_str().unwrap())
            .await
            .expect("explicit matching variant must short-circuit");
        assert_eq!(variant, ModelVariant::Rnnt);

        let has_partial = std::fs::read_dir(dir)
            .unwrap()
            .filter_map(|e| e.ok())
            .any(|e| e.file_name().to_string_lossy().contains(".partial"));
        assert!(!has_partial, "no download for an explicit matching variant");
    }

    /// `ensure_vad_model` short-circuits (no network, no `.partial`) when the
    /// Silero ONNX file is already present in the VAD directory.
    #[tokio::test]
    async fn test_ensure_vad_model_present_no_download() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let dir = tmp.path();
        std::fs::write(dir.join(crate::vad::VAD_MODEL_FILE), b"stub vad").unwrap();

        ensure_vad_model(dir.to_str().unwrap())
            .await
            .expect("present VAD model must short-circuit");

        let partials: Vec<_> = std::fs::read_dir(dir)
            .unwrap()
            .filter_map(|e| e.ok())
            .filter(|e| e.file_name().to_string_lossy().contains(".partial"))
            .collect();
        assert!(partials.is_empty(), "no .partial files: {partials:?}");
        assert_eq!(
            std::fs::read(dir.join(crate::vad::VAD_MODEL_FILE)).unwrap(),
            b"stub vad",
            "existing VAD model must be left untouched"
        );
    }

    /// `default_punct_model_dir` and `default_vad_model_dir` are siblings of the
    /// main model dir under `.gigastt/models`, with the expected leaf names.
    #[test]
    fn test_default_punct_and_vad_dirs_are_model_siblings() {
        let punct = default_punct_model_dir();
        let vad = default_vad_model_dir();
        assert!(
            punct.contains(".gigastt") && punct.ends_with("punct"),
            "punct dir should be under .gigastt and end with 'punct', got: {punct}"
        );
        assert!(
            vad.contains(".gigastt") && vad.ends_with("vad"),
            "vad dir should be under .gigastt and end with 'vad', got: {vad}"
        );
    }

    /// `VAD_MODEL_SHA256` is a 64-char lowercase hex digest (no truncation or
    /// placeholder slipping into a release).
    #[test]
    fn test_vad_model_sha256_shape() {
        assert_eq!(
            VAD_MODEL_SHA256.len(),
            64,
            "VAD_MODEL_SHA256 must be a 64-char hex digest"
        );
        assert!(
            VAD_MODEL_SHA256
                .chars()
                .all(|c| c.is_ascii_digit() || ('a'..='f').contains(&c)),
            "VAD_MODEL_SHA256 must be lowercase hex; got: {VAD_MODEL_SHA256}"
        );
    }

    /// `ensure_model_variant` tolerates a deep, freshly-created model directory:
    /// a complete Rnnt set pre-staged under a nested path is detected and used
    /// as-is (early `Use(...)` return) without any network access.
    #[tokio::test]
    async fn test_ensure_model_variant_uses_complete_set_in_nested_path() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let nested = tmp.path().join("a").join("b").join("models");
        std::fs::create_dir_all(&nested).unwrap();
        for f in ModelVariant::Rnnt.download_files() {
            std::fs::write(nested.join(f), b"stub").unwrap();
        }

        let variant = ensure_model_variant(None, nested.to_str().unwrap())
            .await
            .expect("nested complete install must be used as-is");
        assert_eq!(variant, ModelVariant::Rnnt);
    }
}
