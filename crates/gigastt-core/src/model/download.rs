//! Model presence, ensure/download, and the streaming SHA-256 fetch.

use crate::error::GigasttError;
use crate::sha256::{Sha256, hex_lower};
#[cfg(feature = "net")]
use anyhow::Context;
use anyhow::Result;
use std::path::Path;

#[cfg(unix)]
#[cfg(feature = "net")]
use std::os::fd::AsRawFd;

use super::variant::ModelVariant;
#[cfg(feature = "net")]
use super::variant::PREQUANT_RELEASE_BASE;

/// Stream a file and return its lowercase SHA-256 hex digest.
///
/// Used at engine load (not just download) so a tampered file in
/// `~/.gigastt/models/` cannot be silently mapped.
pub(crate) fn hash_file_sha256(path: &Path) -> std::io::Result<String> {
    use std::io::Read;
    let mut file = std::fs::File::open(path)?;
    let mut hasher = Sha256::new();
    let mut buf = [0u8; 64 * 1024];
    loop {
        let n = file.read(&mut buf)?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    Ok(hex_lower(&hasher.finalize()))
}

/// Refuse `path` when its digest is not `expected`.
pub(crate) fn verify_pinned_checksum(path: &Path, expected: &str) -> Result<(), GigasttError> {
    let actual = hash_file_sha256(path).map_err(|e| GigasttError::ModelLoad {
        path: path.display().to_string(),
        source: Some(e.into()),
    })?;
    if actual != expected {
        return Err(GigasttError::ModelLoad {
            path: path.display().to_string(),
            source: Some(format!("SHA-256 mismatch: expected {expected}, got {actual}").into()),
        });
    }
    Ok(())
}

pub(super) fn home_dir() -> Option<std::path::PathBuf> {
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
#[cfg(feature = "net")]
pub(super) fn acquire_download_lock(dir: &Path) -> Result<std::fs::File> {
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
#[cfg(feature = "net")]
pub async fn ensure_model(model_dir: &str) -> Result<()> {
    ensure_model_variant(None, model_dir).await?;
    Ok(())
}

/// Ensure an appropriate model variant's files exist in `model_dir`,
/// downloading if missing (GitHub Releases for default `rnnt` / `e2e_rnnt`
/// INT8; HuggingFace for CTC heads).
///
/// When `requested` is `Some(v)`, the function enforces that variant `v` is
/// present, downloading it if it isn't (or if the dir holds a different variant).
///
/// When `requested` is `None`, the function respects whatever is already
/// installed: if any variant's complete **prequantized INT8 set** is in
/// `model_dir`, it is used as-is and **no network request is made**. Only when
/// the directory holds no usable INT8 model does it fall back to downloading
/// the default (`Rnnt`) **pre-quantized INT8** set. FP32-only installs are
/// ignored (not considered usable).
///
/// A present `manifest.toml` selects a local custom INT8 pack. Its declared
/// files must exist and its architecture must match an explicit request.
/// Invalid or incomplete custom packs are errors, never download fallbacks.
/// File integrity is still verified by the engine for pinned model basenames.
///
/// Returns the variant that is now ready in `model_dir`.
#[cfg(feature = "net")]
pub async fn ensure_model_variant(
    requested: Option<ModelVariant>,
    model_dir: &str,
) -> Result<ModelVariant> {
    let dir = Path::new(model_dir);

    if let Some(manifest) = super::manifest::ModelManifest::load(dir)? {
        anyhow::ensure!(
            requested.is_none_or(|variant| variant == manifest.architecture),
            "requested model variant conflicts with manifest architecture '{}'",
            manifest.architecture.as_str()
        );
        anyhow::ensure!(
            manifest.prefers_int8(dir),
            "custom model pack requires an existing encoder_int8 file"
        );
        let files = [
            Some(manifest.preferred_encoder_path(dir)),
            Some(manifest.vocab_path(dir)),
            manifest.decoder_path(dir),
            manifest.joint_path(dir),
        ];
        for path in files.into_iter().flatten() {
            anyhow::ensure!(
                path.is_file(),
                "custom model pack file missing: {}",
                path.display()
            );
        }
        tracing::info!(
            "Using custom {:?} model pack at {model_dir}",
            manifest.architecture
        );
        return Ok(manifest.architecture);
    }

    // Determine the variant that is fully usable on disk. `detect_in_dir` only
    // checks for an encoder file, so we filter to variants whose complete INT8
    // set is present (FP32-only is not usable).
    let existing = ModelVariant::detect_in_dir(dir).filter(|&v| is_usable_present(v, dir));

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
    // the download while we were waiting (prequantized INT8 set).
    if is_usable_present(variant, dir) {
        tracing::info!("Model ({variant:?}) found at {model_dir} after lock acquisition");
        return Ok(variant);
    }

    // Default fetch is the lean pre-quantized INT8 bundle for RNN-T heads
    // (GitHub Release). CTC heads already download INT8 from HuggingFace.
    // Use the FP32 HF path only when the lean path is unavailable for a head
    // that has no prequantized set (should not happen for shipped variants).
    if variant.is_ctc() {
        tracing::info!("Model ({variant:?}) not found, downloading from HuggingFace...");
        for file in variant.download_files() {
            download_file(variant, file, dir).await?;
        }
    } else {
        tracing::info!(
            "Model ({variant:?}) not found, downloading pre-quantized INT8 bundle from {PREQUANT_RELEASE_BASE}..."
        );
        for file in variant.prequantized_files() {
            let final_dest = dir.join(file);
            if final_dest.exists() {
                continue;
            }
            let url = format!("{PREQUANT_RELEASE_BASE}/{file}");
            let expected = variant.prequantized_checksum(file);
            stream_to_partial_then_finalize(&url, &final_dest, expected, file).await?;
        }
    }

    tracing::info!("Model download complete");
    Ok(variant)
}

/// Ensure the **FP32 download set** for `requested` (or the variant already on
/// disk, else default `Rnnt`) exists in `model_dir`, fetching from HuggingFace
/// when missing.
///
/// **Packaging / quantize source only** — the product runtime never loads FP32.
/// Prefer [`ensure_model_variant`] / [`ensure_prequantized_model_variant`] for
/// inference. Does not quantize; callers run [`crate::quantize`] separately.
#[cfg(feature = "net")]
pub async fn ensure_fp32_model_variant(
    requested: Option<ModelVariant>,
    model_dir: &str,
) -> Result<ModelVariant> {
    let dir = Path::new(model_dir);
    let existing = ModelVariant::detect_in_dir(dir).filter(|&v| is_model_present(v, dir));
    let variant = match resolve_variant(requested, existing) {
        VariantAction::Use(v) => {
            tracing::info!("Using existing FP32 {v:?} model at {model_dir}");
            return Ok(v);
        }
        VariantAction::Download(v) => v,
    };

    std::fs::create_dir_all(dir).context("Failed to create model directory")?;
    #[cfg(unix)]
    let _lock = acquire_download_lock(dir)?;

    if is_model_present(variant, dir) {
        tracing::info!("FP32 model ({variant:?}) found at {model_dir} after lock");
        return Ok(variant);
    }

    tracing::info!("Downloading FP32 {variant:?} model set from HuggingFace...");
    for file in variant.download_files() {
        download_file(variant, file, dir).await?;
    }
    tracing::info!("FP32 model download complete");
    Ok(variant)
}

/// Ensure the **pre-quantized** INT8 model bundle for `requested` (or the
/// variant already on disk, else the default `Rnnt`) exists in `model_dir`,
/// downloading it from the pinned GitHub Release if missing.
///
/// This is the product download path: INT8 encoder + decoder + joiner + vocab
/// (no FP32, no on-device quantize). Each file is SHA-256-verified and atomically
/// renamed, reusing the same download primitive as [`ensure_model_variant`].
///
/// If the **pre-quantized INT8 set** is already present, it is used as-is.
/// An FP32-only tree is **not** treated as ready (runtime is INT8-only).
#[cfg(feature = "net")]
pub async fn ensure_prequantized_model_variant(
    requested: Option<ModelVariant>,
    model_dir: &str,
) -> Result<ModelVariant> {
    let dir = Path::new(model_dir);
    let variant = requested
        .or_else(|| ModelVariant::detect_in_dir(dir).filter(|&v| is_usable_present(v, dir)))
        .unwrap_or_default();

    if is_prequantized_present(variant, dir) {
        tracing::info!("Using existing {variant:?} INT8 model at {model_dir}");
        return Ok(variant);
    }

    std::fs::create_dir_all(dir).context("Failed to create model directory")?;

    #[cfg(unix)]
    let _lock = acquire_download_lock(dir)?;

    // Re-check after acquiring the lock in case another process finished.
    if is_prequantized_present(variant, dir) {
        tracing::info!("Pre-quantized {variant:?} model found at {model_dir} after lock");
        return Ok(variant);
    }

    tracing::info!("Downloading pre-quantized {variant:?} model from {PREQUANT_RELEASE_BASE}...");

    for file in variant.prequantized_files() {
        let final_dest = dir.join(file);
        if final_dest.exists() {
            continue;
        }
        let url = format!("{PREQUANT_RELEASE_BASE}/{file}");
        let expected = variant.prequantized_checksum(file);
        stream_to_partial_then_finalize(&url, &final_dest, expected, file).await?;
    }

    tracing::info!("Pre-quantized model download complete");
    Ok(variant)
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

/// True when every file in `variant`'s pre-quantized bundle (INT8 encoder,
/// decoder, joiner, vocab) is present in `dir`. The engine runs from this set
/// alone — no FP32 encoder required.
pub fn is_prequantized_present(variant: ModelVariant, dir: &Path) -> bool {
    variant
        .prequantized_files()
        .iter()
        .all(|f| dir.join(f).exists())
}

/// True when the engine can load `variant` from `dir` without a download.
///
/// **INT8 only:** the lean pre-quantized set ([`is_prequantized_present`]) or,
/// for CTC heads, the same INT8-on-disk layout as [`is_model_present`] (CTC
/// download files are already INT8). An FP32-only install is **not** usable.
pub fn is_usable_present(variant: ModelVariant, dir: &Path) -> bool {
    if variant.is_ctc() {
        // CTC download set is the INT8 encoder + vocab (no separate prequant list).
        return is_model_present(variant, dir) || is_prequantized_present(variant, dir);
    }
    is_prequantized_present(variant, dir)
}

/// Append `.partial` to a path; retained for tests that assert the legacy
/// staging name. Production download path uses `partial_path_unique`.
#[cfg(test)]
pub(super) fn partial_path(final_path: &Path) -> std::path::PathBuf {
    let mut s: std::ffi::OsString = final_path.as_os_str().to_owned();
    s.push(".partial");
    std::path::PathBuf::from(s)
}

#[cfg(feature = "net")]
pub(super) mod fetch;
#[cfg(feature = "net")]
use fetch::{download_file, stream_to_partial_then_finalize};

#[cfg(feature = "ane")]
pub(super) mod ane;
#[cfg(all(feature = "net", feature = "ane"))]
pub use ane::ensure_ane_packages;
#[cfg(feature = "ane")]
pub use ane::{ane_package_complete, ane_package_dir_name, default_ane_model_dir, is_ane_present};

#[cfg(feature = "net")]
mod sidecars;
#[cfg(all(feature = "net", feature = "diarization"))]
pub use sidecars::ensure_speaker_model;
#[cfg(feature = "net")]
pub use sidecars::{ensure_punct_model, ensure_vad_model};
