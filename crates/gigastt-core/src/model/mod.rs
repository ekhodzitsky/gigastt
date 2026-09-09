//! Model download and management.
//!
//! Downloads GigaAM v3 ONNX files to `~/.gigastt/models/`.
//! Default `rnnt` / `e2e_rnnt` INT8 comes from GitHub Releases; `ml_ctc` /
//! `ml_ctc_large` and optional sidecars come from HuggingFace.
//! Recognition heads are selectable via [`ModelVariant`]: the plain `rnnt`
//! head (default — lower WER, bare lowercase output) and the `e2e_rnnt` head
//! (punctuation / casing / ITN baked in), plus the multilingual CTC heads.
//!
//! An optional `manifest.toml` in the model directory can override ONNX/vocab
//! basenames and select the architecture for third-party packs (see
//! [`ModelManifest`]). When absent, load paths use the hardcoded
//! [`ModelVariant`] filenames.

mod cache;
mod download;
mod manifest;
mod progress;
mod variant;

#[cfg(feature = "coreml")]
pub(crate) use cache::coreml_cache_dir;
pub use cache::{
    CoremlCachePruneReport, DedupeReport, OptimizedCachePruneReport, dedupe_model_dir,
    optimized_cache_basename, prune_coreml_cache, prune_optimized_cache, prune_optimized_cache_dir,
};
pub use manifest::{MANIFEST_FILE, ManifestFiles, ModelManifest};

#[cfg(all(feature = "net", feature = "ane"))]
pub use download::ensure_ane_packages;
#[cfg(all(feature = "net", feature = "diarization"))]
pub use download::ensure_speaker_model;
pub(crate) use download::verify_pinned_checksum;
pub use download::{
    VariantAction, default_model_dir, default_punct_model_dir, default_vad_model_dir,
    is_model_present, is_prequantized_present, is_usable_present, resolve_variant,
};
#[cfg(feature = "ane")]
pub use download::{
    ane_package_complete, ane_package_dir_name, default_ane_model_dir, is_ane_present,
};
#[cfg(feature = "net")]
pub use download::{
    ensure_fp32_model_variant, ensure_model, ensure_model_variant,
    ensure_prequantized_model_variant, ensure_punct_model, ensure_vad_model,
};

#[cfg(feature = "net")]
pub use progress::classify_download_error;
pub use progress::{
    ProgressErrorKind, ProgressEvent, ProgressMode, emit_progress_event, progress_mode,
    set_progress_mode,
};

#[cfg(feature = "ane")]
pub use variant::ANE_BUCKETS;
pub use variant::ModelVariant;
#[cfg(feature = "diarization")]
pub use variant::SPEAKER_MODEL_FILE;

#[cfg(test)]
mod tests;
