//! Model-file resolution for engine load (variant override + manifest + INT8 paths).
//!
//! Free of ONNX sessions so override / manifest / disk precedence is unit-testable
//! without model weights.

use std::path::Path;

use crate::error::GigasttError;
use crate::model::{ModelManifest, ModelVariant};

/// Resolve which recognition head the engine should load.
///
/// Precedence:
/// 1. Explicit `override_` (from `--model-variant`) always wins.
/// 2. Else `manifest.toml` architecture, when that file is present.
/// 3. Else auto-detect from on-disk encoder filenames (`rnnt` precedence).
///
/// A present-but-invalid `manifest.toml` is a hard error (not silently ignored).
pub(crate) fn resolve_load_variant(
    override_: Option<ModelVariant>,
    model_dir: &Path,
) -> anyhow::Result<Option<ModelVariant>> {
    // Always validate a present manifest so a corrupt pack fails clearly, even
    // when the CLI override selects the architecture.
    let manifest = ModelManifest::load(model_dir)?;
    if let Some(v) = override_ {
        return Ok(Some(v));
    }
    if let Some(m) = manifest {
        return Ok(Some(m.architecture));
    }
    Ok(ModelVariant::detect_in_dir(model_dir))
}

/// Like [`resolve_load_variant`], but maps missing/invalid layouts to
/// [`GigasttError::ModelLoad`] for engine load entry points.
pub(crate) fn resolve_variant_required(
    override_: Option<ModelVariant>,
    model_dir: &Path,
) -> Result<ModelVariant, GigasttError> {
    match resolve_load_variant(override_, model_dir) {
        Ok(Some(v)) => Ok(v),
        Ok(None) => Err(GigasttError::ModelLoad {
            path: model_dir.display().to_string(),
            source: None,
        }),
        Err(e) => Err(GigasttError::ModelLoad {
            path: model_dir.display().to_string(),
            source: Some(e.into()),
        }),
    }
}

/// On-disk model file paths for a load: either from `manifest.toml` or from the
/// hardcoded [`ModelVariant`] basenames when no manifest is present.
pub(crate) struct ResolvedModelFiles {
    pub encoder: std::path::PathBuf,
    pub decoder: Option<std::path::PathBuf>,
    pub joint: Option<std::path::PathBuf>,
    pub vocab: std::path::PathBuf,
    pub using_int8: bool,
    encoder_hash: Option<String>,
}

impl ResolvedModelFiles {
    pub(crate) fn resolve(dir: &Path, variant: ModelVariant) -> anyhow::Result<Self> {
        if let Some(m) = ModelManifest::load(dir)? {
            anyhow::ensure!(
                m.architecture == variant,
                "requested model variant conflicts with manifest architecture"
            );
            let encoder = m.resolve_encoder(dir)?;
            return Ok(Self {
                encoder: encoder.path,
                decoder: m.decoder_path(dir),
                joint: m.joint_path(dir),
                vocab: m.vocab_path(dir),
                using_int8: encoder.using_int8,
                encoder_hash: None,
            });
        }
        Self::from_variant(dir, variant)
    }

    pub(crate) fn from_variant(dir: &Path, variant: ModelVariant) -> anyhow::Result<Self> {
        // Without an explicit manifest, builtin packs require INT8; never fall back.
        let int8 = dir.join(variant.encoder_int8_file());
        anyhow::ensure!(
            int8.is_file(),
            "INT8 encoder not found at {} — a builtin pack requires INT8. \
             Run `gigastt download`, or configure an explicit FP32 E2E manifest.",
            int8.display()
        );
        if variant.is_ctc() {
            Ok(Self {
                encoder: int8,
                decoder: None,
                joint: None,
                vocab: dir.join(variant.vocab_file()),
                using_int8: true,
                encoder_hash: None,
            })
        } else {
            Ok(Self {
                encoder: int8,
                decoder: Some(dir.join(variant.decoder_file())),
                joint: Some(dir.join(variant.joint_file())),
                vocab: dir.join(variant.vocab_file()),
                using_int8: true,
                encoder_hash: None,
            })
        }
    }

    /// Check every resolved file that has a pinned digest. Custom
    /// `manifest.toml` names with no table entry are skipped.
    pub(crate) fn verify_pinned_checksums(
        &mut self,
        variant: ModelVariant,
    ) -> Result<(), GigasttError> {
        let mut files: Vec<&std::path::Path> = vec![&self.encoder, &self.vocab];
        if let Some(d) = self.decoder.as_deref() {
            files.push(d);
        }
        if let Some(j) = self.joint.as_deref() {
            files.push(j);
        }
        for path in files {
            let Some(name) = path.file_name().and_then(|n| n.to_str()) else {
                continue;
            };
            let Some(expected) = variant.checksum(name) else {
                continue;
            };
            crate::model::verify_pinned_checksum(path, expected)?;
            if path == self.encoder {
                // Pinned bundles are self-contained; reuse the verified digest.
                self.encoder_hash = Some(expected.to_owned());
            }
        }
        if self.encoder_hash.is_none() {
            self.encoder_hash =
                crate::model::optimized_source_hash(&self.encoder).map_err(|e| {
                    GigasttError::ModelLoad {
                        path: self.encoder.display().to_string(),
                        source: Some(e.into()),
                    }
                })?;
        }
        Ok(())
    }
}

use crate::runtime::factory::Runtime;
use crate::runtime::tensor::{Shape, Tensor, TensorData};

use super::N_MELS;
use super::pool::SessionTriplet;
use super::sizing;

/// Path to the selected encoder for `variant`. Honors manifest precision
/// when present; falls back to the expected INT8 basename when resolve fails
/// (tests may call this without a full install).
pub(crate) fn encoder_model_path(dir: &Path, variant: ModelVariant) -> std::path::PathBuf {
    ResolvedModelFiles::resolve(dir, variant)
        .map(|files| files.encoder)
        .unwrap_or_else(|_| dir.join(variant.encoder_int8_file()))
}

/// Load up to `pool_size` session triplets in parallel through the given
/// [`Runtime`], tolerating a partial pool down to `min_size`.
pub(crate) fn load_triplets_runtime(
    runtime: &dyn Runtime,
    files: &ResolvedModelFiles,
    variant: ModelVariant,
    pool_size: usize,
    min_size: usize,
) -> anyhow::Result<Vec<SessionTriplet>> {
    let encoder_path = files.encoder.clone();
    // CTC is encoder-only: no decoder/joiner ONNX exists on disk, and the CTC
    // branch in `run_inference` returns right after the encoder run without
    // touching them. Load them only for the RNN-T heads (leaving `None` for
    // CTC avoids holding an unused, never-run session per pool triplet).
    let is_ctc = variant.is_ctc();
    let decoder_path = files.decoder.clone();
    let joiner_path = files.joint.clone();

    let results: Vec<anyhow::Result<SessionTriplet>> = std::thread::scope(|s| {
        let handles: Vec<_> = (0..pool_size)
            .map(|i| {
                let encoder_path = &encoder_path;
                let decoder_path = &decoder_path;
                let joiner_path = &joiner_path;
                s.spawn(move || {
                    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                        tracing::info!(
                            "Loading session triplet {}/{pool_size} (shared runtime)",
                            i + 1
                        );
                        let encoder = runtime
                            .load_session_with_model_hash(
                                encoder_path,
                                true,
                                files.encoder_hash.as_deref(),
                            )
                            .map_err(|e| anyhow::anyhow!(e))?;
                        let (decoder, joiner) = if is_ctc {
                            (None, None)
                        } else {
                            let decoder_path = decoder_path.as_ref().ok_or_else(|| {
                                anyhow::anyhow!(
                                    "decoder ONNX path missing for non-CTC architecture {}",
                                    variant.as_str()
                                )
                            })?;
                            let joiner_path = joiner_path.as_ref().ok_or_else(|| {
                                anyhow::anyhow!(
                                    "joint ONNX path missing for non-CTC architecture {}",
                                    variant.as_str()
                                )
                            })?;
                            let decoder = runtime
                                .load_session(decoder_path, false)
                                .map_err(|e| anyhow::anyhow!(e))?;
                            let joiner = runtime
                                .load_session(joiner_path, false)
                                .map_err(|e| anyhow::anyhow!(e))?;
                            (Some(decoder), Some(joiner))
                        };
                        Ok(SessionTriplet {
                            encoder,
                            decoder,
                            joiner,
                            encoder_inputs: vec![
                                Tensor::new(
                                    Shape::new(vec![1, N_MELS, 1]),
                                    TensorData::F32(vec![0.0; N_MELS]),
                                )?,
                                Tensor::new(Shape::new(vec![1]), TensorData::I64(vec![0]))?,
                            ],
                        })
                    }))
                    .map_err(|_| anyhow::anyhow!("model loading thread panicked"))?
                })
            })
            .collect();
        handles
            .into_iter()
            .map(|h| match h.join() {
                Ok(r) => r,
                Err(_) => Err(anyhow::anyhow!("model loading thread panicked")),
            })
            .collect()
    });
    sizing::finalize_pool_load(results, pool_size, min_size)
}

#[cfg(test)]
mod tests;
