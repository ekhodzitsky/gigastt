//! Optional per-directory model pack manifest (`manifest.toml`).
//!
//! When present, the file names the ONNX/vocab basenames to load and selects
//! the decode architecture. When absent, load paths fall back to the hardcoded
//! [`super::ModelVariant`] filenames so existing model dirs stay byte-identical.

use anyhow::{Context, Result, bail};
use serde::Deserialize;
use std::path::{Path, PathBuf};

use super::ModelVariant;

/// Basename of the optional model-pack manifest inside a model directory.
pub const MANIFEST_FILE: &str = "manifest.toml";

/// Parsed `manifest.toml` for a model pack.
///
/// All file fields are basenames relative to the model directory (not absolute
/// paths). `architecture` selects the decode path via [`ModelVariant`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelManifest {
    /// Decode / recognition head selected by this pack.
    pub architecture: ModelVariant,
    /// Encoder precision; omitted manifest precision defaults to INT8.
    pub precision: ModelPrecision,
    /// ONNX and vocab basenames for this pack.
    pub files: ManifestFiles,
}

/// Encoder precision explicitly selected by a model pack.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ModelPrecision {
    /// Default published encoder.
    #[default]
    Int8,
    /// Explicitly supplied FP32 E2E RNN-T encoder.
    Fp32,
}

/// Validated encoder selection, shared by preflight and engine loading.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResolvedEncoder {
    /// Existing encoder file in the model directory.
    pub path: PathBuf,
    /// Whether the pack selects the INT8 encoder.
    pub using_int8: bool,
}

/// File basenames listed under `[files]` in `manifest.toml`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ManifestFiles {
    /// FP32 (or sole) encoder basename. Required.
    pub encoder: String,
    /// Preferred INT8 encoder basename when that file exists on disk.
    pub encoder_int8: Option<String>,
    /// Decoder basename; empty/absent for encoder-only CTC heads.
    pub decoder: Option<String>,
    /// Joiner basename; empty/absent for encoder-only CTC heads.
    pub joint: Option<String>,
    /// Vocabulary basename. Required.
    pub vocab: String,
}

#[derive(Debug, Deserialize)]
struct RawManifest {
    architecture: String,
    #[serde(default)]
    precision: ModelPrecision,
    files: RawFiles,
}

#[derive(Debug, Deserialize)]
struct RawFiles {
    encoder: String,
    #[serde(default)]
    encoder_int8: Option<String>,
    #[serde(default)]
    decoder: Option<String>,
    #[serde(default)]
    joint: Option<String>,
    vocab: String,
}

impl ModelManifest {
    /// Load `dir/manifest.toml` when present.
    ///
    /// - Missing file → `Ok(None)` (not an error; callers use hardcoded names).
    /// - Present but invalid → `Err` with a clear context message.
    pub fn load(dir: &Path) -> Result<Option<Self>> {
        let path = dir.join(MANIFEST_FILE);
        if !path.is_file() {
            return Ok(None);
        }
        let text = std::fs::read_to_string(&path)
            .with_context(|| format!("failed to read model manifest {}", path.display()))?;
        Self::parse(&text)
            .with_context(|| format!("invalid model manifest {}", path.display()))
            .map(Some)
    }

    /// Parse a manifest TOML document (no filesystem access).
    pub fn parse(text: &str) -> Result<Self> {
        let raw: RawManifest =
            toml::from_str(text).context("failed to parse model manifest TOML")?;
        Self::from_raw(raw)
    }

    fn from_raw(raw: RawManifest) -> Result<Self> {
        let architecture: ModelVariant = raw
            .architecture
            .parse()
            .map_err(|e: String| anyhow::anyhow!("invalid architecture: {e}"))?;

        if raw.precision == ModelPrecision::Fp32 && architecture != ModelVariant::E2eRnnt {
            bail!("precision 'fp32' is supported only for architecture 'e2e_rnnt'");
        }

        let encoder = normalize_required_basename("encoder", &raw.files.encoder)?;
        let vocab = normalize_required_basename("vocab", &raw.files.vocab)?;
        let encoder_int8 = normalize_optional_basename("encoder_int8", raw.files.encoder_int8)?;
        let decoder = normalize_optional_basename("decoder", raw.files.decoder)?;
        let joint = normalize_optional_basename("joint", raw.files.joint)?;

        if !architecture.is_ctc() {
            if decoder.is_none() {
                bail!(
                    "manifest files.decoder is required for architecture '{}'",
                    architecture.as_str()
                );
            }
            if joint.is_none() {
                bail!(
                    "manifest files.joint is required for architecture '{}'",
                    architecture.as_str()
                );
            }
        }

        Ok(Self {
            architecture,
            precision: raw.precision,
            files: ManifestFiles {
                encoder,
                encoder_int8,
                decoder,
                joint,
                vocab,
            },
        })
    }

    /// Resolve the explicitly selected encoder without falling back to another precision.
    ///
    /// # Errors
    /// Returns an error for unsupported FP32 architectures or a missing encoder file.
    pub fn resolve_encoder(&self, dir: &Path) -> Result<ResolvedEncoder> {
        let (name, using_int8) = match self.precision {
            ModelPrecision::Int8 => (
                self.files.encoder_int8.as_deref().ok_or_else(|| {
                    anyhow::anyhow!("manifest files.encoder_int8 is required for precision 'int8'")
                })?,
                true,
            ),
            ModelPrecision::Fp32 => {
                if self.architecture != ModelVariant::E2eRnnt {
                    bail!("precision 'fp32' is supported only for architecture 'e2e_rnnt'");
                }
                (self.files.encoder.as_str(), false)
            }
        };
        let path = dir.join(name);
        if !path.is_file() {
            let field = if using_int8 {
                "encoder_int8"
            } else {
                "encoder"
            };
            bail!("manifest {field} file missing: {}", path.display());
        }
        Ok(ResolvedEncoder { path, using_int8 })
    }

    /// Legacy path hint; explicit FP32 selects `files.encoder`.
    /// Use [`Self::resolve_encoder`] for validated loading without fallback.
    pub fn preferred_encoder_path(&self, dir: &Path) -> PathBuf {
        if self.precision == ModelPrecision::Fp32 {
            return dir.join(&self.files.encoder);
        }
        if let Some(ref int8_name) = self.files.encoder_int8 {
            let int8 = dir.join(int8_name);
            if int8.exists() {
                return int8;
            }
        }
        dir.join(&self.files.encoder)
    }

    /// True when INT8 is selected and its declared file exists.
    /// Use [`Self::resolve_encoder`] to validate that it is a regular file.
    pub fn prefers_int8(&self, dir: &Path) -> bool {
        self.precision == ModelPrecision::Int8
            && self
                .files
                .encoder_int8
                .as_ref()
                .is_some_and(|name| dir.join(name).exists())
    }

    /// Decoder path when configured (non-empty); `None` for CTC / empty.
    pub fn decoder_path(&self, dir: &Path) -> Option<PathBuf> {
        self.files.decoder.as_ref().map(|name| dir.join(name))
    }

    /// Joiner path when configured (non-empty); `None` for CTC / empty.
    pub fn joint_path(&self, dir: &Path) -> Option<PathBuf> {
        self.files.joint.as_ref().map(|name| dir.join(name))
    }

    /// Vocabulary path.
    pub fn vocab_path(&self, dir: &Path) -> PathBuf {
        dir.join(&self.files.vocab)
    }
}

/// Non-empty basename; rejects empty strings and path separators so values stay
/// relative basenames under the model dir.
fn normalize_required_basename(field: &str, value: &str) -> Result<String> {
    let trimmed = value.trim();
    if trimmed.is_empty() {
        bail!("manifest files.{field} must be a non-empty basename");
    }
    validate_basename(field, trimmed)?;
    Ok(trimmed.to_string())
}

fn normalize_optional_basename(field: &str, value: Option<String>) -> Result<Option<String>> {
    let Some(value) = value else {
        return Ok(None);
    };
    let trimmed = value.trim();
    if trimmed.is_empty() {
        return Ok(None);
    }
    validate_basename(field, trimmed)?;
    Ok(Some(trimmed.to_string()))
}

fn validate_basename(field: &str, value: &str) -> Result<()> {
    if value.contains('/') || value.contains('\\') || value.contains("..") {
        bail!(
            "manifest files.{field} must be a basename (got '{value}'); \
             paths and '..' are not allowed"
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests;
