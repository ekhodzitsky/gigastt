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
    /// ONNX and vocab basenames for this pack.
    pub files: ManifestFiles,
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
            files: ManifestFiles {
                encoder,
                encoder_int8,
                decoder,
                joint,
                vocab,
            },
        })
    }

    /// Preferred encoder path: INT8 basename when that file exists, else FP32.
    pub fn preferred_encoder_path(&self, dir: &Path) -> PathBuf {
        if let Some(ref int8_name) = self.files.encoder_int8 {
            let int8 = dir.join(int8_name);
            if int8.exists() {
                return int8;
            }
        }
        dir.join(&self.files.encoder)
    }

    /// True when the preferred encoder path is the INT8 basename and exists.
    pub fn prefers_int8(&self, dir: &Path) -> bool {
        self.files
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
mod tests {
    use super::*;
    use std::fs;

    #[test]
    fn test_parse_valid_rnnt_manifest() {
        let text = r#"
architecture = "rnnt"

[files]
encoder = "v3_rnnt_encoder.onnx"
encoder_int8 = "v3_rnnt_encoder_int8.onnx"
decoder = "v3_rnnt_decoder.onnx"
joint = "v3_rnnt_joint.onnx"
vocab = "v3_vocab.txt"
"#;
        let m = ModelManifest::parse(text).expect("valid manifest");
        assert_eq!(m.architecture, ModelVariant::Rnnt);
        assert_eq!(m.files.encoder, "v3_rnnt_encoder.onnx");
        assert_eq!(
            m.files.encoder_int8.as_deref(),
            Some("v3_rnnt_encoder_int8.onnx")
        );
        assert_eq!(m.files.decoder.as_deref(), Some("v3_rnnt_decoder.onnx"));
        assert_eq!(m.files.joint.as_deref(), Some("v3_rnnt_joint.onnx"));
        assert_eq!(m.files.vocab, "v3_vocab.txt");
    }

    #[test]
    fn test_parse_valid_ctc_manifest_without_decoder_joint() {
        let text = r#"
architecture = "ml_ctc"

[files]
encoder = "multilingual_ctc.onnx"
encoder_int8 = "multilingual_ctc.int8.onnx"
vocab = "multilingual_vocab.txt"
"#;
        let m = ModelManifest::parse(text).expect("ctc manifest");
        assert_eq!(m.architecture, ModelVariant::MlCtc);
        assert!(m.files.decoder.is_none());
        assert!(m.files.joint.is_none());
    }

    #[test]
    fn test_parse_empty_decoder_joint_treated_as_absent() {
        let text = r#"
architecture = "ml_ctc_large"

[files]
encoder = "multilingual_large_ctc.onnx"
decoder = ""
joint = ""
vocab = "multilingual_vocab.txt"
"#;
        let m = ModelManifest::parse(text).expect("empty decoder/joint ok for ctc");
        assert!(m.files.decoder.is_none());
        assert!(m.files.joint.is_none());
    }

    #[test]
    fn test_load_missing_file_returns_none() {
        let dir = tempfile::tempdir().expect("tempdir");
        let loaded = ModelManifest::load(dir.path()).expect("missing is ok");
        assert!(loaded.is_none());
    }

    #[test]
    fn test_load_valid_manifest_from_disk() {
        let dir = tempfile::tempdir().expect("tempdir");
        fs::write(
            dir.path().join(MANIFEST_FILE),
            r#"
architecture = "e2e_rnnt"
[files]
encoder = "custom_encoder.onnx"
decoder = "custom_decoder.onnx"
joint = "custom_joint.onnx"
vocab = "custom_vocab.txt"
"#,
        )
        .unwrap();
        let m = ModelManifest::load(dir.path())
            .expect("load")
            .expect("present");
        assert_eq!(m.architecture, ModelVariant::E2eRnnt);
        assert_eq!(m.files.encoder, "custom_encoder.onnx");
    }

    #[test]
    fn test_invalid_architecture_rejected() {
        let text = r#"
architecture = "whisper"
[files]
encoder = "e.onnx"
decoder = "d.onnx"
joint = "j.onnx"
vocab = "v.txt"
"#;
        let err = ModelManifest::parse(text).expect_err("whisper must fail");
        let msg = format!("{err:#}");
        assert!(
            msg.contains("architecture") || msg.contains("whisper"),
            "error should mention architecture: {msg}"
        );
    }

    #[test]
    fn test_rnnt_requires_decoder_and_joint() {
        let text = r#"
architecture = "rnnt"
[files]
encoder = "e.onnx"
vocab = "v.txt"
"#;
        let err = ModelManifest::parse(text).expect_err("decoder required");
        assert!(
            format!("{err:#}").contains("decoder"),
            "expected decoder error, got {err:#}"
        );
    }

    #[test]
    fn test_reject_path_separators_in_basenames() {
        let text = r#"
architecture = "rnnt"
[files]
encoder = "../escape.onnx"
decoder = "d.onnx"
joint = "j.onnx"
vocab = "v.txt"
"#;
        let err = ModelManifest::parse(text).expect_err("path sep rejected");
        assert!(
            format!("{err:#}").contains("basename"),
            "expected basename error, got {err:#}"
        );
    }

    #[test]
    fn test_preferred_encoder_path_prefers_int8_when_present() {
        let dir = tempfile::tempdir().expect("tempdir");
        fs::write(dir.path().join("enc.onnx"), b"fp32").unwrap();
        fs::write(dir.path().join("enc_int8.onnx"), b"int8").unwrap();
        let m = ModelManifest {
            architecture: ModelVariant::Rnnt,
            files: ManifestFiles {
                encoder: "enc.onnx".into(),
                encoder_int8: Some("enc_int8.onnx".into()),
                decoder: Some("d.onnx".into()),
                joint: Some("j.onnx".into()),
                vocab: "v.txt".into(),
            },
        };
        assert_eq!(
            m.preferred_encoder_path(dir.path()).file_name().unwrap(),
            "enc_int8.onnx"
        );
        assert!(m.prefers_int8(dir.path()));
    }

    #[test]
    fn test_preferred_encoder_path_falls_back_to_fp32() {
        let dir = tempfile::tempdir().expect("tempdir");
        fs::write(dir.path().join("enc.onnx"), b"fp32").unwrap();
        let m = ModelManifest {
            architecture: ModelVariant::Rnnt,
            files: ManifestFiles {
                encoder: "enc.onnx".into(),
                encoder_int8: Some("enc_int8.onnx".into()),
                decoder: Some("d.onnx".into()),
                joint: Some("j.onnx".into()),
                vocab: "v.txt".into(),
            },
        };
        assert_eq!(
            m.preferred_encoder_path(dir.path()).file_name().unwrap(),
            "enc.onnx"
        );
        assert!(!m.prefers_int8(dir.path()));
    }

    #[test]
    fn test_load_invalid_manifest_returns_error() {
        let dir = tempfile::tempdir().expect("tempdir");
        fs::write(dir.path().join(MANIFEST_FILE), "architecture = [\n").unwrap();
        let err = ModelManifest::load(dir.path()).expect_err("invalid toml");
        assert!(
            format!("{err:#}").contains("manifest"),
            "expected manifest context, got {err:#}"
        );
    }
}
