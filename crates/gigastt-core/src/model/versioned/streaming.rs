//! Explicit qualified cached v4 streaming pack; full-context and legacy loaders
//! do not admit this separate ABI.

#[cfg(any(
    test,
    all(
        feature = "v4-mkl",
        target_os = "linux",
        target_arch = "x86_64",
        target_env = "gnu"
    )
))]
use super::read_artifact_bytes;
use super::{
    CHECKPOINT_SHA256, MAX_DESCRIPTOR_BYTES, MODEL_REPOSITORY, MODEL_REVISION, PackArtifact,
    PackFamily, PackHead, PackMode, PackProvenance, SCHEMA_VERSION, SOURCE_REPOSITORY,
    SOURCE_REVISION, TensorElement, TranscriptStyle, hash_field, read_manifest_text,
    valid_basename, valid_digest, verify_artifact,
};
use crate::{
    model::ModelPrecision,
    sha256::{Sha256, hex_lower},
};
use anyhow::{Context, Result, ensure};
use serde::Deserialize;
use std::{
    collections::{BTreeMap, BTreeSet},
    path::{Path, PathBuf},
};

#[path = "streaming_abi.rs"]
mod abi;
pub use abi::{V4_STREAMING_FRONT_CACHE_SHAPES, V4_STREAMING_LAYER_CACHE_SHAPES};

/// Only the three qualified native future contexts are available.
pub const V4_STREAMING_CONTEXTS_MS: &[u16] = &[160, 320, 640];
/// Fixed left cache length in 40ms encoder frames.
pub const V4_STREAMING_LEFT_CONTEXT_FRAMES: usize = 32;
/// Qualified fixed streaming timestamp interval (seconds per encoder frame).
pub const V4_STREAMING_FRAME_SHIFT_SECONDS: f64 = 0.04;

/// Explicit native future-context selection. Changing it resets the stream.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum V4StreamContext {
    /// 160ms (4 encoder frames).
    Ms160,
    /// 320ms (8 encoder frames).
    Ms320,
    /// 640ms (16 encoder frames).
    Ms640,
}
impl V4StreamContext {
    /// Reject any context not measured by the qualified profile.
    pub fn from_millis(ms: u16) -> Result<Self> {
        match ms {
            160 => Ok(Self::Ms160),
            320 => Ok(Self::Ms320),
            640 => Ok(Self::Ms640),
            _ => {
                anyhow::bail!("unsupported v4 streaming future context: require 160, 320 or 640ms")
            }
        }
    }
    /// Configured future context in milliseconds.
    pub const fn future_context_ms(self) -> u16 {
        match self {
            Self::Ms160 => 160,
            Self::Ms320 => 320,
            Self::Ms640 => 640,
        }
    }
    /// Number of 40ms encoder frames per admission block.
    pub const fn context_frames(self) -> usize {
        self.future_context_ms() as usize / 40
    }
    /// Number of mel frames per admission block.
    pub const fn mel_block(self) -> usize {
        self.context_frames() * 4
    }
}

/// Separate cached-streaming ABI (not a full-context encoder profile).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[non_exhaustive]
pub enum StreamingPackAbi {
    /// Official dynamic CPU FP32 split frontend, step and both RNNT heads.
    #[serde(rename = "v4_rnnt_streaming_split_v1")]
    V4RnntStreamingSplitV1,
}

/// Exact exported tensor metadata. `-1` denotes a symbolic ONNX dimension;
/// actual physical batch/cache/sequence shapes are separately checked at run.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StreamingTensorSpec {
    /// Actual ONNX outlet name (and required order in the surrounding list).
    pub name: &'static str,
    /// Exact tensor element type.
    pub element: TensorElement,
    /// Exported metadata axes, including exact rank.
    pub shape: &'static [i64],
}

/// Role-specific pinned graph, tokenizer and official metadata bytes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum StreamingArtifactRole {
    /// Stateful acoustic subsampling frontend graph.
    Frontend,
    /// Stateful cached encoder step graph.
    Step,
    /// Raw predictor.
    RawDecoder,
    /// Raw joiner.
    RawJoint,
    /// Raw SentencePiece model.
    RawTokenizer,
    /// End-to-end predictor.
    E2eDecoder,
    /// End-to-end joiner.
    E2eJoint,
    /// End-to-end SentencePiece model.
    E2eTokenizer,
    /// Official streaming state manifest, provenance only.
    OfficialStateManifest,
    /// Identical official named manifest alias, provenance only.
    OfficialManifestAlias,
    /// Official saved model configuration, provenance only.
    OfficialSavedConfig,
}
impl StreamingArtifactRole {
    /// Stable role name in descriptors/content identities.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Frontend => "frontend",
            Self::Step => "step",
            Self::RawDecoder => "raw_decoder",
            Self::RawJoint => "raw_joint",
            Self::RawTokenizer => "raw_tokenizer",
            Self::E2eDecoder => "e2e_decoder",
            Self::E2eJoint => "e2e_joint",
            Self::E2eTokenizer => "e2e_tokenizer",
            Self::OfficialStateManifest => "official_state_manifest",
            Self::OfficialManifestAlias => "official_manifest_alias",
            Self::OfficialSavedConfig => "official_saved_config",
        }
    }
    /// Trusted exact exported input/output metadata for graph roles only.
    pub const fn graph_specs(
        self,
    ) -> Option<(
        &'static [StreamingTensorSpec],
        &'static [StreamingTensorSpec],
    )> {
        match self {
            Self::Frontend => Some((abi::FRONTEND_INPUTS, abi::FRONTEND_OUTPUTS)),
            Self::Step => Some((abi::STEP_INPUTS, abi::STEP_OUTPUTS)),
            Self::RawDecoder | Self::E2eDecoder => {
                Some((abi::DECODER_INPUTS, abi::DECODER_OUTPUTS))
            }
            Self::RawJoint | Self::E2eJoint => Some((abi::JOINT_INPUTS, abi::JOINT_OUTPUTS)),
            _ => None,
        }
    }
}

const STREAMING_ROLES: [StreamingArtifactRole; 11] = [
    StreamingArtifactRole::Frontend,
    StreamingArtifactRole::Step,
    StreamingArtifactRole::RawDecoder,
    StreamingArtifactRole::RawJoint,
    StreamingArtifactRole::RawTokenizer,
    StreamingArtifactRole::E2eDecoder,
    StreamingArtifactRole::E2eJoint,
    StreamingArtifactRole::E2eTokenizer,
    StreamingArtifactRole::OfficialStateManifest,
    StreamingArtifactRole::OfficialManifestAlias,
    StreamingArtifactRole::OfficialSavedConfig,
];
#[derive(Clone, Copy)]
struct StreamingPin<'a> {
    role: StreamingArtifactRole,
    bytes: u64,
    sha256: &'a str,
}

// Exact qualified official dynamic CPU FP32 batch-one exports from the pinned
// checkpoint. The original JSON manifests and YAML are also verified as bytes;
// their whole/advance graph locators are not imported or executed by this ABI.
const STREAMING_PINS: [StreamingPin<'static>; 11] = [
    StreamingPin {
        role: StreamingArtifactRole::Frontend,
        bytes: 3_228_802,
        sha256: "5fd2b06f0a83b1206d88b2a32c5785bcdf261005cc176aa65b74c4d94ca3ac84",
    },
    StreamingPin {
        role: StreamingArtifactRole::Step,
        bytes: 948_135_265,
        sha256: "75e57be63d3ed747d5e25985c86fbba406ad94b7dda7240659fa299598c3fac4",
    },
    StreamingPin {
        role: StreamingArtifactRole::E2eDecoder,
        bytes: 4_599_970,
        sha256: "087ea392f825547ecc707e37f091d31ab211ec3ca2ee23dd86298d07621d4545",
    },
    StreamingPin {
        role: StreamingArtifactRole::E2eJoint,
        bytes: 2_712_938,
        sha256: "4964b0a350f35c78da1d100705b83fe05dfdf95e8535b970192985385f366ed7",
    },
    StreamingPin {
        role: StreamingArtifactRole::RawDecoder,
        bytes: 4_599_970,
        sha256: "2881cef81aa7f1e559293306bdbc2059be209065f6ce4ac45140b1c818cc8879",
    },
    StreamingPin {
        role: StreamingArtifactRole::RawJoint,
        bytes: 2_712_938,
        sha256: "38815775659e398cf9b0d977b92707557f2dcaea73c482233b7d5c6fb86fe95f",
    },
    StreamingPin {
        role: StreamingArtifactRole::E2eTokenizer,
        bytes: 14_686,
        sha256: "f547bcb38e071694ef1ddb25d9d40c86b4ab88cd0b16ab9fd52f27dd52c48e39",
    },
    StreamingPin {
        role: StreamingArtifactRole::RawTokenizer,
        bytes: 15_374,
        sha256: "dcbe2598d1b374065091832970593a06065d498a43fbb1787de73e180d0556bc",
    },
    StreamingPin {
        role: StreamingArtifactRole::OfficialStateManifest,
        bytes: 6_328,
        sha256: "879597716d08ec7ad2631a87e5c5e18fc265cfc7fb33c729cb32223e69cfb372",
    },
    StreamingPin {
        role: StreamingArtifactRole::OfficialManifestAlias,
        bytes: 6_328,
        sha256: "879597716d08ec7ad2631a87e5c5e18fc265cfc7fb33c729cb32223e69cfb372",
    },
    StreamingPin {
        role: StreamingArtifactRole::OfficialSavedConfig,
        bytes: 1_789,
        sha256: "f7d32c94c7de7fe0e414232cf80caf0f16aaf3ff3581980cc73760737aa9185a",
    },
];

/// Strict explicit streaming descriptor. The full-context descriptor parser
/// is intentionally a different type and rejects this manifest.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields)]
#[non_exhaustive]
pub struct StreamingPackDescriptor {
    /// Manifest schema.
    pub schema_version: u32,
    /// GigaAM v4 only.
    pub family: PackFamily,
    /// RNN-T only.
    pub head: PackHead,
    /// Both raw and e2e styles must be present.
    pub styles: Vec<TranscriptStyle>,
    /// Native streaming only.
    pub mode: PackMode,
    /// FP32 only.
    pub precision: ModelPrecision,
    /// Explicit CPU provider only.
    pub provider: String,
    /// Distinct cached streaming graph ABI.
    pub abi: StreamingPackAbi,
    /// Exact qualified supported contexts, in canonical order.
    pub future_context_ms: Vec<u16>,
    /// Exact cache left-context length.
    pub left_context_frames: usize,
    /// Trusted in-code state schema identifier; not arbitrary cache shapes.
    pub state_schema: String,
    /// Exact upstream source and checkpoint provenance.
    pub provenance: PackProvenance,
    /// Complete role-specific artifact declarations (both heads and metadata).
    pub artifacts: BTreeMap<StreamingArtifactRole, PackArtifact>,
    /// External ONNX data is not supported.
    #[serde(default)]
    pub external_data: Vec<PackArtifact>,
}
impl StreamingPackDescriptor {
    /// Parse the strict local descriptor, without accessing artifact files.
    pub fn parse(text: &str) -> Result<Self> {
        ensure!(
            text.len() as u64 <= MAX_DESCRIPTOR_BYTES,
            "versioned streaming pack manifest is too large"
        );
        let value: Self =
            toml::from_str(text).context("invalid versioned streaming model pack manifest")?;
        value.validate_profile()?;
        Ok(value)
    }
    /// Load a regular bounded `manifest.toml`; no fallback.
    pub fn load(dir: &Path) -> Result<Self> {
        Self::parse(&read_manifest_text(dir)?)
    }
    fn validate_profile(&self) -> Result<()> {
        ensure!(
            self.schema_version == SCHEMA_VERSION
                && self.family == PackFamily::GigaamV4
                && self.head == PackHead::Rnnt
                && self.abi == StreamingPackAbi::V4RnntStreamingSplitV1,
            "unsupported streaming pack family/head/ABI/schema"
        );
        ensure!(
            self.mode == PackMode::Streaming
                && self.precision == ModelPrecision::Fp32
                && self.provider == "cpu",
            "only the explicit CPU FP32 v4 cached streaming profile is qualified"
        );
        ensure!(
            self.styles.len() == 2
                && self.styles.contains(&TranscriptStyle::Raw)
                && self.styles.contains(&TranscriptStyle::E2e),
            "streaming pack requires both distinct raw and e2e styles"
        );
        ensure!(
            self.future_context_ms == V4_STREAMING_CONTEXTS_MS
                && self.left_context_frames == V4_STREAMING_LEFT_CONTEXT_FRAMES
                && self.state_schema == "v4_zapformer_cache_v1",
            "unsupported v4 streaming context or state schema"
        );
        let p = &self.provenance;
        ensure!(
            p.source_repository == SOURCE_REPOSITORY
                && p.source_revision == SOURCE_REVISION
                && p.model_repository == MODEL_REPOSITORY
                && p.model_revision == MODEL_REVISION
                && p.checkpoint_sha256 == CHECKPOINT_SHA256
                && p.license == "MIT",
            "unqualified v4 streaming source/checkpoint provenance"
        );
        ensure!(
            self.external_data.is_empty(),
            "external ONNX data is not qualified by the streaming profile"
        );
        ensure!(
            self.artifacts.len() == STREAMING_ROLES.len(),
            "streaming pack must declare exactly eleven artifact roles"
        );
        let mut names = BTreeSet::new();
        for role in STREAMING_ROLES {
            let artifact = self
                .artifacts
                .get(&role)
                .with_context(|| format!("missing streaming artifact role {}", role.as_str()))?;
            ensure!(
                valid_basename(&artifact.file),
                "streaming artifact {} must use a safe basename",
                role.as_str()
            );
            ensure!(
                names.insert(artifact.file.to_ascii_lowercase()),
                "duplicate streaming artifact basename"
            );
            ensure!(
                artifact.bytes > 0 && valid_digest(&artifact.sha256),
                "invalid size or SHA-256 for streaming artifact {}",
                role.as_str()
            );
        }
        Ok(())
    }
    /// Verify fixed release pins and actual bytes of all eleven local roles.
    /// No network, runtime creation, legacy fallback or caller trust roots.
    pub fn validate_in_dir(&self, dir: &Path) -> Result<QualifiedV4StreamingPack> {
        self.validate_with_pins(dir, &STREAMING_PINS)
    }
    fn validate_with_pins(
        &self,
        dir: &Path,
        pins: &[StreamingPin<'_>],
    ) -> Result<QualifiedV4StreamingPack> {
        self.validate_profile()?;
        let dir = dir
            .canonicalize()
            .context("streaming model pack directory is unavailable")?;
        ensure!(dir.is_dir(), "streaming model pack path is not a directory");
        for role in STREAMING_ROLES {
            let artifact = self
                .artifacts
                .get(&role)
                .context("missing required streaming artifact")?;
            let pin = pins
                .iter()
                .find(|p| p.role == role)
                .context("missing qualified streaming artifact pin")?;
            ensure!(
                artifact.bytes == pin.bytes && artifact.sha256 == pin.sha256,
                "artifact {} is not qualified by the v4 streaming profile",
                role.as_str()
            );
        }
        for role in STREAMING_ROLES {
            let artifact = self
                .artifacts
                .get(&role)
                .context("missing required streaming artifact")?;
            verify_artifact(&dir.join(&artifact.file), artifact)
                .with_context(|| format!("invalid {} streaming artifact", role.as_str()))?;
        }
        Ok(QualifiedV4StreamingPack {
            descriptor: self.clone(),
            dir,
            identity: self.content_identity(),
        })
    }
    fn content_identity(&self) -> String {
        let mut hash = Sha256::new();
        hash_field(&mut hash,b"gigastt/versioned-pack/schema-1/gigaam-v4/rnnt/raw+e2e/streaming/fp32/cpu/v4-rnnt-streaming-split-v1/160+320+640/left32/v4-zapformer-cache-v1");
        let p = &self.provenance;
        for field in [
            &p.source_repository,
            &p.source_revision,
            &p.model_repository,
            &p.model_revision,
            &p.checkpoint_sha256,
            &p.license,
        ] {
            hash_field(&mut hash, field.as_bytes());
        }
        for (role, artifact) in &self.artifacts {
            hash_field(&mut hash, role.as_str().as_bytes());
            hash_field(&mut hash, &artifact.bytes.to_le_bytes());
            hash_field(&mut hash, artifact.sha256.as_bytes());
        }
        hex_lower(&hash.finalize())
    }
}

/// Validation snapshot of the exact local streaming pack, distinct from a
/// full-context pack or an inference session. Consumers reverify actual bytes.
#[derive(Debug)]
pub struct QualifiedV4StreamingPack {
    descriptor: StreamingPackDescriptor,
    dir: PathBuf,
    identity: String,
}
impl QualifiedV4StreamingPack {
    /// Load and verify all local qualified byte artifacts.
    pub fn load(dir: &Path) -> Result<Self> {
        StreamingPackDescriptor::load(dir)?.validate_in_dir(dir)
    }
    /// Immutable verified descriptor.
    pub fn descriptor(&self) -> &StreamingPackDescriptor {
        &self.descriptor
    }
    /// Whole-pack identity includes context/state ABI and both heads.
    pub fn identity(&self) -> &str {
        &self.identity
    }
    #[cfg(any(
        test,
        all(
            feature = "v4-mkl",
            target_os = "linux",
            target_arch = "x86_64",
            target_env = "gnu"
        )
    ))]
    pub(crate) fn read_tokenizer_bytes(&self, style: TranscriptStyle) -> Result<Vec<u8>> {
        self.read_verified_bytes(match style {
            TranscriptStyle::Raw => StreamingArtifactRole::RawTokenizer,
            TranscriptStyle::E2e => StreamingArtifactRole::E2eTokenizer,
        })
    }
    #[cfg(all(
        feature = "v4-mkl",
        target_os = "linux",
        target_arch = "x86_64",
        target_env = "gnu"
    ))]
    pub(crate) fn read_graph_bytes(
        &self,
        role: StreamingArtifactRole,
    ) -> Result<VerifiedV4StreamingGraph> {
        ensure!(
            role.graph_specs().is_some(),
            "streaming metadata/tokenizer role is not an ONNX graph"
        );
        Ok(VerifiedV4StreamingGraph {
            role,
            bytes: self.read_verified_bytes(role)?,
        })
    }
    #[cfg(any(
        test,
        all(
            feature = "v4-mkl",
            target_os = "linux",
            target_arch = "x86_64",
            target_env = "gnu"
        )
    ))]
    fn read_verified_bytes(&self, role: StreamingArtifactRole) -> Result<Vec<u8>> {
        let artifact = self
            .descriptor
            .artifacts
            .get(&role)
            .context("missing validated streaming artifact")?;
        read_artifact_bytes(&self.dir.join(&artifact.file), artifact)
    }
    /// Canonical directory where the qualified pack was validated.
    pub fn directory(&self) -> &Path {
        &self.dir
    }
}

#[cfg(all(
    feature = "v4-mkl",
    target_os = "linux",
    target_arch = "x86_64",
    target_env = "gnu"
))]
pub(crate) struct VerifiedV4StreamingGraph {
    role: StreamingArtifactRole,
    bytes: Vec<u8>,
}
#[cfg(all(
    feature = "v4-mkl",
    target_os = "linux",
    target_arch = "x86_64",
    target_env = "gnu"
))]
impl VerifiedV4StreamingGraph {
    pub(crate) fn role(&self) -> StreamingArtifactRole {
        self.role
    }
    pub(crate) fn bytes(&self) -> &[u8] {
        &self.bytes
    }
}

#[cfg(test)]
#[path = "streaming_tests.rs"]
mod tests;
