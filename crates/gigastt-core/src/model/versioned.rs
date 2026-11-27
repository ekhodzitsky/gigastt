//! Explicit versioned model-pack contracts, separate from legacy `ModelVariant`.
//!
//! The first profile admits only the exact qualified v4 RNN-T FP32 split
//! artifacts, not arbitrary graphs with self-declared hashes. This module
//! validates local bytes and selections; it does not create inference sessions.

use anyhow::{Context, Result, ensure};
use serde::Deserialize;
use std::collections::{BTreeMap, BTreeSet};
use std::io::Read;
use std::path::{Path, PathBuf};

use crate::runtime::ExecutionProviderChoice;
use crate::sha256::{Sha256, hex_lower};

use super::{MANIFEST_FILE, ModelPrecision};

const SCHEMA_VERSION: u32 = 1;
const MAX_DESCRIPTOR_BYTES: u64 = 64 * 1024;
const SOURCE_REPOSITORY: &str = "https://github.com/salute-developers/GigaAM";
const SOURCE_REVISION: &str = "e9bbf2e773eedd217c6f0d3b89b9728c41332b7d";
const MODEL_REPOSITORY: &str = "ai-sage/GigaAM-v4";
const MODEL_REVISION: &str = "9fe3ebf50c9e0cce5eed60d546938632238b6a96";
const CHECKPOINT_SHA256: &str = "2c38690c38edbd9dfc2190f1a2dacdcfbea2a692078457fc145a6034c031840a";

/// Acoustic model generation, independent of the legacy variant names.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[non_exhaustive]
pub enum PackFamily {
    /// GigaAM v4 bilingual Zapformer.
    #[serde(rename = "gigaam_v4")]
    GigaamV4,
}

/// Recognition head architecture in a versioned pack.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum PackHead {
    /// Recurrent neural network transducer.
    Rnnt,
}

/// Independent output styles sharing the acoustic encoder.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum TranscriptStyle {
    /// Unformatted recognition output.
    Raw,
    /// End-to-end formatting with punctuation and casing.
    E2e,
}

impl TranscriptStyle {
    const fn as_str(self) -> &'static str {
        match self {
            Self::Raw => "raw",
            Self::E2e => "e2e",
        }
    }
}

/// Recognition mode. Streaming is a distinct, currently unqualified ABI.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum PackMode {
    /// Offline whole-utterance encoder context.
    FullContext,
    /// Native cached streaming (not admitted by the first profile).
    Streaming,
}

/// Versioned graph and token contract, not merely an ONNX filename convention.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[non_exhaustive]
pub enum PackAbi {
    /// Qualified FP32 split encoder/predictor/joiner v4 RNN-T profile.
    #[serde(rename = "v4_rnnt_split_v1")]
    V4RnntSplitV1,
}

impl PackAbi {
    /// Semantic tensor specification for this ABI. Symbolic ONNX dimension
    /// names are not the actual physical or valid lengths.
    pub const fn specification(self) -> &'static SplitAbi {
        match self {
            Self::V4RnntSplitV1 => &V4_RNNT_SPLIT_ABI,
        }
    }
}

/// Tensor element types used by the qualified split ABI.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TensorElement {
    /// IEEE float32.
    F32,
    /// Signed 64-bit integer.
    I64,
}

/// Semantic tensor dimension (the pack supports batch size one only).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TensorAxis {
    /// Constant dimension.
    Fixed(usize),
    /// Batch dimension.
    Batch,
    /// Input log-mel sequence dimension.
    MelFrames,
    /// Encoder or joiner time dimension.
    Frames,
}

/// A named tensor in the semantic split contract.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TensorSpec {
    /// Actual exported ONNX tensor name.
    pub name: &'static str,
    /// Element type.
    pub element: TensorElement,
    /// Ordered semantic axes, including the exact rank.
    pub axes: &'static [TensorAxis],
}

/// Qualified split ABI; the descriptor derives these invariants from its ABI
/// identifier instead of trusting redundant manifest declarations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SplitAbi {
    /// ONNX opset.
    pub opset: u32,
    /// PCM sample rate in Hz.
    pub sample_rate: usize,
    /// Mel channels.
    pub mel_bins: usize,
    /// FFT and window size in samples.
    pub fft_size: usize,
    /// Mel hop size in samples.
    pub hop_length: usize,
    /// Whether the acoustic frontend centers the STFT.
    pub center: bool,
    /// Encoder temporal subsampling factor.
    pub subsampling: usize,
    /// Actual encoder feature width.
    pub encoder_channels: usize,
    /// Predictor hidden-state width.
    pub predictor_hidden: usize,
    /// Predictor recurrent-layer count.
    pub predictor_layers: usize,
    /// Separate tokenizer format for each head.
    pub tokenizer_format: &'static str,
    /// SentencePiece vocabulary size excluding blank.
    pub vocabulary_pieces: usize,
    /// RNN-T blank label (also the fresh predictor input).
    pub blank_id: usize,
    /// Author maximum symbols per encoder frame.
    pub max_symbols_per_step: usize,
    /// Qualified joiner scoring-window size.
    pub joiner_window: usize,
    /// Maximum validated batch size, regardless of dynamic ONNX dimensions.
    pub max_batch_size: usize,
    /// Encoder input tensors.
    pub encoder_inputs: &'static [TensorSpec],
    /// Encoder output tensors.
    pub encoder_outputs: &'static [TensorSpec],
    /// Predictor input tensors (including recurrent state).
    pub decoder_inputs: &'static [TensorSpec],
    /// Predictor output tensors (including updated state).
    pub decoder_outputs: &'static [TensorSpec],
    /// Joiner inputs.
    pub joint_inputs: &'static [TensorSpec],
    /// Joiner output (log-softmax, not fused token IDs).
    pub joint_outputs: &'static [TensorSpec],
    /// Session-owned encoder cache tensors; empty for full context.
    pub encoder_cache: &'static [TensorSpec],
}

const fn tensor(
    name: &'static str,
    element: TensorElement,
    axes: &'static [TensorAxis],
) -> TensorSpec {
    TensorSpec {
        name,
        element,
        axes,
    }
}

use TensorAxis::{Batch, Fixed, Frames, MelFrames};
use TensorElement::{F32, I64};

/// The first qualified v4 RNN-T metadata-preserving split contract.
pub const V4_RNNT_SPLIT_ABI: SplitAbi = SplitAbi {
    opset: 17,
    sample_rate: 16_000,
    mel_bins: 64,
    fft_size: 320,
    hop_length: 160,
    center: false,
    subsampling: 4,
    encoder_channels: 768,
    predictor_hidden: 320,
    predictor_layers: 1,
    tokenizer_format: "sentencepiece",
    vocabulary_pieces: 1024,
    blank_id: 1024,
    max_symbols_per_step: 10,
    joiner_window: 8,
    max_batch_size: 1,
    encoder_inputs: &[
        tensor("audio_signal", F32, &[Batch, Fixed(64), MelFrames]),
        tensor("length", I64, &[Batch]),
    ],
    encoder_outputs: &[
        tensor("encoded", F32, &[Batch, Fixed(768), Frames]),
        tensor("encoded_len", I64, &[Batch]),
    ],
    decoder_inputs: &[
        tensor("x", I64, &[Batch, Fixed(1)]),
        tensor("hi", F32, &[Fixed(1), Batch, Fixed(320)]),
        tensor("ci", F32, &[Fixed(1), Batch, Fixed(320)]),
    ],
    decoder_outputs: &[
        tensor("dec", F32, &[Batch, Fixed(1), Fixed(320)]),
        tensor("ho", F32, &[Fixed(1), Batch, Fixed(320)]),
        tensor("co", F32, &[Fixed(1), Batch, Fixed(320)]),
    ],
    joint_inputs: &[
        tensor("enc", F32, &[Batch, Fixed(768), Frames]),
        tensor("dec", F32, &[Batch, Fixed(320), Fixed(1)]),
    ],
    joint_outputs: &[tensor(
        "joint",
        F32,
        &[Batch, Frames, Fixed(1), Fixed(1025)],
    )],
    encoder_cache: &[],
};

/// Logical artifact roles. Equal tensor shapes do not make independent heads
/// interchangeable: the release profile pins each role separately.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum ArtifactRole {
    /// Shared acoustic encoder.
    Encoder,
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
}

impl ArtifactRole {
    /// Stable role name used in manifests and content identities.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Encoder => "encoder",
            Self::RawDecoder => "raw_decoder",
            Self::RawJoint => "raw_joint",
            Self::RawTokenizer => "raw_tokenizer",
            Self::E2eDecoder => "e2e_decoder",
            Self::E2eJoint => "e2e_joint",
            Self::E2eTokenizer => "e2e_tokenizer",
        }
    }
}

const REQUIRED_ROLES: [ArtifactRole; 7] = [
    ArtifactRole::Encoder,
    ArtifactRole::RawDecoder,
    ArtifactRole::RawJoint,
    ArtifactRole::RawTokenizer,
    ArtifactRole::E2eDecoder,
    ArtifactRole::E2eJoint,
    ArtifactRole::E2eTokenizer,
];

#[derive(Clone, Copy)]
struct ArtifactPin<'a> {
    role: ArtifactRole,
    bytes: u64,
    sha256: &'a str,
}

// Exact locally qualified FP32 exports from the pinned official HF checkpoint.
// Their actual ONNX ABI and token/frame/confidence parity were validated before
// admission. A different export requires a separately qualified profile, even
// if it declares the same tensor shapes and an internally consistent digest.
const V4_SPLIT_PINS: [ArtifactPin<'static>; 7] = [
    ArtifactPin {
        role: ArtifactRole::Encoder,
        bytes: 951_218_063,
        sha256: "62fd1735d1a3cf45d83d525c1857fd5df807be9c5d38cd61a5af52f9de2888b6",
    },
    ArtifactPin {
        role: ArtifactRole::RawDecoder,
        bytes: 4_599_970,
        sha256: "2881cef81aa7f1e559293306bdbc2059be209065f6ce4ac45140b1c818cc8879",
    },
    ArtifactPin {
        role: ArtifactRole::RawJoint,
        bytes: 2_712_938,
        sha256: "38815775659e398cf9b0d977b92707557f2dcaea73c482233b7d5c6fb86fe95f",
    },
    ArtifactPin {
        role: ArtifactRole::RawTokenizer,
        bytes: 15_374,
        sha256: "dcbe2598d1b374065091832970593a06065d498a43fbb1787de73e180d0556bc",
    },
    ArtifactPin {
        role: ArtifactRole::E2eDecoder,
        bytes: 4_599_970,
        sha256: "087ea392f825547ecc707e37f091d31ab211ec3ca2ee23dd86298d07621d4545",
    },
    ArtifactPin {
        role: ArtifactRole::E2eJoint,
        bytes: 2_712_938,
        sha256: "4964b0a350f35c78da1d100705b83fe05dfdf95e8535b970192985385f366ed7",
    },
    ArtifactPin {
        role: ArtifactRole::E2eTokenizer,
        bytes: 14_686,
        sha256: "f547bcb38e071694ef1ddb25d9d40c86b4ab88cd0b16ab9fd52f27dd52c48e39",
    },
];

/// Authority and integrity declaration for one artifact.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields)]
#[non_exhaustive]
pub struct PackArtifact {
    /// Safe, relative basename in the pack directory.
    pub file: String,
    /// Exact byte length.
    pub bytes: u64,
    /// Lowercase SHA-256 digest.
    pub sha256: String,
}

/// Exact upstream origins of the qualified checkpoint and exporter.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields)]
#[non_exhaustive]
pub struct PackProvenance {
    /// Official upstream source repository.
    pub source_repository: String,
    /// Immutable exporter/source commit.
    pub source_revision: String,
    /// Official model repository identifier.
    pub model_repository: String,
    /// Immutable model revision.
    pub model_revision: String,
    /// SHA-256 of the original safetensors checkpoint.
    pub checkpoint_sha256: String,
    /// Upstream weight/source license identifier.
    pub license: String,
}

/// Parsed explicit versioned `manifest.toml`. Parsing verifies the descriptor
/// contract; only [`Self::validate_in_dir`] verifies qualified artifact bytes.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields)]
#[non_exhaustive]
pub struct VersionedPackDescriptor {
    /// Manifest schema version.
    pub schema_version: u32,
    /// Model generation.
    pub family: PackFamily,
    /// Recognition algorithm.
    pub head: PackHead,
    /// Both independent output styles, irrespective of ordering.
    pub styles: Vec<TranscriptStyle>,
    /// Declared recognition mode.
    pub mode: PackMode,
    /// Declared encoder precision.
    pub precision: ModelPrecision,
    /// Explicit execution provider (CPU only in this profile).
    pub provider: String,
    /// Semantic graph contract.
    pub abi: PackAbi,
    /// Exact model and exporter identity.
    pub provenance: PackProvenance,
    /// All required artifact roles, including the unselected head.
    pub artifacts: BTreeMap<ArtifactRole, PackArtifact>,
    /// External ONNX data inventory; must be empty for this self-contained profile.
    #[serde(default)]
    pub external_data: Vec<PackArtifact>,
}

impl VersionedPackDescriptor {
    /// Parse a versioned descriptor without accessing artifact files.
    pub fn parse(text: &str) -> Result<Self> {
        ensure!(
            text.len() as u64 <= MAX_DESCRIPTOR_BYTES,
            "versioned pack manifest is too large"
        );
        let descriptor: Self =
            toml::from_str(text).context("invalid versioned model pack manifest")?;
        descriptor.validate_profile()?;
        Ok(descriptor)
    }

    /// Load a regular, bounded `manifest.toml`. A missing manifest is an error
    /// here, not permission to fall back to a legacy model directory.
    pub fn load(dir: &Path) -> Result<Self> {
        let path = dir.join(MANIFEST_FILE);
        let meta =
            std::fs::symlink_metadata(&path).context("missing versioned model pack manifest")?;
        ensure!(
            meta.is_file() && meta.len() <= MAX_DESCRIPTOR_BYTES,
            "versioned pack manifest must be a bounded regular file"
        );
        let mut text = String::new();
        std::fs::File::open(&path)?
            .take(MAX_DESCRIPTOR_BYTES + 1)
            .read_to_string(&mut text)?;
        Self::parse(&text)
    }

    fn validate_profile(&self) -> Result<()> {
        ensure!(
            self.schema_version == SCHEMA_VERSION,
            "unsupported model pack schema version"
        );
        ensure!(
            self.family == PackFamily::GigaamV4
                && self.head == PackHead::Rnnt
                && self.abi == PackAbi::V4RnntSplitV1,
            "unsupported model pack family/head/ABI"
        );
        ensure!(
            self.mode == PackMode::FullContext
                && self.precision == ModelPrecision::Fp32
                && self.provider == "cpu",
            "only the explicit CPU FP32 full-context v4 split profile is qualified"
        );
        ensure!(
            self.styles.len() == 2
                && self.styles.contains(&TranscriptStyle::Raw)
                && self.styles.contains(&TranscriptStyle::E2e),
            "versioned v4 pack requires both distinct raw and e2e styles"
        );
        let p = &self.provenance;
        ensure!(
            p.source_repository == SOURCE_REPOSITORY
                && p.source_revision == SOURCE_REVISION
                && p.model_repository == MODEL_REPOSITORY
                && p.model_revision == MODEL_REVISION
                && p.checkpoint_sha256 == CHECKPOINT_SHA256
                && p.license == "MIT",
            "unqualified v4 source/checkpoint provenance"
        );
        ensure!(
            self.external_data.is_empty(),
            "external ONNX data is not qualified by this profile"
        );
        ensure!(
            self.artifacts.len() == REQUIRED_ROLES.len(),
            "versioned v4 pack must declare exactly seven artifact roles"
        );
        let mut filenames = BTreeSet::new();
        for role in REQUIRED_ROLES {
            let artifact = self
                .artifacts
                .get(&role)
                .with_context(|| format!("missing artifact role {}", role.as_str()))?;
            ensure!(
                valid_basename(&artifact.file),
                "artifact {} must use a safe basename",
                role.as_str()
            );
            ensure!(
                filenames.insert(artifact.file.to_ascii_lowercase()),
                "duplicate artifact basename"
            );
            ensure!(
                artifact.bytes > 0 && valid_digest(&artifact.sha256),
                "invalid size or SHA-256 for artifact {}",
                role.as_str()
            );
        }
        Ok(())
    }

    /// Verify both heads and all other declared files against the exact known
    /// release profile, then hash each local regular file. No network, graph
    /// loading, implicit selection, or fallback is performed.
    pub fn validate_in_dir(&self, dir: &Path) -> Result<QualifiedV4Pack> {
        self.validate_with_pins(dir, &V4_SPLIT_PINS)
    }

    // Kept private: tests can exercise byte verification with tiny fixtures,
    // but public callers cannot supply trust roots or bypass role-specific pins.
    fn validate_with_pins(&self, dir: &Path, pins: &[ArtifactPin<'_>]) -> Result<QualifiedV4Pack> {
        self.validate_profile()?;
        let dir = dir
            .canonicalize()
            .context("model pack directory is unavailable")?;
        ensure!(dir.is_dir(), "model pack path is not a directory");
        // Reject all mismatched declarations before reading large graphs.
        for role in REQUIRED_ROLES {
            let artifact = self
                .artifacts
                .get(&role)
                .context("missing required artifact")?;
            let pin = pins
                .iter()
                .find(|p| p.role == role)
                .context("missing qualified artifact pin")?;
            ensure!(
                artifact.bytes == pin.bytes && artifact.sha256 == pin.sha256,
                "artifact {} is not qualified by the v4 split profile",
                role.as_str()
            );
        }
        for role in REQUIRED_ROLES {
            let artifact = self
                .artifacts
                .get(&role)
                .context("missing required artifact")?;
            verify_artifact(&dir.join(&artifact.file), artifact)
                .with_context(|| format!("invalid {} artifact", role.as_str()))?;
        }
        Ok(QualifiedV4Pack {
            descriptor: self.clone(),
            dir,
            identity: self.content_identity(),
        })
    }

    fn content_identity(&self) -> String {
        let mut hash = Sha256::new();
        hash_field(&mut hash, b"gigastt/versioned-pack/schema-1/gigaam-v4/rnnt/raw+e2e/full-context/fp32/cpu/v4-rnnt-split-v1");
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
        // A basename is only a local locator. Role + exact bytes identify
        // content, independently of paths, TOML ordering, or selected style.
        for (role, artifact) in &self.artifacts {
            hash_field(&mut hash, role.as_str().as_bytes());
            hash_field(&mut hash, &artifact.bytes.to_le_bytes());
            hash_field(&mut hash, artifact.sha256.as_bytes());
        }
        hex_lower(&hash.finalize())
    }
}

fn valid_basename(name: &str) -> bool {
    !name.eq_ignore_ascii_case(MANIFEST_FILE)
        && name.len() <= 255
        && name
            .as_bytes()
            .first()
            .is_some_and(u8::is_ascii_alphanumeric)
        && !name.ends_with('.')
        && name
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'.' | b'_' | b'-'))
}

fn valid_digest(digest: &str) -> bool {
    digest.len() == 64
        && digest
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}

fn verify_artifact(path: &Path, artifact: &PackArtifact) -> Result<()> {
    let meta = std::fs::symlink_metadata(path).context("artifact missing")?;
    ensure!(
        meta.is_file() && meta.len() == artifact.bytes,
        "artifact must be a regular file with the declared size"
    );
    let file = std::fs::File::open(path)?;
    let current = file.metadata()?;
    ensure!(
        current.is_file() && current.len() == artifact.bytes,
        "artifact changed before verification"
    );
    let limit = artifact
        .bytes
        .checked_add(1)
        .context("artifact size overflow")?;
    let mut reader = file.take(limit);
    let mut hash = Sha256::new();
    let mut buf = [0u8; 64 * 1024];
    let mut bytes = 0u64;
    loop {
        let read = reader.read(&mut buf)?;
        if read == 0 {
            break;
        }
        bytes += read as u64;
        hash.update(&buf[..read]);
    }
    ensure!(
        bytes == artifact.bytes,
        "artifact size changed while reading"
    );
    ensure!(
        hex_lower(&hash.finalize()) == artifact.sha256,
        "artifact SHA-256 mismatch"
    );
    Ok(())
}

fn hash_field(hash: &mut Sha256, bytes: &[u8]) {
    hash.update(&(bytes.len() as u64).to_le_bytes());
    hash.update(bytes);
}

/// Validation snapshot of the exact qualified CPU FP32 v4 split pack.
///
/// This is not an inference session or a guarantee about future mutable disk
/// contents. A later loader must consume unchanged, verified artifact bytes.
#[derive(Debug)]
pub struct QualifiedV4Pack {
    descriptor: VersionedPackDescriptor,
    dir: PathBuf,
    identity: String,
}

/// Explicit selection to validate against a qualified pack, not an execution
/// request. There is no automatic provider, precision, or style fallback.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub struct PackSelection {
    /// Selected transcript style.
    pub style: TranscriptStyle,
    /// Recognition mode.
    pub mode: PackMode,
    /// Requested precision.
    pub precision: ModelPrecision,
    /// Explicit provider choice.
    pub provider: ExecutionProviderChoice,
    /// Input batch size.
    pub batch_size: usize,
}

impl PackSelection {
    /// Explicit CPU FP32 full-context batch-one request for the chosen style.
    pub const fn cpu_full_context(style: TranscriptStyle) -> Self {
        Self {
            style,
            mode: PackMode::FullContext,
            precision: ModelPrecision::Fp32,
            provider: ExecutionProviderChoice::Cpu,
            batch_size: 1,
        }
    }
}

/// Paths and cache identity for a validated style. The identity includes the
/// complete pack (both heads), not merely the selected encoder or head.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct V4HeadFiles {
    /// Shared acoustic encoder path.
    pub encoder: PathBuf,
    /// Selected style predictor path.
    pub decoder: PathBuf,
    /// Selected style joiner path.
    pub joint: PathBuf,
    /// Selected style SentencePiece model path.
    pub tokenizer: PathBuf,
    /// Content identity of the whole pack plus explicit style.
    pub cache_identity: String,
}

#[cfg(all(
    feature = "v4-mkl",
    target_os = "linux",
    target_arch = "x86_64",
    target_env = "gnu"
))]
pub(crate) struct VerifiedV4Graph {
    role: ArtifactRole,
    bytes: Vec<u8>,
}
#[cfg(all(
    feature = "v4-mkl",
    target_os = "linux",
    target_arch = "x86_64",
    target_env = "gnu"
))]
impl VerifiedV4Graph {
    pub(crate) fn role(&self) -> ArtifactRole {
        self.role
    }
    pub(crate) fn bytes(&self) -> &[u8] {
        &self.bytes
    }
}

impl QualifiedV4Pack {
    /// Load and verify a versioned local pack, without inference or download.
    pub fn load(dir: &Path) -> Result<Self> {
        VersionedPackDescriptor::load(dir)?.validate_in_dir(dir)
    }

    /// Immutable descriptor of the validated bytes.
    pub fn descriptor(&self) -> &VersionedPackDescriptor {
        &self.descriptor
    }

    /// Re-read the selected small tokenizer and verify the bytes passed to its
    /// parser; a validated pack snapshot alone does not protect mutable files.
    pub(crate) fn read_tokenizer_bytes(&self, style: TranscriptStyle) -> Result<Vec<u8>> {
        let role = match style {
            TranscriptStyle::Raw => ArtifactRole::RawTokenizer,
            TranscriptStyle::E2e => ArtifactRole::E2eTokenizer,
        };
        self.read_verified_artifact_bytes(role)
    }

    #[cfg(all(
        feature = "v4-mkl",
        target_os = "linux",
        target_arch = "x86_64",
        target_env = "gnu"
    ))]
    pub(crate) fn read_graph_bytes(&self, role: ArtifactRole) -> Result<VerifiedV4Graph> {
        ensure!(
            !matches!(
                role,
                ArtifactRole::RawTokenizer | ArtifactRole::E2eTokenizer
            ),
            "tokenizer role is not an ONNX graph"
        );
        Ok(VerifiedV4Graph {
            role,
            bytes: self.read_verified_artifact_bytes(role)?,
        })
    }

    fn read_verified_artifact_bytes(&self, role: ArtifactRole) -> Result<Vec<u8>> {
        let artifact = self
            .descriptor
            .artifacts
            .get(&role)
            .context("missing validated artifact")?;
        let path = self.dir.join(&artifact.file);
        let meta = std::fs::symlink_metadata(&path)?;
        ensure!(
            meta.is_file() && meta.len() == artifact.bytes,
            "artifact is not a regular file of the verified size"
        );
        let file = std::fs::File::open(&path)?;
        let meta = file.metadata()?;
        ensure!(
            meta.is_file() && meta.len() == artifact.bytes,
            "artifact changed before reading"
        );
        let mut bytes = Vec::new();
        let limit = artifact
            .bytes
            .checked_add(1)
            .context("artifact size overflow")?;
        bytes
            .try_reserve_exact(usize::try_from(limit)?)
            .context("allocate verified artifact buffer")?;
        file.take(limit).read_to_end(&mut bytes)?;
        let mut hash = Sha256::new();
        hash.update(&bytes);
        ensure!(
            bytes.len() as u64 == artifact.bytes && hex_lower(&hash.finalize()) == artifact.sha256,
            "artifact bytes changed after pack validation"
        );
        Ok(bytes)
    }

    /// Stable whole-pack content identity, separate from the legacy optimized
    /// encoder cache's source-file digest.
    pub fn identity(&self) -> &str {
        &self.identity
    }

    /// Validate supported selection and return the corresponding artifact paths.
    /// This method does not create or enable the v4 runtime.
    pub fn select(&self, request: PackSelection) -> Result<V4HeadFiles> {
        ensure!(
            request.mode == self.descriptor.mode
                && request.precision == self.descriptor.precision
                && request.provider == ExecutionProviderChoice::Cpu
                && request.batch_size == self.descriptor.abi.specification().max_batch_size,
            "unsupported v4 selection: require full-context FP32, explicit CPU, batch one"
        );
        ensure!(
            self.descriptor.styles.contains(&request.style),
            "selected style is not available"
        );
        let (decoder, joint, tokenizer) = match request.style {
            TranscriptStyle::Raw => (
                ArtifactRole::RawDecoder,
                ArtifactRole::RawJoint,
                ArtifactRole::RawTokenizer,
            ),
            TranscriptStyle::E2e => (
                ArtifactRole::E2eDecoder,
                ArtifactRole::E2eJoint,
                ArtifactRole::E2eTokenizer,
            ),
        };
        let mut hash = Sha256::new();
        hash_field(&mut hash, b"gigastt/versioned-pack/selection-v1");
        hash_field(&mut hash, self.identity.as_bytes());
        hash_field(&mut hash, request.style.as_str().as_bytes());
        let path = |role: ArtifactRole| -> Result<PathBuf> {
            let artifact = self
                .descriptor
                .artifacts
                .get(&role)
                .context("missing validated artifact")?;
            Ok(self.dir.join(&artifact.file))
        };
        Ok(V4HeadFiles {
            encoder: path(ArtifactRole::Encoder)?,
            decoder: path(decoder)?,
            joint: path(joint)?,
            tokenizer: path(tokenizer)?,
            cache_identity: hex_lower(&hash.finalize()),
        })
    }
}

#[cfg(test)]
mod tests;
