//! Recognition-head identity: filenames, checksums, HF / Release pins.

use std::path::Path;

/// HuggingFace repo hosting the RNN-T heads' shared v3 ONNX files. The
/// Multilingual CTC head ships from its own repo (see [`ModelVariant::hf_repo`]).
pub(super) const HF_REPO: &str = "istupakov/gigaam-v3-onnx";

/// Base URL of the pinned GitHub Release hosting the **pre-quantized** INT8
/// model bundle (INT8 encoder + decoder + joiner + vocab, per variant). Lets
/// integrators skip the ~844 MB FP32 encoder download AND the ~2-minute
/// on-device quantization (and need no `protoc`). The release tag pins the
/// revision; bump it together with the INT8 checksums when re-quantizing.
#[cfg(feature = "net")]
pub(super) const PREQUANT_RELEASE_BASE: &str =
    "https://github.com/ekhodzitsky/gigastt/releases/download/models-v3-2026-10-09";

/// Base URL of the pinned GitHub Release hosting the per-bucket palettized
/// **ANE** (Core ML) encoder packages, one deterministic `.tar` per mel bucket.
/// The release tag must match the `release-ane.yml` workflow's default tag; bump
/// it together with [`ANE_TAR_CHECKSUMS`] when re-converting.
#[cfg(all(feature = "net", feature = "ane"))]
pub(super) const ANE_RELEASE_BASE: &str =
    "https://github.com/ekhodzitsky/gigastt/releases/download/ane-v3-2026-06-24";

/// Mel-frame bucket ladder for the ANE encoder packages. MUST equal the convert
/// script's `--buckets` default (`scripts/convert_gigaam_ane.py`): the Rust side
/// pads each clip's mel up to the smallest bucket >= its length and runs the
/// matching fixed-window package.
#[cfg(feature = "ane")]
pub const ANE_BUCKETS: &[usize] = &[512, 768, 1536, 3000];

/// Per-bucket SHA-256 of the deterministic `.mlpackage.tar` published by
/// `release-ane.yml`. Each digest is simultaneously the content-identity
/// fingerprint and the download pin for that bucket's `.tar`.
///
/// Pinned to the `ane-v3-2026-06-24` release (built by `release-ane.yml`). On a
/// re-release, refill each entry from the printed `SHA256SUMS.txt` and bump
/// [`ANE_RELEASE_BASE`]'s tag in the same change. An empty string means "not
/// published yet" and makes `ensure_ane_packages` bail for that bucket.
#[cfg(all(feature = "net", feature = "ane"))]
pub(super) const ANE_TAR_CHECKSUMS: &[(usize, &str)] = &[
    (
        512,
        "307739d76bebe9805d36e695db030bcf4e71b0b105670609cdcbd3cdc4d4c629",
    ),
    (
        768,
        "111bd2722c46d41c0984e246752782f05892017990f50837ee6342b0dc41b5be",
    ),
    (
        1536,
        "dabb0ee21e064a79621f047c795d81f33ef95358c43157a7d242cd9a504b2e93",
    ),
    (
        3000,
        "7499327eccb326f18014c222adce11f323fbaf3ff76dea7f7c0820f9adb834d4",
    ),
];

/// HuggingFace repo hosting the optional RUPunct punctuation model (MIT).
#[cfg(feature = "net")]
pub(super) const PUNCT_HF_REPO: &str = "ekhodzitsky/rupunct-small-onnx";

/// Direct URL for the optional Silero v5 VAD model (MIT), pinned to a release
/// tag. SHA-256 below guards integrity regardless of the host.
#[cfg(feature = "net")]
pub(super) const VAD_MODEL_URL: &str =
    "https://github.com/snakers4/silero-vad/raw/v5.1.2/src/silero_vad/data/silero_vad.onnx";

/// SHA-256 of the pinned Silero v5.1.2 `silero_vad.onnx` (verified 2026-06-19).
#[cfg(feature = "net")]
pub(super) const VAD_MODEL_SHA256: &str =
    "2623a2953f6ff3d2c1e61740c6cdb7168133479b267dfef114a4a3cc5bdd788f";

/// The three files the punctuation pass needs, with their pinned SHA-256
/// checksums. Filenames mirror the `PUNCT_*` constants in [`crate::punctuation`].
/// Verified against the canonical HuggingFace copies on 2026-06-19.
#[cfg(feature = "net")]
pub(super) const PUNCT_FILES: &[(&str, &str)] = &[
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

/// Selectable GigaAM recognition head.
///
/// The RNN-T heads ship in the shared v3 HuggingFace repo (`HF_REPO`); the
/// Multilingual CTC head ships from its own repo (see [`ModelVariant::hf_repo`]).
/// All heads share the mel frontend and inference pipeline; they differ in their
/// ONNX files, vocabulary, and recognition head.
///
/// - [`ModelVariant::Rnnt`] (default): plain RNN-T head. Lower WER on the
///   golos_crowd_1k set (3.29% vs 9.65%) but emits bare lowercase Russian with
///   no punctuation / casing / ITN. Uses a 34-token character vocabulary.
/// - [`ModelVariant::E2eRnnt`]: end-to-end head with punctuation, casing, and
///   inverse text normalization baked in. Uses a 1025-token BPE vocabulary.
/// - [`ModelVariant::MlCtc`]: GigaAM Multilingual charwise-CTC head (220M),
///   encoder-only, 71-class multilingual char vocab (ru/en/kk/ky/uz), bare
///   lowercase output. Downloads istupakov's pre-quantized INT8 encoder.
///
/// Real upstream filenames are kept on disk (no canonical-prefix rename). An
/// explicit `--model-variant` (the resolved variant threaded into the engine
/// loader) selects the head; when none is given the engine auto-detects it from
/// the encoder file present in the model directory (`rnnt` precedence when more
/// than one head's files coexist).
///
/// `#[non_exhaustive]`: recognition heads are added over time (this is an
/// opt-in, additive catalog), so downstream matches must include a wildcard arm.
/// New heads are shipped as minor releases, not breaking changes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[non_exhaustive]
pub enum ModelVariant {
    /// Plain RNN-T head (default). Bare lowercase output, lower WER.
    #[default]
    Rnnt,
    /// End-to-end RNN-T head with punctuation / casing / ITN.
    E2eRnnt,
    /// GigaAM Multilingual CTC head (220M); single encoder-only ONNX, 71-class
    /// char vocab, blank id 70.
    MlCtc,
    /// GigaAM Multilingual CTC head, large (600M) encoder; same 71-class char
    /// vocab and CTC decoding as [`ModelVariant::MlCtc`], higher WER headroom.
    MlCtcLarge,
}

impl ModelVariant {
    /// All known recognition heads, in auto-detection precedence order
    /// (`Rnnt` first, mirroring the engine's default).
    pub const ALL: [ModelVariant; 4] = [
        ModelVariant::Rnnt,
        ModelVariant::E2eRnnt,
        ModelVariant::MlCtc,
        ModelVariant::MlCtcLarge,
    ];

    /// Basename of the FP32 encoder ONNX file for this variant.
    pub fn encoder_file(self) -> &'static str {
        match self {
            ModelVariant::Rnnt => "v3_rnnt_encoder.onnx",
            ModelVariant::E2eRnnt => "v3_e2e_rnnt_encoder.onnx",
            ModelVariant::MlCtc => "multilingual_ctc.onnx",
            ModelVariant::MlCtcLarge => "multilingual_large_ctc.onnx",
        }
    }

    /// Basename of the INT8 quantized encoder ONNX file. For the RNN-T heads
    /// this is generated locally by the native quantizer; for the Multilingual
    /// CTC head it is downloaded pre-quantized from HuggingFace.
    pub fn encoder_int8_file(self) -> &'static str {
        match self {
            ModelVariant::Rnnt => "v3_rnnt_encoder_int8.onnx",
            ModelVariant::E2eRnnt => "v3_e2e_rnnt_encoder_int8.onnx",
            ModelVariant::MlCtc => "multilingual_ctc.int8.onnx",
            ModelVariant::MlCtcLarge => "multilingual_large_ctc.int8.onnx",
        }
    }

    /// Basename of the decoder ONNX file for this variant.
    pub fn decoder_file(self) -> &'static str {
        match self {
            ModelVariant::Rnnt => "v3_rnnt_decoder.onnx",
            ModelVariant::E2eRnnt => "v3_e2e_rnnt_decoder.onnx",
            // CTC is encoder-only: no decoder/joiner ONNX exists. This empty
            // path is never loaded — the CTC branch in `run_inference` returns
            // before the decoder/joiner sessions are touched.
            ModelVariant::MlCtc | ModelVariant::MlCtcLarge => "",
        }
    }

    /// Basename of the joiner ONNX file for this variant.
    pub fn joint_file(self) -> &'static str {
        match self {
            ModelVariant::Rnnt => "v3_rnnt_joint.onnx",
            ModelVariant::E2eRnnt => "v3_e2e_rnnt_joint.onnx",
            // CTC is encoder-only: no decoder/joiner ONNX exists (see
            // `decoder_file`). Never loaded.
            ModelVariant::MlCtc | ModelVariant::MlCtcLarge => "",
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
            // Both CTC heads share the identical 71-token multilingual vocab.
            ModelVariant::MlCtc | ModelVariant::MlCtcLarge => "multilingual_vocab.txt",
        }
    }

    /// Files downloaded from HuggingFace for this variant. RNN-T heads ship
    /// encoder (FP32) + decoder + joiner + vocab, and the INT8 encoder is
    /// generated locally. The Multilingual CTC head is encoder-only and ships a
    /// ready-made INT8 encoder upstream, so it downloads that INT8 encoder + vocab
    /// directly (no FP32 download, no on-device quantization).
    pub fn download_files(self) -> Vec<&'static str> {
        match self {
            ModelVariant::Rnnt | ModelVariant::E2eRnnt => vec![
                self.encoder_file(),
                self.decoder_file(),
                self.joint_file(),
                self.vocab_file(),
            ],
            ModelVariant::MlCtc | ModelVariant::MlCtcLarge => {
                vec![self.encoder_int8_file(), self.vocab_file()]
            }
        }
    }

    /// HuggingFace repo hosting this variant's ONNX files. The RNN-T heads live
    /// in the shared v3 repo; the Multilingual CTC head ships from istupakov's
    /// dedicated `gigaam-multilingual-ctc-onnx` repo.
    pub fn hf_repo(self) -> &'static str {
        match self {
            ModelVariant::Rnnt | ModelVariant::E2eRnnt => HF_REPO,
            ModelVariant::MlCtc => "istupakov/gigaam-multilingual-ctc-onnx",
            ModelVariant::MlCtcLarge => "istupakov/gigaam-multilingual-large-ctc-onnx",
        }
    }

    /// Pinned SHA-256 checksum for a downloaded file, or `None` when no checksum
    /// is pinned for it. Verified against the canonical HuggingFace copies.
    pub fn checksum(self, filename: &str) -> Option<&'static str> {
        let table = match self {
            ModelVariant::Rnnt => RNNT_CHECKSUMS,
            ModelVariant::E2eRnnt => E2E_RNNT_CHECKSUMS,
            ModelVariant::MlCtc => ML_CTC_CHECKSUMS,
            ModelVariant::MlCtcLarge => ML_CTC_LARGE_CHECKSUMS,
        };
        table
            .iter()
            .find(|(name, _)| *name == filename)
            .and_then(|(_, hash)| *hash)
    }

    /// SHA-256 of the pre-quantized INT8 encoder for this variant. For the RNN-T
    /// heads this is gigastt's own quantizer output, published in the pinned
    /// GitHub Release (`PREQUANT_RELEASE_BASE`); bump it together with the release
    /// tag on re-quantization. For the Multilingual CTC head it is istupakov's
    /// upstream INT8 encoder, downloaded directly from HuggingFace.
    pub fn encoder_int8_checksum(self) -> &'static str {
        match self {
            ModelVariant::Rnnt => {
                "1d5a6f580b692e38ba35ed98d0f54648808f27b0a147dc11bd8978c15dc1a18f"
            }
            ModelVariant::E2eRnnt => {
                "16805903b102d8b27bb0253de044f25e33a8a5bf07fce0c4da4ac1a5a7544091"
            }
            // Downloaded pre-quantized from istupakov's HuggingFace repo (not our
            // GitHub Release); this is the SHA-256 of `multilingual_ctc.int8.onnx`.
            ModelVariant::MlCtc => {
                "e08e27ae5669b39f0c378fae101bbbb9a80505f74f9b66719c309bf5b894a480"
            }
            // SHA-256 of `multilingual_large_ctc.int8.onnx`, from
            // istupakov/gigaam-multilingual-large-ctc-onnx.
            ModelVariant::MlCtcLarge => {
                "b2ad9c38fc04197ba758105d33f7404fd13d977958722e0f49e3f3e22521f1c6"
            }
        }
    }

    /// Files in the pre-quantized bundle published on GitHub Releases: the INT8
    /// encoder (no FP32 download, no on-device quantization) plus the decoder,
    /// joiner, and vocab. The engine runs from these alone — it prefers the INT8
    /// encoder when present.
    pub fn prequantized_files(self) -> Vec<&'static str> {
        match self {
            ModelVariant::Rnnt | ModelVariant::E2eRnnt => vec![
                self.encoder_int8_file(),
                self.decoder_file(),
                self.joint_file(),
                self.vocab_file(),
            ],
            ModelVariant::MlCtc | ModelVariant::MlCtcLarge => {
                vec![self.encoder_int8_file(), self.vocab_file()]
            }
        }
    }

    /// Pinned SHA-256 for a pre-quantized bundle file. The INT8 encoder uses
    /// [`ModelVariant::encoder_int8_checksum`]; the decoder/joiner/vocab are
    /// byte-identical to the FP32 download set, so they reuse
    /// [`ModelVariant::checksum`].
    pub fn prequantized_checksum(self, filename: &str) -> Option<&'static str> {
        if filename == self.encoder_int8_file() {
            Some(self.encoder_int8_checksum())
        } else {
            self.checksum(filename)
        }
    }

    /// Detect which variant's files are present in `dir` by probing for the
    /// encoder file (FP32 or generated INT8). Returns `None` when neither
    /// variant's encoder is present. `Rnnt` takes precedence when several
    /// heads' encoders coexist, mirroring the engine's default.
    pub fn detect_in_dir(dir: &Path) -> Option<Self> {
        Self::ALL.into_iter().find(|&variant| {
            dir.join(variant.encoder_file()).exists()
                || dir.join(variant.encoder_int8_file()).exists()
        })
    }

    /// Stable model identifier surfaced by the REST API (`/health`,
    /// `/v1/models`). Distinguishes the two heads so a client can tell which
    /// one is actually loaded instead of always seeing the e2e id.
    pub fn model_id(self) -> &'static str {
        match self {
            ModelVariant::Rnnt => "gigaam-v3-rnnt",
            ModelVariant::E2eRnnt => "gigaam-v3-e2e-rnnt",
            ModelVariant::MlCtc => "gigaam-multilingual-ctc",
            ModelVariant::MlCtcLarge => "gigaam-multilingual-large-ctc",
        }
    }

    /// Short variant token (`rnnt` / `e2e_rnnt`) — the value accepted by
    /// `--model-variant` and echoed in the REST `variant` field.
    pub fn as_str(self) -> &'static str {
        match self {
            ModelVariant::Rnnt => "rnnt",
            ModelVariant::E2eRnnt => "e2e_rnnt",
            ModelVariant::MlCtc => "ml_ctc",
            ModelVariant::MlCtcLarge => "ml_ctc_large",
        }
    }

    /// Human-readable model name for `/v1/models`.
    pub fn display_name(self) -> &'static str {
        match self {
            ModelVariant::Rnnt => "GigaAM v3 RNN-T",
            ModelVariant::E2eRnnt => "GigaAM v3 E2E RNN-T",
            ModelVariant::MlCtc => "GigaAM Multilingual CTC",
            ModelVariant::MlCtcLarge => "GigaAM Multilingual CTC (large)",
        }
    }

    /// True for the encoder-only CTC heads (greedy CTC decode, no prediction
    /// network / joiner). Both the 220M and 600M Multilingual heads are CTC.
    pub fn is_ctc(self) -> bool {
        matches!(self, ModelVariant::MlCtc | ModelVariant::MlCtcLarge)
    }
}

impl std::str::FromStr for ModelVariant {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.trim().to_ascii_lowercase().as_str() {
            "rnnt" => Ok(ModelVariant::Rnnt),
            "e2e_rnnt" | "e2e-rnnt" => Ok(ModelVariant::E2eRnnt),
            "ml_ctc" | "ml-ctc" => Ok(ModelVariant::MlCtc),
            "ml_ctc_large" | "ml-ctc-large" => Ok(ModelVariant::MlCtcLarge),
            other => Err(format!(
                "unknown model variant '{other}' \
                 (expected 'rnnt', 'e2e_rnnt', 'ml_ctc', or 'ml_ctc_large')"
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

/// SHA-256 checksums for the GigaAM Multilingual CTC head's downloaded files
/// (the pre-quantized INT8 encoder + vocab; it is encoder-only). Computed from
/// the canonical copies at `istupakov/gigaam-multilingual-ctc-onnx` on
/// 2026-07-17.
const ML_CTC_CHECKSUMS: &[(&str, Option<&str>)] = &[
    (
        "multilingual_ctc.int8.onnx",
        Some("e08e27ae5669b39f0c378fae101bbbb9a80505f74f9b66719c309bf5b894a480"),
    ),
    (
        "multilingual_vocab.txt",
        Some("4d130287892e1099fedfb3f93c4b4cf8a263151158801680b28977d1be4133f4"),
    ),
];

/// SHA-256 checksums for the GigaAM Multilingual CTC *large* (600M) head's
/// downloaded files (pre-quantized INT8 encoder + the shared vocab, which is
/// byte-identical to the 220M head's). Computed from the canonical copies at
/// `istupakov/gigaam-multilingual-large-ctc-onnx` on 2026-07-17.
const ML_CTC_LARGE_CHECKSUMS: &[(&str, Option<&str>)] = &[
    (
        "multilingual_large_ctc.int8.onnx",
        Some("b2ad9c38fc04197ba758105d33f7404fd13d977958722e0f49e3f3e22521f1c6"),
    ),
    (
        "multilingual_vocab.txt",
        Some("4d130287892e1099fedfb3f93c4b4cf8a263151158801680b28977d1be4133f4"),
    ),
];

#[cfg(feature = "diarization")]
pub(super) const SPEAKER_HF_REPO: &str = "onnx-community/wespeaker-voxceleb-resnet34-LM";
#[cfg(feature = "diarization")]
pub const SPEAKER_MODEL_FILE: &str = "wespeaker_resnet34.onnx";

/// SHA-256 of the upstream speaker-diarization model (`onnx/model.onnx` at
/// `onnx-community/wespeaker-voxceleb-resnet34-LM`, 26 535 549 bytes).
/// Verified against the canonical HuggingFace copy on 2026-04-20; if the
/// upstream model is ever rotated, update this constant alongside the
/// SPEAKER_MODEL_FILE bump.
#[cfg(feature = "diarization")]
pub(super) const SPEAKER_MODEL_SHA256: &str =
    "3955447b0499dc9e0a4541a895df08b03c69098eba4e56c02b5603e9f7f4fcbb";
