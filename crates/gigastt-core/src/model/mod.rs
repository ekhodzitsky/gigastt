//! Model download and management.
//!
//! Downloads GigaAM v3 RNN-T ONNX files from HuggingFace to `~/.gigastt/models/`.
//! Two recognition heads are selectable via [`ModelVariant`]: the plain `rnnt`
//! head (default — lower WER, bare lowercase output) and the `e2e_rnnt` head
//! (punctuation / casing / ITN baked in).

#[cfg(feature = "net")]
use anyhow::Context;
use anyhow::Result;
#[cfg(feature = "net")]
use futures_util::StreamExt;
#[cfg(feature = "net")]
use sha2::{Digest, Sha256};
use std::path::Path;
use std::sync::atomic::{AtomicU8, Ordering};
#[cfg(feature = "net")]
use tokio::io::AsyncWriteExt;

#[cfg(unix)]
#[cfg(feature = "net")]
use std::os::fd::AsRawFd;

/// Progress reporting mode for `gigastt download` (and any other caller that
/// sets it process-wide): `Human` keeps the interactive `\r` stderr reporter,
/// `Json` emits NDJSON events — one [`ProgressEvent`] per line — on stdout
/// for sidecar integrators.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[non_exhaustive]
pub enum ProgressMode {
    /// Interactive human reporter: a `\r`-redrawn percentage line on stderr.
    #[default]
    Human,
    /// Machine-readable NDJSON events on stdout; nothing else may write there.
    Json,
}

impl ProgressMode {
    /// Stable token accepted by `--progress` / `GIGASTT_DOWNLOAD_PROGRESS`.
    pub fn as_str(self) -> &'static str {
        match self {
            ProgressMode::Human => "human",
            ProgressMode::Json => "json",
        }
    }
}

impl std::str::FromStr for ProgressMode {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.trim().to_ascii_lowercase().as_str() {
            "human" => Ok(ProgressMode::Human),
            "json" => Ok(ProgressMode::Json),
            other => Err(format!(
                "unknown progress mode '{other}' (expected 'human' or 'json')"
            )),
        }
    }
}

/// Failure category surfaced in the NDJSON `error` event and mapped to the
/// documented `gigastt download` exit-code taxonomy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum ProgressErrorKind {
    /// Network failure: unreachable host, TLS, broken stream, HTTP error status.
    Network,
    /// Local filesystem failure: create/write/rename of a staging or model file.
    Disk,
    /// SHA-256 mismatch on a staged download (corrupt or tampered artefact).
    Checksum,
    /// Cancelled by the operator (SIGINT / Ctrl-C).
    Interrupted,
    /// Anything else; keeps the four primary kinds stable to match on.
    Other,
}

impl ProgressErrorKind {
    /// `gigastt download` process exit code for this failure category. `0` is
    /// success; every category keeps the historical `!= 0` failure contract.
    pub fn exit_code(self) -> i32 {
        // BSD `sysexits`-flavored codes, deliberately avoiding 2: clap exits 2
        // on argument/usage errors (before any event can be emitted), and an
        // integrator keying retries off "network" must be able to tell the two
        // apart.
        match self {
            // EX_UNAVAILABLE: the remote end could not be reached / served.
            ProgressErrorKind::Network => 69,
            // EX_IOERR: local create/write/rename failure.
            ProgressErrorKind::Disk => 74,
            // EX_DATAERR: SHA-256 mismatch on a staged download.
            ProgressErrorKind::Checksum => 65,
            // Conventional 128 + SIGINT.
            ProgressErrorKind::Interrupted => 130,
            // Historical generic failure code (anyhow's `Termination`).
            ProgressErrorKind::Other => 1,
        }
    }
}

/// Machine-readable `gigastt download` progress event, serialized as a single
/// NDJSON line on stdout when [`ProgressMode::Json`] is active. One line = one
/// event; the `phase` tag is the discriminator integrators match on.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize)]
#[serde(tag = "phase", rename_all = "snake_case")]
#[non_exhaustive]
pub enum ProgressEvent {
    /// Byte progress of one file's download (throttled to ~200 ms per file,
    /// plus an unconditional event at 100%).
    Download {
        file: String,
        bytes_done: u64,
        bytes_total: u64,
    },
    /// The on-device INT8 quantization pass started for `file` (~2 min, no
    /// byte progress — its presence tells a sidecar the CLI is busy, not hung).
    Quantize { file: String },
    /// SHA-256 verification of a staged download started.
    Verify { file: String },
    /// All requested artefacts are ready; emitted once, last.
    Done { model_dir: String },
    /// Fatal failure, emitted right before the non-zero exit.
    Error {
        kind: ProgressErrorKind,
        message: String,
    },
}

impl ProgressEvent {
    /// Serialize as one NDJSON line (no trailing newline). This POD enum
    /// cannot realistically fail serialization; a minimal valid `error`
    /// object is the fallback rather than panicking on a progress path.
    pub fn to_ndjson(&self) -> String {
        serde_json::to_string(self).unwrap_or_else(|_| {
            "{\"phase\":\"error\",\"kind\":\"other\",\"message\":\"progress event serialization failed\"}"
                .to_string()
        })
    }
}

/// Process-wide download progress mode, set once by the CLI before any
/// download call. Library users get the `Human` default (identical to the
/// historical `\r` reporter) unless they opt into JSON events.
static PROGRESS_MODE: AtomicU8 = AtomicU8::new(ProgressMode::Human as u8);

/// Set the process-wide download progress mode. Call once at startup, before
/// any `ensure_*` download function runs.
pub fn set_progress_mode(mode: ProgressMode) {
    PROGRESS_MODE.store(mode as u8, Ordering::Relaxed);
}

/// The process-wide download progress mode (`Human` unless set).
pub fn progress_mode() -> ProgressMode {
    match PROGRESS_MODE.load(Ordering::Relaxed) {
        1 => ProgressMode::Json,
        _ => ProgressMode::Human,
    }
}

/// Emit `event` as one NDJSON line on stdout — but only in
/// [`ProgressMode::Json`]; in `Human` mode this is a no-op so call sites never
/// branch on the mode. Public because phases only the CLI can see (quantize
/// entry, terminal done/error) are emitted from the binary crate.
///
/// Write failures (e.g. the reader closed the pipe) are ignored: progress
/// reporting must never take down an in-flight download.
pub fn emit_progress_event(event: &ProgressEvent) {
    if progress_mode() != ProgressMode::Json {
        return;
    }
    use std::io::Write;
    let stdout = std::io::stdout();
    let mut lock = stdout.lock();
    let _ = writeln!(lock, "{}", event.to_ndjson());
    let _ = lock.flush();
}

/// Classify a failed download for the NDJSON `error` event / exit-code
/// taxonomy. Typed root causes win (reqwest → network, io → disk); the
/// remaining cases are recognized by the stable messages of the two
/// `anyhow::bail!` sites in this module (SHA-256 mismatch, HTTP error status).
#[cfg(feature = "net")]
pub fn classify_download_error(err: &anyhow::Error) -> ProgressErrorKind {
    for cause in err.chain() {
        if cause.downcast_ref::<reqwest::Error>().is_some() {
            return ProgressErrorKind::Network;
        }
        if cause.downcast_ref::<std::io::Error>().is_some() {
            return ProgressErrorKind::Disk;
        }
    }
    let msg = format!("{err:#}");
    if msg.contains("SHA-256 mismatch") {
        return ProgressErrorKind::Checksum;
    }
    // `stream_to_partial_then_finalize` bails with "…: HTTP <status>" on a
    // non-2xx response; that is a network-class failure for integrators.
    if msg.contains("HTTP ") {
        return ProgressErrorKind::Network;
    }
    ProgressErrorKind::Other
}

/// Throttle for NDJSON `download` events: at most one per 200 ms per file,
/// plus an unconditional event at 100% so integrators always see completion.
#[cfg(feature = "net")]
const JSON_PROGRESS_THROTTLE: std::time::Duration = std::time::Duration::from_millis(200);

/// Where progress output goes. `Human` renders the legacy `\r` stderr line;
/// `Json` routes [`ProgressEvent`]s to the emitter (process stdout in the CLI,
/// a captured buffer in tests).
#[cfg(feature = "net")]
struct ProgressSink {
    mode: ProgressMode,
    emit: Box<dyn Fn(&ProgressEvent) + Send + Sync>,
}

#[cfg(feature = "net")]
impl ProgressSink {
    /// Sink honouring the process-wide [`ProgressMode`] set by the CLI.
    fn global() -> Self {
        Self {
            mode: progress_mode(),
            emit: Box::new(emit_progress_event),
        }
    }

    /// Human-mode sink for tests that exercise the legacy renderer.
    #[cfg(test)]
    fn human() -> Self {
        Self {
            mode: ProgressMode::Human,
            emit: Box::new(|_| {}),
        }
    }

    /// Capturing Json-mode sink: returns the sink plus the shared event log.
    #[cfg(test)]
    fn capturing() -> (Self, std::sync::Arc<std::sync::Mutex<Vec<ProgressEvent>>>) {
        let log = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        let sink_log = std::sync::Arc::clone(&log);
        (
            Self {
                mode: ProgressMode::Json,
                emit: Box::new(move |e| {
                    if let Ok(mut guard) = sink_log.lock() {
                        guard.push(e.clone());
                    }
                }),
            },
            log,
        )
    }

    fn event(&self, event: &ProgressEvent) {
        (self.emit)(event);
    }
}

/// Per-file download progress reporter. Human mode keeps the historical
/// stderr `\r` rendering byte-for-byte; Json mode emits throttled NDJSON
/// `download` events through the sink (first chunk immediately, then at most
/// one per [`JSON_PROGRESS_THROTTLE`], and always exactly one at 100%).
#[cfg(feature = "net")]
struct DownloadProgress {
    total: u64,
    current: u64,
    last_percent: u8,
    last_json_emit: Option<std::time::Instant>,
    json_final_emitted: bool,
}

#[cfg(feature = "net")]
impl DownloadProgress {
    fn new(total: u64) -> Self {
        Self {
            total,
            current: 0,
            last_percent: 0,
            last_json_emit: None,
            json_final_emitted: false,
        }
    }

    /// The legacy human line for the current state, or `None` when the
    /// percentage did not move (the historical throttle).
    fn human_tick(&mut self) -> Option<String> {
        let percent = (self.current * 100)
            .checked_div(self.total)
            .map(|p| p as u8)
            .unwrap_or(0);
        if percent == self.last_percent {
            return None;
        }
        self.last_percent = percent;
        Some(format!(
            "\rDownloading... {percent}% ({:.1}MB / {:.1}MB)",
            self.current as f64 / 1_048_576.0,
            self.total as f64 / 1_048_576.0
        ))
    }

    /// The legacy human completion line (redraws over the progress line).
    fn human_finish(&self) -> String {
        format!(
            "\rDownload complete ({:.1}MB)                    ",
            self.current as f64 / 1_048_576.0
        )
    }

    fn update(&mut self, bytes: u64, sink: &ProgressSink, label: &str) {
        self.current += bytes;
        match sink.mode {
            ProgressMode::Human => {
                if let Some(line) = self.human_tick() {
                    eprint!("{line}");
                }
            }
            ProgressMode::Json => {
                let complete = self.total > 0 && self.current >= self.total;
                let due = self
                    .last_json_emit
                    .is_none_or(|t| t.elapsed() >= JSON_PROGRESS_THROTTLE);
                if (complete && !self.json_final_emitted) || (due && !complete) {
                    sink.event(&ProgressEvent::Download {
                        file: label.to_string(),
                        bytes_done: self.current,
                        bytes_total: self.total,
                    });
                    self.last_json_emit = Some(std::time::Instant::now());
                    if complete {
                        self.json_final_emitted = true;
                    }
                }
            }
        }
    }

    fn finish(&mut self, sink: &ProgressSink, label: &str) {
        match sink.mode {
            ProgressMode::Human => eprintln!("{}", self.human_finish()),
            ProgressMode::Json => {
                // An unknown (chunked) total never hits the 100% branch in
                // `update`; close the file out with exactly one final event.
                if !self.json_final_emitted {
                    self.json_final_emitted = true;
                    sink.event(&ProgressEvent::Download {
                        file: label.to_string(),
                        bytes_done: self.current,
                        bytes_total: self.total,
                    });
                }
            }
        }
    }
}

/// HuggingFace repo hosting the RNN-T heads' shared v3 ONNX files. The
/// Multilingual CTC head ships from its own repo (see [`ModelVariant::hf_repo`]).
const HF_REPO: &str = "istupakov/gigaam-v3-onnx";

/// Base URL of the pinned GitHub Release hosting the **pre-quantized** INT8
/// model bundle (INT8 encoder + decoder + joiner + vocab, per variant). Lets
/// integrators skip the ~844 MB FP32 encoder download AND the ~2-minute
/// on-device quantization (and need no `protoc`). The release tag pins the
/// revision; bump it together with the INT8 checksums when re-quantizing.
#[cfg(feature = "net")]
const PREQUANT_RELEASE_BASE: &str =
    "https://github.com/ekhodzitsky/gigastt/releases/download/models-v3-2026-06-22";

/// Base URL of the pinned GitHub Release hosting the per-bucket palettized
/// **ANE** (Core ML) encoder packages, one deterministic `.tar` per mel bucket.
/// The release tag must match the `release-ane.yml` workflow's default tag; bump
/// it together with [`ANE_TAR_CHECKSUMS`] when re-converting.
#[cfg(all(feature = "net", feature = "ane"))]
const ANE_RELEASE_BASE: &str =
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
const ANE_TAR_CHECKSUMS: &[(usize, &str)] = &[
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
const PUNCT_HF_REPO: &str = "ekhodzitsky/rupunct-small-onnx";

/// Direct URL for the optional Silero v5 VAD model (MIT), pinned to a release
/// tag. SHA-256 below guards integrity regardless of the host.
#[cfg(feature = "net")]
const VAD_MODEL_URL: &str =
    "https://github.com/snakers4/silero-vad/raw/v5.1.2/src/silero_vad/data/silero_vad.onnx";

/// SHA-256 of the pinned Silero v5.1.2 `silero_vad.onnx` (verified 2026-06-19).
#[cfg(feature = "net")]
const VAD_MODEL_SHA256: &str = "2623a2953f6ff3d2c1e61740c6cdb7168133479b267dfef114a4a3cc5bdd788f";

/// The three files the punctuation pass needs, with their pinned SHA-256
/// checksums. Filenames mirror the `PUNCT_*` constants in [`crate::punctuation`].
/// Verified against the canonical HuggingFace copies on 2026-06-19.
#[cfg(feature = "net")]
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
                "c52665e9d96c4ca3a153c063d2ee9af6c567fe2975ca50fd038b75bbf2f60e7f"
            }
            ModelVariant::E2eRnnt => {
                "cf51b300af47cea099e17c806f8fecce2c46e9e8deb4709ec203f8970a067389"
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
    /// variant's encoder is present. `Rnnt` takes precedence when (anomalously)
    /// both encoders coexist, mirroring the engine's default.
    pub fn detect_in_dir(dir: &Path) -> Option<Self> {
        [
            ModelVariant::Rnnt,
            ModelVariant::E2eRnnt,
            ModelVariant::MlCtc,
            ModelVariant::MlCtcLarge,
        ]
        .into_iter()
        .find(|&variant| {
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
#[cfg(feature = "net")]
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
#[cfg(feature = "net")]
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
#[cfg(feature = "net")]
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

/// Ensure the **pre-quantized** INT8 model bundle for `requested` (or the
/// variant already on disk, else the default `Rnnt`) exists in `model_dir`,
/// downloading it from the pinned GitHub Release if missing.
///
/// This is the lean integration path: it fetches the INT8 encoder + decoder +
/// joiner + vocab directly, so integrators skip the ~844 MB FP32 encoder
/// download AND the ~2-minute on-device quantization (and need no `protoc`).
/// Each file is SHA-256-verified and atomically renamed, reusing the same
/// download primitive as [`ensure_model_variant`].
///
/// If a usable model (pre-quantized OR FP32 set) is already present, it is used
/// as-is and no network request is made. Returns the ready variant.
#[cfg(feature = "net")]
pub async fn ensure_prequantized_model_variant(
    requested: Option<ModelVariant>,
    model_dir: &str,
) -> Result<ModelVariant> {
    let dir = Path::new(model_dir);
    let variant = requested
        .or_else(|| ModelVariant::detect_in_dir(dir))
        .unwrap_or_default();

    if is_prequantized_present(variant, dir) || is_model_present(variant, dir) {
        tracing::info!("Using existing {variant:?} model at {model_dir}");
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

/// Directory name of the unpacked `.mlpackage` for a given mel bucket.
#[cfg(feature = "ane")]
pub fn ane_package_dir_name(bucket: usize) -> String {
    format!("gigaam_v3_encoder_{bucket}.mlpackage")
}

/// Filename of the published `.tar` artifact for a given mel bucket.
#[cfg(all(feature = "net", feature = "ane"))]
fn ane_tar_name(bucket: usize) -> String {
    format!("{}.tar", ane_package_dir_name(bucket))
}

/// Pinned `.tar` SHA-256 for `bucket`, or `None` when unreleased (the empty
/// sentinel in [`ANE_TAR_CHECKSUMS`]).
#[cfg(all(feature = "net", feature = "ane"))]
fn ane_tar_checksum(bucket: usize) -> Option<&'static str> {
    ANE_TAR_CHECKSUMS
        .iter()
        .find(|(b, _)| *b == bucket)
        .and_then(|(_, sum)| if sum.is_empty() { None } else { Some(*sum) })
}

/// Return the default ANE-model directory (`~/.gigastt/models/ane/`), a sibling
/// of [`default_model_dir`].
///
/// Holds the per-bucket palettized Core ML encoder packages the macOS ANE
/// backend runs. The packages auto-download via [`ensure_ane_packages`] when the
/// ANE path is requested (`gigastt download --ane`).
#[cfg(feature = "ane")]
pub fn default_ane_model_dir() -> String {
    home_dir()
        .map(|h| {
            h.join(".gigastt")
                .join("models")
                .join("ane")
                .to_string_lossy()
                .into_owned()
        })
        .unwrap_or_else(|| ".gigastt/models/ane".into())
}

/// True when `pkg_dir` is a fully-formed Core ML `.mlpackage` directory.
///
/// Requires every structurally-load-bearing member, not just the
/// `Manifest.json` marker: the manifest, the serialized model spec, and the
/// weights blob. A package missing any of these cannot load on the ANE, so
/// treating it as "present" would wedge the download path forever. Observed
/// layout (real published package, all three buckets identical):
///   `Manifest.json`
///   `Data/com.apple.CoreML/model.mlmodel`
///   `Data/com.apple.CoreML/weights/weight.bin`
#[cfg(feature = "ane")]
pub fn ane_package_complete(pkg_dir: &Path) -> bool {
    pkg_dir.is_dir()
        && pkg_dir.join("Manifest.json").is_file()
        && pkg_dir
            .join("Data")
            .join("com.apple.CoreML")
            .join("model.mlmodel")
            .is_file()
        && pkg_dir
            .join("Data")
            .join("com.apple.CoreML")
            .join("weights")
            .join("weight.bin")
            .is_file()
}

/// True when every bucket's unpacked `.mlpackage` is present and complete in
/// `dir` (see [`ane_package_complete`] for the structural requirements).
#[cfg(feature = "ane")]
pub fn is_ane_present(dir: &Path) -> bool {
    ANE_BUCKETS
        .iter()
        .all(|&b| ane_package_complete(&dir.join(ane_package_dir_name(b))))
}

/// Ensure the per-bucket ANE Core ML encoder packages exist in `model_dir`,
/// downloading each bucket's deterministic `.tar` from the pinned GitHub Release
/// and unpacking it to reconstruct the `.mlpackage` directory.
///
/// Each `.tar` is SHA-256-verified against [`ANE_TAR_CHECKSUMS`] (one digest =
/// content identity = download pin) before being unpacked with the `tar` crate's
/// default path-traversal guard, then the `.tar` is removed. Buckets whose
/// `.mlpackage` is already present are skipped. Reuses the same streaming
/// download + atomic-rename + lock infra as [`ensure_prequantized_model_variant`].
///
/// Bails with a clear message when the release is not yet published (sentinel
/// checksums).
#[cfg(all(feature = "net", feature = "ane"))]
pub async fn ensure_ane_packages(model_dir: &str) -> Result<()> {
    let dir = Path::new(model_dir);

    if is_ane_present(dir) {
        tracing::info!("ANE encoder packages found at {model_dir}");
        return Ok(());
    }

    std::fs::create_dir_all(dir).context("Failed to create ANE model directory")?;

    #[cfg(unix)]
    let _lock = acquire_download_lock(dir)?;

    // Re-check after acquiring the lock in case another process finished.
    if is_ane_present(dir) {
        tracing::info!("ANE encoder packages found at {model_dir} after lock");
        return Ok(());
    }

    tracing::info!("Downloading ANE encoder packages from {ANE_RELEASE_BASE}...");

    for &bucket in ANE_BUCKETS {
        let pkg_name = ane_package_dir_name(bucket);
        if ane_package_complete(&dir.join(&pkg_name)) {
            continue;
        }
        let checksum = require_ane_tar_checksum(bucket)?;
        let tar_name = ane_tar_name(bucket);
        let tar_dest = dir.join(&tar_name);
        let url = format!("{ANE_RELEASE_BASE}/{tar_name}");
        stream_to_partial_then_finalize(&url, &tar_dest, Some(checksum), &tar_name).await?;

        // Extract atomically: unpack into a unique staging dir on the SAME
        // filesystem, then rename the reconstructed package into place so the
        // present-check only ever observes a fully-formed `.mlpackage`. A torn
        // unpack (disk-full / SIGKILL) leaves only the staging dir + the `.tar`,
        // both of which we remove on every error path so a retry starts clean.
        tracing::info!("Unpacking {tar_name} into {model_dir}");
        if let Err(e) = extract_ane_tar_atomic(&tar_dest, dir, &pkg_name) {
            let _ = std::fs::remove_file(&tar_dest);
            return Err(e);
        }
        // `.tar` removed only AFTER a successful rename, so a failed run above
        // retains it for retry.
        std::fs::remove_file(&tar_dest)
            .with_context(|| format!("Failed to remove {}", tar_dest.display()))?;
    }

    tracing::info!("ANE encoder packages download complete");
    Ok(())
}

/// Unpack `tar_dest` into a unique staging dir under `dir` (same filesystem →
/// atomic rename), then move the reconstructed `<pkg_name>` package into
/// `dir/<pkg_name>` with a single `rename`. The package only ever appears at
/// its final path fully-formed.
///
/// The deterministic `.tar`'s arcnames are prefixed with `<pkg_name>/`, so the
/// reconstructed package lands at `staging/<pkg_name>`. `tar::Archive::unpack`
/// keeps its default path-traversal guard (entries escaping the target are
/// rejected). On any failure the staging dir is removed before bailing so a
/// retry starts clean; the caller removes the `.tar`.
#[cfg(all(feature = "net", feature = "ane"))]
fn extract_ane_tar_atomic(tar_dest: &Path, dir: &Path, pkg_name: &str) -> Result<()> {
    // Unique per-process staging dir, same pid+nanos scheme as
    // `partial_path_unique`, kept under `dir` so the final rename is atomic.
    let stamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    let staging = dir.join(format!(".extract.{}.{}", std::process::id(), stamp));

    // Best-effort staging cleanup before any `?`-bail.
    let cleanup_staging = || {
        let _ = std::fs::remove_dir_all(&staging);
    };

    if let Err(e) = std::fs::create_dir_all(&staging)
        .with_context(|| format!("Failed to create staging dir {}", staging.display()))
    {
        cleanup_staging();
        return Err(e);
    }

    let unpack = (|| -> Result<()> {
        let tar_file = std::fs::File::open(tar_dest)
            .with_context(|| format!("Failed to open {}", tar_dest.display()))?;
        tar::Archive::new(tar_file)
            .unpack(&staging)
            .with_context(|| format!("Failed to unpack {}", tar_dest.display()))?;

        let src = staging.join(pkg_name);
        let dest = dir.join(pkg_name);
        // A torn package from a prior aborted run (the strengthened present-check
        // now rejects it) must be cleared before rename, or `rename` fails with
        // "directory not empty".
        if dest.exists() {
            std::fs::remove_dir_all(&dest)
                .with_context(|| format!("Failed to remove stale {}", dest.display()))?;
        }
        std::fs::rename(&src, &dest)
            .with_context(|| format!("Failed to rename {} -> {}", src.display(), dest.display()))?;
        Ok(())
    })();

    cleanup_staging();
    unpack
}

/// Resolve the pinned `.tar` checksum for `bucket`, bailing with the
/// not-yet-published message when it is a sentinel. Factored out so the
/// sentinel-bail branch is unit-testable without the network / async path.
#[cfg(all(feature = "net", feature = "ane"))]
fn require_ane_tar_checksum(bucket: usize) -> Result<&'static str> {
    ane_tar_checksum(bucket).ok_or_else(|| {
        anyhow::anyhow!(
            "ANE encoder release not yet published; run the Release ANE workflow \
             (release-ane.yml), then pin the per-bucket .tar SHA-256 from \
             SHA256SUMS.txt in ANE_TAR_CHECKSUMS"
        )
    })
}

/// Ensure the speaker diarization model exists in `model_dir`, downloading from HuggingFace if missing.
///
/// Downloads `wespeaker_resnet34.onnx` from `onnx-community/wespeaker-voxceleb-resnet34-LM`
/// into `<model_dir>/wespeaker_resnet34.onnx.partial`, verifies its SHA-256 against
/// `SPEAKER_MODEL_SHA256`, and atomically renames it into place. On checksum mismatch or
/// crash the final path is never observable, so a subsequent `ensure_speaker_model` call
/// will re-download from scratch rather than loading a tampered model.
#[cfg(feature = "diarization")]
#[cfg(feature = "net")]
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
#[cfg(feature = "net")]
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
#[cfg(feature = "net")]
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

/// True when every file in `variant`'s pre-quantized bundle (INT8 encoder,
/// decoder, joiner, vocab) is present in `dir`. The engine runs from this set
/// alone — no FP32 encoder required.
pub fn is_prequantized_present(variant: ModelVariant, dir: &Path) -> bool {
    variant
        .prequantized_files()
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
#[cfg(feature = "net")]
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
#[cfg(feature = "net")]
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
#[cfg(feature = "net")]
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

#[cfg(feature = "net")]
async fn download_file(variant: ModelVariant, filename: &str, dir: &Path) -> Result<()> {
    let url = format!(
        "https://huggingface.co/{}/resolve/main/{filename}",
        variant.hf_repo()
    );
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
#[cfg(feature = "net")]
async fn stream_to_partial_then_finalize(
    url: &str,
    final_dest: &Path,
    expected_sha256: Option<&str>,
    label: &str,
) -> Result<()> {
    stream_to_partial_then_finalize_with_sink(
        url,
        final_dest,
        expected_sha256,
        label,
        &ProgressSink::global(),
    )
    .await
}

/// Sink-parameterized core of [`stream_to_partial_then_finalize`] so tests
/// can capture the emitted [`ProgressEvent`]s instead of parsing process
/// stdout.
#[cfg(feature = "net")]
async fn stream_to_partial_then_finalize_with_sink(
    url: &str,
    final_dest: &Path,
    expected_sha256: Option<&str>,
    label: &str,
    sink: &ProgressSink,
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
        progress.update(chunk.len() as u64, sink, label);
    }

    file.flush().await?;
    drop(file);
    progress.finish(sink, label);
    tracing::info!("Wrote partial {} ({downloaded} bytes)", partial.display());

    if expected_sha256.is_some() {
        sink.event(&ProgressEvent::Verify {
            file: label.to_string(),
        });
    }
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
        let sink = ProgressSink::human();
        let mut progress = DownloadProgress::new(1_000_000);
        // Should not panic on normal update.
        progress.update(500_000, &sink, "model.onnx");
        assert_eq!(progress.current, 500_000);
        assert_eq!(progress.last_percent, 50);
        progress.finish(&sink, "model.onnx");
    }

    #[test]
    fn test_download_progress_zero_total() {
        let sink = ProgressSink::human();
        let mut progress = DownloadProgress::new(0);
        // Must not divide by zero.
        progress.update(100, &sink, "model.onnx");
        assert_eq!(progress.last_percent, 0);
        progress.finish(&sink, "model.onnx");
    }

    // ── progress events (NDJSON / human sink) ───────────────────────────────

    #[test]
    fn test_progress_mode_from_str() {
        use std::str::FromStr;
        assert_eq!(
            ProgressMode::from_str("human").unwrap(),
            ProgressMode::Human
        );
        assert_eq!(ProgressMode::from_str("json").unwrap(), ProgressMode::Json);
        assert_eq!(
            ProgressMode::from_str(" JSON ").unwrap(),
            ProgressMode::Json
        );
        assert!(ProgressMode::from_str("xml").is_err());
        assert_eq!(ProgressMode::default(), ProgressMode::Human);
        assert_eq!(ProgressMode::Json.as_str(), "json");
    }

    /// The NDJSON wire shape is the integrator contract: one line per event,
    /// `phase` as the discriminator, exactly the fields sidecars match on.
    #[test]
    fn test_progress_event_ndjson_schema() {
        let cases: Vec<(ProgressEvent, &str)> = vec![
            (
                ProgressEvent::Download {
                    file: "v3_rnnt_encoder.onnx".to_string(),
                    bytes_done: 50,
                    bytes_total: 100,
                },
                "{\"phase\":\"download\",\"file\":\"v3_rnnt_encoder.onnx\",\"bytes_done\":50,\"bytes_total\":100}",
            ),
            (
                ProgressEvent::Quantize {
                    file: "v3_rnnt_encoder.onnx".to_string(),
                },
                "{\"phase\":\"quantize\",\"file\":\"v3_rnnt_encoder.onnx\"}",
            ),
            (
                ProgressEvent::Verify {
                    file: "v3_vocab.txt".to_string(),
                },
                "{\"phase\":\"verify\",\"file\":\"v3_vocab.txt\"}",
            ),
            (
                ProgressEvent::Done {
                    model_dir: "/home/u/.gigastt/models".to_string(),
                },
                "{\"phase\":\"done\",\"model_dir\":\"/home/u/.gigastt/models\"}",
            ),
            (
                ProgressEvent::Error {
                    kind: ProgressErrorKind::Network,
                    message: "connection refused".to_string(),
                },
                "{\"phase\":\"error\",\"kind\":\"network\",\"message\":\"connection refused\"}",
            ),
            (
                ProgressEvent::Error {
                    kind: ProgressErrorKind::Interrupted,
                    message: "SIGINT".to_string(),
                },
                "{\"phase\":\"error\",\"kind\":\"interrupted\",\"message\":\"SIGINT\"}",
            ),
        ];
        for (event, want) in cases {
            let line = event.to_ndjson();
            assert_eq!(line, want);
            // Every line must round-trip as a JSON object with a `phase` tag.
            let parsed: serde_json::Value =
                serde_json::from_str(&line).expect("NDJSON line must parse");
            assert!(parsed.get("phase").is_some(), "phase tag missing: {line}");
        }
    }

    #[test]
    fn test_progress_error_kind_exit_codes_keep_nonzero_contract() {
        assert_eq!(ProgressErrorKind::Other.exit_code(), 1);
        assert_eq!(ProgressErrorKind::Network.exit_code(), 69);
        assert_eq!(ProgressErrorKind::Disk.exit_code(), 74);
        assert_eq!(ProgressErrorKind::Checksum.exit_code(), 65);
        assert_eq!(ProgressErrorKind::Interrupted.exit_code(), 130);
        // 2 stays reserved for clap usage errors: a misconfigured invocation
        // must never look like a transient (retryable) download failure.
        for kind in [
            ProgressErrorKind::Other,
            ProgressErrorKind::Network,
            ProgressErrorKind::Disk,
            ProgressErrorKind::Checksum,
            ProgressErrorKind::Interrupted,
        ] {
            assert_ne!(kind.exit_code(), 2, "{kind:?} must not collide with clap");
        }
        for kind in [
            ProgressErrorKind::Network,
            ProgressErrorKind::Disk,
            ProgressErrorKind::Checksum,
            ProgressErrorKind::Interrupted,
            ProgressErrorKind::Other,
        ] {
            assert_ne!(kind.exit_code(), 0, "{kind:?} must keep != 0");
        }
    }

    /// Human mode must stay byte-for-byte identical to the legacy `\r`
    /// reporter: same format strings, same trailing-space padding.
    #[test]
    fn test_download_progress_human_render_matches_legacy() {
        let mut progress = DownloadProgress::new(10 * 1_048_576);
        // current=0, percent 0 == last_percent 0 -> no redraw.
        assert_eq!(progress.human_tick(), None);
        progress.current = 5 * 1_048_576;
        assert_eq!(
            progress.human_tick().as_deref(),
            Some("\rDownloading... 50% (5.0MB / 10.0MB)")
        );
        // Same percentage -> throttled (no redraw).
        assert_eq!(progress.human_tick(), None);
        progress.current = 10 * 1_048_576;
        assert_eq!(
            progress.human_tick().as_deref(),
            Some("\rDownloading... 100% (10.0MB / 10.0MB)")
        );
        assert_eq!(
            progress.human_finish(),
            "\rDownload complete (10.0MB)                    "
        );
    }

    /// Json mode: first chunk emits immediately, rapid chunks within the 200 ms
    /// window are throttled, and 100% always emits exactly once.
    #[test]
    fn test_download_progress_json_first_throttled_then_final() {
        let (sink, log) = ProgressSink::capturing();
        let mut progress = DownloadProgress::new(1_000);
        // 100 chunks of 10 bytes, all well inside the throttle window.
        for _ in 0..100 {
            progress.update(10, &sink, "model.onnx");
        }
        let events = log.lock().unwrap();
        assert_eq!(
            events.as_slice(),
            [
                ProgressEvent::Download {
                    file: "model.onnx".to_string(),
                    bytes_done: 10,
                    bytes_total: 1_000,
                },
                ProgressEvent::Download {
                    file: "model.onnx".to_string(),
                    bytes_done: 1_000,
                    bytes_total: 1_000,
                },
            ]
        );
        // finish() must not duplicate the already-emitted 100% event.
        drop(events);
        progress.finish(&sink, "model.onnx");
        assert_eq!(log.lock().unwrap().len(), 2);
    }

    /// Json mode: once the throttle window has elapsed, mid-file progress
    /// emits again (integrators get a steady cadence on long downloads).
    #[test]
    fn test_download_progress_json_emits_after_throttle_window() {
        let (sink, log) = ProgressSink::capturing();
        let mut progress = DownloadProgress::new(1_000);
        progress.update(10, &sink, "model.onnx");
        // Backdate the last emission past the throttle window.
        progress.last_json_emit = Some(
            std::time::Instant::now()
                - JSON_PROGRESS_THROTTLE
                - std::time::Duration::from_millis(50),
        );
        progress.update(10, &sink, "model.onnx");
        let events = log.lock().unwrap();
        assert_eq!(events.len(), 2, "elapsed window must re-emit: {events:?}");
        assert_eq!(
            events[1],
            ProgressEvent::Download {
                file: "model.onnx".to_string(),
                bytes_done: 20,
                bytes_total: 1_000,
            }
        );
    }

    /// Json mode with an unknown (chunked) total: throttled events carry
    /// `bytes_total: 0`, and `finish` closes the file with one final event.
    #[test]
    fn test_download_progress_json_zero_total_emits_final_on_finish() {
        let (sink, log) = ProgressSink::capturing();
        let mut progress = DownloadProgress::new(0);
        progress.update(512, &sink, "model.onnx");
        progress.finish(&sink, "model.onnx");
        let events = log.lock().unwrap();
        assert_eq!(
            events.as_slice(),
            [
                ProgressEvent::Download {
                    file: "model.onnx".to_string(),
                    bytes_done: 512,
                    bytes_total: 0,
                },
                ProgressEvent::Download {
                    file: "model.onnx".to_string(),
                    bytes_done: 512,
                    bytes_total: 0,
                },
            ]
        );
    }

    /// Human mode never routes events to the sink (the `\r` line is the whole
    /// output), so an integrator never sees a stray JSON line.
    #[test]
    fn test_download_progress_human_sink_emits_no_events() {
        let (sink, log) = ProgressSink::capturing();
        let sink = ProgressSink {
            mode: ProgressMode::Human,
            ..sink
        };
        let mut progress = DownloadProgress::new(1_000);
        progress.update(1_000, &sink, "model.onnx");
        progress.finish(&sink, "model.onnx");
        assert!(log.lock().unwrap().is_empty());
    }

    #[test]
    fn test_classify_download_error_checksum_message() {
        let err = anyhow::anyhow!("SHA-256 mismatch for encoder.onnx: expected aa, got bb");
        assert_eq!(classify_download_error(&err), ProgressErrorKind::Checksum);
    }

    #[test]
    fn test_classify_download_error_http_status_is_network() {
        let err = anyhow::anyhow!("Download failed for model.onnx: HTTP 404");
        assert_eq!(classify_download_error(&err), ProgressErrorKind::Network);
    }

    #[test]
    fn test_classify_download_error_io_root_cause_is_disk() {
        let io = std::io::Error::new(std::io::ErrorKind::PermissionDenied, "denied");
        let err = anyhow::Error::from(io).context("Failed to create partial model file");
        assert_eq!(classify_download_error(&err), ProgressErrorKind::Disk);
    }

    #[test]
    fn test_classify_download_error_other() {
        let err =
            anyhow::anyhow!("Failed to acquire download lock (another process is downloading)");
        assert_eq!(classify_download_error(&err), ProgressErrorKind::Other);
    }

    /// A real reqwest connection failure classifies as network via the typed
    /// root cause, not message matching.
    #[tokio::test]
    #[cfg_attr(miri, ignore = "tokio runtime is unsupported under Miri")]
    async fn test_classify_download_error_reqwest_is_network() {
        let err = reqwest::Client::new()
            .get("http://127.0.0.1:9/unreachable")
            .send()
            .await
            .expect_err("port 9 must refuse");
        let err = anyhow::Error::from(err).context("HTTP request failed");
        assert_eq!(classify_download_error(&err), ProgressErrorKind::Network);
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
    #[cfg_attr(miri, ignore = "tokio runtime is unsupported under Miri")]
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
    #[cfg_attr(miri, ignore = "tokio runtime is unsupported under Miri")]
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
    #[cfg_attr(miri, ignore = "tokio runtime is unsupported under Miri")]
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

    /// End-to-end NDJSON contract on a local HTTP stub: the download of one
    /// file emits a 100% `download` event followed by a `verify` event, and
    /// every line round-trips through a JSON parser (true NDJSON).
    #[tokio::test]
    #[cfg_attr(miri, ignore = "tokio runtime is unsupported under Miri")]
    async fn test_stream_to_partial_then_finalize_json_event_sequence() {
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
        let expected = sha256_hex(payload);
        let (sink, log) = ProgressSink::capturing();

        stream_to_partial_then_finalize_with_sink(
            &url,
            &final_path,
            Some(&expected),
            "model.onnx",
            &sink,
        )
        .await
        .expect("download should succeed");

        let events = log.lock().unwrap();
        assert_eq!(
            events.as_slice(),
            [
                ProgressEvent::Download {
                    file: "model.onnx".to_string(),
                    bytes_done: payload.len() as u64,
                    bytes_total: payload.len() as u64,
                },
                ProgressEvent::Verify {
                    file: "model.onnx".to_string(),
                },
            ]
        );
        // The integrator view: serialize each event to a line and parse it
        // back — the stream must be well-formed NDJSON with a phase tag.
        for event in events.iter() {
            let parsed: serde_json::Value =
                serde_json::from_str(&event.to_ndjson()).expect("event must be NDJSON");
            assert!(parsed.get("phase").is_some());
        }
    }

    /// No checksum pinned → no `verify` event (verification did not happen).
    #[tokio::test]
    #[cfg_attr(miri, ignore = "tokio runtime is unsupported under Miri")]
    async fn test_stream_to_partial_then_finalize_json_no_verify_without_checksum() {
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
        let (sink, log) = ProgressSink::capturing();

        stream_to_partial_then_finalize_with_sink(&url, &final_path, None, "model.onnx", &sink)
            .await
            .expect("download should succeed");

        let events = log.lock().unwrap();
        assert!(
            events
                .iter()
                .all(|e| !matches!(e, ProgressEvent::Verify { .. })),
            "no checksum -> no verify event: {events:?}"
        );
        assert!(
            events
                .iter()
                .any(|e| matches!(e, ProgressEvent::Download { .. })),
            "download events expected: {events:?}"
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
    #[cfg_attr(miri, ignore = "tokio runtime is unsupported under Miri")]
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
        assert_eq!(
            ModelVariant::from_str("ml_ctc").unwrap(),
            ModelVariant::MlCtc
        );
        assert_eq!(
            ModelVariant::from_str("ML-CTC").unwrap(),
            ModelVariant::MlCtc
        );
        assert_eq!(
            ModelVariant::from_str("ml_ctc_large").unwrap(),
            ModelVariant::MlCtcLarge
        );
        assert_eq!(
            ModelVariant::from_str("ML-CTC-LARGE").unwrap(),
            ModelVariant::MlCtcLarge
        );
        assert!(ModelVariant::from_str("whisper").is_err());
    }

    #[test]
    fn test_model_variant_ml_ctc_file_mapping() {
        let v = ModelVariant::MlCtc;
        // Real istupakov filenames (gigaam-multilingual-ctc-onnx).
        assert_eq!(v.encoder_file(), "multilingual_ctc.onnx");
        assert_eq!(v.encoder_int8_file(), "multilingual_ctc.int8.onnx");
        assert_eq!(v.vocab_file(), "multilingual_vocab.txt");
        // Encoder-only: no decoder/joiner ONNX exists.
        assert_eq!(v.decoder_file(), "");
        assert_eq!(v.joint_file(), "");
        // Downloads the pre-quantized INT8 encoder directly + vocab.
        assert_eq!(
            v.download_files(),
            ["multilingual_ctc.int8.onnx", "multilingual_vocab.txt"]
        );
        assert_eq!(v.hf_repo(), "istupakov/gigaam-multilingual-ctc-onnx");
        assert_eq!(v.as_str(), "ml_ctc");
        assert_eq!(v.model_id(), "gigaam-multilingual-ctc");
    }

    #[test]
    fn test_hf_repo_per_variant() {
        assert_eq!(ModelVariant::Rnnt.hf_repo(), "istupakov/gigaam-v3-onnx");
        assert_eq!(ModelVariant::E2eRnnt.hf_repo(), "istupakov/gigaam-v3-onnx");
        assert_eq!(
            ModelVariant::MlCtc.hf_repo(),
            "istupakov/gigaam-multilingual-ctc-onnx"
        );
        assert_eq!(
            ModelVariant::MlCtcLarge.hf_repo(),
            "istupakov/gigaam-multilingual-large-ctc-onnx"
        );
    }

    #[test]
    fn test_model_variant_ml_ctc_large_file_mapping() {
        let v = ModelVariant::MlCtcLarge;
        assert_eq!(v.encoder_file(), "multilingual_large_ctc.onnx");
        assert_eq!(v.encoder_int8_file(), "multilingual_large_ctc.int8.onnx");
        // Vocab is byte-identical to (and shares the filename with) the 220M head.
        assert_eq!(v.vocab_file(), "multilingual_vocab.txt");
        assert_eq!(v.vocab_file(), ModelVariant::MlCtc.vocab_file());
        assert_eq!(v.decoder_file(), "");
        assert_eq!(v.joint_file(), "");
        assert_eq!(
            v.download_files(),
            ["multilingual_large_ctc.int8.onnx", "multilingual_vocab.txt"]
        );
        assert_eq!(v.as_str(), "ml_ctc_large");
        assert_eq!(v.model_id(), "gigaam-multilingual-large-ctc");
        assert!(v.is_ctc());
        assert!(ModelVariant::MlCtc.is_ctc());
        assert!(!ModelVariant::Rnnt.is_ctc());
    }

    #[test]
    fn test_detect_in_dir_ml_ctc_large_by_int8_encoder() {
        let tmp = tempfile::tempdir().expect("tempdir");
        std::fs::write(tmp.path().join("multilingual_large_ctc.int8.onnx"), b"int8").unwrap();
        assert_eq!(
            ModelVariant::detect_in_dir(tmp.path()),
            Some(ModelVariant::MlCtcLarge)
        );
    }

    #[test]
    fn test_detect_in_dir_ml_ctc_by_int8_encoder() {
        let tmp = tempfile::tempdir().expect("tempdir");
        std::fs::write(tmp.path().join("multilingual_ctc.int8.onnx"), b"int8").unwrap();
        assert_eq!(
            ModelVariant::detect_in_dir(tmp.path()),
            Some(ModelVariant::MlCtc)
        );
    }

    #[test]
    fn test_model_variant_checksums_are_pinned() {
        // Every downloaded file for every variant has a pinned 64-char hex
        // checksum — security parity, no placeholder slipping into a release.
        for variant in [
            ModelVariant::Rnnt,
            ModelVariant::E2eRnnt,
            ModelVariant::MlCtc,
            ModelVariant::MlCtcLarge,
        ] {
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
    #[cfg_attr(miri, ignore = "tokio runtime is unsupported under Miri")]
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
    #[cfg_attr(miri, ignore = "tokio runtime is unsupported under Miri")]
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
    #[cfg_attr(miri, ignore = "tokio runtime is unsupported under Miri")]
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
    #[cfg_attr(miri, ignore = "tokio runtime is unsupported under Miri")]
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
    #[cfg_attr(miri, ignore = "tokio runtime is unsupported under Miri")]
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

    // ── pre-quantized bundle ────────────────────────────────────────────────

    #[test]
    fn test_prequantized_files_mapping() {
        assert_eq!(
            ModelVariant::Rnnt.prequantized_files(),
            [
                "v3_rnnt_encoder_int8.onnx",
                "v3_rnnt_decoder.onnx",
                "v3_rnnt_joint.onnx",
                "v3_vocab.txt",
            ]
        );
        assert_eq!(
            ModelVariant::E2eRnnt.prequantized_files(),
            [
                "v3_e2e_rnnt_encoder_int8.onnx",
                "v3_e2e_rnnt_decoder.onnx",
                "v3_e2e_rnnt_joint.onnx",
                "v3_e2e_rnnt_vocab.txt",
            ]
        );
        // CTC is encoder-only: pre-quantized set is just the INT8 encoder + vocab.
        assert_eq!(
            ModelVariant::MlCtc.prequantized_files(),
            ["multilingual_ctc.int8.onnx", "multilingual_vocab.txt"]
        );
        assert_eq!(
            ModelVariant::MlCtcLarge.prequantized_files(),
            ["multilingual_large_ctc.int8.onnx", "multilingual_vocab.txt"]
        );
    }

    #[test]
    fn test_encoder_int8_checksums_are_pinned() {
        for variant in [
            ModelVariant::Rnnt,
            ModelVariant::E2eRnnt,
            ModelVariant::MlCtc,
            ModelVariant::MlCtcLarge,
        ] {
            let sum = variant.encoder_int8_checksum();
            assert_eq!(
                sum.len(),
                64,
                "{variant:?} INT8 checksum must be 64 hex chars, got: {sum}"
            );
            assert!(
                sum.chars()
                    .all(|c| c.is_ascii_digit() || ('a'..='f').contains(&c)),
                "{variant:?} INT8 checksum must be lowercase hex, got: {sum}"
            );
        }
    }

    #[test]
    fn test_prequantized_checksum_int8_encoder_and_reuses_fp32_for_rest() {
        let v = ModelVariant::Rnnt;
        // Encoder → the INT8-specific checksum.
        assert_eq!(
            v.prequantized_checksum(v.encoder_int8_file()),
            Some(v.encoder_int8_checksum())
        );
        // Decoder/joiner/vocab → the same pins as the FP32 download set.
        for f in [v.decoder_file(), v.joint_file(), v.vocab_file()] {
            assert_eq!(v.prequantized_checksum(f), v.checksum(f));
            assert!(v.prequantized_checksum(f).is_some(), "{f} must be pinned");
        }
    }

    #[test]
    fn test_is_prequantized_present() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let dir = tmp.path();
        assert!(!is_prequantized_present(ModelVariant::Rnnt, dir));
        for f in ModelVariant::Rnnt.prequantized_files() {
            std::fs::write(dir.join(f), b"x").unwrap();
        }
        assert!(is_prequantized_present(ModelVariant::Rnnt, dir));
        // The FP32 set is absent (no FP32 encoder), yet the prequantized set is
        // complete — the two presence checks are independent.
        assert!(!is_model_present(ModelVariant::Rnnt, dir));
    }

    /// `ensure_prequantized_model_variant` short-circuits (no network, no
    /// `.partial`) when the pre-quantized set is already present.
    #[tokio::test]
    #[cfg_attr(miri, ignore = "tokio runtime is unsupported under Miri")]
    async fn test_ensure_prequantized_present_no_download() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let dir = tmp.path();
        for f in ModelVariant::Rnnt.prequantized_files() {
            std::fs::write(dir.join(f), b"stub").unwrap();
        }

        let variant = ensure_prequantized_model_variant(None, dir.to_str().unwrap())
            .await
            .expect("present prequantized set must short-circuit");
        assert_eq!(variant, ModelVariant::Rnnt);

        let has_partial = std::fs::read_dir(dir)
            .unwrap()
            .filter_map(|e| e.ok())
            .any(|e| e.file_name().to_string_lossy().contains(".partial"));
        assert!(
            !has_partial,
            "no download when the prequantized set is present"
        );
    }

    // ── ANE packages ────────────────────────────────────────────────────────

    #[cfg(feature = "ane")]
    #[test]
    fn test_ane_buckets_ladder_pinned() {
        // Must match the convert script's --buckets default.
        assert_eq!(ANE_BUCKETS, &[512, 768, 1536, 3000]);
    }

    /// Every shipped bucket must clear the ANE-residency floor (~288 mel frames):
    /// below it the fixed-shape graph falls off the Neural Engine onto the CPU EP
    /// (measured in the conversion spike), so a too-small bucket would silently
    /// regress to CPU. 512 (the smallest) clears 288; this guards future ladder
    /// edits from adding a bucket below the residency floor.
    #[cfg(feature = "ane")]
    #[test]
    fn test_ane_buckets_above_residency_floor() {
        const ANE_RESIDENCY_FLOOR: usize = 288;
        for &b in ANE_BUCKETS {
            assert!(
                b >= ANE_RESIDENCY_FLOOR,
                "ANE bucket {b} is below the {ANE_RESIDENCY_FLOOR}-mel residency floor — it would evict to CPU"
            );
        }
    }

    #[cfg(all(feature = "net", feature = "ane"))]
    #[test]
    fn test_ane_tar_checksums_shape() {
        // Exactly one entry per bucket; each entry is either the empty
        // (unreleased) sentinel or a valid 64-char lowercase-hex digest.
        assert_eq!(ANE_TAR_CHECKSUMS.len(), ANE_BUCKETS.len());
        for &b in ANE_BUCKETS {
            let entries: Vec<_> = ANE_TAR_CHECKSUMS
                .iter()
                .filter(|(bucket, _)| *bucket == b)
                .collect();
            assert_eq!(entries.len(), 1, "exactly one ANE checksum entry for {b}");
            let sum = entries[0].1;
            if sum.is_empty() {
                continue; // genuine unreleased state
            }
            assert_eq!(
                sum.len(),
                64,
                "ANE {b} checksum must be 64 hex chars: {sum}"
            );
            assert!(
                sum.chars()
                    .all(|c| c.is_ascii_digit() || ('a'..='f').contains(&c)),
                "ANE {b} checksum must be lowercase hex: {sum}"
            );
        }
    }

    #[cfg(feature = "ane")]
    #[test]
    fn test_ane_filename_helpers() {
        assert_eq!(ane_package_dir_name(768), "gigaam_v3_encoder_768.mlpackage");
        assert_eq!(ane_tar_name(768), "gigaam_v3_encoder_768.mlpackage.tar");
    }

    #[cfg(feature = "ane")]
    #[test]
    fn test_default_ane_model_dir_is_model_sibling() {
        let ane = default_ane_model_dir();
        assert!(
            ane.contains(".gigastt") && ane.ends_with("ane"),
            "ane dir should be under .gigastt and end with 'ane', got: {ane}"
        );
    }

    /// Stage the FULL structurally-required file set Core ML writes into a
    /// `.mlpackage` (manifest + model spec + weights blob) under a bucket dir.
    #[cfg(feature = "ane")]
    fn stage_complete_ane_package(pkg: &Path) {
        let coreml = pkg.join("Data").join("com.apple.CoreML");
        std::fs::create_dir_all(coreml.join("weights")).unwrap();
        std::fs::write(pkg.join("Manifest.json"), b"{}").unwrap();
        std::fs::write(coreml.join("model.mlmodel"), b"spec").unwrap();
        std::fs::write(coreml.join("weights").join("weight.bin"), b"w").unwrap();
    }

    #[cfg(feature = "ane")]
    #[test]
    fn test_is_ane_present_false_on_empty_then_true_when_staged() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let dir = tmp.path();
        assert!(!is_ane_present(dir), "empty dir has no ANE packages");

        for &b in ANE_BUCKETS {
            stage_complete_ane_package(&dir.join(ane_package_dir_name(b)));
        }
        assert!(is_ane_present(dir), "all buckets fully staged → present");
    }

    /// A torn package (only `Manifest.json`, no model spec / weights) must NOT
    /// be reported complete — otherwise the download path wedges forever.
    #[cfg(feature = "ane")]
    #[test]
    fn test_ane_package_complete_false_when_torn() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let dir = tmp.path();
        let pkg = dir.join(ane_package_dir_name(768));
        std::fs::create_dir_all(&pkg).unwrap();
        std::fs::write(pkg.join("Manifest.json"), b"{}").unwrap();

        assert!(
            !ane_package_complete(&pkg),
            "manifest-only package is torn, not complete"
        );

        // Stage the other buckets fully; the torn 768 bucket must still drag
        // the whole-dir check to false.
        for &b in &ANE_BUCKETS[1..] {
            stage_complete_ane_package(&dir.join(ane_package_dir_name(b)));
        }
        assert!(!is_ane_present(dir), "torn bucket → not present");
    }

    /// Build a deterministic `.tar` (a `<pkg_name>/` dir whose arcnames are
    /// prefixed with the package name) holding the full required file set,
    /// written at `tar_path`. Mirrors what `release-ane.yml` publishes.
    #[cfg(feature = "ane")]
    fn build_ane_tar(tar_path: &Path, pkg_name: &str) {
        let tmp = tempfile::tempdir().expect("tempdir");
        let pkg = tmp.path().join(pkg_name);
        stage_complete_ane_package(&pkg);

        let file = std::fs::File::create(tar_path).unwrap();
        let mut builder = tar::Builder::new(file);
        builder.append_dir_all(pkg_name, &pkg).unwrap();
        builder.finish().unwrap();
    }

    /// Building a deterministic tar (a `gigaam_v3_encoder_768.mlpackage/` dir
    /// with the full file set) and unpacking it with `tar::Archive` reconstructs
    /// the directory + files — proves the extract step end-to-end, no network.
    #[cfg(feature = "ane")]
    #[test]
    fn test_ane_tar_roundtrip_extract() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let tar_path = tmp.path().join("pkg.tar");
        build_ane_tar(&tar_path, "gigaam_v3_encoder_768.mlpackage");

        let out = tmp.path().join("out");
        std::fs::create_dir_all(&out).unwrap();
        let file = std::fs::File::open(&tar_path).unwrap();
        tar::Archive::new(file).unpack(&out).unwrap();

        let extracted = out.join("gigaam_v3_encoder_768.mlpackage");
        assert!(
            ane_package_complete(&extracted),
            "extracted .mlpackage must be complete"
        );
    }

    /// `extract_ane_tar_atomic` reconstructs the package at its final path and
    /// leaves no `.extract.*` staging dir behind on success.
    #[cfg(all(feature = "net", feature = "ane"))]
    #[test]
    fn test_extract_ane_tar_atomic_no_staging_leak() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let dir = tmp.path();
        let pkg_name = ane_package_dir_name(768);
        let tar_dest = dir.join(ane_tar_name(768));
        build_ane_tar(&tar_dest, &pkg_name);

        extract_ane_tar_atomic(&tar_dest, dir, &pkg_name).expect("atomic extract");

        assert!(
            ane_package_complete(&dir.join(&pkg_name)),
            "package must land complete at its final path"
        );
        let leaked = std::fs::read_dir(dir)
            .unwrap()
            .filter_map(|e| e.ok())
            .any(|e| e.file_name().to_string_lossy().starts_with(".extract."));
        assert!(
            !leaked,
            "no .extract.* staging dir may remain after success"
        );
    }

    /// Every shipped bucket resolves to its pinned `.tar` checksum; a bucket with
    /// no pin (an empty sentinel, or one outside the ladder) surfaces the
    /// actionable "not yet published" bail rather than downloading unverified.
    #[cfg(all(feature = "net", feature = "ane"))]
    #[test]
    fn test_require_ane_tar_checksum_resolves_pinned_and_bails_unpinned() {
        // Each ladder bucket is pinned to its release `.tar` SHA-256.
        for &b in ANE_BUCKETS {
            let sum = require_ane_tar_checksum(b).expect("shipped bucket must be pinned");
            assert_eq!(sum.len(), 64, "checksum must be 64 hex chars, got: {sum}");
        }
        // A bucket with no pin (here: outside the ladder) takes the bail path.
        let err = require_ane_tar_checksum(99_999).expect_err("unpinned bucket must bail");
        assert!(
            format!("{err}").contains("not yet published"),
            "unexpected error: {err}"
        );
    }
}
