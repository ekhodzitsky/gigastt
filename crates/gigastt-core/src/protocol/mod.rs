//! WebSocket protocol messages for gigastt.

use serde::{Deserialize, Serialize};

/// Current WebSocket protocol version (semver-lite: major.minor).
pub const PROTOCOL_VERSION: &str = "1.2";
/// Oldest accepted client version; newer versions add optional fields.
pub const MIN_PROTOCOL_VERSION: &str = "1.0";

/// Server → Client messages.
#[derive(Debug, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
#[non_exhaustive]
pub enum ServerMessage {
    /// Server is ready to accept audio.
    Ready {
        /// Model identifier (e.g., `"gigaam-v3-e2e-rnnt"`).
        model: String,
        /// Default audio sample rate in Hz (48000 for backward compatibility).
        sample_rate: u32,
        /// Protocol version string (e.g., `"1.0"`).
        version: String,
        /// Supported input sample rates (omitted from JSON if empty for backward compat).
        #[serde(skip_serializing_if = "Vec::is_empty")]
        supported_rates: Vec<u32>,
        /// Whether this server *can* diarize — a speaker model is loaded.
        /// `Ready` precedes `Configure`, so this is a capability advert, not
        /// session state: `Configure { diarization: true }` against a `false`
        /// here is a graceful no-op (same convention as `punctuation`), and
        /// this field is how a client knows that in advance. Omitted from JSON
        /// when false.
        #[serde(skip_serializing_if = "std::ops::Not::not")]
        diarization: bool,
        /// Minimum protocol version accepted by this server. Lets clients
        /// discover compatibility without trial-and-error. Omitted when equal
        /// to `version` (i.e. only one version is supported) for backward compat.
        #[serde(skip_serializing_if = "Option::is_none")]
        min_protocol_version: Option<String>,
        /// Maximum wall-clock session duration in seconds (server
        /// `--max-session-secs`; `0` = no cap). Always sent so clients can
        /// plan a reconnect before the server closes the socket with
        /// `max_session_duration_exceeded`. Additive; older clients ignore it.
        max_session_secs: u64,
        /// Idle timeout in seconds (server `--idle-timeout-secs`): the server
        /// closes the session when no frame arrives within this window.
        /// Always sent; additive, older clients ignore it.
        idle_timeout_secs: u64,
    },

    /// Partial (interim) transcript — may change with more audio.
    Partial(crate::inference::TranscriptSegment),

    /// Final transcript — utterance is complete (endpointing detected or stream flushed).
    Final(crate::inference::TranscriptSegment),

    /// Error occurred during processing.
    Error {
        /// Human-readable error description (internal details are hidden).
        message: String,
        /// Machine-readable error code (e.g., `"inference_error"`).
        code: String,
        /// Suggested delay (milliseconds) before retry. Present only for transient
        /// backpressure errors (e.g. pool saturation). Optional; omitted from JSON
        /// when absent to preserve backward-compatible payloads.
        #[serde(skip_serializing_if = "Option::is_none")]
        retry_after_ms: Option<u32>,
    },
}

/// Client → Server text messages (optional control commands).
#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
#[non_exhaustive]
pub enum ClientMessage {
    /// Request server to stop and finalize.
    Stop,
    /// Configure session parameters (must be sent before first audio frame).
    ///
    /// `#[non_exhaustive]`: the wire protocol evolves by adding optional
    /// fields only (existing fields are never renamed or removed), so this
    /// variant gains fields in minor releases — always match it with a `..`
    /// rest pattern.
    #[non_exhaustive]
    Configure {
        /// Audio sample rate in Hz (e.g., 8000, 16000, 24000, 44100, 48000). Optional.
        #[serde(default)]
        sample_rate: Option<u32>,
        /// Enable speaker diarization for this session. Optional.
        #[serde(default)]
        diarization: Option<bool>,
        /// Protocol version the client wants to speak (e.g., `"1.0"`).
        /// When omitted the server defaults to the current version.
        /// When present but unsupported, the server replies with an error
        /// (`unsupported_protocol_version`) listing the supported range.
        #[serde(default)]
        protocol_version: Option<String>,
        /// Per-session punctuation/casing-restoration override applied to
        /// `final` segments only (`partial` payloads always stay raw).
        /// Omitted = server default (on iff the server has a punctuator
        /// attached). A `true` on a server without a punctuation model is a
        /// graceful no-op. Optional.
        #[serde(default)]
        punctuation: Option<bool>,
        /// Per-session inverse text normalization override (number-words →
        /// digits) applied to `final` segments only. Omitted = server default.
        /// Optional.
        #[serde(default)]
        itn: Option<bool>,
        /// Streaming utterance-end policy: `"auto"` | `"assistant"` | `"manual"`.
        /// Omitted = server boot default (`--endpoint-mode`). Optional/unknown
        /// values are rejected with `invalid_endpoint_mode`. Optional.
        #[serde(default)]
        endpoint_mode: Option<String>,
        /// Per-session minimum trailing silence (ms) for VAD endpointing.
        /// Overrides server `--vad-min-silence-ms` for this connection only.
        /// No effect when the server has no VAD loaded. Optional.
        #[serde(default)]
        min_silence_ms: Option<u32>,
        /// Text commitment policy: `auto`, `on_finalize`, or `stable_prefix`.
        /// Omitted = keep the previous policy (initially `auto`). Commitment
        /// is scoped to each utterance and resets after Final.
        #[serde(default)]
        commit_policy: Option<String>,
    },
}

#[cfg(test)]
mod tests;
