//! Explicit native v4 WebSocket dialect. Legacy messages remain unchanged.

use crate::transcript::{EndpointReason, WordInfo, aggregate_confidence, now_timestamp};
use serde::Serialize;

/// Protocol used only by the native v4 WebSocket endpoint.
pub const PROTOCOL_VERSION: &str = "1.4";
/// Optional upgrade negotiation used by clients that require the v4 dialect.
pub const SUBPROTOCOL: &str = "gigastt-native-v4";

/// Additional native-only fields in a Ready message.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ReadyCapability {
    pub model_family: &'static str,
    pub model_head: &'static str,
    pub model_precision: &'static str,
    pub transcript_style: &'static str,
    pub future_context_ms: u16,
    pub supported_future_context_ms: &'static [u16],
    pub max_packet_bytes: usize,
    pub endpoint_mode: &'static str,
}

/// Metadata for a newly emitted native nonblank token.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Token {
    pub id: usize,
    pub frame: usize,
    pub confidence: f64,
}

/// Native packet delta; the enclosing partial/final remains a snapshot.
#[derive(Debug, Clone, Serialize)]
pub struct Delta {
    pub packet_one_based: usize,
    pub samples: usize,
    pub encoded_frames: usize,
    pub text: String,
    pub words: Vec<WordInfo>,
    pub tokens: Vec<Token>,
}

/// Borrowed transcript view with additive packet metadata, serialized only on
/// the explicitly provisioned native v4 endpoint.
#[derive(Debug, Serialize)]
pub struct Transcript<'a> {
    #[serde(rename = "type")]
    kind: &'static str,
    text: &'a str,
    committed: &'a str,
    tentative: &'static str,
    words: &'a [WordInfo],
    is_final: bool,
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    speech_final: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    endpoint_reason: Option<EndpointReason>,
    timestamp: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    confidence: Option<f32>,
    native_delta: &'a Delta,
}
impl<'a> Transcript<'a> {
    /// Use append-only completed snapshots and a single true stop final.
    pub fn new(text: &'a str, words: &'a [WordInfo], delta: &'a Delta, final_packet: bool) -> Self {
        Self {
            kind: if final_packet { "final" } else { "partial" },
            text,
            committed: text,
            tentative: "",
            words,
            is_final: final_packet,
            speech_final: final_packet,
            endpoint_reason: final_packet.then_some(EndpointReason::Stop),
            timestamp: now_timestamp(),
            confidence: aggregate_confidence(words),
            native_delta: delta,
        }
    }
}
