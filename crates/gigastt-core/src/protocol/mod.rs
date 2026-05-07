//! WebSocket protocol messages for gigastt.

use serde::{Deserialize, Serialize};

/// Current WebSocket protocol version (semver-lite: major.minor).
pub const PROTOCOL_VERSION: &str = "1.0";

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
        /// Whether diarization is active for this session. Omitted from JSON when false.
        #[serde(skip_serializing_if = "std::ops::Not::not")]
        diarization: bool,
        /// Minimum protocol version accepted by this server. Lets clients
        /// discover compatibility without trial-and-error. Omitted when equal
        /// to `version` (i.e. only one version is supported) for backward compat.
        #[serde(skip_serializing_if = "Option::is_none")]
        min_protocol_version: Option<String>,
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
    },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_protocol_version_constant() {
        assert_eq!(PROTOCOL_VERSION, "1.0");
    }

    #[test]
    fn test_ready_serialization_includes_version() {
        let msg = ServerMessage::Ready {
            model: "test-model".into(),
            sample_rate: 48000,
            version: PROTOCOL_VERSION.into(),
            supported_rates: vec![],
            diarization: false,
            min_protocol_version: Some(PROTOCOL_VERSION.into()),
        };
        let json = serde_json::to_string(&msg).unwrap();
        let v: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(v["type"], "ready");
        assert_eq!(v["version"], "1.0");
        assert_eq!(v["model"], "test-model");
        assert_eq!(v["sample_rate"], 48000);
        assert_eq!(v["min_protocol_version"], "1.0");
    }

    #[test]
    fn test_partial_serialization_no_version() {
        let msg = ServerMessage::Partial(crate::inference::TranscriptSegment {
            text: "hello".into(),
            timestamp: 1.0,
            words: vec![],
            is_final: false,
        });
        let json = serde_json::to_string(&msg).unwrap();
        let v: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(v["type"], "partial");
        assert!(v.get("version").is_none());
    }

    #[test]
    fn test_final_serialization_no_version() {
        let msg = ServerMessage::Final(crate::inference::TranscriptSegment {
            text: "hello".into(),
            timestamp: 1.0,
            words: vec![],
            is_final: true,
        });
        let json = serde_json::to_string(&msg).unwrap();
        let v: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(v["type"], "final");
        assert!(v.get("version").is_none());
    }

    #[test]
    fn test_error_serialization_no_version() {
        let msg = ServerMessage::Error {
            message: "fail".into(),
            code: "err".into(),
            retry_after_ms: None,
        };
        let json = serde_json::to_string(&msg).unwrap();
        let v: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(v["type"], "error");
        assert!(v.get("version").is_none());
        assert!(
            v.get("retry_after_ms").is_none(),
            "retry_after_ms must be omitted when None"
        );
    }

    #[test]
    fn test_error_serialization_with_retry_after() {
        let msg = ServerMessage::Error {
            message: "Server busy, try again later".into(),
            code: "timeout".into(),
            retry_after_ms: Some(30_000),
        };
        let json = serde_json::to_string(&msg).unwrap();
        let v: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(v["type"], "error");
        assert_eq!(v["code"], "timeout");
        assert_eq!(v["retry_after_ms"], 30_000);
    }

    #[test]
    fn test_client_message_stop_deserialize() {
        let json = r#"{"type":"stop"}"#;
        let msg: ClientMessage = serde_json::from_str(json).unwrap();
        assert!(matches!(msg, ClientMessage::Stop));
    }

    #[test]
    fn test_client_message_configure_deserialize() {
        let json = r#"{"type":"configure","sample_rate":8000}"#;
        let msg: ClientMessage = serde_json::from_str(json).unwrap();
        match msg {
            ClientMessage::Configure { sample_rate, .. } => assert_eq!(sample_rate, Some(8000)),
            _ => panic!("Expected Configure"),
        }
    }

    #[test]
    fn test_ready_supported_rates_serialization() {
        let msg = ServerMessage::Ready {
            model: "test".into(),
            sample_rate: 48000,
            version: "1.0".into(),
            supported_rates: vec![8000, 16000, 24000, 44100, 48000],
            diarization: false,
            min_protocol_version: None,
        };
        let json = serde_json::to_string(&msg).unwrap();
        let v: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(v["supported_rates"].as_array().unwrap().len(), 5);
    }

    #[test]
    fn test_ready_empty_supported_rates_omitted() {
        let msg = ServerMessage::Ready {
            model: "test".into(),
            sample_rate: 48000,
            version: "1.0".into(),
            supported_rates: vec![],
            diarization: false,
            min_protocol_version: None,
        };
        let json = serde_json::to_string(&msg).unwrap();
        let v: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert!(v.get("supported_rates").is_none());
    }

    #[test]
    fn test_word_info_speaker_none_omitted() {
        let word = crate::inference::WordInfo {
            word: "hello".into(),
            start: 0.0,
            end: 1.0,
            confidence: 0.9,
            speaker: None,
        };
        let json = serde_json::to_string(&word).unwrap();
        let v: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert!(v.get("speaker").is_none());
    }

    #[test]
    fn test_word_info_speaker_present() {
        let word = crate::inference::WordInfo {
            word: "hello".into(),
            start: 0.0,
            end: 1.0,
            confidence: 0.9,
            speaker: Some(2),
        };
        let json = serde_json::to_string(&word).unwrap();
        let v: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(v["speaker"], 2);
    }

    #[test]
    fn test_configure_diarization_deserialize() {
        let json = r#"{"type":"configure","diarization":true}"#;
        let msg: ClientMessage = serde_json::from_str(json).unwrap();
        match msg {
            ClientMessage::Configure { diarization, .. } => assert_eq!(diarization, Some(true)),
            _ => panic!("Expected Configure"),
        }
    }

    #[test]
    fn test_configure_sample_rate_only() {
        let json = r#"{"type":"configure","sample_rate":8000}"#;
        let msg: ClientMessage = serde_json::from_str(json).unwrap();
        match msg {
            ClientMessage::Configure {
                sample_rate,
                diarization,
                ..
            } => {
                assert_eq!(sample_rate, Some(8000));
                assert_eq!(diarization, None);
            }
            _ => panic!("Expected Configure"),
        }
    }

    #[test]
    fn test_configure_protocol_version_deserialize() {
        let json = r#"{"type":"configure","protocol_version":"1.0"}"#;
        let msg: ClientMessage = serde_json::from_str(json).unwrap();
        match msg {
            ClientMessage::Configure {
                protocol_version, ..
            } => assert_eq!(protocol_version, Some("1.0".into())),
            _ => panic!("Expected Configure"),
        }
    }

    #[test]
    fn test_configure_protocol_version_absent() {
        let json = r#"{"type":"configure","sample_rate":16000}"#;
        let msg: ClientMessage = serde_json::from_str(json).unwrap();
        match msg {
            ClientMessage::Configure {
                protocol_version, ..
            } => assert_eq!(protocol_version, None),
            _ => panic!("Expected Configure"),
        }
    }

    #[test]
    fn test_ready_min_protocol_version_omitted_when_none() {
        let msg = ServerMessage::Ready {
            model: "test".into(),
            sample_rate: 48000,
            version: "1.0".into(),
            supported_rates: vec![],
            diarization: false,
            min_protocol_version: None,
        };
        let json = serde_json::to_string(&msg).unwrap();
        let v: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert!(v.get("min_protocol_version").is_none());
    }

    #[test]
    fn test_ready_diarization_false_omitted() {
        let msg = ServerMessage::Ready {
            model: "test".into(),
            sample_rate: 48000,
            version: "1.0".into(),
            supported_rates: vec![],
            diarization: false,
            min_protocol_version: None,
        };
        let json = serde_json::to_string(&msg).unwrap();
        let v: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert!(v.get("diarization").is_none());
    }
}
