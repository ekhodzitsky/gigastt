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
        /// Expected audio sample rate in Hz (typically 48000).
        sample_rate: u32,
        /// Protocol version string (e.g., `"1.0"`).
        version: String,
    },

    /// Partial (interim) transcript — may change with more audio.
    Partial {
        /// Current recognized text (may be revised in subsequent partials).
        text: String,
        /// Unix timestamp when this partial was produced.
        timestamp: f64,
        /// Per-word timing and confidence (omitted from JSON if empty).
        #[serde(skip_serializing_if = "Vec::is_empty")]
        words: Vec<crate::inference::WordInfo>,
    },

    /// Final transcript — utterance is complete (endpointing detected or stream flushed).
    Final {
        /// Final recognized text for this utterance.
        text: String,
        /// Unix timestamp when this final was produced.
        timestamp: f64,
        /// Per-word timing and confidence (omitted from JSON if empty).
        #[serde(skip_serializing_if = "Vec::is_empty")]
        words: Vec<crate::inference::WordInfo>,
    },

    /// Error occurred during processing.
    Error {
        /// Human-readable error description (internal details are hidden).
        message: String,
        /// Machine-readable error code (e.g., `"inference_error"`).
        code: String,
    },
}

/// Client → Server text messages (optional control commands).
#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
#[non_exhaustive]
pub enum ClientMessage {
    /// Request server to stop and finalize.
    Stop,
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
        };
        let json = serde_json::to_string(&msg).unwrap();
        let v: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(v["type"], "ready");
        assert_eq!(v["version"], "1.0");
        assert_eq!(v["model"], "test-model");
        assert_eq!(v["sample_rate"], 48000);
    }

    #[test]
    fn test_partial_serialization_no_version() {
        let msg = ServerMessage::Partial {
            text: "hello".into(),
            timestamp: 1.0,
            words: vec![],
        };
        let json = serde_json::to_string(&msg).unwrap();
        let v: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(v["type"], "partial");
        assert!(v.get("version").is_none());
    }

    #[test]
    fn test_final_serialization_no_version() {
        let msg = ServerMessage::Final {
            text: "hello".into(),
            timestamp: 1.0,
            words: vec![],
        };
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
        };
        let json = serde_json::to_string(&msg).unwrap();
        let v: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(v["type"], "error");
        assert!(v.get("version").is_none());
    }

    #[test]
    fn test_client_message_stop_deserialize() {
        let json = r#"{"type":"stop"}"#;
        let msg: ClientMessage = serde_json::from_str(json).unwrap();
        assert!(matches!(msg, ClientMessage::Stop));
    }
}
