//! WebSocket protocol messages for gigastt.

use serde::{Deserialize, Serialize};

/// Current WebSocket protocol version (semver-lite: major.minor).
pub const PROTOCOL_VERSION: &str = "1.0";

/// Server → Client messages.
#[derive(Debug, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ServerMessage {
    /// Server is ready to accept audio.
    Ready {
        model: String,
        sample_rate: u32,
        version: String,
    },

    /// Partial (interim) transcript — may change with more audio.
    Partial {
        text: String,
        timestamp: f64,
    },

    /// Final transcript — utterance is complete, with punctuation.
    Final {
        text: String,
        timestamp: f64,
    },

    /// Error occurred.
    Error {
        message: String,
        code: String,
    },
}

/// Client → Server text messages (optional control commands).
#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
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
