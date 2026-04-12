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
    /// Configure session parameters (must be sent before first audio frame).
    Configure {
        /// Audio sample rate in Hz (e.g., 8000, 16000, 24000, 44100, 48000). Optional.
        #[serde(default)]
        sample_rate: Option<u32>,
        /// Enable speaker diarization for this session. Optional.
        #[serde(default)]
        diarization: Option<bool>,
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
            ClientMessage::Configure { sample_rate, diarization } => {
                assert_eq!(sample_rate, Some(8000));
                assert_eq!(diarization, None);
            }
            _ => panic!("Expected Configure"),
        }
    }

    #[test]
    fn test_ready_diarization_false_omitted() {
        let msg = ServerMessage::Ready {
            model: "test".into(),
            sample_rate: 48000,
            version: "1.0".into(),
            supported_rates: vec![],
            diarization: false,
        };
        let json = serde_json::to_string(&msg).unwrap();
        let v: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert!(v.get("diarization").is_none());
    }
}
