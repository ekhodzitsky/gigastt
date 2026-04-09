//! WebSocket protocol messages for gigastt.

use serde::{Deserialize, Serialize};

/// Server → Client messages.
#[derive(Debug, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ServerMessage {
    /// Server is ready to accept audio.
    Ready {
        model: String,
        sample_rate: u32,
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
