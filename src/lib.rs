//! # gigastt
//!
//! Local speech-to-text powered by GigaAM v3 e2e_rnnt — on-device Russian speech
//! recognition via ONNX Runtime. No cloud APIs, no API keys, full privacy.
//!
//! ## Quick start
//!
//! ```ignore
//! use gigastt::inference::Engine;
//!
//! let engine = Engine::load("~/.gigastt/models")?;
//!
//! // File transcription
//! let text = engine.transcribe_file("audio.wav")?;
//!
//! // Streaming recognition
//! let mut state = engine.create_state(/* diarization_enabled (diarization feature only): */ false);
//! let segments = engine.process_chunk(&audio_16khz, &mut state)?;
//! ```
//!
//! ## Modules
//!
//! - [`inference`] — ONNX inference engine, streaming state, audio utilities
//! - [`error`] — Typed error types ([`GigasttError`](error::GigasttError))
//! - [`protocol`] — WebSocket JSON message types
//! - [`server`] — WebSocket server entry point
//! - [`model`] — Model download and management

pub mod error;
pub mod inference;
pub mod model;
pub mod protocol;
pub mod server;
