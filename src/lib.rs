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
//! let mut state = engine.create_state(/* diarization_enabled: */ false);
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
//!
//! ## Feature flags
//!
//! | Feature | Default | Description |
//! |---------|---------|-------------|
//! | `server` | yes | HTTP + WebSocket server (`axum`) |
//! | `diarization` | yes | Speaker diarization (`polyvoice`) |
//! | `coreml` | no | CoreML execution provider (macOS ARM64) |
//! | `cuda` | no | CUDA execution provider (Linux x86_64) |
//! | `ffi` | no | C-ABI FFI for mobile integration |

pub mod error;
pub mod inference;
pub mod model;
pub mod onnx_proto;
pub mod protocol;
pub mod quantize;

#[cfg(feature = "server")]
pub mod server;

#[cfg(feature = "ffi")]
pub mod ffi;
