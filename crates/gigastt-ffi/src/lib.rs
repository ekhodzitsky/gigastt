#![deny(unsafe_code)]
#![deny(unsafe_op_in_unsafe_fn)]

//! C-ABI FFI layer for Android / JNI integration.
//!
//! Exposes a minimal surface so that Kotlin (or any other JNI consumer) can:
//! 1. Load the inference engine (`gigastt_engine_new`).
//! 2. Transcribe a WAV file (`gigastt_transcribe_file`).
//! 3. Stream audio in real-time (`gigastt_stream_new`, `gigastt_stream_process_chunk`,
//!    `gigastt_stream_flush`).
//! 4. Free the returned C string (`gigastt_string_free`).
//! 5. Tear down the engine (`gigastt_engine_free`).
//!
//! C entry points are explicitly unsafe; null and handle checks do not validate
//! foreign memory. Engine and stream handles are opaque table IDs, never
//! dereferenced by callers. For pointer/lifetime contracts, see each exported
//! function's Safety section and the generated C header.

mod boundary;
mod engine;
mod file;
#[forbid(unsafe_code)]
mod handles;
mod quantize;
mod stream;
mod v4;

pub use boundary::gigastt_string_free;
pub use engine::{
    gigastt_engine_free, gigastt_engine_new, gigastt_engine_new_with_pool_size,
    gigastt_engine_new_with_threads,
};
pub use file::gigastt_transcribe_file;
pub use handles::{GigasttEngine, GigasttStream, GigasttV4Engine};
pub use quantize::gigastt_quantize_model;
pub use stream::{
    gigastt_stream_flush, gigastt_stream_free, gigastt_stream_new, gigastt_stream_process_chunk,
};
pub use v4::{
    GigasttV4EngineResult, gigastt_v4_engine_free, gigastt_v4_engine_new, gigastt_v4_supported,
    gigastt_v4_transcribe_file,
};

#[cfg(test)]
#[allow(unsafe_code)] // Calls the unsafe C ABI with test-owned data.
mod tests;
