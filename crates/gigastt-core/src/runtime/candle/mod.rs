//! Pure-Rust Candle inference backend (Metal on Apple Silicon, else CPU).
//!
//! ISOLATION: all `candle_core`/`candle_nn` usage MUST stay inside this module
//! (`runtime/candle/`). The rest of the crate talks only to the RuntimeFactory/
//! Runtime/RuntimeSession traits. Gated behind `feature = "candle"`.
pub mod config;
pub mod conformer;
pub mod factory;
pub mod runtime;
pub mod session;
pub mod tensor;
