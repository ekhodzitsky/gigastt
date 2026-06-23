//! Native Core ML / Apple Neural Engine backend (encoder on ANE; decoder/joiner on ort).
//! ISOLATION: all `objc2_core_ml` usage MUST stay inside this module (added in a later phase).
//! Gated behind `feature = "ane"`. macOS-only at runtime.
pub mod factory;
pub mod runtime;
