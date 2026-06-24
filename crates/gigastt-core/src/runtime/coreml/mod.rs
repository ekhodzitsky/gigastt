//! Native Core ML / Apple Neural Engine backend (encoder on ANE; decoder/joiner on ort).
//! ISOLATION: all `objc2_core_ml` usage MUST stay inside this module (added in a later phase).
//! Gated behind `feature = "ane"`. macOS-only at runtime.
pub mod factory;
pub mod runtime;

/// objc2-core-ml bridge (Phase 2a spike): compile+load a bucket `.mlpackage`
/// and run a Float16 prediction on the Apple Neural Engine. macOS-only.
#[cfg(target_os = "macos")]
pub mod bridge;
