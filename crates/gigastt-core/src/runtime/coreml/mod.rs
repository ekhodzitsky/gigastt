//! Native Core ML / Apple Neural Engine backend (encoder on ANE; decoder/joiner on ort).
//! ISOLATION: all `objc2_core_ml` usage MUST stay inside this module.
//! Gated behind `feature = "ane"`; the live runtime is additionally macOS-only
//! (the Apple frameworks only link on macOS).

/// The composite ANE factory/runtime + encoder session are macOS-only: they
/// hold and predict on `MLModel` handles produced by [`bridge`]. On non-macOS
/// the `ane` feature degrades to the ort path (see `production_factory`).
#[cfg(target_os = "macos")]
pub mod encoder_session;
#[cfg(target_os = "macos")]
pub mod factory;
#[cfg(target_os = "macos")]
pub mod runtime;

/// objc2-core-ml bridge: compile+load a bucket `.mlpackage` and run a Float16
/// prediction on the Apple Neural Engine. macOS-only.
#[cfg(target_os = "macos")]
pub mod bridge;
