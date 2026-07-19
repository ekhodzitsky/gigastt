pub mod error;
pub mod factory;
pub mod mock;
pub mod ort;
pub mod session;
pub mod tensor;

#[cfg(feature = "candle")]
pub mod candle;

#[cfg(feature = "ane")]
pub mod coreml;

/// Returns a Candle factory (Metal on Apple Silicon, CPU otherwise).
#[cfg(feature = "candle")]
pub fn candle_factory() -> Box<dyn RuntimeFactory> {
    Box::new(candle::factory::CandleFactory::new())
}

/// Returns the composite ANE factory (encoder on the Neural Engine, decoder /
/// joiner on ort). macOS-only; off macOS the `ane` feature degrades to the ort
/// path, so this helper is not provided there.
#[cfg(all(feature = "ane", target_os = "macos"))]
pub fn ane_factory() -> Box<dyn RuntimeFactory> {
    Box::new(coreml::factory::AneFactory::new())
}

#[allow(unused_imports)]
pub use error::RuntimeError;
#[allow(unused_imports)]
pub use factory::{Runtime, RuntimeFactory};
pub(crate) use ort::factory::production_factory_variant;
#[allow(unused_imports)]
pub use ort::factory::{cpu_factory, production_factory};
#[allow(unused_imports)]
pub use session::RuntimeSession;
#[allow(unused_imports)]
pub use tensor::{ElementType, Shape, Tensor, TensorData, TensorDataView, TensorView};
