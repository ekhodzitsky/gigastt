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

/// Returns an ANE factory stub (Phase 0: not yet implemented).
#[cfg(feature = "ane")]
pub fn ane_factory() -> Box<dyn RuntimeFactory> {
    Box::new(coreml::factory::AneFactory::new())
}

#[allow(unused_imports)]
pub use error::RuntimeError;
#[allow(unused_imports)]
pub use factory::{Runtime, RuntimeFactory};
#[allow(unused_imports)]
pub use ort::factory::{cpu_factory, production_factory};
#[allow(unused_imports)]
pub use session::RuntimeSession;
#[allow(unused_imports)]
pub use tensor::{ElementType, Shape, Tensor, TensorData, TensorDataView, TensorView};
