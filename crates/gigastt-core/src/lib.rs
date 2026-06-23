#[cfg(all(
    feature = "candle",
    any(feature = "coreml", feature = "cuda", feature = "nnapi")
))]
compile_error!("feature `candle` is mutually exclusive with `coreml`/`cuda`/`nnapi`");

pub mod error;
pub mod export;
pub mod inference;
pub mod itn;
pub mod lexicon;
pub mod model;
pub mod onnx_proto;
pub mod protocol;
pub mod punctuation;
pub mod quantize;
pub(crate) mod runtime;
pub mod vad;

pub use runtime::cpu_factory;

/// Runtime abstraction surface needed to drive backends directly (e.g. parity
/// tests that construct and compare the ort and candle encoder sessions).
pub mod runtime_api {
    #[cfg(feature = "candle")]
    pub use crate::runtime::candle_factory;
    pub use crate::runtime::{
        Runtime, RuntimeError, RuntimeFactory, RuntimeSession, Shape, Tensor, TensorData,
        TensorDataView, cpu_factory, production_factory,
    };
}
