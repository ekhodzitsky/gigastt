#[cfg(all(
    feature = "candle",
    any(feature = "coreml", feature = "cuda", feature = "nnapi")
))]
compile_error!("feature `candle` is mutually exclusive with `coreml`/`cuda`/`nnapi`");

#[cfg(all(
    feature = "ane",
    any(
        feature = "coreml",
        feature = "cuda",
        feature = "nnapi",
        feature = "candle"
    )
))]
compile_error!("feature `ane` is mutually exclusive with `coreml`/`cuda`/`nnapi`/`candle`");

pub mod error;
pub mod export;
pub mod inference;
pub mod itn;
pub mod lexicon;
pub mod model;
pub mod protocol;
pub mod punctuation;

/// INT8 quantizer, re-exported so `gigastt_core::quantize::quantize_model`
/// keeps working. Behind the default-on `quantize` feature: lean embedders that
/// side-load a pre-quantized model can turn it off and drop `prost` — and the
/// `protoc` build requirement — entirely.
#[cfg(feature = "quantize")]
pub use gigastt_quantize as quantize;

/// ONNX protobuf types used by the quantizer. Re-exported at the historical
/// `gigastt_core::onnx_proto` path so existing dependents keep compiling.
/// Nested modules are re-exported explicitly so paths like
/// `onnx_proto::tensor_proto::DataType` remain reachable.
#[cfg(feature = "quantize")]
pub mod onnx_proto {
    pub use gigastt_quantize::onnx_proto::AttributeProto;
    pub use gigastt_quantize::onnx_proto::DeviceConfigurationProto;
    pub use gigastt_quantize::onnx_proto::FunctionProto;
    pub use gigastt_quantize::onnx_proto::GraphProto;
    pub use gigastt_quantize::onnx_proto::IntIntListEntryProto;
    pub use gigastt_quantize::onnx_proto::ModelProto;
    pub use gigastt_quantize::onnx_proto::NodeDeviceConfigurationProto;
    pub use gigastt_quantize::onnx_proto::NodeProto;
    pub use gigastt_quantize::onnx_proto::OperatorSetIdProto;
    pub use gigastt_quantize::onnx_proto::OperatorStatus;
    pub use gigastt_quantize::onnx_proto::ShardedDimProto;
    pub use gigastt_quantize::onnx_proto::ShardingSpecProto;
    pub use gigastt_quantize::onnx_proto::SimpleShardedDimProto;
    pub use gigastt_quantize::onnx_proto::SparseTensorProto;
    pub use gigastt_quantize::onnx_proto::StringStringEntryProto;
    pub use gigastt_quantize::onnx_proto::TensorAnnotation;
    pub use gigastt_quantize::onnx_proto::TensorProto;
    pub use gigastt_quantize::onnx_proto::TensorShapeProto;
    pub use gigastt_quantize::onnx_proto::TrainingInfoProto;
    pub use gigastt_quantize::onnx_proto::TypeProto;
    pub use gigastt_quantize::onnx_proto::ValueInfoProto;
    pub use gigastt_quantize::onnx_proto::Version;
    pub use gigastt_quantize::onnx_proto::attribute_proto;
    pub use gigastt_quantize::onnx_proto::simple_sharded_dim_proto;
    pub use gigastt_quantize::onnx_proto::tensor_proto;
    pub use gigastt_quantize::onnx_proto::tensor_shape_proto;
    pub use gigastt_quantize::onnx_proto::type_proto;
}

pub(crate) mod runtime;
mod sha256;
pub mod vad;
mod wordpiece;

pub use runtime::cpu_factory;
pub use runtime::ort::selection::ExecutionProviderChoice;

/// Model-free mock engines for tests. Behind `__internals` (and `cfg(test)`
/// inside this crate). Not a stable public surface.
#[cfg(any(test, feature = "__internals"))]
pub mod test_support;

/// Runtime abstraction surface needed to drive backends directly (e.g. parity
/// tests that construct and compare the ort and candle encoder sessions).
pub mod runtime_api {
    #[cfg(all(feature = "ane", target_os = "macos"))]
    pub use crate::runtime::ane_factory;
    #[cfg(feature = "candle")]
    pub use crate::runtime::candle_factory;
    pub use crate::runtime::{
        Runtime, RuntimeError, RuntimeFactory, RuntimeSession, Shape, Tensor, TensorData,
        TensorDataView, cpu_factory, production_factory,
    };
}
