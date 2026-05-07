//! ONNX protobuf types, generated at build time from `proto/onnx.proto`.
//!
//! The `.proto` file is vendored in `proto/` (MIT-licensed, copied from
//! <https://github.com/onnx/onnx>). `build.rs` runs `prost-build` on every
//! build; generated code lives under `OUT_DIR/onnx.rs` and is inlined here
//! so callers can write `crate::onnx_proto::ModelProto` instead of a
//! deeper package-qualified path.
//!
//! Replaces `onnx-pb 0.1.4` (unmaintained, pinned to `prost 0.6`).

#![allow(clippy::all)]
#![allow(missing_docs)]

include!(concat!(env!("OUT_DIR"), "/onnx.rs"));
