//! Build script that regenerates ONNX protobuf types from `proto/onnx.proto`.
//!
//! Replaces the legacy `onnx-pb 0.1.4` crate (last published in 2020, pulls
//! `prost 0.6` transitively → `RUSTSEC-2021-0073`). Vendoring the `.proto`
//! file plus modern `prost-build` gives us a supported supply chain and a
//! single-source-of-truth update path: when ONNX ships a new opset, bump
//! `proto/onnx.proto` and the types regenerate on the next build.
//!
//! Requires `protoc` in `PATH` (macOS: `brew install protobuf`,
//! Debian/Ubuntu: `apt install protobuf-compiler`). The `prost-build` crate
//! does not bundle `protoc` in its default features — keeping it external
//! avoids pulling the C++ compiler and ~40 MiB of archives into `cargo
//! install`. The generated module lands at `OUT_DIR/onnx.rs` and is included
//! by `src/onnx_proto.rs`.

fn main() -> Result<(), Box<dyn std::error::Error>> {
    prost_build::Config::new()
        .compile_protos(&["proto/onnx.proto"], &["proto"])
        .map_err(|e| -> Box<dyn std::error::Error> {
            format!(
                "prost-build failed to compile proto/onnx.proto: {e}. \
                 Ensure `protoc` is installed and on PATH."
            )
            .into()
        })?;
    println!("cargo:rerun-if-changed=proto/onnx.proto");
    println!("cargo:rerun-if-changed=build.rs");
    Ok(())
}
