# gigastt-quantize

Native Rust INT8 dynamic quantizer for ONNX encoder graphs, extracted from
[`gigastt`](https://github.com/ekhodzitsky/gigastt).

Produces per-channel `MatMulInteger` graphs, shrinking the GigaAM v3 encoder
from ~844 MiB to ~305 MiB. Convolutions stay in FP32: dynamic activation
quantization in `ConvInteger` can increase phrase loss and run slower.
`QuantizeOps { conv: true }` retains the experimental legacy recipe.
A conversion with no quantizable weights fails instead of labelling FP32 as INT8.

Requires `protoc` in `PATH` at build time: the ONNX protobuf types are
regenerated from the vendored `proto/onnx.proto` via `prost-build`.

```rust
gigastt_quantize::quantize_model(&input_path, &output_path)?;
```

MIT.
