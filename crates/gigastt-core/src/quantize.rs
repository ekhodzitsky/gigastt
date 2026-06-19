//! Dynamic INT8 (QOperator) quantization for ONNX encoder models.
//!
//! Native Rust replacement for `scripts/quantize.py`. Auto-invoked after
//! `gigastt download` and `gigastt serve` (see `src/main.rs`); also exposed
//! as the `gigastt quantize` subcommand.
//!
//! This emits the **dynamic INT8 (QOperator)** form that ONNX Runtime's
//! `quantize_dynamic(..., weight_type=QInt8)` produces:
//! `DynamicQuantizeLinear` on activations feeding integer compute kernels
//! (`MatMulInteger` / `ConvInteger`), with a per-channel float rescale on the
//! int32 output. This is fundamentally faster than weight-only `QDQ`
//! (`DequantizeLinear` → float `MatMul`/`Conv`), which stores int8 weights but
//! dequantizes them back to float at load and runs the heavy ops in FP32.
//!
//! The protobuf types come from `crate::onnx_proto`, which is generated at
//! build time from `proto/onnx.proto` via `prost-build` (see `build.rs`).
//! Fields that are `optional` in proto2 surface as `Option<T>` in prost
//! 0.13, so we lean on the generated accessor methods (`data_type()`,
//! `name()`, `op_type()`, …) for reads and wrap writes in `Some(_)`.

use anyhow::{Context, Result};
use prost::Message;
use std::collections::{HashMap, HashSet};
use std::path::Path;

use crate::onnx_proto::{AttributeProto, ModelProto, NodeProto, TensorProto};

/// ONNX data types (from onnx.proto `TensorProto.DataType`).
const FLOAT: i32 = 1;
const INT8: i32 = 3;
const INT64: i32 = 7;

/// ONNX attribute types (from onnx.proto `AttributeProto.AttributeType`).
const ATTR_INT: i32 = 2;

/// `Cast` `to` value for FLOAT (matches `TensorProto.DataType.FLOAT`).
const CAST_TO_FLOAT: i64 = FLOAT as i64;

/// Minimum opset (domain "") for `DynamicQuantizeLinear` (≥11).
/// `MatMulInteger` / `ConvInteger` need ≥10, so 11 covers everything we emit.
const MIN_OPSET: i64 = 11;

/// Node types whose weights benefit from INT8 quantization.
const QUANTIZABLE_OPS: &[&str] = &["MatMul", "Conv", "Gemm"];

/// Minimum number of elements in a tensor to quantize (skip small biases).
const MIN_ELEMENTS: usize = 1024;

/// A weight that has been quantized once and may be shared across ops.
struct QuantizedWeight {
    /// `{weight}_quantized` — INT8 initializer name.
    q_name: String,
    /// `{weight}_scale` — FLOAT [N] initializer name.
    s_name: String,
    /// `{weight}_zero_point` — INT8 [N] zeros initializer name.
    zp_name: String,
}

/// Quantize an ONNX model's float32 weights to dynamic INT8 (QOperator form).
///
/// For each quantizable weight tensor (MatMul/Conv/Gemm) the original float op
/// is **replaced** by a `DynamicQuantizeLinear` → `MatMulInteger`/`ConvInteger`
/// → `Cast` → per-channel `Mul` (+ optional bias `Add`) chain, with the weight
/// stored as a per-channel-symmetric INT8 initializer.
pub fn quantize_model(input: &Path, output: &Path) -> Result<()> {
    let model_bytes = std::fs::read(input).context("Failed to read ONNX model")?;
    let mut model =
        ModelProto::decode(&model_bytes[..]).context("Failed to decode ONNX protobuf")?;

    // Ensure opset (domain "") is high enough for the integer ops we emit.
    bump_opset(&mut model);

    let graph = model.graph.as_mut().context("Model has no graph")?;

    // Build map: initializer_name → index.
    let init_map: HashMap<String, usize> = graph
        .initializer
        .iter()
        .enumerate()
        .map(|(i, t)| (t.name().to_string(), i))
        .collect();

    // Collect quantization targets: (node_index, weight_input_index, weight_name, init_index).
    let mut targets = Vec::new();
    for (ni, node) in graph.node.iter().enumerate() {
        if !QUANTIZABLE_OPS.contains(&node.op_type()) {
            continue;
        }
        // Weight is input[1] for MatMul/Conv/Gemm.
        if node.input.len() < 2 {
            continue;
        }
        let weight_name = &node.input[1];
        if let Some(&init_idx) = init_map.get(weight_name) {
            let init = &graph.initializer[init_idx];
            if init.data_type() != FLOAT {
                continue;
            }
            let num_elements: i64 = init.dims.iter().product();
            if num_elements > 0 && num_elements as usize >= MIN_ELEMENTS {
                targets.push((ni, 1usize, weight_name.clone(), init_idx));
            }
        }
    }

    tracing::info!(
        "Found {} quantizable weight tensors in {} nodes",
        targets.len(),
        graph.node.len()
    );

    // First pass: quantize each distinct weight once (shared-weight dedup).
    let mut new_initializers = Vec::new();
    let mut quantized: HashMap<String, QuantizedWeight> = HashMap::new();

    for (node_idx, _input_idx, weight_name, init_idx) in &targets {
        if quantized.contains_key(weight_name) {
            continue;
        }

        let init = &graph.initializer[*init_idx];
        let float_data = extract_float_data(init)?;
        let dims = init.dims.clone();

        if dims.is_empty() {
            continue;
        }

        let expected_elements: usize = dims.iter().map(|&d: &i64| d.max(0) as usize).product();
        if expected_elements != float_data.len() {
            tracing::warn!(
                "Skipping tensor '{}': shape mismatch (dims={:?}, data={})",
                init.name(),
                dims,
                float_data.len()
            );
            continue;
        }

        // Pick the per-output-channel axis from the consuming op's semantics.
        // Quantizing along the wrong axis groups unrelated output channels under
        // one scale, silently inflating quantization error (and WER): a Conv
        // weight is `[out_channels, ...]` (axis 0), a MatMul weight is
        // `[..., K, N]` (output channel = last dim N), and a Gemm weight is
        // `[K, N]` or — when `transB=1` — `[N, K]`, so N's axis flips with it.
        let node = &graph.node[*node_idx];
        let axis = per_channel_axis(node.op_type(), node, dims.len());
        let channels = dims[axis].max(0) as usize;
        if channels == 0 {
            continue;
        }

        let (quantized_data, scales) = quantize_per_channel(&float_data, &dims, axis);
        let q_name = format!("{weight_name}_quantized");
        let s_name = format!("{weight_name}_scale");
        let zp_name = format!("{weight_name}_zero_point");

        // Quantized weight tensor (INT8).
        new_initializers.push(TensorProto {
            name: Some(q_name.clone()),
            dims,
            data_type: Some(INT8),
            raw_data: Some(quantized_data.iter().map(|&v| v as u8).collect()),
            ..Default::default()
        });

        // Per-channel scale (FLOAT [N]) — this carries the per-channel accuracy
        // and is applied as a float rescale on the integer op's int32 output.
        new_initializers.push(TensorProto {
            name: Some(s_name.clone()),
            dims: vec![channels as i64],
            data_type: Some(FLOAT),
            float_data: scales,
            ..Default::default()
        });

        // Weight zero-point: a per-TENSOR scalar INT8 zero. ONNX Runtime's CPU
        // `MatMulInteger` / `ConvInteger` kernels reject a per-channel weight
        // zero-point ("Non per-tensor quantization is not supported now"), and
        // for symmetric quantization the zero-point is 0 on every channel, so a
        // scalar 0 is numerically exact — the per-channel scale above keeps the
        // accuracy.
        new_initializers.push(TensorProto {
            name: Some(zp_name.clone()),
            dims: vec![],
            data_type: Some(INT8),
            raw_data: Some(vec![0u8]),
            ..Default::default()
        });

        quantized.insert(
            weight_name.clone(),
            QuantizedWeight {
                q_name,
                s_name,
                zp_name,
            },
        );
    }

    // Second pass: build the replacement op chain for every quantizable node.
    // `replacements[node_idx] = Vec<NodeProto>` substitutes the original node.
    let mut replacements: HashMap<usize, Vec<NodeProto>> = HashMap::new();
    let mut chain_initializers = Vec::new();

    for (node_idx, _input_idx, weight_name, _init_idx) in &targets {
        let Some(qw) = quantized.get(weight_name) else {
            continue; // weight was skipped above (shape mismatch / zero channels)
        };
        let node = &graph.node[*node_idx];
        let op_type = node.op_type();

        let chain = match op_type {
            "Conv" => build_conv_chain(node, qw, &graph.initializer, &mut chain_initializers),
            // MatMul and Gemm-as-MatMul: B is the per-channel weight on its N axis.
            _ => build_matmul_chain(node, qw),
        };
        replacements.insert(*node_idx, chain);
    }

    // Reassemble the node list: drop replaced float ops, splice in their chains,
    // and prepend any leftover quantizable nodes' chains in graph order.
    let original_nodes = std::mem::take(&mut graph.node);
    let mut rebuilt = Vec::with_capacity(original_nodes.len());
    for (idx, node) in original_nodes.into_iter().enumerate() {
        if let Some(chain) = replacements.remove(&idx) {
            rebuilt.extend(chain);
        } else {
            rebuilt.push(node);
        }
    }
    graph.node = rebuilt;

    // Remove original float weight initializers that we quantized.
    let quantized_weight_names: HashSet<&str> = quantized.keys().map(|s| s.as_str()).collect();
    graph
        .initializer
        .retain(|t| !quantized_weight_names.contains(t.name()));

    // Add quantized weight initializers + chain reshape/shape initializers.
    graph.initializer.extend(new_initializers);
    graph.initializer.extend(chain_initializers);

    // Write quantized model (atomic: write to partial, then rename).
    // Uses the `.partial` suffix convention shared with `src/model/mod.rs`
    // downloads so both pipelines leave identical breadcrumbs after a crash.
    let mut output_bytes = Vec::new();
    model
        .encode(&mut output_bytes)
        .context("Failed to encode quantized model")?;
    let mut partial_os: std::ffi::OsString = output.as_os_str().to_owned();
    partial_os.push(".partial");
    let partial = std::path::PathBuf::from(partial_os);
    std::fs::write(&partial, &output_bytes).context("Failed to write quantized model")?;
    std::fs::rename(&partial, output).context("Failed to finalize quantized model")?;

    let in_mb = model_bytes.len() as f64 / (1024.0 * 1024.0);
    let out_mb = output_bytes.len() as f64 / (1024.0 * 1024.0);
    tracing::info!(
        "Quantized: {in_mb:.0}MB → {out_mb:.0}MB ({:.1}x smaller)",
        in_mb / out_mb
    );

    Ok(())
}

/// Ensure the model imports opset (domain "") ≥ [`MIN_OPSET`]; bump it if lower
/// and add the default operator set if the model declares none.
fn bump_opset(model: &mut ModelProto) {
    let default = model
        .opset_import
        .iter_mut()
        .find(|o| o.domain() == "" || o.domain() == "ai.onnx");
    match default {
        Some(o) => {
            if o.version() < MIN_OPSET {
                o.version = Some(MIN_OPSET);
            }
        }
        None => {
            model
                .opset_import
                .push(crate::onnx_proto::OperatorSetIdProto {
                    domain: Some(String::new()),
                    version: Some(MIN_OPSET),
                });
        }
    }
}

/// Build the dynamic-INT8 replacement chain for a `MatMul`/`Gemm`:
/// `Y = (Cast(MatMulInteger(DynQ(A), W)) * (a_scale * W_scale))`.
fn build_matmul_chain(node: &NodeProto, qw: &QuantizedWeight) -> Vec<NodeProto> {
    let base = node_base_name(node);
    let a_input = node.input[0].clone();
    let y_output = node.output[0].clone();

    let a_q = format!("{base}_a_q");
    let a_scale = format!("{base}_a_scale");
    let a_zp = format!("{base}_a_zp");
    let mm_i32 = format!("{base}_mm_i32");
    let mm_f32 = format!("{base}_mm_f32");
    let scale_vec = format!("{base}_scale_vec");

    vec![
        // DynamicQuantizeLinear(A) → (a_q: uint8, a_scale: f32 scalar, a_zp: uint8 scalar)
        NodeProto {
            op_type: Some("DynamicQuantizeLinear".into()),
            input: vec![a_input],
            output: vec![a_q.clone(), a_scale.clone(), a_zp.clone()],
            name: Some(format!("{base}_dynq")),
            ..Default::default()
        },
        // MatMulInteger((A - a_zp) @ (W - 0)) → int32. A is uint8, W int8.
        NodeProto {
            op_type: Some("MatMulInteger".into()),
            input: vec![a_q, qw.q_name.clone(), a_zp, qw.zp_name.clone()],
            output: vec![mm_i32.clone()],
            name: Some(format!("{base}_matmulinteger")),
            ..Default::default()
        },
        // Cast int32 → float.
        cast_to_float(&mm_i32, &mm_f32, &format!("{base}_cast")),
        // Combined per-channel scale: a_scale (scalar) * W_scale ([N]) → [N].
        NodeProto {
            op_type: Some("Mul".into()),
            input: vec![a_scale, qw.s_name.clone()],
            output: vec![scale_vec.clone()],
            name: Some(format!("{base}_scale_mul")),
            ..Default::default()
        },
        // Rescale: mm_f32 ([..., N]) * scale_vec ([N]) → Y (original output name).
        NodeProto {
            op_type: Some("Mul".into()),
            input: vec![mm_f32, scale_vec],
            output: vec![y_output],
            name: Some(format!("{base}_rescale")),
            ..Default::default()
        },
    ]
}

/// Build the dynamic-INT8 replacement chain for a `Conv`:
/// `Y = Cast(ConvInteger(DynQ(A), W)) * reshape(a_scale * W_scale) [+ reshape(bias)]`.
///
/// The Conv's spatial attributes (`strides`, `pads`, `dilations`, `group`,
/// `kernel_shape`, `auto_pad`) are copied verbatim onto `ConvInteger`, which
/// takes no bias — any bias is folded back as a trailing `Add`.
fn build_conv_chain(
    node: &NodeProto,
    qw: &QuantizedWeight,
    initializers: &[TensorProto],
    chain_initializers: &mut Vec<TensorProto>,
) -> Vec<NodeProto> {
    let base = node_base_name(node);
    let a_input = node.input[0].clone();
    let y_output = node.output[0].clone();

    // Conv weight is [C_out, C_in/groups, *kernel]; the conv output is NCHW-like
    // with the channel on axis 1. The reshape rank matches the weight rank.
    let weight_init = initializers.iter().find(|t| t.name() == node.input[1]);
    let weight_rank = weight_init.map(|t| t.dims.len()).unwrap_or(4).max(2);
    let c_out = weight_init.map(|t| t.dims[0].max(0)).unwrap_or(0);

    let a_q = format!("{base}_a_q");
    let a_scale = format!("{base}_a_scale");
    let a_zp = format!("{base}_a_zp");
    let ci_i32 = format!("{base}_ci_i32");
    let ci_f32 = format!("{base}_ci_f32");
    let scale_c = format!("{base}_scale_c");
    let scale_reshaped = format!("{base}_scale_reshaped");
    let scaled = format!("{base}_scaled");

    let mut nodes = vec![
        // DynamicQuantizeLinear(A) → (a_q: uint8, a_scale: f32, a_zp: uint8)
        NodeProto {
            op_type: Some("DynamicQuantizeLinear".into()),
            input: vec![a_input],
            output: vec![a_q.clone(), a_scale.clone(), a_zp.clone()],
            name: Some(format!("{base}_dynq")),
            ..Default::default()
        },
        // ConvInteger(A, W, a_zp, W_zp) → int32, carrying the original conv attrs.
        NodeProto {
            op_type: Some("ConvInteger".into()),
            input: vec![a_q, qw.q_name.clone(), a_zp, qw.zp_name.clone()],
            output: vec![ci_i32.clone()],
            name: Some(format!("{base}_convinteger")),
            attribute: copy_conv_attrs(node),
            ..Default::default()
        },
        // Cast int32 → float.
        cast_to_float(&ci_i32, &ci_f32, &format!("{base}_cast")),
        // Combined per-channel scale: a_scale (scalar) * W_scale ([C_out]) → [C_out].
        NodeProto {
            op_type: Some("Mul".into()),
            input: vec![a_scale, qw.s_name.clone()],
            output: vec![scale_c.clone()],
            name: Some(format!("{base}_scale_mul")),
            ..Default::default()
        },
    ];

    // Reshape scale [C_out] → [1, C_out, 1, ...] so it broadcasts over the
    // channel axis (axis 1) of the conv output.
    let scale_shape = channel_broadcast_shape(c_out, weight_rank);
    chain_initializers.push(int64_shape_initializer(
        &format!("{base}_scale_shape"),
        &scale_shape,
    ));
    nodes.push(reshape_node(
        &scale_c,
        &format!("{base}_scale_shape"),
        &scale_reshaped,
        &format!("{base}_scale_reshape"),
    ));

    // Does the conv carry a bias (input[2] = float initializer)?
    let bias_name = node.input.get(2).filter(|n| !n.is_empty()).cloned();
    let has_bias = bias_name
        .as_deref()
        .map(|b| initializers.iter().any(|t| t.name() == b))
        .unwrap_or(false);

    if has_bias {
        // scaled = ci_f32 * reshaped_scale
        nodes.push(NodeProto {
            op_type: Some("Mul".into()),
            input: vec![ci_f32, scale_reshaped],
            output: vec![scaled.clone()],
            name: Some(format!("{base}_rescale")),
            ..Default::default()
        });
        // Reshape bias [C_out] → [1, C_out, 1, ...] and add.
        let bias = bias_name.expect("has_bias implies a bias name");
        let bias_reshaped = format!("{base}_bias_reshaped");
        let bias_shape = channel_broadcast_shape(c_out, weight_rank);
        chain_initializers.push(int64_shape_initializer(
            &format!("{base}_bias_shape"),
            &bias_shape,
        ));
        nodes.push(reshape_node(
            &bias,
            &format!("{base}_bias_shape"),
            &bias_reshaped,
            &format!("{base}_bias_reshape"),
        ));
        nodes.push(NodeProto {
            op_type: Some("Add".into()),
            input: vec![scaled, bias_reshaped],
            output: vec![y_output],
            name: Some(format!("{base}_bias_add")),
            ..Default::default()
        });
    } else {
        // No bias: the rescale produces Y directly.
        nodes.push(NodeProto {
            op_type: Some("Mul".into()),
            input: vec![ci_f32, scale_reshaped],
            output: vec![y_output],
            name: Some(format!("{base}_rescale")),
            ..Default::default()
        });
    }

    nodes
}

/// `[1, C_out, 1, ...]` broadcast shape for a conv output of the given rank.
fn channel_broadcast_shape(c_out: i64, rank: usize) -> Vec<i64> {
    let mut shape = vec![1i64; rank.max(2)];
    shape[1] = c_out;
    shape
}

/// Build an INT64 1-D shape initializer (for `Reshape`'s second input).
fn int64_shape_initializer(name: &str, shape: &[i64]) -> TensorProto {
    let mut raw = Vec::with_capacity(shape.len() * 8);
    for &v in shape {
        raw.extend_from_slice(&v.to_le_bytes());
    }
    TensorProto {
        name: Some(name.into()),
        dims: vec![shape.len() as i64],
        data_type: Some(INT64),
        raw_data: Some(raw),
        ..Default::default()
    }
}

/// A `Reshape` node `out = Reshape(data, shape_init)`.
fn reshape_node(data: &str, shape_init: &str, out: &str, name: &str) -> NodeProto {
    NodeProto {
        op_type: Some("Reshape".into()),
        input: vec![data.into(), shape_init.into()],
        output: vec![out.into()],
        name: Some(name.into()),
        ..Default::default()
    }
}

/// A `Cast` node to FLOAT.
fn cast_to_float(input: &str, output: &str, name: &str) -> NodeProto {
    NodeProto {
        op_type: Some("Cast".into()),
        input: vec![input.into()],
        output: vec![output.into()],
        name: Some(name.into()),
        attribute: vec![AttributeProto {
            name: Some("to".into()),
            i: Some(CAST_TO_FLOAT),
            r#type: Some(ATTR_INT),
            ..Default::default()
        }],
        ..Default::default()
    }
}

/// Copy the spatial attributes a `ConvInteger` understands from a `Conv`.
/// `ConvInteger` shares Conv's `strides`/`pads`/`dilations`/`group`/
/// `kernel_shape`/`auto_pad`; everything else (e.g. fused activations) is
/// dropped because there is no float weight to fuse against anymore.
fn copy_conv_attrs(node: &NodeProto) -> Vec<AttributeProto> {
    const CONV_ATTRS: &[&str] = &[
        "strides",
        "pads",
        "dilations",
        "group",
        "kernel_shape",
        "auto_pad",
    ];
    node.attribute
        .iter()
        .filter(|a| CONV_ATTRS.contains(&a.name()))
        .cloned()
        .collect()
}

/// Stable base name for the nodes/tensors we synthesize for a quantized op.
/// Falls back to the op's first output (always present and unique) when the
/// node itself is unnamed, so generated tensor names never collide.
fn node_base_name(node: &NodeProto) -> String {
    let raw = if !node.name().is_empty() {
        node.name().to_string()
    } else {
        node.output
            .first()
            .cloned()
            .unwrap_or_else(|| node.op_type().to_string())
    };
    sanitize(&raw)
}

/// Make a string safe as part of an ONNX tensor name (no `/` or `:`).
fn sanitize(s: &str) -> String {
    s.replace(['/', ':'], "_")
}

/// Extract float32 data from a TensorProto initializer.
fn extract_float_data(tensor: &TensorProto) -> Result<Vec<f32>> {
    if !tensor.float_data.is_empty() {
        return Ok(tensor.float_data.clone());
    }
    if let Some(raw) = tensor.raw_data.as_deref()
        && !raw.is_empty()
    {
        anyhow::ensure!(
            raw.len().is_multiple_of(4),
            "Tensor '{}' raw_data length {} is not aligned to 4 bytes",
            tensor.name(),
            raw.len()
        );
        let num_floats = raw.len() / 4;
        let mut data = Vec::with_capacity(num_floats);
        for chunk in raw.chunks_exact(4) {
            data.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
        }
        return Ok(data);
    }
    anyhow::bail!("Tensor '{}' has no float data", tensor.name());
}

/// Per-output-channel axis for a quantizable weight, chosen from the consuming
/// op's semantics. The scale tensor carries one entry per index along this axis,
/// so it must line up with the operator's *output* channels to keep
/// quantization error low:
/// - `Conv` weight `[out_channels, in/groups, *kernel]` → axis 0.
/// - `Gemm` weight `[K, N]` (`transB=0`) or `[N, K]` (`transB=1`) → N's axis.
/// - `MatMul` (and the fallback) weight `[..., K, N]` → the last axis (N).
fn per_channel_axis(op_type: &str, node: &NodeProto, rank: usize) -> usize {
    let last = rank.saturating_sub(1);
    match op_type {
        "Conv" => 0,
        "Gemm" => {
            if attr_int(node, "transB").unwrap_or(0) != 0 {
                0
            } else {
                last.min(1)
            }
        }
        // MatMul and any other matmul-shaped op: output channel is the last dim.
        _ => last,
    }
}

/// Read an integer attribute by name from a node, if present.
fn attr_int(node: &NodeProto, name: &str) -> Option<i64> {
    node.attribute
        .iter()
        .find(|a| a.name() == name)
        .and_then(|a| a.i)
}

/// Symmetric per-channel INT8 quantization of `data` (row-major, shaped `dims`)
/// along `axis`. Returns the quantized values in the original element order plus
/// one scale per channel (`dims[axis]` entries). All-zero channels get scale 1.0
/// to avoid division by zero. Quantizing along an arbitrary axis requires a
/// strided gather, so this generalises the previous axis-0-only contiguous-block
/// path.
fn quantize_per_channel(data: &[f32], dims: &[i64], axis: usize) -> (Vec<i8>, Vec<f32>) {
    let channels = (dims[axis].max(0) as usize).max(1);
    // Number of contiguous elements between successive indices along `axis`.
    let stride: usize = dims[axis + 1..]
        .iter()
        .map(|&d| d.max(0) as usize)
        .product::<usize>()
        .max(1);

    let mut abs_max = vec![0.0f32; channels];
    for (f, &v) in data.iter().enumerate() {
        let ch = (f / stride) % channels;
        abs_max[ch] = abs_max[ch].max(v.abs());
    }
    let scales: Vec<f32> = abs_max
        .iter()
        .map(|&m| if m == 0.0 { 1.0 } else { m / 127.0 })
        .collect();

    let mut quantized = vec![0i8; data.len()];
    for (f, &v) in data.iter().enumerate() {
        let ch = (f / stride) % channels;
        quantized[f] = (v / scales[ch]).round().clamp(-128.0, 127.0) as i8;
    }
    (quantized, scales)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Round-trip a model through `quantize_model` and return the output graph.
    fn quantize_roundtrip(model: ModelProto) -> crate::onnx_proto::GraphProto {
        let tmp_dir = tempfile::tempdir().unwrap();
        let input_path = tmp_dir.path().join("input.onnx");
        let output_path = tmp_dir.path().join("output.onnx");
        let mut bytes = Vec::new();
        model.encode(&mut bytes).unwrap();
        std::fs::write(&input_path, &bytes).unwrap();
        quantize_model(&input_path, &output_path).unwrap();
        let out_bytes = std::fs::read(&output_path).unwrap();
        ModelProto::decode(&out_bytes[..]).unwrap().graph.unwrap()
    }

    fn matmul_model(weight_name: &str, dims: Vec<i64>, n_elems: usize) -> ModelProto {
        let float_data: Vec<f32> = (0..n_elems).map(|i| i as f32 * 0.001).collect();
        let weight = TensorProto {
            name: Some(weight_name.into()),
            dims,
            data_type: Some(FLOAT),
            float_data,
            ..Default::default()
        };
        let node = NodeProto {
            op_type: Some("MatMul".into()),
            input: vec!["input".into(), weight_name.into()],
            output: vec!["output".into()],
            ..Default::default()
        };
        ModelProto {
            ir_version: Some(8),
            opset_import: vec![crate::onnx_proto::OperatorSetIdProto {
                domain: Some(String::new()),
                version: Some(17),
            }],
            graph: Some(crate::onnx_proto::GraphProto {
                name: Some("test".into()),
                initializer: vec![weight],
                node: vec![node],
                ..Default::default()
            }),
            ..Default::default()
        }
    }

    #[test]
    fn test_extract_float_data_from_float_data_field() {
        let tensor = TensorProto {
            name: Some("test".into()),
            float_data: vec![1.0, 2.0, 3.0],
            ..Default::default()
        };
        let data = extract_float_data(&tensor).unwrap();
        assert_eq!(data, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_extract_float_data_from_raw_data() {
        let mut raw = Vec::new();
        raw.extend_from_slice(&1.0f32.to_le_bytes());
        raw.extend_from_slice(&(-2.5f32).to_le_bytes());
        let tensor = TensorProto {
            name: Some("test".into()),
            raw_data: Some(raw),
            ..Default::default()
        };
        let data = extract_float_data(&tensor).unwrap();
        assert_eq!(data, vec![1.0, -2.5]);
    }

    #[test]
    fn test_extract_float_data_empty() {
        let tensor = TensorProto {
            name: Some("empty".into()),
            ..Default::default()
        };
        assert!(extract_float_data(&tensor).is_err());
    }

    #[test]
    fn test_symmetric_quantization_values() {
        // Verify scale/quantized value computation.
        let val = 1.27f32;
        let scale = val.abs() / 127.0; // = 0.01
        let q = (val / scale).round().clamp(-128.0, 127.0) as i8;
        assert_eq!(q, 127);

        let val2 = -1.27f32;
        let q2 = (val2 / scale).round().clamp(-128.0, 127.0) as i8;
        assert_eq!(q2, -127);
    }

    #[test]
    fn test_zero_scale_handling() {
        // All-zero tensor should get scale=1.0 (not division by zero).
        let data = vec![0.0f32; 100];
        let abs_max = data.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
        let scale = if abs_max == 0.0 { 1.0 } else { abs_max / 127.0 };
        assert_eq!(scale, 1.0);
    }

    #[test]
    fn test_roundtrip_encode_decode_minimal_model() {
        // End-to-end sanity: a tiny ModelProto round-trips through the
        // generated prost codec without losing fields.
        let model = ModelProto {
            ir_version: Some(8),
            producer_name: Some("gigastt-test".into()),
            graph: Some(crate::onnx_proto::GraphProto {
                name: Some("tiny".into()),
                node: vec![NodeProto {
                    op_type: Some("Identity".into()),
                    input: vec!["x".into()],
                    output: vec!["y".into()],
                    ..Default::default()
                }],
                ..Default::default()
            }),
            ..Default::default()
        };
        let mut bytes = Vec::new();
        model.encode(&mut bytes).unwrap();
        let decoded = ModelProto::decode(&bytes[..]).unwrap();
        assert_eq!(decoded.ir_version(), 8);
        assert_eq!(decoded.producer_name(), "gigastt-test");
        let g = decoded.graph.as_ref().unwrap();
        assert_eq!(g.name(), "tiny");
        assert_eq!(g.node.len(), 1);
        assert_eq!(g.node[0].op_type(), "Identity");
    }

    #[test]
    fn test_extract_float_data_raw_misaligned() {
        let tensor = TensorProto {
            name: Some("misaligned".into()),
            raw_data: Some(vec![0x01, 0x02, 0x03]),
            ..Default::default()
        };
        let err = extract_float_data(&tensor).unwrap_err().to_string();
        assert!(
            err.contains("not aligned to 4 bytes"),
            "Error should mention alignment: {err}"
        );
    }

    #[test]
    fn test_quantize_model_matmul_emits_integer_chain() {
        let g = quantize_roundtrip(matmul_model("weight", vec![32, 32], 1024));

        // No weight-only DequantizeLinear path remains.
        assert_eq!(
            g.node
                .iter()
                .filter(|n| n.op_type() == "DequantizeLinear")
                .count(),
            0,
            "Dynamic-INT8 form must not emit DequantizeLinear"
        );
        // The original float MatMul is replaced (no float MatMul left).
        assert_eq!(
            g.node.iter().filter(|n| n.op_type() == "MatMul").count(),
            0,
            "Original float MatMul should be removed"
        );

        // Exactly one of each integer-path op.
        let dynq: Vec<_> = g
            .node
            .iter()
            .filter(|n| n.op_type() == "DynamicQuantizeLinear")
            .collect();
        assert_eq!(dynq.len(), 1);
        let mmi: Vec<_> = g
            .node
            .iter()
            .filter(|n| n.op_type() == "MatMulInteger")
            .collect();
        assert_eq!(mmi.len(), 1);

        // DynamicQuantizeLinear: input = original activation, 3 outputs.
        let dynq = dynq[0];
        assert_eq!(dynq.input, vec!["input".to_string()]);
        assert_eq!(dynq.output.len(), 3);
        let (a_q, a_scale, a_zp) = (&dynq.output[0], &dynq.output[1], &dynq.output[2]);

        // MatMulInteger: [a_q, W_quantized, a_zp, W_zero_point] → mm_i32.
        let mmi = mmi[0];
        assert_eq!(
            mmi.input,
            vec![
                a_q.clone(),
                "weight_quantized".to_string(),
                a_zp.clone(),
                "weight_zero_point".to_string(),
            ]
        );
        let mm_i32 = &mmi.output[0];

        // Cast(mm_i32 → f32) with to=FLOAT.
        let cast = g
            .node
            .iter()
            .find(|n| n.op_type() == "Cast" && n.input == vec![mm_i32.clone()])
            .expect("Cast node feeding off MatMulInteger output");
        let to = cast.attribute.iter().find(|a| a.name() == "to").unwrap();
        assert_eq!(to.i, Some(CAST_TO_FLOAT));
        let mm_f32 = &cast.output[0];

        // scale_vec = Mul(a_scale, weight_scale).
        let scale_mul = g
            .node
            .iter()
            .find(|n| {
                n.op_type() == "Mul" && n.input == vec![a_scale.clone(), "weight_scale".to_string()]
            })
            .expect("scale Mul(a_scale, weight_scale)");
        let scale_vec = &scale_mul.output[0];

        // Final Mul(mm_f32, scale_vec) → original output name.
        let rescale = g
            .node
            .iter()
            .find(|n| n.op_type() == "Mul" && n.input == vec![mm_f32.clone(), scale_vec.clone()])
            .expect("final rescale Mul");
        assert_eq!(
            rescale.output,
            vec!["output".to_string()],
            "Final Mul must produce the original op's output name"
        );

        // Quantized weight set present; original float weight removed.
        let init_names: Vec<_> = g.initializer.iter().map(|t| t.name()).collect();
        assert!(!init_names.contains(&"weight"), "float weight removed");
        assert!(init_names.contains(&"weight_quantized"));
        assert!(init_names.contains(&"weight_scale"));
        assert!(init_names.contains(&"weight_zero_point"));
    }

    #[test]
    fn test_quantize_model_weight_types_and_scale_length() {
        let g = quantize_roundtrip(matmul_model("weight", vec![32, 32], 1024));

        let wq = g
            .initializer
            .iter()
            .find(|t| t.name() == "weight_quantized")
            .unwrap();
        assert_eq!(wq.data_type(), INT8, "weight stored as INT8");
        assert_eq!(wq.dims, vec![32, 32]);

        let ws = g
            .initializer
            .iter()
            .find(|t| t.name() == "weight_scale")
            .unwrap();
        assert_eq!(ws.data_type(), FLOAT);
        assert_eq!(ws.dims, vec![32], "per-channel scale length == N");
        assert_eq!(ws.float_data.len(), 32);

        let wzp = g
            .initializer
            .iter()
            .find(|t| t.name() == "weight_zero_point")
            .unwrap();
        assert_eq!(wzp.data_type(), INT8, "weight zero-point is INT8");
        assert_eq!(
            wzp.dims,
            Vec::<i64>::new(),
            "weight zero-point is a per-tensor scalar (ORT integer kernels reject per-channel)"
        );
        assert_eq!(
            wzp.raw_data.as_deref(),
            Some(&[0u8][..]),
            "symmetric → scalar zero"
        );
    }

    #[test]
    fn test_quantize_model_small_tensor_skipped() {
        let g = quantize_roundtrip(matmul_model("small_weight", vec![16, 16], 256));

        assert_eq!(
            g.node
                .iter()
                .filter(|n| n.op_type() == "MatMulInteger")
                .count(),
            0,
            "Small tensor should be skipped"
        );
        // Original float MatMul + weight untouched.
        assert_eq!(g.node.iter().filter(|n| n.op_type() == "MatMul").count(), 1);
        let init_names: Vec<_> = g.initializer.iter().map(|t| t.name()).collect();
        assert!(init_names.contains(&"small_weight"));
        assert!(!init_names.contains(&"small_weight_quantized"));
    }

    #[test]
    fn test_quantize_model_shared_weights() {
        let float_data: Vec<f32> = (0..1024).map(|i| i as f32 * 0.001).collect();
        let weight = TensorProto {
            name: Some("shared_weight".into()),
            dims: vec![32, 32],
            data_type: Some(FLOAT),
            float_data,
            ..Default::default()
        };
        let node1 = NodeProto {
            op_type: Some("MatMul".into()),
            input: vec!["a".into(), "shared_weight".into()],
            output: vec!["b".into()],
            name: Some("mm1".into()),
            ..Default::default()
        };
        let node2 = NodeProto {
            op_type: Some("MatMul".into()),
            input: vec!["c".into(), "shared_weight".into()],
            output: vec!["d".into()],
            name: Some("mm2".into()),
            ..Default::default()
        };
        let model = ModelProto {
            ir_version: Some(8),
            opset_import: vec![crate::onnx_proto::OperatorSetIdProto {
                domain: Some(String::new()),
                version: Some(17),
            }],
            graph: Some(crate::onnx_proto::GraphProto {
                name: Some("test".into()),
                initializer: vec![weight],
                node: vec![node1, node2],
                ..Default::default()
            }),
            ..Default::default()
        };
        let g = quantize_roundtrip(model);

        // ONE quantized weight set, shared by both ops.
        let init_names: Vec<_> = g.initializer.iter().map(|t| t.name()).collect();
        assert_eq!(
            init_names
                .iter()
                .filter(|&&n| n == "shared_weight_quantized")
                .count(),
            1,
            "Shared weight quantized exactly once"
        );
        assert!(!init_names.contains(&"shared_weight"));

        // But a per-op DynamicQuantizeLinear + MatMulInteger each.
        assert_eq!(
            g.node
                .iter()
                .filter(|n| n.op_type() == "DynamicQuantizeLinear")
                .count(),
            2,
            "Each consuming op gets its own DynamicQuantizeLinear"
        );
        assert_eq!(
            g.node
                .iter()
                .filter(|n| n.op_type() == "MatMulInteger")
                .count(),
            2,
        );
        // Both MatMulInteger nodes reference the single shared quantized weight.
        for mmi in g.node.iter().filter(|n| n.op_type() == "MatMulInteger") {
            assert_eq!(mmi.input[1], "shared_weight_quantized");
            assert_eq!(mmi.input[3], "shared_weight_zero_point");
        }
        // Outputs preserved.
        let outputs: HashSet<&str> = g
            .node
            .iter()
            .filter(|n| n.op_type() == "Mul")
            .flat_map(|n| n.output.iter().map(|s| s.as_str()))
            .collect();
        assert!(outputs.contains("b"));
        assert!(outputs.contains("d"));
    }

    #[test]
    fn test_per_channel_axis_selection() {
        // Conv: output channels on axis 0.
        let conv = NodeProto {
            op_type: Some("Conv".into()),
            ..Default::default()
        };
        assert_eq!(per_channel_axis("Conv", &conv, 4), 0);

        // MatMul: output channel is the last dim.
        let matmul = NodeProto {
            op_type: Some("MatMul".into()),
            ..Default::default()
        };
        assert_eq!(per_channel_axis("MatMul", &matmul, 2), 1);
        assert_eq!(per_channel_axis("MatMul", &matmul, 3), 2);

        // Gemm transB=1: B is [N, K] → N on axis 0.
        let gemm_tb = NodeProto {
            op_type: Some("Gemm".into()),
            attribute: vec![AttributeProto {
                name: Some("transB".into()),
                i: Some(1),
                r#type: Some(2),
                ..Default::default()
            }],
            ..Default::default()
        };
        assert_eq!(per_channel_axis("Gemm", &gemm_tb, 2), 0);

        // Gemm transB=0 (default): B is [K, N] → N on axis 1.
        let gemm = NodeProto {
            op_type: Some("Gemm".into()),
            ..Default::default()
        };
        assert_eq!(per_channel_axis("Gemm", &gemm, 2), 1);
    }

    #[test]
    fn test_quantize_per_channel_groups_along_axis() {
        // Row-major [2, 3]; column 0 is large, columns 1/2 are tiny.
        let data = vec![10.0, 0.1, 0.1, 10.0, 0.1, 0.1];
        let dims = [2i64, 3];

        // axis 1 (per-column): each column owns its scale, so the tiny columns
        // keep full int8 resolution.
        let (q1, s1) = quantize_per_channel(&data, &dims, 1);
        assert_eq!(s1.len(), 3);
        assert!((s1[0] - 10.0 / 127.0).abs() < 1e-9);
        assert!((s1[1] - 0.1 / 127.0).abs() < 1e-9);
        assert_eq!(
            q1[1], 127,
            "0.1 under its own column scale → full-scale 127"
        );

        // axis 0 (per-row): 0.1 shares a row scale with 10.0 and is crushed.
        let (q0, s0) = quantize_per_channel(&data, &dims, 0);
        assert_eq!(s0.len(), 2);
        assert_eq!(q0[1], 1, "0.1 under the row scale (10/127) collapses to 1");
    }

    #[test]
    fn test_quantize_model_matmul_scale_is_n_axis() {
        // MatMul weight [32, 64] → per-channel scale length == N (64).
        let g = quantize_roundtrip(matmul_model("weight", vec![32, 64], 32 * 64));
        let scale = g
            .initializer
            .iter()
            .find(|t| t.name() == "weight_scale")
            .unwrap();
        assert_eq!(
            scale.dims,
            vec![64],
            "MatMul scale length is the N (last) axis"
        );
    }

    #[test]
    fn test_quantize_model_conv_chain() {
        // Conv weight [C_out=8, C_in=4, k=3] (1-D conv, rank 3, 96 elems < 1024)
        // — bump element count by widening the kernel so the gate fires.
        let c_out = 8i64;
        let dims = vec![c_out, 16, 8]; // 1024 elements
        let n_elems = (c_out * 16 * 8) as usize;
        let float_data: Vec<f32> = (0..n_elems).map(|i| (i as f32 * 0.001) - 0.5).collect();
        let weight = TensorProto {
            name: Some("conv_w".into()),
            dims: dims.clone(),
            data_type: Some(FLOAT),
            float_data,
            ..Default::default()
        };
        let bias = TensorProto {
            name: Some("conv_b".into()),
            dims: vec![c_out],
            data_type: Some(FLOAT),
            float_data: vec![0.25; c_out as usize],
            ..Default::default()
        };
        let conv = NodeProto {
            op_type: Some("Conv".into()),
            input: vec!["x".into(), "conv_w".into(), "conv_b".into()],
            output: vec!["y".into()],
            name: Some("conv0".into()),
            attribute: vec![
                AttributeProto {
                    name: Some("strides".into()),
                    ints: vec![1],
                    r#type: Some(7),
                    ..Default::default()
                },
                AttributeProto {
                    name: Some("kernel_shape".into()),
                    ints: vec![8],
                    r#type: Some(7),
                    ..Default::default()
                },
            ],
            ..Default::default()
        };
        let model = ModelProto {
            ir_version: Some(8),
            opset_import: vec![crate::onnx_proto::OperatorSetIdProto {
                domain: Some(String::new()),
                version: Some(17),
            }],
            graph: Some(crate::onnx_proto::GraphProto {
                name: Some("test".into()),
                initializer: vec![weight, bias],
                node: vec![conv],
                ..Default::default()
            }),
            ..Default::default()
        };
        let g = quantize_roundtrip(model);

        // No float Conv / no DequantizeLinear left.
        assert_eq!(g.node.iter().filter(|n| n.op_type() == "Conv").count(), 0);
        assert_eq!(
            g.node
                .iter()
                .filter(|n| n.op_type() == "DequantizeLinear")
                .count(),
            0
        );

        // ConvInteger present, attrs copied.
        let ci = g
            .node
            .iter()
            .find(|n| n.op_type() == "ConvInteger")
            .expect("ConvInteger node");
        assert_eq!(ci.input[1], "conv_w_quantized");
        assert_eq!(ci.input[3], "conv_w_zero_point");
        assert!(
            ci.attribute.iter().any(|a| a.name() == "strides"),
            "strides copied to ConvInteger"
        );
        assert!(ci.attribute.iter().any(|a| a.name() == "kernel_shape"));

        // Per-channel scale length == C_out.
        let scale = g
            .initializer
            .iter()
            .find(|t| t.name() == "conv_w_scale")
            .unwrap();
        assert_eq!(scale.dims, vec![c_out], "conv scale length == C_out");

        // Bias path: an Add produces the original output 'y'.
        let add = g
            .node
            .iter()
            .find(|n| n.op_type() == "Add")
            .expect("bias Add node");
        assert_eq!(add.output, vec!["y".to_string()]);

        // Reshape shape initializers broadcast over channel axis: [1, C_out, 1].
        let scale_shape = g
            .initializer
            .iter()
            .find(|t| t.name() == "conv0_scale_shape")
            .expect("scale shape initializer");
        let shape_vals: Vec<i64> = scale_shape
            .raw_data
            .as_deref()
            .unwrap()
            .chunks_exact(8)
            .map(|c| i64::from_le_bytes(c.try_into().unwrap()))
            .collect();
        assert_eq!(shape_vals, vec![1, c_out, 1]);
    }

    #[test]
    fn test_quantize_model_conv_no_bias() {
        let c_out = 8i64;
        let dims = vec![c_out, 16, 8];
        let n_elems = (c_out * 16 * 8) as usize;
        let float_data: Vec<f32> = (0..n_elems).map(|i| (i as f32 * 0.001) - 0.5).collect();
        let weight = TensorProto {
            name: Some("conv_w".into()),
            dims,
            data_type: Some(FLOAT),
            float_data,
            ..Default::default()
        };
        let conv = NodeProto {
            op_type: Some("Conv".into()),
            input: vec!["x".into(), "conv_w".into()],
            output: vec!["y".into()],
            name: Some("conv0".into()),
            ..Default::default()
        };
        let model = ModelProto {
            ir_version: Some(8),
            opset_import: vec![crate::onnx_proto::OperatorSetIdProto {
                domain: Some(String::new()),
                version: Some(17),
            }],
            graph: Some(crate::onnx_proto::GraphProto {
                name: Some("test".into()),
                initializer: vec![weight],
                node: vec![conv],
                ..Default::default()
            }),
            ..Default::default()
        };
        let g = quantize_roundtrip(model);

        // No bias: no Add, the final rescale Mul produces 'y' directly.
        assert_eq!(g.node.iter().filter(|n| n.op_type() == "Add").count(), 0);
        let rescale = g
            .node
            .iter()
            .find(|n| n.op_type() == "Mul" && n.output == vec!["y".to_string()])
            .expect("rescale Mul producing y");
        assert_eq!(rescale.output, vec!["y".to_string()]);
    }

    #[test]
    fn test_bump_opset_raises_low_version() {
        let mut model = ModelProto {
            opset_import: vec![crate::onnx_proto::OperatorSetIdProto {
                domain: Some(String::new()),
                version: Some(9),
            }],
            ..Default::default()
        };
        bump_opset(&mut model);
        assert_eq!(model.opset_import[0].version(), MIN_OPSET);
    }

    #[test]
    fn test_bump_opset_preserves_high_version() {
        let mut model = ModelProto {
            opset_import: vec![crate::onnx_proto::OperatorSetIdProto {
                domain: Some(String::new()),
                version: Some(17),
            }],
            ..Default::default()
        };
        bump_opset(&mut model);
        assert_eq!(model.opset_import[0].version(), 17);
    }

    #[test]
    fn test_bump_opset_adds_default_when_missing() {
        let mut model = ModelProto::default();
        bump_opset(&mut model);
        assert_eq!(model.opset_import.len(), 1);
        assert_eq!(model.opset_import[0].domain(), "");
        assert_eq!(model.opset_import[0].version(), MIN_OPSET);
    }
}
