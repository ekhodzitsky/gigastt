//! Dynamic INT8 (QOperator) quantization for ONNX encoder models.
//!
//! Packaging-only conversion exposed as the `gigastt quantize` subcommand.
//! Runtime commands download a prequantized bundle instead.
//!
//! This emits the **dynamic INT8 (QOperator)** form that ONNX Runtime's
//! `quantize_dynamic(..., weight_type=QInt8)` produces:
//! `DynamicQuantizeLinear` on activations feeding integer compute kernels
//! (`MatMulInteger` / `ConvInteger`), with a per-channel float rescale on the
//! int32 output. Integer kernels are not necessarily faster on every CPU;
//! model changes require measured quality, latency, and memory comparisons.
//!
//! The protobuf types come from `crate::onnx_proto`, which is generated at
//! build time from `proto/onnx.proto` via `prost-build` (see `build.rs`).
//! Fields that are `optional` in proto2 surface as `Option<T>` in prost
//! 0.13, so we lean on the generated accessor methods (`data_type()`,
//! `name()`, `op_type()`, …) for reads and wrap writes in `Some(_)`.

pub mod onnx_proto;

use anyhow::{Context, Result};
use prost::Message;
use std::collections::{HashMap, HashSet};
use std::path::Path;

#[cfg(test)]
pub(crate) use crate::onnx_proto::AttributeProto;
use crate::onnx_proto::{ModelProto, NodeProto, TensorProto};

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

/// Packaging filter for one local rebuild.
///
/// [`Default`] quantizes `MatMul` and `Gemm`, leaving convolutions in float.
/// Dynamic Conv quantization is an explicit experimental opt-in: it can
/// increase phrase loss and latency even on CPUs with VNNI.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct QuantizeOps {
    pub conv: bool,
}

impl QuantizeOps {
    fn allows(self, op: &str) -> bool {
        match op {
            "MatMul" | "Gemm" => true,
            "Conv" => self.conv,
            _ => false,
        }
    }
}

/// Minimum number of elements in a tensor to quantize (skip small biases).
const MIN_ELEMENTS: usize = 1024;

mod graph;
mod weights;

use graph::{build_conv_chain, build_matmul_chain, bump_opset};
use weights::{QuantizedWeight, extract_float_data, per_channel_axis, quantize_per_channel};

/// Quantize an ONNX model's float32 weights to dynamic INT8 (QOperator form).
///
/// For each quantizable weight tensor (MatMul/Conv/Gemm) the original float op
/// is **replaced** by a `DynamicQuantizeLinear` → `MatMulInteger`/`ConvInteger`
/// → `Cast` → per-channel `Mul` (+ optional bias `Add`) chain, with the weight
/// stored as a per-channel-symmetric INT8 initializer.
pub fn quantize_model(input: &Path, output: &Path) -> Result<()> {
    quantize_model_with(input, output, QuantizeOps::default())
}

/// [`quantize_model`] with explicit operator selection. The default leaves
/// Conv in FP32; validate model quality and resources before publication.
pub fn quantize_model_with(input: &Path, output: &Path, ops: QuantizeOps) -> Result<()> {
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
        if !ops.allows(node.op_type()) {
            continue;
        }
        // The integer chain implements plain A @ B. Preserve general Gemm
        // semantics (bias, transposes and scaling) instead of dropping them.
        if node.op_type() == "Gemm"
            && (!node.attribute.is_empty()
                || node.input.get(2).is_some_and(|bias| !bias.is_empty()))
        {
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

    anyhow::ensure!(
        !quantized.is_empty(),
        "No weights were quantized; refusing to write a float-only model as INT8"
    );

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

    // A weight can also feed an unquantized op or be a graph output. Keep it
    // while still referenced; removing it makes mixed-precision graphs invalid.
    let quantized_weight_names: HashSet<&str> = quantized.keys().map(|s| s.as_str()).collect();
    let live_names: HashSet<&str> = graph
        .node
        .iter()
        .flat_map(|node| node.input.iter().map(String::as_str))
        .chain(graph.output.iter().map(|output| output.name()))
        .collect();
    graph
        .initializer
        .retain(|t| !quantized_weight_names.contains(t.name()) || live_names.contains(t.name()));

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

#[cfg(test)]
mod tests;
