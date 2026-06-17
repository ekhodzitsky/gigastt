//! INT8 static weight quantization for ONNX encoder models (QDQ node insertion).
//!
//! Native Rust replacement for `scripts/quantize.py`. Auto-invoked after
//! `gigastt download` and `gigastt serve` (see `src/main.rs`); also exposed
//! as the `gigastt quantize` subcommand.
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

/// Node types whose weights benefit from INT8 quantization.
const QUANTIZABLE_OPS: &[&str] = &["MatMul", "Conv", "Gemm"];

/// Minimum number of elements in a tensor to quantize (skip small biases).
const MIN_ELEMENTS: usize = 1024;

/// Quantize an ONNX model's float32 weights to int8 per-channel using QDQ format.
///
/// For each quantizable weight tensor (MatMul/Conv/Gemm), inserts a
/// `DequantizeLinear` node between the quantized int8 weight and the
/// original operator, with per-channel scale and zero_point initializers.
pub fn quantize_model(input: &Path, output: &Path) -> Result<()> {
    let model_bytes = std::fs::read(input).context("Failed to read ONNX model")?;
    let mut model =
        ModelProto::decode(&model_bytes[..]).context("Failed to decode ONNX protobuf")?;
    let graph = model.graph.as_mut().context("Model has no graph")?;

    // Build map: initializer_name → index.
    let init_map: HashMap<String, usize> = graph
        .initializer
        .iter()
        .enumerate()
        .map(|(i, t)| (t.name().to_string(), i))
        .collect();

    // Collect quantization targets: (node_index, input_index, weight_name, init_index).
    let mut targets = Vec::new();
    for (ni, node) in graph.node.iter().enumerate() {
        if !QUANTIZABLE_OPS.contains(&node.op_type()) {
            continue;
        }
        // Weight is typically input[1] for MatMul/Conv/Gemm.
        for (ii, input_name) in node.input.iter().enumerate() {
            if ii == 0 {
                continue; // Skip activation input.
            }
            if let Some(&init_idx) = init_map.get(input_name) {
                let init = &graph.initializer[init_idx];
                if init.data_type() != FLOAT {
                    continue;
                }
                let num_elements: i64 = init.dims.iter().product();
                if num_elements > 0 && num_elements as usize >= MIN_ELEMENTS {
                    targets.push((ni, ii, input_name.clone(), init_idx));
                }
            }
        }
    }

    tracing::info!(
        "Found {} quantizable weight tensors in {} nodes",
        targets.len(),
        graph.node.len()
    );

    // For each target: quantize weights, create DequantizeLinear node, rewire graph.
    let mut new_nodes = Vec::new();
    let mut new_initializers = Vec::new();
    let mut quantized_names: HashSet<String> = HashSet::new();

    for (node_idx, _input_idx, weight_name, init_idx) in &targets {
        // Skip already-quantized shared weights (avoid duplicate initializers).
        if !quantized_names.insert(weight_name.clone()) {
            continue;
        }

        let init = &graph.initializer[*init_idx];
        let float_data = extract_float_data(init)?;
        let dims = &init.dims;

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

        // Per-channel symmetric quantization along `axis`.
        let (quantized_data, scales) = quantize_per_channel(&float_data, dims, axis);
        let zero_points = vec![0i8; channels];

        // Create new initializer names.
        let q_name = format!("{weight_name}_quantized");
        let s_name = format!("{weight_name}_scale");
        let zp_name = format!("{weight_name}_zero_point");
        let dq_output = format!("{weight_name}_dequantized");

        // Quantized weight tensor (INT8).
        new_initializers.push(TensorProto {
            name: Some(q_name.clone()),
            dims: dims.clone(),
            data_type: Some(INT8),
            raw_data: Some(quantized_data.iter().map(|&v| v as u8).collect()),
            ..Default::default()
        });

        // Scale tensor (FLOAT, per-channel).
        new_initializers.push(TensorProto {
            name: Some(s_name.clone()),
            dims: vec![channels as i64],
            data_type: Some(FLOAT),
            float_data: scales,
            ..Default::default()
        });

        // Zero-point tensor (INT8, all zeros for symmetric).
        new_initializers.push(TensorProto {
            name: Some(zp_name.clone()),
            dims: vec![channels as i64],
            data_type: Some(INT8),
            raw_data: Some(zero_points.iter().map(|&v| v as u8).collect()),
            ..Default::default()
        });

        // DequantizeLinear node.
        new_nodes.push(NodeProto {
            op_type: Some("DequantizeLinear".into()),
            input: vec![q_name, s_name, zp_name],
            output: vec![dq_output.clone()],
            name: Some(format!("dequant_{weight_name}")),
            attribute: vec![AttributeProto {
                name: Some("axis".into()),
                i: Some(axis as i64), // per-channel on the op's output-channel axis
                r#type: Some(2),      // AttributeType::INT
                ..Default::default()
            }],
            ..Default::default()
        });
    }

    // Apply input rewiring.
    for (node_idx, input_idx, weight_name, _) in &targets {
        let dq_output = format!("{weight_name}_dequantized");
        graph.node[*node_idx].input[*input_idx] = dq_output;
    }

    // Remove original float initializers for quantized weights.
    graph
        .initializer
        .retain(|t| !quantized_names.contains(t.name()));

    // Add new initializers (quantized weights, scales, zero_points).
    graph.initializer.extend(new_initializers);

    // Insert DequantizeLinear nodes before existing nodes.
    let mut all_nodes = new_nodes;
    all_nodes.append(&mut graph.node);
    graph.node = all_nodes;

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
    fn test_quantize_model_matmul_end_to_end() {
        let float_data: Vec<f32> = (0..1024).map(|i| i as f32 * 0.001).collect();
        let weight = TensorProto {
            name: Some("weight".into()),
            dims: vec![32, 32],
            data_type: Some(FLOAT),
            float_data: float_data.clone(),
            ..Default::default()
        };
        let node = NodeProto {
            op_type: Some("MatMul".into()),
            input: vec!["input".into(), "weight".into()],
            output: vec!["output".into()],
            ..Default::default()
        };
        let graph = crate::onnx_proto::GraphProto {
            name: Some("test".into()),
            initializer: vec![weight],
            node: vec![node],
            ..Default::default()
        };
        let model = ModelProto {
            ir_version: Some(8),
            graph: Some(graph),
            ..Default::default()
        };

        let tmp_dir = tempfile::tempdir().unwrap();
        let input_path = tmp_dir.path().join("input.onnx");
        let output_path = tmp_dir.path().join("output.onnx");

        let mut bytes = Vec::new();
        model.encode(&mut bytes).unwrap();
        std::fs::write(&input_path, &bytes).unwrap();

        quantize_model(&input_path, &output_path).unwrap();

        let out_bytes = std::fs::read(&output_path).unwrap();
        let out_model = ModelProto::decode(&out_bytes[..]).unwrap();
        let g = out_model.graph.as_ref().unwrap();

        let dq_nodes: Vec<_> = g
            .node
            .iter()
            .filter(|n| n.op_type() == "DequantizeLinear")
            .collect();
        assert_eq!(dq_nodes.len(), 1, "Expected one DequantizeLinear node");

        let init_names: Vec<_> = g.initializer.iter().map(|t| t.name()).collect();
        assert!(
            !init_names.contains(&"weight"),
            "Original float weight should be removed"
        );
        assert!(init_names.contains(&"weight_quantized"));
        assert!(init_names.contains(&"weight_scale"));
        assert!(init_names.contains(&"weight_zero_point"));

        let matmul = g.node.iter().find(|n| n.op_type() == "MatMul").unwrap();
        assert_eq!(matmul.input[1], "weight_dequantized");
    }

    #[test]
    fn test_quantize_model_small_tensor_skipped() {
        let float_data: Vec<f32> = (0..256).map(|i| i as f32 * 0.001).collect();
        let weight = TensorProto {
            name: Some("small_weight".into()),
            dims: vec![16, 16],
            data_type: Some(FLOAT),
            float_data,
            ..Default::default()
        };
        let node = NodeProto {
            op_type: Some("MatMul".into()),
            input: vec!["input".into(), "small_weight".into()],
            output: vec!["output".into()],
            ..Default::default()
        };
        let graph = crate::onnx_proto::GraphProto {
            name: Some("test".into()),
            initializer: vec![weight],
            node: vec![node],
            ..Default::default()
        };
        let model = ModelProto {
            ir_version: Some(8),
            graph: Some(graph),
            ..Default::default()
        };

        let tmp_dir = tempfile::tempdir().unwrap();
        let input_path = tmp_dir.path().join("input.onnx");
        let output_path = tmp_dir.path().join("output.onnx");

        let mut bytes = Vec::new();
        model.encode(&mut bytes).unwrap();
        std::fs::write(&input_path, &bytes).unwrap();

        quantize_model(&input_path, &output_path).unwrap();

        let out_bytes = std::fs::read(&output_path).unwrap();
        let out_model = ModelProto::decode(&out_bytes[..]).unwrap();
        let g = out_model.graph.as_ref().unwrap();

        let dq_count = g
            .node
            .iter()
            .filter(|n| n.op_type() == "DequantizeLinear")
            .count();
        assert_eq!(dq_count, 0, "Small tensor should be skipped");

        let init_names: Vec<_> = g.initializer.iter().map(|t| t.name()).collect();
        assert!(init_names.contains(&"small_weight"));
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
            ..Default::default()
        };
        let node2 = NodeProto {
            op_type: Some("MatMul".into()),
            input: vec!["c".into(), "shared_weight".into()],
            output: vec!["d".into()],
            ..Default::default()
        };
        let graph = crate::onnx_proto::GraphProto {
            name: Some("test".into()),
            initializer: vec![weight],
            node: vec![node1, node2],
            ..Default::default()
        };
        let model = ModelProto {
            ir_version: Some(8),
            graph: Some(graph),
            ..Default::default()
        };

        let tmp_dir = tempfile::tempdir().unwrap();
        let input_path = tmp_dir.path().join("input.onnx");
        let output_path = tmp_dir.path().join("output.onnx");

        let mut bytes = Vec::new();
        model.encode(&mut bytes).unwrap();
        std::fs::write(&input_path, &bytes).unwrap();

        quantize_model(&input_path, &output_path).unwrap();

        let out_bytes = std::fs::read(&output_path).unwrap();
        let out_model = ModelProto::decode(&out_bytes[..]).unwrap();
        let g = out_model.graph.as_ref().unwrap();

        let dq_count = g
            .node
            .iter()
            .filter(|n| n.op_type() == "DequantizeLinear")
            .count();
        assert_eq!(
            dq_count, 1,
            "Shared weight should produce a single DequantizeLinear node"
        );

        let matmul_nodes: Vec<_> = g.node.iter().filter(|n| n.op_type() == "MatMul").collect();
        assert_eq!(matmul_nodes.len(), 2);
        assert_eq!(matmul_nodes[0].input[1], "shared_weight_dequantized");
        assert_eq!(matmul_nodes[1].input[1], "shared_weight_dequantized");
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
    fn test_quantize_model_matmul_emits_last_axis() {
        // MatMul weight [32, 32] → per-channel axis must be the last (1), not 0.
        let float_data: Vec<f32> = (0..1024).map(|i| i as f32 * 0.001).collect();
        let weight = TensorProto {
            name: Some("weight".into()),
            dims: vec![32, 32],
            data_type: Some(FLOAT),
            float_data,
            ..Default::default()
        };
        let node = NodeProto {
            op_type: Some("MatMul".into()),
            input: vec!["input".into(), "weight".into()],
            output: vec!["output".into()],
            ..Default::default()
        };
        let graph = crate::onnx_proto::GraphProto {
            name: Some("test".into()),
            initializer: vec![weight],
            node: vec![node],
            ..Default::default()
        };
        let model = ModelProto {
            ir_version: Some(8),
            graph: Some(graph),
            ..Default::default()
        };

        let tmp_dir = tempfile::tempdir().unwrap();
        let input_path = tmp_dir.path().join("input.onnx");
        let output_path = tmp_dir.path().join("output.onnx");
        let mut bytes = Vec::new();
        model.encode(&mut bytes).unwrap();
        std::fs::write(&input_path, &bytes).unwrap();

        quantize_model(&input_path, &output_path).unwrap();

        let out_bytes = std::fs::read(&output_path).unwrap();
        let out_model = ModelProto::decode(&out_bytes[..]).unwrap();
        let g = out_model.graph.as_ref().unwrap();

        let dq = g
            .node
            .iter()
            .find(|n| n.op_type() == "DequantizeLinear")
            .unwrap();
        let axis = dq
            .attribute
            .iter()
            .find(|a| a.name() == "axis")
            .and_then(|a| a.i)
            .unwrap();
        assert_eq!(
            axis, 1,
            "MatMul weight quantized per-channel on the last (N) axis"
        );

        // Scale length must equal the channel count along that axis.
        let scale = g
            .initializer
            .iter()
            .find(|t| t.name() == "weight_scale")
            .unwrap();
        assert_eq!(scale.dims, vec![32]);
    }
}
