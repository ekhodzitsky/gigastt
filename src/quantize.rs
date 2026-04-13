//! INT8 static weight quantization for ONNX encoder models (QDQ node insertion).
//!
//! Replaces `scripts/quantize.py` with a native Rust implementation.
//! Enabled via `--features quantize`.

use anyhow::{Context, Result};
use onnx_pb::{ModelProto, NodeProto, TensorProto};
use prost::Message;
use std::collections::{HashMap, HashSet};
use std::path::Path;

/// ONNX data types (from onnx.proto3 TensorProto.DataType enum)
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

    // Build map: initializer_name → index
    let init_map: HashMap<String, usize> = graph
        .initializer
        .iter()
        .enumerate()
        .map(|(i, t)| (t.name.clone(), i))
        .collect();

    // Collect quantization targets: (node_index, input_index, weight_name, init_index)
    let mut targets = Vec::new();
    for (ni, node) in graph.node.iter().enumerate() {
        if !QUANTIZABLE_OPS.contains(&node.op_type.as_str()) {
            continue;
        }
        // Weight is typically input[1] for MatMul/Conv/Gemm
        for (ii, input_name) in node.input.iter().enumerate() {
            if ii == 0 {
                continue; // Skip activation input
            }
            if let Some(&init_idx) = init_map.get(input_name) {
                let init = &graph.initializer[init_idx];
                if init.data_type != FLOAT {
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

    // For each target: quantize weights, create DequantizeLinear node, rewire graph
    let mut new_nodes = Vec::new();
    let mut new_initializers = Vec::new();
    let mut quantized_names: HashSet<String> = HashSet::new();

    for (_node_idx, _input_idx, weight_name, init_idx) in &targets {
        // CRITICAL fix: skip already-quantized shared weights (avoid duplicate initializers)
        if !quantized_names.insert(weight_name.clone()) {
            continue;
        }

        let init = &graph.initializer[*init_idx];
        let float_data = extract_float_data(init)?;
        let dims = &init.dims;

        if dims.is_empty() {
            continue;
        }

        let channels = dims[0] as usize;
        if channels == 0 {
            continue;
        }
        let expected_elements: usize = dims.iter().map(|&d| d.max(0) as usize).product();
        if expected_elements != float_data.len() {
            tracing::warn!(
                "Skipping tensor '{}': shape mismatch (dims={:?}, data={})",
                init.name,
                dims,
                float_data.len()
            );
            continue;
        }
        let channel_size = float_data.len() / channels;

        // Per-channel symmetric quantization
        let mut quantized_data = Vec::with_capacity(float_data.len());
        let mut scales = Vec::with_capacity(channels);
        let zero_points = vec![0i8; channels];

        for ch in 0..channels {
            let start = ch * channel_size;
            let end = start + channel_size;
            let channel_data = &float_data[start..end];

            let abs_max = channel_data.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
            let scale = if abs_max == 0.0 { 1.0 } else { abs_max / 127.0 };
            scales.push(scale);

            for &val in channel_data {
                let q = (val / scale).round().clamp(-128.0, 127.0) as i8;
                quantized_data.push(q);
            }
        }

        // Create new initializer names
        let q_name = format!("{weight_name}_quantized");
        let s_name = format!("{weight_name}_scale");
        let zp_name = format!("{weight_name}_zero_point");
        let dq_output = format!("{weight_name}_dequantized");

        // Quantized weight tensor (INT8)
        new_initializers.push(TensorProto {
            name: q_name.clone(),
            dims: dims.clone(),
            data_type: INT8,
            raw_data: quantized_data.iter().map(|&v| v as u8).collect(),
            ..Default::default()
        });

        // Scale tensor (FLOAT, per-channel)
        new_initializers.push(TensorProto {
            name: s_name.clone(),
            dims: vec![channels as i64],
            data_type: FLOAT,
            float_data: scales,
            ..Default::default()
        });

        // Zero-point tensor (INT8, all zeros for symmetric)
        new_initializers.push(TensorProto {
            name: zp_name.clone(),
            dims: vec![channels as i64],
            data_type: INT8,
            raw_data: zero_points.iter().map(|&v| v as u8).collect(),
            ..Default::default()
        });

        // DequantizeLinear node
        new_nodes.push(NodeProto {
            op_type: "DequantizeLinear".into(),
            input: vec![q_name, s_name, zp_name],
            output: vec![dq_output.clone()],
            name: format!("dequant_{}", weight_name),
            attribute: vec![onnx_pb::AttributeProto {
                name: "axis".into(),
                i: 0,      // per-channel on axis 0
                r#type: 2, // AttributeType::INT
                ..Default::default()
            }],
            ..Default::default()
        });

        // Rewire: original node's weight input now points to dequantized output
        quantized_names.insert(weight_name.clone());

        // We need to update the node's input — collect for later
        // (can't mutate graph.node while iterating targets that reference it by index)
    }

    // Apply input rewiring
    for (node_idx, input_idx, weight_name, _) in &targets {
        let dq_output = format!("{weight_name}_dequantized");
        graph.node[*node_idx].input[*input_idx] = dq_output;
    }

    // Remove original float initializers for quantized weights
    graph
        .initializer
        .retain(|t| !quantized_names.contains(&t.name));

    // Add new initializers (quantized weights, scales, zero_points)
    graph.initializer.extend(new_initializers);

    // Insert DequantizeLinear nodes before existing nodes
    let mut all_nodes = new_nodes;
    all_nodes.append(&mut graph.node);
    graph.node = all_nodes;

    // Write quantized model (atomic: write to tmp, then rename)
    let mut output_bytes = Vec::new();
    model
        .encode(&mut output_bytes)
        .context("Failed to encode quantized model")?;
    let tmp = output.with_extension("onnx.tmp");
    std::fs::write(&tmp, &output_bytes).context("Failed to write quantized model")?;
    std::fs::rename(&tmp, output).context("Failed to finalize quantized model")?;

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
    if !tensor.raw_data.is_empty() {
        anyhow::ensure!(
            tensor.raw_data.len().is_multiple_of(4),
            "Tensor '{}' raw_data length {} is not aligned to 4 bytes",
            tensor.name,
            tensor.raw_data.len()
        );
        let num_floats = tensor.raw_data.len() / 4;
        let mut data = Vec::with_capacity(num_floats);
        for chunk in tensor.raw_data.chunks_exact(4) {
            data.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
        }
        return Ok(data);
    }
    anyhow::bail!("Tensor '{}' has no float data", tensor.name);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_float_data_from_float_data_field() {
        let tensor = TensorProto {
            name: "test".into(),
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
            name: "test".into(),
            raw_data: raw,
            ..Default::default()
        };
        let data = extract_float_data(&tensor).unwrap();
        assert_eq!(data, vec![1.0, -2.5]);
    }

    #[test]
    fn test_extract_float_data_empty() {
        let tensor = TensorProto {
            name: "empty".into(),
            ..Default::default()
        };
        assert!(extract_float_data(&tensor).is_err());
    }

    #[test]
    fn test_symmetric_quantization_values() {
        // Verify scale/quantized value computation
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
        // All-zero tensor should get scale=1.0 (not division by zero)
        let data = vec![0.0f32; 100];
        let abs_max = data.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
        let scale = if abs_max == 0.0 { 1.0 } else { abs_max / 127.0 };
        assert_eq!(scale, 1.0);
    }
}
