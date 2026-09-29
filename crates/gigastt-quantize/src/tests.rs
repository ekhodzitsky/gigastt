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
fn test_skip_conv_keeps_float_conv_and_quantizes_matmul() {
    let mut model = matmul_model("w_mm", vec![32, 32], 1024);
    let graph = model.graph.as_mut().unwrap();
    graph.initializer.push(TensorProto {
        name: Some("w_conv".into()),
        dims: vec![32, 32, 1, 1],
        data_type: Some(FLOAT),
        float_data: vec![0.01; 1024],
        ..Default::default()
    });
    graph.initializer.push(TensorProto {
        name: Some("w_gemm".into()),
        dims: vec![32, 32],
        data_type: Some(FLOAT),
        float_data: vec![0.02; 1024],
        ..Default::default()
    });
    graph.node.push(NodeProto {
        op_type: Some("Conv".into()),
        input: vec!["input".into(), "w_conv".into()],
        output: vec!["conv_out".into()],
        ..Default::default()
    });
    graph.node.push(NodeProto {
        op_type: Some("Gemm".into()),
        input: vec!["input".into(), "w_gemm".into()],
        output: vec!["gemm_out".into()],
        ..Default::default()
    });

    let tmp = tempfile::tempdir().unwrap();
    let input = tmp.path().join("in.onnx");
    let output = tmp.path().join("out.onnx");
    let mut bytes = Vec::new();
    model.encode(&mut bytes).unwrap();
    std::fs::write(&input, bytes).unwrap();
    quantize_model_with(&input, &output, QuantizeOps { conv: false }).unwrap();

    let decoded = ModelProto::decode(&std::fs::read(&output).unwrap()[..]).unwrap();
    let graph = decoded.graph.unwrap();
    let ops: Vec<&str> = graph.node.iter().map(|node| node.op_type()).collect();
    assert!(ops.contains(&"Conv"), "{ops:?}");
    assert!(ops.contains(&"MatMulInteger"), "{ops:?}");
    assert!(!ops.contains(&"ConvInteger"), "{ops:?}");
    assert!(!ops.contains(&"MatMul"), "{ops:?}");
    assert!(!ops.contains(&"Gemm"), "{ops:?}");
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
        .as_chunks::<8>()
        .0
        .iter()
        .map(|c| i64::from_le_bytes(*c))
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
