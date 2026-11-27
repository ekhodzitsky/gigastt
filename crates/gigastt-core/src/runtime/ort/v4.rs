//! Explicit verified v4 CPU session construction. No legacy cache/fallback.

use ort::{
    session::{Session, builder::GraphOptimizationLevel},
    value::{Outlet, TensorElementType},
};

use super::{factory::shared_environment, session::OrtSession};
use crate::model::{
    ArtifactRole, StreamingTensorSpec, TensorAxis, TensorElement, TensorSpec, V4_RNNT_SPLIT_ABI,
    V4_STREAMING_FRONT_CACHE_SHAPES, V4_STREAMING_LAYER_CACHE_SHAPES, VerifiedV4Graph,
    VerifiedV4StreamingGraph,
};
use crate::runtime::{RuntimeError, RuntimeSession};

fn load_error(role: ArtifactRole, message: impl std::fmt::Display) -> RuntimeError {
    RuntimeError::LoadFailed {
        path: role.as_str().into(),
        message: message.to_string(),
    }
}

fn validate_outlets(
    actual: &[Outlet],
    expected: &[TensorSpec],
    encoder_output: bool,
    role: ArtifactRole,
) -> Result<(), RuntimeError> {
    if actual.len() != expected.len() {
        return Err(load_error(role, "unexpected v4 graph input/output count"));
    }
    for (actual, expected) in actual.iter().zip(expected) {
        if actual.name() != expected.name {
            return Err(load_error(role, "unexpected v4 graph tensor name/order"));
        }
        let ty = match expected.element {
            TensorElement::F32 => TensorElementType::Float32,
            TensorElement::I64 => TensorElementType::Int64,
        };
        if actual.dtype().tensor_type() != Some(ty) {
            return Err(load_error(role, "unexpected v4 graph tensor type"));
        }
        let shape = actual
            .dtype()
            .tensor_shape()
            .ok_or_else(|| load_error(role, "non-tensor v4 graph outlet"))?;
        if shape.len() != expected.axes.len() {
            return Err(load_error(role, "unexpected v4 graph tensor rank"));
        }
        for (index, (&dim, axis)) in shape.iter().zip(expected.axes).enumerate() {
            let valid = match axis {
                // Exact qualified encoder output uses a dynamic symbolic channel
                // axis in its exported metadata; actual width is checked at run.
                TensorAxis::Fixed(n)
                    if encoder_output && expected.name == "encoded" && index == 1 =>
                {
                    dim == -1 || dim == *n as i64
                }
                TensorAxis::Fixed(n) => dim == *n as i64,
                TensorAxis::Batch | TensorAxis::MelFrames | TensorAxis::Frames => dim == -1,
            };
            if !valid {
                return Err(load_error(
                    role,
                    format!(
                        "unexpected v4 graph dimension {}[{index}]={dim}",
                        expected.name
                    ),
                ));
            }
        }
    }
    Ok(())
}

/// The graph wrapper can only be produced by the exact qualified pack's
/// byte-verification method. ORT parses these same bytes, not a reopened file.
pub(crate) fn load_verified_cpu(
    graph: VerifiedV4Graph,
) -> Result<Box<dyn RuntimeSession>, RuntimeError> {
    let role = graph.role();
    let abi = &V4_RNNT_SPLIT_ABI;
    let (inputs, outputs, encoder_output) = match role {
        ArtifactRole::Encoder => (abi.encoder_inputs, abi.encoder_outputs, true),
        ArtifactRole::RawDecoder | ArtifactRole::E2eDecoder => {
            (abi.decoder_inputs, abi.decoder_outputs, false)
        }
        ArtifactRole::RawJoint | ArtifactRole::E2eJoint => {
            (abi.joint_inputs, abi.joint_outputs, false)
        }
        ArtifactRole::RawTokenizer | ArtifactRole::E2eTokenizer => {
            return Err(load_error(role, "tokenizer is not an ONNX graph"));
        }
    };
    let session = build_verified_cpu(graph.bytes(), |message| load_error(role, message))?;
    validate_outlets(session.inputs(), inputs, false, role)?;
    validate_outlets(session.outputs(), outputs, encoder_output, role)?;
    Ok(Box::new(OrtSession::from_session(session)))
}

fn build_verified_cpu(
    bytes: &[u8],
    error: impl Fn(String) -> RuntimeError,
) -> Result<Session, RuntimeError> {
    let environment = shared_environment()?;
    let mut builder = Session::builder(&environment)
        .map_err(|e| error(e.to_string()))?
        .with_no_environment_execution_providers()
        .map_err(|e| error(e.to_string()))?
        .with_execution_providers([ort::ep::CPU::default()
            .with_arena_allocator(true)
            .build()
            .error_on_failure()])
        .map_err(|e| error(e.to_string()))?
        .with_optimization_level(GraphOptimizationLevel::Level3)
        .map_err(|e| error(e.to_string()))?
        .with_parallel_execution(false)
        .map_err(|e| error(e.to_string()))?
        .with_intra_threads(1)
        .map_err(|e| error(e.to_string()))?
        .with_inter_threads(1)
        .map_err(|e| error(e.to_string()))?;
    let session = builder
        .commit_from_memory(bytes)
        .map_err(|e| error(e.to_string()))?;
    Ok(session)
}

/// Load only exact role-pinned streaming graphs, not arbitrary ONNX bytes.
pub(crate) fn load_verified_streaming_cpu(
    graph: VerifiedV4StreamingGraph,
) -> Result<Box<dyn RuntimeSession>, RuntimeError> {
    let role = graph.role();
    let error = |message: String| RuntimeError::LoadFailed {
        path: role.as_str().into(),
        message,
    };
    let (inputs, outputs) = role
        .graph_specs()
        .ok_or_else(|| error("not a v4 streaming ONNX role".into()))?;
    let session = build_verified_cpu(graph.bytes(), error)?;
    validate_streaming_outlets(session.inputs(), inputs, false).map_err(error)?;
    validate_streaming_outlets(session.outputs(), outputs, true).map_err(error)?;
    Ok(Box::new(OrtSession::from_session(session)))
}

fn validate_streaming_outlets(
    actual: &[Outlet],
    expected: &[StreamingTensorSpec],
    output: bool,
) -> Result<(), String> {
    if actual.len() != expected.len() {
        return Err("unexpected v4 streaming graph input/output count".into());
    }
    for (actual, expected) in actual.iter().zip(expected) {
        let ty = match expected.element {
            TensorElement::F32 => TensorElementType::Float32,
            TensorElement::I64 => TensorElementType::Int64,
        };
        if actual.name() != expected.name || actual.dtype().tensor_type() != Some(ty) {
            return Err(format!(
                "unexpected v4 streaming graph outlet {} type/name",
                expected.name
            ));
        }
        let shape = actual
            .dtype()
            .tensor_shape()
            .ok_or_else(|| format!("non-tensor v4 streaming outlet {}", expected.name))?;
        let shape_ok = shape.len() == expected.shape.len()
            && shape
                .iter()
                .zip(expected.shape)
                .enumerate()
                .all(|(axis, (&got, &want))| {
                    got == want
                        || (output
                            && want == -1
                            && inferred_streaming_output_axis(expected.name, axis) == Some(got))
                });
        if !shape_ok {
            return Err(format!(
                "unexpected v4 streaming outlet {} shape {:?}",
                expected.name,
                shape.as_ref()
            ));
        }
    }
    Ok(())
}

// ORT can resolve a symbolic output axis to its known physical geometry.
// Do not accept a fixed temporal width or any unqualified specialization.
fn inferred_streaming_output_axis(name: &str, axis: usize) -> Option<i64> {
    let known = if name == "pre_encoded" || name == "encoded" {
        match (name, axis) {
            (_, 0) => 1,
            ("encoded", 1) | ("pre_encoded", 2) => 768,
            _ => return None,
        }
    } else {
        let (index, shapes) = if let Some(index) = name.strip_prefix("next_front_") {
            (index, V4_STREAMING_FRONT_CACHE_SHAPES)
        } else {
            (
                name.strip_prefix("next_cache_")?,
                V4_STREAMING_LAYER_CACHE_SHAPES,
            )
        };
        *shapes.get(index.parse::<usize>().ok()?)?.get(axis)?
    };
    i64::try_from(known).ok()
}
