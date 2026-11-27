//! Explicit verified v4 CPU session construction. No legacy cache/fallback.

use ort::{
    session::{Session, builder::GraphOptimizationLevel},
    value::{Outlet, TensorElementType},
};

use super::{factory::shared_environment, session::OrtSession};
use crate::model::{
    ArtifactRole, TensorAxis, TensorElement, TensorSpec, V4_RNNT_SPLIT_ABI, VerifiedV4Graph,
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
    let environment = shared_environment()?;
    let mut builder = Session::builder(&environment)
        .map_err(|e| load_error(role, e))?
        .with_no_environment_execution_providers()
        .map_err(|e| load_error(role, e))?
        .with_execution_providers([ort::ep::CPU::default()
            .with_arena_allocator(true)
            .build()
            .error_on_failure()])
        .map_err(|e| load_error(role, e))?
        .with_optimization_level(GraphOptimizationLevel::Level3)
        .map_err(|e| load_error(role, e))?
        .with_parallel_execution(false)
        .map_err(|e| load_error(role, e))?
        .with_intra_threads(1)
        .map_err(|e| load_error(role, e))?
        .with_inter_threads(1)
        .map_err(|e| load_error(role, e))?;
    let session = builder
        .commit_from_memory(graph.bytes())
        .map_err(|e| load_error(role, e))?;
    validate_outlets(session.inputs(), inputs, false, role)?;
    validate_outlets(session.outputs(), outputs, encoder_output, role)?;
    Ok(Box::new(OrtSession::from_session(session)))
}
