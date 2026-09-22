//! Model-free test engines for in-crate tests and the private `__internals`
//! feature (server / FFI unit tests).
//!
//! Not part of the stable public API. The encoder session accepts any audio
//! length and emits a single blank frame so REST / SSE / jobs / file wrappers
//! can run without ONNX weights.

use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

use crate::error::GigasttError;
use crate::inference::{Engine, PRED_HIDDEN};
use crate::runtime::mock::{MockFactory, MockSession};
use crate::runtime::tensor::{Shape, Tensor, TensorData};

const ENC_DIM: usize = 768;

/// Write the INT8 rnnt filenames the engine loader expects (empty ONNX bytes).
pub fn write_rnnt_layout(dir: &Path) -> std::io::Result<()> {
    std::fs::write(dir.join("v3_rnnt_encoder_int8.onnx"), b"")?;
    std::fs::write(dir.join("v3_rnnt_decoder.onnx"), b"")?;
    std::fs::write(dir.join("v3_rnnt_joint.onnx"), b"")?;
    std::fs::write(dir.join("v3_vocab.txt"), "\u{2581}hi\n<blk>\n")?;
    Ok(())
}

/// PCM16 mono WAV with a standard 44-byte header.
pub fn pcm16_wav(samples: &[i16], sample_rate: u32) -> Vec<u8> {
    let data_size = (samples.len() * 2) as u32;
    let file_size = 36 + data_size;
    let mut wav = Vec::with_capacity(44 + samples.len() * 2);
    wav.extend_from_slice(b"RIFF");
    wav.extend_from_slice(&file_size.to_le_bytes());
    wav.extend_from_slice(b"WAVE");
    wav.extend_from_slice(b"fmt ");
    wav.extend_from_slice(&16u32.to_le_bytes());
    wav.extend_from_slice(&1u16.to_le_bytes());
    wav.extend_from_slice(&1u16.to_le_bytes());
    wav.extend_from_slice(&sample_rate.to_le_bytes());
    wav.extend_from_slice(&(sample_rate * 2).to_le_bytes());
    wav.extend_from_slice(&2u16.to_le_bytes());
    wav.extend_from_slice(&16u16.to_le_bytes());
    wav.extend_from_slice(b"data");
    wav.extend_from_slice(&data_size.to_le_bytes());
    for s in samples {
        wav.extend_from_slice(&s.to_le_bytes());
    }
    wav
}

/// Scripted rnnt factory: unconstrained encoder (any T) + fixed decoder/joiner.
pub fn rnnt_factory() -> MockFactory {
    let mut sessions: HashMap<String, Arc<MockSession>> = HashMap::new();
    sessions.insert(
        "v3_rnnt_encoder_int8".into(),
        Arc::new(MockSession::unconstrained(vec![
            Tensor::new(
                Shape::new(vec![1, ENC_DIM, 1]),
                TensorData::F32(vec![0.0; ENC_DIM]),
            )
            .expect("encoder out"),
            Tensor::new(Shape::new(vec![1]), TensorData::I64(vec![1])).expect("enc len"),
        ])),
    );
    sessions.insert(
        "v3_rnnt_decoder".into(),
        Arc::new(MockSession::new(
            vec![
                Shape::new(vec![1, 1]),
                Shape::new(vec![1, 1, PRED_HIDDEN]),
                Shape::new(vec![1, 1, PRED_HIDDEN]),
            ],
            vec![
                Tensor::new(
                    Shape::new(vec![1, 1, PRED_HIDDEN]),
                    TensorData::F32(vec![0.0; PRED_HIDDEN]),
                )
                .expect("dec h"),
                Tensor::new(
                    Shape::new(vec![1, 1, PRED_HIDDEN]),
                    TensorData::F32(vec![0.0; PRED_HIDDEN]),
                )
                .expect("dec c"),
                Tensor::new(
                    Shape::new(vec![1, 1, PRED_HIDDEN]),
                    TensorData::F32(vec![0.0; PRED_HIDDEN]),
                )
                .expect("dec out"),
            ],
        )),
    );
    sessions.insert(
        "v3_rnnt_joint".into(),
        Arc::new(MockSession::new(
            vec![
                Shape::new(vec![1, ENC_DIM, 1]),
                Shape::new(vec![1, PRED_HIDDEN, 1]),
            ],
            vec![
                Tensor::new(Shape::new(vec![1, 1, 2]), TensorData::F32(vec![0.0; 2]))
                    .expect("joint"),
            ],
        )),
    );
    MockFactory::new(sessions)
}

/// Load an INT8 rnnt engine from `dir` (must already hold [`write_rnnt_layout`]).
pub fn load_rnnt_engine(dir: &Path, pool_size: usize) -> Result<Engine, GigasttError> {
    Engine::load_with_factory(
        dir,
        None,
        pool_size.max(1),
        1,
        0,
        Box::new(rnnt_factory()),
        1,
        true,
        "cpu",
    )
}

/// Convenience for this crate's own unit tests (`tempfile` is a dev-dep).
#[cfg(test)]
pub fn rnnt_engine() -> (Engine, tempfile::TempDir) {
    let tmp = tempfile::tempdir().expect("tempdir");
    write_rnnt_layout(tmp.path()).expect("rnnt layout");
    let engine = load_rnnt_engine(tmp.path(), 1).expect("mock rnnt engine");
    (engine, tmp)
}
