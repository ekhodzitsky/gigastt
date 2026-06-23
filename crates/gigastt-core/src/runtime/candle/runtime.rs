#![allow(dead_code)]

use crate::runtime::{error::RuntimeError, factory::Runtime, session::RuntimeSession};

use super::config::EncoderConfig;
use super::conformer::ConformerEncoder;
use super::factory::CandleDevice;
use super::session::{DecoderSession, EncoderSession, JoinerSession};

/// Candle runtime owning a device handle.
pub struct CandleRuntime {
    device: candle_core::Device,
}

impl CandleRuntime {
    pub fn new(dev: CandleDevice) -> Result<Self, RuntimeError> {
        let device = match dev {
            CandleDevice::Metal => candle_core::Device::new_metal(0).map_err(|e| {
                RuntimeError::InferenceFailed(format!("candle Metal device init failed: {e}"))
            })?,
            CandleDevice::Cpu => candle_core::Device::Cpu,
        };
        Ok(Self { device })
    }
}

impl Runtime for CandleRuntime {
    fn load_session(
        &self,
        model_path: &std::path::Path,
        is_encoder: bool,
    ) -> Result<Box<dyn RuntimeSession>, RuntimeError> {
        // The converted Candle weights live in a `candle/` subdirectory next to
        // the ONNX models: `<model_dir>/candle/{encoder,decoder,joiner}.safetensors`.
        // Encoder is flagged explicitly; otherwise dispatch by filename.
        let dir = model_path.parent().ok_or_else(|| {
            RuntimeError::InferenceFailed("model path has no parent directory".to_string())
        })?;

        if is_encoder {
            let vb = self.var_builder(&dir.join("candle/encoder.safetensors"))?;
            let enc = ConformerEncoder::load(&EncoderConfig::v3_rnnt(), vb).map_err(|e| {
                RuntimeError::LoadFailed {
                    path: dir.join("candle/encoder.safetensors"),
                    message: e.to_string(),
                }
            })?;
            return Ok(Box::new(EncoderSession::new(enc, self.device.clone())));
        }

        let name = model_path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("")
            .to_ascii_lowercase();

        if name.contains("decoder") {
            let st_path = dir.join("candle/decoder.safetensors");
            let vb = self.var_builder(&st_path)?;
            let session = DecoderSession::load(vb, self.device.clone()).map_err(|e| {
                RuntimeError::LoadFailed {
                    path: st_path,
                    message: e.to_string(),
                }
            })?;
            return Ok(Box::new(session));
        }
        if name.contains("joint") || name.contains("joiner") {
            let st_path = dir.join("candle/joiner.safetensors");
            let vb = self.var_builder(&st_path)?;
            let session = JoinerSession::load(vb, self.device.clone()).map_err(|e| {
                RuntimeError::LoadFailed {
                    path: st_path,
                    message: e.to_string(),
                }
            })?;
            return Ok(Box::new(session));
        }

        Err(RuntimeError::InferenceFailed(format!(
            "candle backend cannot classify model file: {name}"
        )))
    }
}

impl CandleRuntime {
    /// Build an F32 `VarBuilder` over a sibling `candle/*.safetensors`, with a
    /// clear error when the converted weights are missing.
    fn var_builder(
        &self,
        st: &std::path::Path,
    ) -> Result<candle_nn::VarBuilder<'static>, RuntimeError> {
        if !st.exists() {
            return Err(RuntimeError::LoadFailed {
                path: st.to_path_buf(),
                message: format!(
                    "candle weights not found ({}); run scripts/convert_gigaam_candle.py",
                    st.file_name().and_then(|n| n.to_str()).unwrap_or("?")
                ),
            });
        }
        // SAFETY: `from_mmaped_safetensors` memory-maps `st` and requires the
        // file to stay valid and unchanged for the VarBuilder's lifetime. `st`
        // lives under `~/.gigastt/models/candle/`, is produced by gigastt's own
        // converter, and is not mutated or truncated by gigastt while loaded. The
        // preceding `exists()` check is advisory only; the mmap call itself is the
        // authoritative error path (a missing/corrupt file returns `Err` here).
        unsafe {
            candle_nn::VarBuilder::from_mmaped_safetensors(
                std::slice::from_ref(&st.to_path_buf()),
                candle_core::DType::F32,
                &self.device,
            )
            .map_err(|e| RuntimeError::LoadFailed {
                path: st.to_path_buf(),
                message: e.to_string(),
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::factory::Runtime;

    fn cpu_runtime() -> CandleRuntime {
        CandleRuntime::new(CandleDevice::Cpu).expect("CPU device always available")
    }

    #[test]
    fn test_load_session_missing_candle_dir_is_load_failed() {
        let tmp = tempfile::tempdir().unwrap();
        // No `candle/` subdir exists; the encoder weights resolution must fail
        // with LoadFailed (not InferenceFailed) so it's classified consistently
        // with the ort encoder load path. (`Box<dyn RuntimeSession>` is not Debug,
        // so match on the Result rather than using expect_err.)
        let model_path = tmp.path().join("v3_rnnt_encoder.onnx");
        match cpu_runtime().load_session(&model_path, true) {
            Ok(_) => panic!("missing weights must fail"),
            Err(RuntimeError::LoadFailed { .. }) => {}
            Err(other) => panic!("expected LoadFailed, got {other:?}"),
        }
    }

    #[test]
    fn test_load_session_bogus_filename_is_classification_error() {
        let tmp = tempfile::tempdir().unwrap();
        // A non-encoder file whose name is neither decoder nor joint/joiner must
        // hit the classification error branch.
        let model_path = tmp.path().join("something_else.onnx");
        match cpu_runtime().load_session(&model_path, false) {
            Ok(_) => panic!("unclassifiable file must fail"),
            Err(RuntimeError::InferenceFailed(msg)) => {
                assert!(
                    msg.contains("cannot classify"),
                    "expected classification error, got: {msg}"
                );
            }
            Err(other) => {
                panic!("expected InferenceFailed classification error, got {other:?}")
            }
        }
    }
}
