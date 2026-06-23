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
            let vb = self.var_builder(&dir.join("candle/decoder.safetensors"))?;
            return Ok(Box::new(DecoderSession::load(vb, self.device.clone())?));
        }
        if name.contains("joint") || name.contains("joiner") {
            let vb = self.var_builder(&dir.join("candle/joiner.safetensors"))?;
            return Ok(Box::new(JoinerSession::load(vb, self.device.clone())?));
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
