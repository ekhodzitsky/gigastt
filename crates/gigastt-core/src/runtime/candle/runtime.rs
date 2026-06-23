#![allow(dead_code)]

use crate::runtime::{error::RuntimeError, factory::Runtime, session::RuntimeSession};

use super::config::EncoderConfig;
use super::conformer::ConformerEncoder;
use super::factory::CandleDevice;
use super::session::EncoderSession;

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
        if !is_encoder {
            return Err(RuntimeError::InferenceFailed(format!(
                "candle backend not yet implemented for {}",
                model_path.display()
            )));
        }

        // The converted Candle weights live in a `candle/` subdirectory next to
        // the ONNX models: `<model_dir>/candle/encoder.safetensors`.
        let st = model_path
            .parent()
            .map(|p| p.join("candle/encoder.safetensors"))
            .ok_or_else(|| {
                RuntimeError::InferenceFailed(
                    "encoder model path has no parent directory".to_string(),
                )
            })?;

        if !st.exists() {
            return Err(RuntimeError::LoadFailed {
                path: st.clone(),
                message: "candle encoder weights (candle/encoder.safetensors) not found"
                    .to_string(),
            });
        }

        let vb = unsafe {
            candle_nn::VarBuilder::from_mmaped_safetensors(
                std::slice::from_ref(&st),
                candle_core::DType::F32,
                &self.device,
            )
            .map_err(|e| RuntimeError::LoadFailed {
                path: st.clone(),
                message: e.to_string(),
            })?
        };

        let enc = ConformerEncoder::load(&EncoderConfig::v3_rnnt(), vb).map_err(|e| {
            RuntimeError::LoadFailed {
                path: st,
                message: e.to_string(),
            }
        })?;

        Ok(Box::new(EncoderSession::new(enc, self.device.clone())))
    }
}
