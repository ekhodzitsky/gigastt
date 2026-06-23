#![allow(dead_code)]

use crate::runtime::{error::RuntimeError, factory::Runtime, session::RuntimeSession};

use super::factory::CandleDevice;

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
        _is_encoder: bool,
    ) -> Result<Box<dyn RuntimeSession>, RuntimeError> {
        Err(RuntimeError::InferenceFailed(format!(
            "candle backend not yet implemented for {}",
            model_path.display()
        )))
    }
}
