#![allow(dead_code)]

use crate::runtime::{
    error::RuntimeError,
    factory::{Runtime, RuntimeFactory},
};

use super::runtime::CandleRuntime;

/// Which hardware device the Candle backend runs on.
#[derive(Clone, Copy)]
pub enum CandleDevice {
    Cpu,
    Metal,
}

/// Factory that creates a `CandleRuntime` for the selected device.
pub struct CandleFactory {
    device: CandleDevice,
}

impl CandleFactory {
    /// Prefer Metal (Apple Silicon Neural Engine / GPU); fall back to CPU.
    pub fn new() -> Self {
        let device = if candle_core::Device::new_metal(0).is_ok() {
            CandleDevice::Metal
        } else {
            CandleDevice::Cpu
        };
        Self { device }
    }

    /// CPU-only factory.
    pub fn cpu() -> Self {
        Self {
            device: CandleDevice::Cpu,
        }
    }
}

impl RuntimeFactory for CandleFactory {
    fn create(&self, _intra_threads: usize) -> Result<Box<dyn Runtime>, RuntimeError> {
        let runtime = CandleRuntime::new(self.device)?;
        Ok(Box::new(runtime))
    }

    fn cpu_fallback(&self) -> Box<dyn RuntimeFactory> {
        Box::new(CandleFactory::cpu())
    }
}
