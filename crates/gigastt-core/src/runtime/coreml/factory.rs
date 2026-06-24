#![allow(dead_code)]

use crate::runtime::{
    error::RuntimeError,
    factory::{Runtime, RuntimeFactory},
    ort::factory::OrtFactory,
};

use super::runtime::AneRuntime;

/// Factory that creates a composite [`AneRuntime`]: the encoder runs on the
/// Apple Neural Engine (per-bucket `.mlpackage`), while the decoder/joiner and
/// the encoder fallback delegate to an inner ort CPU runtime.
pub struct AneFactory;

impl AneFactory {
    pub fn new() -> Self {
        Self
    }
}

impl RuntimeFactory for AneFactory {
    fn create(&self, intra_threads: usize) -> Result<Box<dyn Runtime>, RuntimeError> {
        // Inner ort CPU runtime: serves decoder/joiner (always) and the
        // variable-length encoder fallback for clips outside the fill-floor.
        let ort = OrtFactory::cpu().create(intra_threads)?;
        Ok(Box::new(AneRuntime::new(ort)))
    }

    fn cpu_fallback(&self) -> Box<dyn RuntimeFactory> {
        crate::runtime::ort::factory::cpu_factory()
    }
}
