#![allow(dead_code)]

use crate::runtime::{
    error::RuntimeError,
    factory::{Runtime, RuntimeFactory},
};

use super::runtime::AneRuntime;

/// Factory that creates an `AneRuntime` (stub in Phase 0).
pub struct AneFactory;

impl AneFactory {
    pub fn new() -> Self {
        Self
    }
}

impl RuntimeFactory for AneFactory {
    fn create(&self, _intra_threads: usize) -> Result<Box<dyn Runtime>, RuntimeError> {
        Ok(Box::new(AneRuntime::new()))
    }

    fn cpu_fallback(&self) -> Box<dyn RuntimeFactory> {
        crate::runtime::ort::factory::cpu_factory()
    }
}
