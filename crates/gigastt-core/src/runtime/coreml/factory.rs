use crate::runtime::{
    error::RuntimeError,
    factory::{Runtime, RuntimeFactory},
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
        // Inner CPU runtime via the re-exported `crate::runtime::cpu_factory`
        // seam rather than the concrete ort factory type — going through the
        // re-export keeps the concrete ort types confined to their own module
        // per the CI isolation guard. Serves decoder/joiner (always) and the
        // variable-length encoder fallback for clips outside the fill-floor.
        let inner = crate::runtime::cpu_factory().create(intra_threads)?;
        Ok(Box::new(AneRuntime::new(inner)))
    }

    fn cpu_fallback(&self) -> Box<dyn RuntimeFactory> {
        crate::runtime::cpu_factory()
    }
}
