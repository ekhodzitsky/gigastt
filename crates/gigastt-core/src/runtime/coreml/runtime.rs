#![allow(dead_code)]

use crate::runtime::{error::RuntimeError, factory::Runtime, session::RuntimeSession};

/// ANE/CoreML runtime stub.
///
/// Phase 0: `load_session` returns a clear "not yet implemented" error.
/// Phase 2 will add `objc2_core_ml` calls inside this module.
pub struct AneRuntime;

impl AneRuntime {
    pub fn new() -> Self {
        Self
    }
}

impl Runtime for AneRuntime {
    fn load_session(
        &self,
        model_path: &std::path::Path,
        _is_encoder: bool,
    ) -> Result<Box<dyn RuntimeSession>, RuntimeError> {
        Err(RuntimeError::LoadFailed {
            path: model_path.to_path_buf(),
            message: "ANE backend not yet implemented; Phase 2".to_string(),
        })
    }
}
