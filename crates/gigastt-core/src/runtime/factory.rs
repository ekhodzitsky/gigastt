use super::{error::RuntimeError, session::RuntimeSession};

/// Creates a `Runtime` configured for a specific execution provider.
pub trait RuntimeFactory: Send + Sync + 'static {
    fn create(&self, intra_threads: usize) -> Result<Box<dyn Runtime>, RuntimeError>;
}

/// Owns loaded sessions. One runtime per `Engine`.
pub trait Runtime: Send + Sync + 'static {
    fn load_session(
        &self,
        model_path: &std::path::Path,
    ) -> Result<Box<dyn RuntimeSession>, RuntimeError>;
}
