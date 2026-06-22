use super::{error::RuntimeError, tensor::Tensor};

/// One loaded model session: encoder, decoder, or joiner.
pub trait RuntimeSession: Send + Sync + 'static {
    fn run(&self, inputs: &[Tensor]) -> Result<Vec<Tensor>, RuntimeError>;
}
