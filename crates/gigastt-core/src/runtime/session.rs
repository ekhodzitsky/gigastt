use super::{error::RuntimeError, tensor::Tensor};

/// One loaded model session: encoder, decoder, or joiner.
#[allow(dead_code)]
pub trait RuntimeSession: Send + Sync + 'static {
    fn run(&self, inputs: Vec<Tensor>) -> Result<Vec<Tensor>, RuntimeError>;
}
