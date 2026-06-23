//! Candle-backed encoder [`RuntimeSession`].

use candle_core::Device;

use crate::runtime::{error::RuntimeError, session::RuntimeSession, tensor::Tensor};

use super::conformer::ConformerEncoder;

/// Wraps a loaded Candle Conformer encoder behind the [`RuntimeSession`] seam.
pub struct EncoderSession {
    enc: ConformerEncoder,
    device: Device,
}

impl EncoderSession {
    pub(crate) fn new(enc: ConformerEncoder, device: Device) -> Self {
        Self { enc, device }
    }
}

impl RuntimeSession for EncoderSession {
    /// Encoder contract: `inputs[0] = audio_signal [1, 64, T] F32`,
    /// `inputs[1] = length [1]` (ignored; batch is always 1 here).
    /// Returns `[1, 768, T/4] F32` (channels-first), matching the ort backend.
    fn run(&self, inputs: &[Tensor]) -> Result<Vec<Tensor>, RuntimeError> {
        if inputs.is_empty() {
            return Err(RuntimeError::InvalidInputCount {
                expected: 1,
                got: inputs.len(),
            });
        }
        let mel = super::tensor::to_candle(&inputs[0], &self.device)?;
        let out = self
            .enc
            .forward(&mel)
            .map_err(|e| RuntimeError::InferenceFailed(e.to_string()))?;
        Ok(vec![super::tensor::from_candle(&out)?])
    }
}
