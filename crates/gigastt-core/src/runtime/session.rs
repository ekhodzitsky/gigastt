use super::{error::RuntimeError, tensor::Tensor};

/// One loaded model session: encoder, decoder, or joiner.
pub trait RuntimeSession: Send + Sync + 'static {
    fn run(&self, inputs: &[Tensor]) -> Result<Vec<Tensor>, RuntimeError>;

    /// Low-latency encoder path used by streaming windows.
    ///
    /// Default delegates to [`Self::run`]. The ANE encoder overrides this to
    /// accept a lower pad fill floor so short streaming windows can pad into the
    /// smallest eligible bucket instead of falling back to ort.
    fn run_low_latency(&self, inputs: &[Tensor]) -> Result<Vec<Tensor>, RuntimeError> {
        self.run(inputs)
    }

    /// True when this encoder runs on the ANE fixed-shape pad-up path.
    ///
    /// Used to pick a longer long-form chunk window (30s fills ANE bucket 3000
    /// nearly full; ort keeps 24s for peak activation memory). Default false.
    fn is_ane_encoder(&self) -> bool {
        false
    }
}
