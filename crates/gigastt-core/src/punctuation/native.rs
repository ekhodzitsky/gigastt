//! Experimental native RUPunct adapter. Model provisioning stays explicit.
use anyhow::Result;
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};

/// Opt-in rupunct backend with gigastt's fail-soft restoration contract.
///
/// This does not replace or download the standard ONNX punctuation model.
/// One request uses one window worker to avoid multiplying the server's
/// concurrent requests by the library's automatic window parallelism. Native
/// GEMM uses Rayon separately on Linux; this window limit does not cap that pool.
pub struct NativePunctuator {
    model: rupunct::Punctuator,
    failures: AtomicU64,
}

impl NativePunctuator {
    /// Load a locally provisioned Q8/F32 safetensors checkpoint.
    pub fn load(model_dir: &Path) -> Result<Self> {
        Ok(Self {
            model: rupunct::Punctuator::load(model_dir)?,
            failures: AtomicU64::new(0),
        })
    }
    /// Restore text, returning the original input on failure.
    pub fn restore(&self, text: &str) -> String {
        let result = self.model.restore_with(
            text,
            rupunct::RestoreOpts {
                threads: 1,
                ..Default::default()
            },
        );
        finish_restore(text, result.map_err(Into::into), &self.failures)
    }
    /// Calls that failed and returned their original input.
    pub fn failed_restores(&self) -> u64 {
        self.failures.load(Ordering::Relaxed)
    }
}

fn finish_restore(text: &str, result: Result<String>, failures: &AtomicU64) -> String {
    match result {
        Ok(output) => output,
        Err(error) => {
            failures.fetch_add(1, Ordering::Relaxed);
            tracing::warn!("Native punctuation restore failed, returning bare text: {error:#}");
            text.to_owned()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_native_failure_preserves_input_and_reports_failure() {
        let failures = AtomicU64::new(0);
        let input = "  привет\nмир ";
        assert_eq!(
            finish_restore(input, Err(anyhow::anyhow!("inference failed")), &failures),
            input
        );
        assert_eq!(failures.load(Ordering::Relaxed), 1);
    }
    #[test]
    fn test_native_success_does_not_report_failure() {
        let failures = AtomicU64::new(0);
        assert_eq!(
            finish_restore("привет", Ok("Привет!".into()), &failures),
            "Привет!"
        );
        assert_eq!(failures.load(Ordering::Relaxed), 0);
    }
    #[test]
    fn test_native_missing_model_returns_load_error() {
        let directory = tempfile::tempdir().expect("temporary directory");
        assert!(NativePunctuator::load(directory.path()).is_err());
    }
}
