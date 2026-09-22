//! Operator choice of ORT execution provider, and what this build can bind.
//!
//! `auto` may fall back to CPU. An exact provider must be compiled into the
//! binary; the load path then refuses to rebuild the pool on CPU.

use crate::model::ModelVariant;

use super::factory::{BackendKind, select_backend};

/// What the operator asked for. `Auto` is the default.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExecutionProviderChoice {
    Auto,
    Cpu,
    Coreml,
    Cuda,
}

impl ExecutionProviderChoice {
    pub fn parse(s: &str) -> Result<Self, String> {
        match s.trim().to_ascii_lowercase().as_str() {
            "auto" => Ok(Self::Auto),
            "cpu" => Ok(Self::Cpu),
            "coreml" => Ok(Self::Coreml),
            "cuda" => Ok(Self::Cuda),
            other => Err(format!(
                "unknown execution provider '{other}' (expected auto, cpu, coreml, cuda)"
            )),
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::Cpu => "cpu",
            Self::Coreml => "coreml",
            Self::Cuda => "cuda",
        }
    }

    /// `auto` may rebuild sessions on CPU after a CoreML/CUDA failure.
    /// An exact choice may not.
    pub fn allows_cpu_fallback(self) -> bool {
        matches!(self, Self::Auto)
    }
}

/// Provider this binary will try to bind before any runtime fallback.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BoundProvider {
    Cpu,
    Coreml,
    Cuda,
}

impl BoundProvider {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Cpu => "cpu",
            Self::Coreml => "coreml",
            Self::Cuda => "cuda",
        }
    }
}

/// Compiled-in ORT default: CoreML, else CUDA, else CPU.
pub fn compiled_ort_default() -> BoundProvider {
    if cfg!(feature = "coreml") {
        BoundProvider::Coreml
    } else if cfg!(feature = "cuda") {
        BoundProvider::Cuda
    } else {
        BoundProvider::Cpu
    }
}

/// Name reported for `auto` after backend selection (candle / ane / ort).
pub fn auto_backend_name(variant: ModelVariant) -> &'static str {
    match select_backend(Some(variant)) {
        BackendKind::Candle => "candle",
        BackendKind::Ane => "ane",
        BackendKind::Ort => compiled_ort_default().as_str(),
    }
}

/// Exact provider, or an error when this build cannot load it.
pub fn resolve_ort_provider(choice: ExecutionProviderChoice) -> Result<BoundProvider, String> {
    match choice {
        ExecutionProviderChoice::Auto => Ok(compiled_ort_default()),
        ExecutionProviderChoice::Cpu => Ok(BoundProvider::Cpu),
        ExecutionProviderChoice::Coreml => {
            if cfg!(feature = "coreml") {
                Ok(BoundProvider::Coreml)
            } else {
                Err(
                    "execution provider 'coreml' was requested but this build does not include it \
                     (rebuild with --features coreml, or use auto/cpu)"
                        .into(),
                )
            }
        }
        ExecutionProviderChoice::Cuda => {
            if cfg!(feature = "cuda") {
                Ok(BoundProvider::Cuda)
            } else {
                Err(
                    "execution provider 'cuda' was requested but this build does not include it \
                     (rebuild with --features cuda, or use auto/cpu)"
                        .into(),
                )
            }
        }
    }
}

/// After the primary provider fails to load or to pass warmup: `auto` binds
/// CPU; an exact request fails.
pub fn bind_after_primary_failure(choice: ExecutionProviderChoice) -> Result<&'static str, ()> {
    if choice.allows_cpu_fallback() {
        Ok("cpu")
    } else {
        Err(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_execution_provider_names() {
        assert_eq!(
            ExecutionProviderChoice::parse("auto").unwrap(),
            ExecutionProviderChoice::Auto
        );
        assert_eq!(
            ExecutionProviderChoice::parse("CPU").unwrap(),
            ExecutionProviderChoice::Cpu
        );
        assert_eq!(
            ExecutionProviderChoice::parse("coreml").unwrap(),
            ExecutionProviderChoice::Coreml
        );
        assert_eq!(
            ExecutionProviderChoice::parse("cuda").unwrap(),
            ExecutionProviderChoice::Cuda
        );
        assert!(ExecutionProviderChoice::parse("vulkan").is_err());
    }

    #[test]
    fn test_auto_may_fall_back_and_exact_may_not() {
        assert_eq!(
            bind_after_primary_failure(ExecutionProviderChoice::Auto),
            Ok("cpu")
        );
        assert!(bind_after_primary_failure(ExecutionProviderChoice::Coreml).is_err());
        assert!(bind_after_primary_failure(ExecutionProviderChoice::Cuda).is_err());
        assert!(bind_after_primary_failure(ExecutionProviderChoice::Cpu).is_err());
    }

    #[test]
    fn test_exact_cpu_binds_cpu() {
        assert_eq!(
            resolve_ort_provider(ExecutionProviderChoice::Cpu).unwrap(),
            BoundProvider::Cpu
        );
        assert_eq!(BoundProvider::Cpu.as_str(), "cpu");
    }

    #[test]
    fn test_exact_unavailable_provider_fails() {
        let coreml = resolve_ort_provider(ExecutionProviderChoice::Coreml);
        let cuda = resolve_ort_provider(ExecutionProviderChoice::Cuda);
        if cfg!(feature = "coreml") {
            assert_eq!(coreml.unwrap(), BoundProvider::Coreml);
        } else {
            let msg = coreml.unwrap_err();
            assert!(msg.contains("coreml"), "{msg}");
        }
        if cfg!(feature = "cuda") {
            assert_eq!(cuda.unwrap(), BoundProvider::Cuda);
        } else {
            let msg = cuda.unwrap_err();
            assert!(msg.contains("cuda"), "{msg}");
        }
    }
}
