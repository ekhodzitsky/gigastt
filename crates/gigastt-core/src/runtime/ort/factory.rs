#![allow(dead_code)]

use std::sync::OnceLock;

use crate::runtime::{
    error::RuntimeError,
    factory::{Runtime, RuntimeFactory},
};

use super::session::OrtRuntime;

#[cfg(all(feature = "coreml", feature = "cuda"))]
compile_error!("features `coreml` and `cuda` are mutually exclusive");

/// `ort` execution provider selector.
#[derive(Clone, Copy)]
pub enum OrtExecutionProvider {
    Cpu,
    CoreML,
    Cuda,
    Nnapi,
}

impl OrtExecutionProvider {
    pub(crate) fn to_ort(self) -> ort::ep::ExecutionProviderDispatch {
        match self {
            Self::Cpu => ort::ep::CPU::default().build(),
            #[cfg(feature = "coreml")]
            Self::CoreML => ort::ep::CoreML::default().build(),
            #[cfg(not(feature = "coreml"))]
            Self::CoreML => ort::ep::CPU::default().build(),
            #[cfg(feature = "cuda")]
            Self::Cuda => ort::ep::CUDA::default().build(),
            #[cfg(not(feature = "cuda"))]
            Self::Cuda => ort::ep::CPU::default().build(),
            #[cfg(feature = "nnapi")]
            Self::Nnapi => ort::ep::NNAPI::default().build(),
            #[cfg(not(feature = "nnapi"))]
            Self::Nnapi => ort::ep::CPU::default().build(),
        }
    }
}

/// Factory that creates an `ort` runtime configured for a specific provider.
pub struct OrtFactory {
    provider: OrtExecutionProvider,
}

impl OrtFactory {
    pub fn cpu() -> Self {
        Self {
            provider: OrtExecutionProvider::Cpu,
        }
    }

    pub fn coreml() -> Self {
        Self {
            provider: OrtExecutionProvider::CoreML,
        }
    }

    pub fn cuda() -> Self {
        Self {
            provider: OrtExecutionProvider::Cuda,
        }
    }

    pub fn nnapi() -> Self {
        Self {
            provider: OrtExecutionProvider::Nnapi,
        }
    }
}

static ORT_INIT: OnceLock<bool> = OnceLock::new();

fn ensure_ort_initialized() {
    let initialized_by_us = ORT_INIT.get_or_init(|| ort::init().with_name("gigastt").commit());
    if !initialized_by_us {
        tracing::warn!(
            "ort environment was already configured before gigastt initialization; execution provider settings may not apply"
        );
    }
}

impl RuntimeFactory for OrtFactory {
    fn create(&self, intra_threads: usize) -> Result<Box<dyn Runtime>, RuntimeError> {
        ensure_ort_initialized();
        Ok(Box::new(OrtRuntime::new(intra_threads, self.provider)))
    }
}

/// Returns the default `ort` factory for the active compile-time feature flags.
pub fn default_factory(intra_threads: usize) -> Box<dyn RuntimeFactory> {
    let _ = intra_threads;
    if cfg!(feature = "coreml") {
        Box::new(OrtFactory::coreml())
    } else if cfg!(feature = "cuda") {
        Box::new(OrtFactory::cuda())
    } else if cfg!(feature = "nnapi") {
        Box::new(OrtFactory::nnapi())
    } else {
        Box::new(OrtFactory::cpu())
    }
}
