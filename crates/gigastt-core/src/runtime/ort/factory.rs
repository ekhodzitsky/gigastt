use crate::runtime::{
    error::RuntimeError,
    factory::{Runtime, RuntimeFactory},
};

use super::session::OrtRuntime;

/// `ort` execution provider selector.
#[derive(Clone)]
#[allow(dead_code)]
pub enum OrtExecutionProvider {
    Cpu,
    #[cfg(feature = "coreml")]
    CoreML,
    #[cfg(feature = "cuda")]
    Cuda,
    #[cfg(feature = "nnapi")]
    Nnapi,
}

impl OrtExecutionProvider {
    fn to_ort(&self) -> ort::ep::ExecutionProviderDispatch {
        match self {
            Self::Cpu => ort::ep::CPU::default().build(),
            #[cfg(feature = "coreml")]
            Self::CoreML => ort::ep::CoreML::default().build(),
            #[cfg(feature = "cuda")]
            Self::Cuda => ort::ep::CUDA::default().build(),
            #[cfg(feature = "nnapi")]
            Self::Nnapi => ort::ep::NNAPI::default().build(),
        }
    }
}

/// Factory that creates an `ort` runtime configured for a specific provider.
#[allow(dead_code)]
pub struct OrtFactory {
    provider: OrtExecutionProvider,
}

impl OrtFactory {
    #[allow(dead_code)]
    pub fn cpu() -> Self {
        Self {
            provider: OrtExecutionProvider::Cpu,
        }
    }

    #[cfg(feature = "coreml")]
    #[allow(dead_code)]
    pub fn coreml() -> Self {
        Self {
            provider: OrtExecutionProvider::CoreML,
        }
    }

    #[cfg(feature = "cuda")]
    #[allow(dead_code)]
    pub fn cuda() -> Self {
        Self {
            provider: OrtExecutionProvider::Cuda,
        }
    }

    #[cfg(feature = "nnapi")]
    #[allow(dead_code)]
    pub fn nnapi() -> Self {
        Self {
            provider: OrtExecutionProvider::Nnapi,
        }
    }
}

impl RuntimeFactory for OrtFactory {
    fn create(&self, intra_threads: usize) -> Result<Box<dyn Runtime>, RuntimeError> {
        ort::init()
            .with_name("gigastt")
            .with_execution_providers([self.provider.to_ort()])
            .commit();
        Ok(Box::new(OrtRuntime::new(
            intra_threads,
            self.provider.to_ort(),
        )))
    }
}

/// Returns the default `ort` factory for the active compile-time feature flags.
#[allow(dead_code)]
pub fn default_factory(_intra_threads: usize) -> Box<dyn RuntimeFactory> {
    #[cfg(feature = "coreml")]
    return Box::new(OrtFactory::coreml());
    #[cfg(feature = "cuda")]
    return Box::new(OrtFactory::cuda());
    #[cfg(feature = "nnapi")]
    return Box::new(OrtFactory::nnapi());
    #[cfg(not(any(feature = "coreml", feature = "cuda", feature = "nnapi")))]
    Box::new(OrtFactory::cpu())
}
