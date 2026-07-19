#![allow(dead_code)]

use std::path::{Path, PathBuf};
use std::sync::{Arc, OnceLock};

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
    /// Returns the execution-provider list to register when loading a session.
    ///
    /// `model_path` is used to derive provider-specific cache directories (e.g.
    /// CoreML's `coreml_cache/` next to the model).
    pub(crate) fn execution_providers(
        self,
        model_path: &Path,
    ) -> Vec<ort::ep::ExecutionProviderDispatch> {
        match self {
            Self::Cpu => vec![ort::ep::CPU::default().build()],
            Self::CoreML => {
                let cache_dir = model_path
                    .parent()
                    .map(|p| p.join("coreml_cache"))
                    .unwrap_or_else(|| PathBuf::from("coreml_cache"));
                let coreml_ep = ort::ep::CoreML::default()
                    .with_model_format(ort::ep::coreml::ModelFormat::MLProgram)
                    .with_static_input_shapes(true)
                    .with_compute_units(ort::ep::coreml::ComputeUnits::CPUAndNeuralEngine)
                    .with_specialization_strategy(
                        ort::ep::coreml::SpecializationStrategy::FastPrediction,
                    )
                    .with_model_cache_dir(cache_dir.to_string_lossy())
                    .build();
                vec![coreml_ep, ort::ep::CPU::default().build()]
            }
            Self::Cuda => vec![
                ort::ep::CUDA::default().build(),
                ort::ep::CPU::default().build(),
            ],
            Self::Nnapi => vec![
                ort::ep::NNAPI::default().build(),
                ort::ep::CPU::default().build(),
            ],
        }
    }

    /// Whether this provider is the plain CPU execution provider.
    pub(crate) fn is_cpu(self) -> bool {
        matches!(self, Self::Cpu)
    }
}

/// Factory that creates an `ort` runtime configured for a specific provider.
pub struct OrtFactory {
    provider: OrtExecutionProvider,
    prepacked: Option<Arc<ort::session::builder::PrepackedWeights>>,
    optimized_cache_dir: Option<PathBuf>,
}

impl OrtFactory {
    fn with_provider(provider: OrtExecutionProvider) -> Self {
        Self {
            provider,
            prepacked: None,
            optimized_cache_dir: None,
        }
    }

    pub fn cpu() -> Self {
        Self::with_provider(OrtExecutionProvider::Cpu)
    }

    pub fn coreml() -> Self {
        Self::with_provider(OrtExecutionProvider::CoreML)
    }

    pub fn cuda() -> Self {
        Self::with_provider(OrtExecutionProvider::Cuda)
    }

    pub fn nnapi() -> Self {
        Self::with_provider(OrtExecutionProvider::Nnapi)
    }

    pub fn with_prepacked_weights(
        mut self,
        prepacked: Arc<ort::session::builder::PrepackedWeights>,
    ) -> Self {
        self.prepacked = Some(prepacked);
        self
    }

    pub fn with_optimized_cache_dir(mut self, dir: impl Into<PathBuf>) -> Self {
        self.optimized_cache_dir = Some(dir.into());
        self
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
        Ok(Box::new(OrtRuntime::new(
            intra_threads,
            self.provider,
            self.prepacked.clone(),
            self.optimized_cache_dir.clone(),
        )))
    }

    fn cpu_fallback(&self) -> Box<dyn RuntimeFactory> {
        Box::new(OrtFactory::cpu())
    }
}

/// Returns the default factory for the active compile-time feature flags.
///
/// When `feature = "candle"` is enabled, returns a `CandleFactory` (Metal on
/// Apple Silicon, CPU otherwise). Otherwise returns an `OrtFactory` selected
/// by the active execution-provider feature.
///
/// NOTE: the Candle backend is rnnt-only (34-token char vocab,
/// `EncoderConfig::v3_rnnt()`); it cannot serve an `e2e_rnnt` model. This entry
/// point has no model directory to detect the variant from, so it always returns
/// `CandleFactory` under the feature — callers that know the directory should use
/// [`production_factory`], which falls back to the ort factory for non-rnnt
/// models.
pub fn default_factory() -> Box<dyn RuntimeFactory> {
    #[cfg(feature = "candle")]
    {
        Box::new(crate::runtime::candle::factory::CandleFactory::new())
    }
    #[cfg(all(feature = "ane", target_os = "macos"))]
    {
        Box::new(crate::runtime::coreml::factory::AneFactory::new())
    }
    #[cfg(not(any(feature = "candle", all(feature = "ane", target_os = "macos"))))]
    {
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
}

/// Returns a CPU-only `ort` factory for auxiliary models.
pub fn cpu_factory() -> Box<dyn RuntimeFactory> {
    Box::new(OrtFactory::cpu())
}

/// Returns a production `ort` factory that preserves the provider selection and
/// disk-cache layout used by the engine before the runtime abstraction.
///
/// Public, stable 1-arg form: selects the backend from the variant detected on
/// disk. The engine calls the crate-internal `production_factory_variant`
/// instead, passing the head it has already resolved so an explicit
/// `--model-variant` is honored.
pub fn production_factory(model_dir: &Path) -> Box<dyn RuntimeFactory> {
    production_factory_variant(
        model_dir,
        crate::model::ModelVariant::detect_in_dir(model_dir),
    )
}

/// Like [`production_factory`], but the caller supplies the resolved recognition
/// head. The rnnt-only candle/ane backends are gated on `variant` directly —
/// re-detecting from disk here would reintroduce the multi-head bug where an
/// explicit `--model-variant` override is overruled by `rnnt`-precedence
/// detection. `None` (nothing resolved/detected) selects the ort factory, never
/// the rnnt-only backends — matching the historical `production_factory`.
pub(crate) fn production_factory_variant(
    model_dir: &Path,
    variant: Option<crate::model::ModelVariant>,
) -> Box<dyn RuntimeFactory> {
    // The candle/ane backends are rnnt-only, so they fire only for an explicit
    // (or detected) `Rnnt` head. `is_rnnt` gates only those feature-gated blocks;
    // on plain ort builds (cpu / coreml / cuda) it is otherwise unused.
    let is_rnnt = variant == Some(crate::model::ModelVariant::Rnnt);
    let _ = is_rnnt;
    // The Candle backend is rnnt-only (34-token char vocab,
    // `EncoderConfig::v3_rnnt()`); for any other head it would produce wrong
    // output / fail to load. Use it only for the `Rnnt` head, otherwise fall back
    // to the ort factory below.
    #[cfg(feature = "candle")]
    {
        if is_rnnt {
            return Box::new(crate::runtime::candle::factory::CandleFactory::new());
        }
    }
    // The ANE backend is rnnt-only (same restriction as Candle) and macOS-only
    // (Apple frameworks); use it only for the `Rnnt` head, otherwise fall back to
    // the ort factory below.
    #[cfg(all(feature = "ane", target_os = "macos"))]
    {
        if is_rnnt {
            return Box::new(crate::runtime::coreml::factory::AneFactory::new());
        }
    }

    let factory = if cfg!(feature = "coreml") {
        OrtFactory::coreml()
    } else if cfg!(feature = "cuda") {
        OrtFactory::cuda()
    } else {
        OrtFactory::cpu().with_optimized_cache_dir(model_dir.join("optimized_cache"))
    };
    Box::new(factory)
}
