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
///
/// As of `ort` 2.0.0-rc.13 the CoreML / CUDA / NNAPI providers live behind
/// `ort`'s own `coreml` / `cuda` / `nnapi` Cargo features, so the variants that
/// name them are compiled in only when our matching feature (which enables the
/// upstream one) is on. A default CPU build carries only `Cpu`. This type is
/// crate-internal (`pub(crate) mod runtime`), so the feature-conditional variant
/// set is not part of the public API.
#[derive(Clone, Copy)]
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
    /// Returns the execution-provider list to register when loading a session.
    ///
    /// `model_path` is used to derive provider-specific cache directories (e.g.
    /// CoreML's version-scoped `coreml_cache/ort-<minor>/` next to the model).
    pub(crate) fn execution_providers(
        self,
        model_path: &Path,
    ) -> Vec<ort::ep::ExecutionProviderDispatch> {
        // Each non-CPU arm names a provider type that `ort` 2.0.0-rc.13 gates
        // behind its own feature, so the arm is gated on the matching feature —
        // the default build compiles a single `Cpu` arm and never references a
        // type that is configured out. `model_path` is only read by the CoreML
        // arm; the `let _` below keeps it accounted for on builds without it.
        #[cfg(not(feature = "coreml"))]
        let _ = model_path;
        match self {
            Self::Cpu => vec![ort::ep::CPU::default().build()],
            #[cfg(feature = "coreml")]
            Self::CoreML => {
                // Version-scoped: `coreml_cache/ort-<minor>/`. The CoreML EP keys
                // its compiled bundles by graph hash only, so an ORT upgrade would
                // otherwise load a bundle a different ONNX Runtime compiled and
                // fail into a silent CPU fallback. Scoping by ORT version makes the
                // upgrade miss the stale entry and recompile once (self-healing).
                let cache_dir = match model_path.parent() {
                    Some(p) => crate::model::coreml_cache_dir(p),
                    None => crate::model::coreml_cache_dir(Path::new(".")),
                };
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
            #[cfg(feature = "cuda")]
            Self::Cuda => vec![
                ort::ep::CUDA::default().build(),
                ort::ep::CPU::default().build(),
            ],
            #[cfg(feature = "nnapi")]
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

    #[cfg(feature = "coreml")]
    pub fn coreml() -> Self {
        Self::with_provider(OrtExecutionProvider::CoreML)
    }

    #[cfg(feature = "cuda")]
    pub fn cuda() -> Self {
        Self::with_provider(OrtExecutionProvider::Cuda)
    }

    #[cfg(feature = "nnapi")]
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
    // Select the provider with `#[cfg]`, not a runtime `cfg!()`: since rc.13 the
    // accelerated constructors don't exist unless their feature is on, so a
    // `cfg!()` branch that merely evaluates false at runtime would still have to
    // compile a call to a function that isn't there. The `not(...)` guards keep
    // exactly one block active for any feature combination (coreml precedes cuda
    // precedes nnapi; coreml+cuda is already a `compile_error!`).
    #[cfg(not(any(feature = "candle", all(feature = "ane", target_os = "macos"))))]
    {
        #[cfg(feature = "coreml")]
        {
            Box::new(OrtFactory::coreml())
        }
        #[cfg(all(feature = "cuda", not(feature = "coreml")))]
        {
            Box::new(OrtFactory::cuda())
        }
        #[cfg(all(feature = "nnapi", not(feature = "coreml"), not(feature = "cuda")))]
        {
            Box::new(OrtFactory::nnapi())
        }
        #[cfg(not(any(feature = "coreml", feature = "cuda", feature = "nnapi")))]
        {
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

/// Which runtime backend [`production_factory_variant`] selects for a resolved head.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum BackendKind {
    /// Default `ort` backend (CPU / CoreML EP / CUDA EP, by compile-time feature).
    Ort,
    /// Pure-Rust Candle backend — rnnt-only.
    Candle,
    /// Apple Neural Engine backend — rnnt-only, macOS-only.
    Ane,
}

/// Pure backend selection for [`production_factory_variant`]. The rnnt-only
/// Candle/ANE backends are chosen ONLY for a resolved `Rnnt` head on a build that
/// compiled them in; every other head — and `None` — uses the `ort` backend.
/// Extracted so the `--model-variant` → backend gate (the candle/ane half of the
/// multi-head fix) is unit-testable without model files.
pub(crate) fn select_backend(variant: Option<crate::model::ModelVariant>) -> BackendKind {
    let is_rnnt = variant == Some(crate::model::ModelVariant::Rnnt);
    #[cfg(feature = "candle")]
    if is_rnnt {
        return BackendKind::Candle;
    }
    #[cfg(all(feature = "ane", target_os = "macos"))]
    if is_rnnt {
        return BackendKind::Ane;
    }
    let _ = is_rnnt;
    BackendKind::Ort
}

/// Like [`production_factory`], but the caller supplies the resolved recognition
/// head. The rnnt-only candle/ane backends are gated on `variant` directly (see
/// [`select_backend`]) — re-detecting from disk here would reintroduce the
/// multi-head bug where an explicit `--model-variant` override is overruled by
/// `rnnt`-precedence detection. `None` (nothing resolved/detected) selects the ort
/// factory, never the rnnt-only backends — matching the historical
/// `production_factory`.
pub(crate) fn production_factory_variant(
    model_dir: &Path,
    variant: Option<crate::model::ModelVariant>,
) -> Box<dyn RuntimeFactory> {
    let backend = select_backend(variant);
    // The Candle/ANE backends are rnnt-only (34-token char vocab,
    // `EncoderConfig::v3_rnnt()`); for any other head they would produce wrong
    // output / fail to load, so `select_backend` only picks them for `Rnnt`.
    #[cfg(feature = "candle")]
    if backend == BackendKind::Candle {
        return Box::new(crate::runtime::candle::factory::CandleFactory::new());
    }
    #[cfg(all(feature = "ane", target_os = "macos"))]
    if backend == BackendKind::Ane {
        return Box::new(crate::runtime::coreml::factory::AneFactory::new());
    }
    let _ = backend;

    // `#[cfg]` rather than a runtime `cfg!()` for the same reason as
    // `default_factory`: the accelerated constructors are compiled out without
    // their feature. Only the CPU branch reads `model_dir`, so it is marked used
    // on the accelerated builds. This path selects coreml/cuda/cpu only — nnapi
    // is a mobile target reached through `default_factory`, not the server.
    #[cfg(feature = "coreml")]
    let factory = OrtFactory::coreml();
    #[cfg(all(feature = "cuda", not(feature = "coreml")))]
    let factory = OrtFactory::cuda();
    #[cfg(not(any(feature = "coreml", feature = "cuda")))]
    let factory = {
        // Shared PrepackedWeights across every session this factory creates.
        // ORT still materializes per-session initializers for most graphs; the
        // container shares prepacked kernel buffers when the EP supports it.
        // Enabled as the weight-share spike: remeasure pool1→2 RSS after deploy
        // (see specs/research theories T-002 / T-021). Safe no-op if unused.
        let prepacked = std::sync::Arc::new(ort::session::builder::PrepackedWeights::new());
        OrtFactory::cpu()
            .with_optimized_cache_dir(model_dir.join("optimized_cache"))
            .with_prepacked_weights(prepacked)
    };
    #[cfg(any(feature = "coreml", feature = "cuda"))]
    let _ = model_dir;
    Box::new(factory)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::ModelVariant;

    // The rnnt-only candle/ane backends must NEVER be selected for a non-rnnt head
    // or for `None`, on any build — this is the exact gate the multi-head
    // `--model-variant` fix added so candle/ane honor the resolved head instead of
    // re-detecting `rnnt` from disk. Model-free; runs in the PR-gating unit tests.
    #[test]
    fn select_backend_non_rnnt_and_none_are_always_ort() {
        assert_eq!(
            select_backend(Some(ModelVariant::E2eRnnt)),
            BackendKind::Ort
        );
        assert_eq!(select_backend(Some(ModelVariant::MlCtc)), BackendKind::Ort);
        assert_eq!(
            select_backend(Some(ModelVariant::MlCtcLarge)),
            BackendKind::Ort
        );
        assert_eq!(select_backend(None), BackendKind::Ort);
    }

    // On the default ort builds (cpu / coreml / cuda) even `Rnnt` uses the ort
    // backend — the rnnt-only accelerated backends aren't compiled in.
    #[cfg(not(any(feature = "candle", all(feature = "ane", target_os = "macos"))))]
    #[test]
    fn select_backend_rnnt_is_ort_without_accelerated_backend() {
        assert_eq!(select_backend(Some(ModelVariant::Rnnt)), BackendKind::Ort);
    }

    #[test]
    fn test_cpu_factory_can_attach_prepacked_weights() {
        let pw = std::sync::Arc::new(ort::session::builder::PrepackedWeights::new());
        let f = OrtFactory::cpu().with_prepacked_weights(pw);
        // create() must succeed without a model path (runtime shell only).
        let rt = f.create(1).expect("cpu runtime with prepacked");
        drop(rt);
    }

    // On a candle build, `Rnnt` picks the Candle backend but every other head (and
    // `None`) still falls through to ort — proving the fix's `is_rnnt` gate.
    #[cfg(feature = "candle")]
    #[test]
    fn select_backend_candle_only_for_rnnt() {
        assert_eq!(
            select_backend(Some(ModelVariant::Rnnt)),
            BackendKind::Candle
        );
        assert_eq!(
            select_backend(Some(ModelVariant::E2eRnnt)),
            BackendKind::Ort
        );
        assert_eq!(select_backend(Some(ModelVariant::MlCtc)), BackendKind::Ort);
        assert_eq!(select_backend(None), BackendKind::Ort);
    }

    // Same for the ANE (macOS) build.
    #[cfg(all(feature = "ane", target_os = "macos"))]
    #[test]
    fn select_backend_ane_only_for_rnnt() {
        assert_eq!(select_backend(Some(ModelVariant::Rnnt)), BackendKind::Ane);
        assert_eq!(
            select_backend(Some(ModelVariant::E2eRnnt)),
            BackendKind::Ort
        );
        assert_eq!(select_backend(None), BackendKind::Ort);
    }
}
