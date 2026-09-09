//! Engine boot recipe shared by `serve` and the offline CLI paths
//! (`transcribe`, `transcribe-batch`, `watch`).
//!
//! One place builds a fully-configured [`Engine`](gigastt_core::inference::Engine)
//! from CLI-shaped options (model dir, pool sizes, punctuation, ITN, VAD,
//! hotwords, threads). Post-processor chains
//! (`.with_punctuator().with_itn().with_vad().with_hotwords()`) live here so
//! serve first-boot, admin reload, and offline commands stay byte-identical.

use gigastt_core::inference;
use gigastt_core::model::{self, ModelVariant};

mod modes;
mod sidecars;

pub use modes::{
    ItnMode, PunctuationMode, parse_itn_mode, parse_punctuation_mode,
    resolve_encoder_intra_threads, resolve_itn, resolve_punctuation,
};
pub use sidecars::{
    DEFAULT_HOTWORDS_BOOST, build_vad_config, ensure_int8_encoder, maybe_download_punct_model,
    maybe_download_vad_model, maybe_load_punctuator, maybe_load_vad, parse_hotwords_file,
    resolve_hotwords,
};

/// Log RSS after engine load (platform-specific; best-effort).
pub fn log_rss() {
    #[cfg(target_os = "linux")]
    {
        if let Ok(status) = std::fs::read_to_string("/proc/self/status")
            && let Some(line) = status.lines().find(|l| l.starts_with("VmRSS:"))
        {
            tracing::info!("{}", line.trim());
        }
    }
    // On macOS/other platforms, use `ps` as a simple cross-platform fallback
    #[cfg(not(target_os = "linux"))]
    {
        if let Ok(output) = std::process::Command::new("ps")
            .args(["-o", "rss=", "-p", &std::process::id().to_string()])
            .output()
            && let Ok(rss) = String::from_utf8_lossy(&output.stdout)
                .trim()
                .parse::<u64>()
        {
            tracing::info!(rss_mb = rss / 1024, "memory_after_load");
        }
    }
}

/// Log a concise summary of the active ANE (Core ML / Apple Neural Engine)
/// encoder backend at startup. No-op outside `--features ane`.
///
/// ANE is rnnt-only and macOS-only: it engages only when the resolved head is
/// `rnnt` (mirroring [`gigastt_core::production_factory`]'s variant gate); an
/// `e2e_rnnt` (or multilingual CTC) model transparently stays on the ort
/// encoder. When engaged, encoder windows pad up to a fixed bucket and run on
/// the ANE in both file mode (50% fill floor) and streaming (zero fill floor,
/// so short streaming windows pad into the smallest bucket); only windows
/// outside the bucket ladder fall back to the CPU/ort encoder.
#[cfg(feature = "ane")]
pub fn log_ane_backend(resolved: ModelVariant) {
    if resolved == ModelVariant::Rnnt {
        tracing::info!(
            "ANE encoder backend active (Core ML / Apple Neural Engine, macOS ARM64): \
             encoder windows pad up to fixed buckets and run on the ANE (file mode: 50% \
             fill floor; streaming: zero floor, pads into the smallest bucket); only \
             windows outside the bucket ladder fall back to the CPU/ort encoder"
        );
    } else {
        tracing::info!(
            "ANE encoder backend requested but the loaded head is {}; ANE is rnnt-only, \
             so this model runs on the ort encoder",
            resolved.as_str()
        );
    }
}

// ---------------------------------------------------------------------------
// EngineRecipe
// ---------------------------------------------------------------------------

/// CLI-shaped options that fully configure an [`inference::Engine`].
///
/// Used by `serve` (including the admin-reload [`crate::server::EngineBuilder`]
/// closure) and the offline `transcribe` / batch / watch paths so post-processor
/// chains stay in one place.
#[derive(Debug, Clone)]
pub struct EngineRecipe {
    pub model_dir: String,
    pub model_variant: Option<ModelVariant>,
    pub punctuation: PunctuationMode,
    pub punct_model_dir: String,
    pub itn: ItnMode,
    pub hotwords_file: Option<String>,
    pub hotwords_default: bool,
    pub hotwords_boost: Option<f32>,
    pub vad: bool,
    pub vad_threshold: Option<f32>,
    pub vad_min_silence_ms: Option<u32>,
    pub vad_model_dir: String,
    pub encoder_intra_threads: Option<usize>,
    pub pool_size: usize,
    /// Minimum triplets required to boot (serve degraded-pool floor). Offline
    /// paths always use `1`.
    pub pool_min_size: usize,
    /// Triplets reserved for batch REST jobs (serve only; offline uses `0`).
    pub batch_pool_size: usize,
    /// When true, run [`ensure_int8_encoder`] before load (packaging / tests).
    /// Product serve/download leave this false — they fetch lean INT8 only.
    pub quantize: bool,
    /// When `quantize` is true and INT8 is missing: if true, error (no FP32
    /// load); if false, attempt on-device quantize from a local FP32 encoder.
    pub skip_quantize: bool,
    /// Optional endpoint-mode token (`auto` / `assistant` / …). `None` leaves
    /// the engine default (offline paths). Serve always sets this.
    pub endpoint_mode: Option<String>,
    /// Optional streaming window cap in seconds. `None` leaves the engine
    /// default (2.5 s); serve always sets this. Offline paths ignore it (they
    /// use file transcription, not the streaming window).
    pub stream_max_window_secs: Option<f64>,
    /// Stable-prefix commits at the window cap (serve flag; engine default is
    /// on, offline recipes leave it off — they never slide a streaming window).
    pub stream_stable_prefix: bool,
    /// Max pooled triplets one file decode may hold for overlapping-window
    /// parallelism. `1` keeps the serial loop (default).
    pub file_window_concurrency: usize,
    /// Override for the CPU encoder's ORT optimized-graph cache directory
    /// (serve `--optimized-cache-dir`). `None` keeps the default
    /// `<model_dir>/optimized_cache`.
    pub optimized_cache_dir: Option<String>,
}

impl EngineRecipe {
    /// Offline defaults: single-slot floor, no batch pool, no quantize side
    /// effect, no endpoint mode override.
    #[allow(clippy::too_many_arguments)]
    pub fn offline(
        model_dir: String,
        model_variant: Option<ModelVariant>,
        punctuation: PunctuationMode,
        punct_model_dir: String,
        itn: ItnMode,
        hotwords_file: Option<String>,
        hotwords_default: bool,
        hotwords_boost: Option<f32>,
        vad: bool,
        vad_threshold: Option<f32>,
        vad_min_silence_ms: Option<u32>,
        vad_model_dir: String,
        encoder_intra_threads: Option<usize>,
        pool_size: usize,
    ) -> Self {
        Self {
            model_dir,
            model_variant,
            punctuation,
            punct_model_dir,
            itn,
            hotwords_file,
            hotwords_default,
            hotwords_boost,
            vad,
            vad_threshold,
            vad_min_silence_ms,
            vad_model_dir,
            encoder_intra_threads,
            pool_size,
            pool_min_size: 1,
            batch_pool_size: 0,
            quantize: false,
            skip_quantize: true,
            endpoint_mode: None,
            stream_max_window_secs: None,
            stream_stable_prefix: false,
            file_window_concurrency: 1,
            optimized_cache_dir: None,
        }
    }

    /// Cap overlapping-window parallelism for file transcription.
    pub fn with_file_window_concurrency(mut self, n: usize) -> Self {
        self.file_window_concurrency = n.max(1);
        self
    }

    /// Resolve the head from the explicit flag or files on disk (no network).
    /// Used by the synchronous serve builder / admin reload path.
    pub fn resolve_variant_local(&self) -> ModelVariant {
        self.model_variant
            .or_else(|| model::ModelVariant::detect_in_dir(std::path::Path::new(&self.model_dir)))
            .unwrap_or_default()
    }

    /// Download side assets (punctuation / VAD) when the corresponding pass is
    /// enabled. Graceful: failures are logged; loaders then fall back.
    pub async fn ensure_side_assets(&self, resolved: ModelVariant) {
        maybe_download_punct_model(self.punctuation, &self.punct_model_dir, resolved).await;
        maybe_download_vad_model(self.vad, &self.vad_model_dir).await;
    }

    /// Synchronous engine build from on-disk state.
    ///
    /// Used by serve first-boot (after async asset ensure) and
    /// `POST /v1/admin/reload`. Detects the variant without network; optionally
    /// quantizes when `self.quantize` is set.
    pub fn build_engine(&self) -> anyhow::Result<inference::Engine> {
        let resolved = self.resolve_variant_local();
        if self.quantize {
            ensure_int8_encoder(resolved, &self.model_dir, self.skip_quantize)?;
        }
        self.finish_build(resolved)
    }

    /// Offline path: ensure the model (may download), pull side assets, then
    /// build without quantizing.
    pub async fn load_offline_engine(&self) -> anyhow::Result<inference::Engine> {
        let resolved = model::ensure_model_variant(self.model_variant, &self.model_dir).await?;
        self.ensure_side_assets(resolved).await;
        self.finish_build(resolved)
    }

    /// Attach post-processors and load ONNX sessions for a known resolved head.
    fn finish_build(&self, resolved: ModelVariant) -> anyhow::Result<inference::Engine> {
        let punctuator = maybe_load_punctuator(self.punctuation, &self.punct_model_dir, resolved);
        let hotwords = resolve_hotwords(self.hotwords_file.as_deref(), self.hotwords_default);
        let total_slots = self.pool_size.saturating_add(self.batch_pool_size);
        let resolved_intra_threads = resolve_encoder_intra_threads(
            self.encoder_intra_threads,
            total_slots,
            std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(1),
        );
        let mut engine = inference::Engine::load_with_pools_threads_variant_cache(
            &self.model_dir,
            Some(resolved),
            self.pool_size,
            self.pool_min_size,
            self.batch_pool_size,
            resolved_intra_threads,
            self.optimized_cache_dir
                .as_deref()
                .map(std::path::PathBuf::from),
        )?
        .with_punctuator(punctuator)
        .with_itn(resolve_itn(self.itn, resolved))
        .with_vad(
            maybe_load_vad(self.vad, &self.vad_model_dir),
            build_vad_config(self.vad_threshold, self.vad_min_silence_ms),
        );
        if let Some(ref token) = self.endpoint_mode {
            let mode = inference::EndpointMode::parse_token(token)
                .unwrap_or(inference::EndpointMode::Auto);
            engine = engine.with_endpoint_mode(mode);
        }
        if let Some(secs) = self.stream_max_window_secs {
            engine = engine.with_stream_max_window_secs(secs);
        }
        engine = engine.with_stream_stable_prefix(self.stream_stable_prefix);
        engine = engine.with_file_window_concurrency(self.file_window_concurrency);
        if let Some(pairs) = hotwords {
            engine = engine.with_hotwords(
                &pairs,
                self.hotwords_boost.unwrap_or(DEFAULT_HOTWORDS_BOOST),
            );
        }
        #[cfg(feature = "ane")]
        log_ane_backend(resolved);
        log_rss();
        Ok(engine)
    }
}

#[cfg(test)]
mod tests;
