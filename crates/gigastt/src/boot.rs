//! Engine boot recipe shared by `serve` and the offline CLI paths
//! (`transcribe`, `transcribe-batch`, `watch`).
//!
//! One place builds a fully-configured [`Engine`](gigastt_core::inference::Engine)
//! from CLI-shaped options (model dir, pool sizes, punctuation, ITN, VAD,
//! hotwords, threads). Post-processor chains
//! (`.with_punctuator().with_itn().with_vad().with_hotwords()`) live here so
//! serve first-boot, admin reload, and offline commands stay byte-identical.

use anyhow::Context;
use gigastt_core::inference;
use gigastt_core::model::{self, ModelVariant};

// ---------------------------------------------------------------------------
// CLI mode enums (shared by clap and the recipe)
// ---------------------------------------------------------------------------

/// Whether to run the optional punctuation / casing restoration pass.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PunctuationMode {
    /// Always attempt to load + apply the punct model.
    On,
    /// Never apply punctuation (pass-through bare output).
    Off,
    /// Decide from the active model variant: on for `rnnt` (bare output),
    /// off for `e2e_rnnt` (punctuation already baked into the head).
    Auto,
}

impl std::str::FromStr for PunctuationMode {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.trim().to_ascii_lowercase().as_str() {
            "on" | "true" | "1" | "yes" => Ok(PunctuationMode::On),
            "off" | "false" | "0" | "no" => Ok(PunctuationMode::Off),
            "auto" => Ok(PunctuationMode::Auto),
            other => Err(format!(
                "unknown punctuation mode '{other}' (expected 'on', 'off', or 'auto')"
            )),
        }
    }
}

/// clap value parser for `--punctuation`.
pub fn parse_punctuation_mode(s: &str) -> Result<PunctuationMode, String> {
    s.parse()
}

/// Whether to run the optional inverse text normalization pass
/// (Russian number-words → digits). Mirrors [`PunctuationMode`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ItnMode {
    /// Always apply ITN.
    On,
    /// Never apply ITN (pass-through number-words).
    Off,
    /// Decide from the active model variant: on for `rnnt` (spells numbers as
    /// words), off for `e2e_rnnt` (ITN already baked into the head).
    Auto,
}

impl std::str::FromStr for ItnMode {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.trim().to_ascii_lowercase().as_str() {
            "on" | "true" | "1" | "yes" => Ok(ItnMode::On),
            "off" | "false" | "0" | "no" => Ok(ItnMode::Off),
            "auto" => Ok(ItnMode::Auto),
            other => Err(format!(
                "unknown ITN mode '{other}' (expected 'on', 'off', or 'auto')"
            )),
        }
    }
}

/// clap value parser for `--itn`.
pub fn parse_itn_mode(s: &str) -> Result<ItnMode, String> {
    s.parse()
}

/// Resolve `--itn` against the active model variant: `auto` enables ITN only
/// for the bare `rnnt` head (the `e2e_rnnt` head already digitizes numbers).
pub fn resolve_itn(mode: ItnMode, variant: ModelVariant) -> bool {
    match mode {
        ItnMode::On => true,
        ItnMode::Off => false,
        ItnMode::Auto => variant == ModelVariant::Rnnt,
    }
}

/// Resolve `--punctuation` against the active model variant: `auto` enables the
/// pass only for the bare `rnnt` head (`e2e_rnnt` already punctuates).
pub fn resolve_punctuation(mode: PunctuationMode, variant: ModelVariant) -> bool {
    match mode {
        PunctuationMode::On => true,
        PunctuationMode::Off => false,
        // e2e_rnnt already emits punctuation/casing, so only the bare rnnt head
        // benefits from the restoration pass.
        PunctuationMode::Auto => variant == ModelVariant::Rnnt,
    }
}

// ---------------------------------------------------------------------------
// Thread budgeting
// ---------------------------------------------------------------------------

/// Resolve the encoder intra-op thread count when the operator left the flag /
/// env unset. `requested == Some(v)` (an explicit flag/env value, including `1`)
/// is honoured verbatim and only passes through the engine's oversubscription
/// clamp downstream. `None` (unset) spreads the logical CPUs across the
/// concurrently-running pool triplets: `max(1, logical_cpus / total_pool_slots)`,
/// so a default install uses every core instead of one. `total_pool_slots` is the
/// effective number of triplets that can run at once (serve: `pool_size +
/// batch_pool_size`; offline transcribe: `1`).
///
/// Pure and total so the budgeting math is unit-tested without touching ORT or
/// the real CPU count.
pub fn resolve_encoder_intra_threads(
    requested: Option<usize>,
    total_pool_slots: usize,
    logical_cpus: usize,
) -> usize {
    match requested {
        Some(explicit) => explicit,
        None => {
            let slots = total_pool_slots.max(1);
            let cpus = logical_cpus.max(1);
            (cpus / slots).max(1)
        }
    }
}

// ---------------------------------------------------------------------------
// Punctuation / VAD / hotwords loaders
// ---------------------------------------------------------------------------

/// Load the punctuation restorer when the pass resolves to ENABLED.
///
/// Graceful fallback: when the punct model dir / files are absent or the model
/// fails to load, a warning is logged once and `None` is returned so
/// transcription proceeds with bare text — the punct pass is strictly optional
/// and never blocks recognition.
pub fn maybe_load_punctuator(
    mode: PunctuationMode,
    punct_model_dir: &str,
    variant: ModelVariant,
) -> Option<gigastt_core::punctuation::Punctuator> {
    if !resolve_punctuation(mode, variant) {
        return None;
    }
    let factory = gigastt_core::cpu_factory();
    match gigastt_core::punctuation::Punctuator::load_with_factory(
        std::path::Path::new(punct_model_dir),
        &*factory,
    ) {
        Ok(p) => {
            tracing::info!("Punctuation restoration enabled (model dir: {punct_model_dir})");
            Some(p)
        }
        Err(e) => {
            tracing::warn!(
                "Punctuation model unavailable at {punct_model_dir} ({e:#}); \
                 continuing without punctuation restoration"
            );
            None
        }
    }
}

/// When the punctuation pass resolves to ENABLED and the punct model files are
/// absent in `punct_model_dir`, auto-download them from the
/// `ekhodzitsky/rupunct-small-onnx` HuggingFace repo so the pass works out of
/// the box.
///
/// Graceful: a download failure is logged as a warning and swallowed — the
/// subsequent [`maybe_load_punctuator`] call then falls back to bare text. The
/// punct pass never blocks transcription.
pub async fn maybe_download_punct_model(
    mode: PunctuationMode,
    punct_model_dir: &str,
    variant: ModelVariant,
) {
    if !resolve_punctuation(mode, variant) {
        return;
    }
    if let Err(e) = model::ensure_punct_model(punct_model_dir).await {
        tracing::warn!(
            "Punctuation model download failed for {punct_model_dir} ({e:#}); \
             continuing without punctuation restoration"
        );
    }
}

/// Build a [`gigastt_core::vad::VadConfig`] from CLI overrides, falling back to
/// the library defaults for any option left unset.
pub fn build_vad_config(
    threshold: Option<f32>,
    min_silence_ms: Option<u32>,
) -> gigastt_core::vad::VadConfig {
    let mut cfg = gigastt_core::vad::VadConfig::default();
    if let Some(t) = threshold {
        cfg.threshold = t.clamp(0.0, 1.0);
    }
    if let Some(ms) = min_silence_ms {
        cfg.min_silence_ms = ms;
    }
    cfg
}

/// Load the Silero VAD when `--vad` is set. Graceful: a missing or broken model
/// logs a warning and returns `None`, so transcription proceeds without VAD
/// (silence is not skipped; endpointing falls back to the decoder heuristic).
pub fn maybe_load_vad(enabled: bool, vad_model_dir: &str) -> Option<gigastt_core::vad::SileroVad> {
    if !enabled {
        return None;
    }
    let path = std::path::Path::new(vad_model_dir).join(gigastt_core::vad::VAD_MODEL_FILE);
    let factory = gigastt_core::cpu_factory();
    match gigastt_core::vad::SileroVad::load_with_factory(&path, &*factory) {
        Ok(v) => {
            tracing::info!("VAD enabled (model dir: {vad_model_dir})");
            Some(v)
        }
        Err(e) => {
            tracing::warn!(
                "VAD model unavailable at {vad_model_dir} ({e:#}); continuing without VAD"
            );
            None
        }
    }
}

/// When `--vad` is set and the Silero model is absent, auto-download it.
/// Graceful: a download failure is logged and swallowed — [`maybe_load_vad`]
/// then falls back to no VAD. VAD never blocks transcription.
pub async fn maybe_download_vad_model(enabled: bool, vad_model_dir: &str) {
    if !enabled {
        return;
    }
    if let Err(e) = model::ensure_vad_model(vad_model_dir).await {
        tracing::warn!(
            "VAD model download failed for {vad_model_dir} ({e:#}); continuing without VAD"
        );
    }
}

/// Default additive logit boost for hotword continuation tokens when
/// `--hotwords-boost` is unset.
pub const DEFAULT_HOTWORDS_BOOST: f32 = 5.0;

/// Parse a hotwords file: one phrase per line, optional `\t<weight>` suffix.
/// Blank lines and `#`-prefixed comment lines are skipped. A malformed weight
/// falls back to `1.0` (the phrase is still kept). Returns the `(phrase, weight)`
/// pairs, or an error only when the file can't be read.
pub fn parse_hotwords_file(path: &str) -> anyhow::Result<Vec<(String, f32)>> {
    let content = std::fs::read_to_string(path)
        .with_context(|| format!("failed to read hotwords file: {path}"))?;
    let mut pairs = Vec::new();
    for line in content.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let (phrase, weight) = match line.split_once('\t') {
            Some((p, w)) => (p.trim(), w.trim().parse::<f32>().unwrap_or(1.0)),
            None => (line, 1.0),
        };
        if !phrase.is_empty() {
            pairs.push((phrase.to_string(), weight));
        }
    }
    Ok(pairs)
}

/// Resolve the hotword pack from CLI options: phrases from `--hotwords-file`
/// (if any) plus the built-in lexicon when `--hotwords-default` is set. Returns
/// `None` when neither source yields any phrase (biasing stays off). A file read
/// error is logged and treated as "no file phrases" so biasing never blocks
/// transcription.
pub fn resolve_hotwords(
    hotwords_file: Option<&str>,
    hotwords_default: bool,
) -> Option<Vec<(String, f32)>> {
    let mut pairs = Vec::new();
    if let Some(path) = hotwords_file {
        match parse_hotwords_file(path) {
            Ok(p) => pairs.extend(p),
            Err(e) => tracing::warn!("{e:#}; continuing without file hotwords"),
        }
    }
    if hotwords_default {
        pairs.extend(gigastt_core::lexicon::default_hotword_pairs());
    }
    if pairs.is_empty() { None } else { Some(pairs) }
}

// ---------------------------------------------------------------------------
// INT8 + logging
// ---------------------------------------------------------------------------

/// Ensure the INT8 encoder exists for `variant`, producing it via the native
/// Rust quantization pipeline if missing. Honoured by `serve` and `download`.
/// First-time quantization takes ~2 minutes on the FP32 encoder.
pub fn ensure_int8_encoder(
    variant: ModelVariant,
    model_dir: &str,
    skip: bool,
) -> anyhow::Result<()> {
    let dir = std::path::Path::new(model_dir);
    let int8_path = dir.join(variant.encoder_int8_file());
    if int8_path.exists() {
        return Ok(());
    }
    if skip {
        tracing::info!(
            "Skipping INT8 quantization (--skip-quantize). Engine will load the FP32 encoder."
        );
        return Ok(());
    }
    let input = dir.join(variant.encoder_file());
    if !input.exists() {
        anyhow::bail!(
            "Cannot quantize: FP32 encoder not found at {}",
            input.display()
        );
    }
    tracing::info!("Quantizing encoder to INT8 (~2 min, one-time)…");
    // Surface the ~2-minute pass as its own phase so a sidecar watching the
    // NDJSON stream does not read it as a hang.
    model::emit_progress_event(&model::ProgressEvent::Quantize {
        file: variant.encoder_file().to_string(),
    });
    gigastt_core::quantize::quantize_model(&input, &int8_path)?;
    tracing::info!("INT8 encoder saved to {}", int8_path.display());
    Ok(())
}

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
/// `e2e_rnnt` model transparently stays on the ort encoder. When engaged it
/// serves file-mode transcription by padding the mel window up to a fixed
/// bucket; streaming / short windows below the fill floor fall back to the
/// CPU/ort encoder (no ANE benefit, no crash).
#[cfg(feature = "ane")]
pub fn log_ane_backend(resolved: ModelVariant) {
    if resolved == ModelVariant::Rnnt {
        tracing::info!(
            "ANE encoder backend active (Core ML / Apple Neural Engine, macOS ARM64): \
             file-mode transcription pads up to fixed buckets; streaming / short windows \
             below the fill floor fall back to the CPU/ort encoder"
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
    /// When true, run [`ensure_int8_encoder`] before load (serve / download).
    /// Offline paths leave this false so they never quantize as a side effect.
    pub quantize: bool,
    /// Passed to [`ensure_int8_encoder`] when `quantize` is true.
    pub skip_quantize: bool,
    /// Optional endpoint-mode token (`auto` / `assistant` / …). `None` leaves
    /// the engine default (offline paths). Serve always sets this.
    pub endpoint_mode: Option<String>,
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
        }
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
        let mut engine = inference::Engine::load_with_pools_threads_variant(
            &self.model_dir,
            Some(resolved),
            self.pool_size,
            self.pool_min_size,
            self.batch_pool_size,
            resolved_intra_threads,
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
mod tests {
    use super::*;

    #[test]
    fn test_punctuation_mode_from_str() {
        use std::str::FromStr;
        assert_eq!(
            PunctuationMode::from_str("on").unwrap(),
            PunctuationMode::On
        );
        assert_eq!(
            PunctuationMode::from_str("OFF").unwrap(),
            PunctuationMode::Off
        );
        assert_eq!(
            PunctuationMode::from_str(" auto ").unwrap(),
            PunctuationMode::Auto
        );
        assert!(PunctuationMode::from_str("maybe").is_err());
    }

    #[test]
    fn test_itn_mode_from_str() {
        use std::str::FromStr;
        assert_eq!(ItnMode::from_str("on").unwrap(), ItnMode::On);
        assert_eq!(ItnMode::from_str("OFF").unwrap(), ItnMode::Off);
        assert_eq!(ItnMode::from_str(" auto ").unwrap(), ItnMode::Auto);
        assert!(ItnMode::from_str("maybe").is_err());
    }

    #[test]
    fn test_parse_punctuation_mode_value_parser() {
        assert_eq!(
            parse_punctuation_mode("auto").unwrap(),
            PunctuationMode::Auto
        );
        assert!(parse_punctuation_mode("garbage").is_err());
    }

    #[test]
    fn test_parse_itn_mode_value_parser() {
        assert_eq!(parse_itn_mode("off").unwrap(), ItnMode::Off);
        assert_eq!(parse_itn_mode("auto").unwrap(), ItnMode::Auto);
        assert!(parse_itn_mode("garbage").is_err());
    }

    #[test]
    fn test_resolve_itn_auto_per_variant() {
        // auto → on for the bare rnnt head, off for the already-ITN e2e head.
        assert!(resolve_itn(ItnMode::Auto, ModelVariant::Rnnt));
        assert!(!resolve_itn(ItnMode::Auto, ModelVariant::E2eRnnt));
        // on/off override the variant.
        assert!(resolve_itn(ItnMode::On, ModelVariant::E2eRnnt));
        assert!(!resolve_itn(ItnMode::Off, ModelVariant::Rnnt));
    }

    #[test]
    fn test_resolve_punctuation_per_variant() {
        // auto → on for bare rnnt, off for the already-punctuated e2e head.
        assert!(resolve_punctuation(
            PunctuationMode::Auto,
            ModelVariant::Rnnt
        ));
        assert!(!resolve_punctuation(
            PunctuationMode::Auto,
            ModelVariant::E2eRnnt
        ));
        // on/off override the variant.
        assert!(resolve_punctuation(
            PunctuationMode::On,
            ModelVariant::E2eRnnt
        ));
        assert!(!resolve_punctuation(
            PunctuationMode::Off,
            ModelVariant::Rnnt
        ));
    }

    #[test]
    fn test_resolve_encoder_intra_threads_defaults_by_pool() {
        // Unset → logical CPUs spread across the concurrently-running triplets.
        assert_eq!(resolve_encoder_intra_threads(None, 2, 10), 5);
        assert_eq!(resolve_encoder_intra_threads(None, 1, 10), 10);
        // Never drop below one thread, even on a single-core box or a pool that
        // is wider than the CPU count.
        assert_eq!(resolve_encoder_intra_threads(None, 1, 1), 1);
        assert_eq!(resolve_encoder_intra_threads(None, 8, 4), 1);
        // A zero slot count (defensive) still yields at least one thread.
        assert_eq!(resolve_encoder_intra_threads(None, 0, 10), 10);
    }

    #[test]
    fn test_resolve_encoder_intra_threads_explicit_passthrough() {
        // An explicit value (including 1) is honoured verbatim; the engine's own
        // clamp still applies downstream.
        assert_eq!(resolve_encoder_intra_threads(Some(1), 2, 10), 1);
        assert_eq!(resolve_encoder_intra_threads(Some(4), 2, 10), 4);
        assert_eq!(resolve_encoder_intra_threads(Some(16), 1, 4), 16);
    }

    #[test]
    fn test_maybe_load_punctuator_off_skips_load() {
        // `off` must never touch the filesystem / model dir.
        assert!(
            maybe_load_punctuator(PunctuationMode::Off, "/nonexistent", ModelVariant::Rnnt)
                .is_none()
        );
    }

    #[test]
    fn test_maybe_load_punctuator_auto_e2e_skips_load() {
        // `auto` + e2e_rnnt → punctuation disabled (head already punctuates),
        // so no load is attempted even if the dir is missing.
        assert!(
            maybe_load_punctuator(PunctuationMode::Auto, "/nonexistent", ModelVariant::E2eRnnt)
                .is_none()
        );
    }

    #[test]
    fn test_maybe_load_punctuator_missing_model_falls_back_to_none() {
        // `on` + missing model dir → graceful fallback to None (warn, no panic).
        let tmp = tempfile::tempdir().unwrap();
        let missing = tmp.path().join("absent");
        assert!(
            maybe_load_punctuator(
                PunctuationMode::On,
                missing.to_str().unwrap(),
                ModelVariant::Rnnt
            )
            .is_none()
        );
    }

    #[test]
    fn test_parse_hotwords_file_lines_and_weights() {
        let tmp = tempfile::NamedTempFile::new().unwrap();
        std::fs::write(
            tmp.path(),
            b"# comment\n\nsynergy\nyoutube\t2.5\n  spaced  \nbadweight\tnope\n",
        )
        .unwrap();
        let pairs = parse_hotwords_file(tmp.path().to_str().unwrap()).unwrap();
        assert_eq!(
            pairs,
            vec![
                ("synergy".to_string(), 1.0),
                ("youtube".to_string(), 2.5),
                ("spaced".to_string(), 1.0),
                ("badweight".to_string(), 1.0), // malformed weight → 1.0, phrase kept
            ]
        );
    }

    #[test]
    fn test_resolve_hotwords_none_when_unset() {
        assert!(resolve_hotwords(None, false).is_none());
    }

    #[test]
    fn test_resolve_hotwords_default_pack_only() {
        let pairs = resolve_hotwords(None, true).expect("default pack present");
        assert_eq!(pairs.len(), gigastt_core::lexicon::DEFAULT_HOTWORDS.len());
    }

    #[test]
    fn test_resolve_hotwords_file_plus_default() {
        let tmp = tempfile::NamedTempFile::new().unwrap();
        std::fs::write(tmp.path(), "мойбренд\n").unwrap();
        let pairs = resolve_hotwords(tmp.path().to_str().unwrap().into(), true).unwrap();
        assert_eq!(
            pairs.len(),
            1 + gigastt_core::lexicon::DEFAULT_HOTWORDS.len()
        );
        assert_eq!(pairs[0].0, "мойбренд");
    }

    #[test]
    fn test_resolve_hotwords_missing_file_is_graceful() {
        // Missing file → warning + treated as no file phrases (None here).
        assert!(resolve_hotwords(Some("/nonexistent/hw.txt"), false).is_none());
    }

    #[test]
    fn test_ensure_int8_encoder_already_exists() {
        let tmp = tempfile::tempdir().unwrap();
        let int8_path = tmp.path().join("v3_rnnt_encoder_int8.onnx");
        std::fs::write(&int8_path, b"fake").unwrap();
        ensure_int8_encoder(ModelVariant::Rnnt, tmp.path().to_str().unwrap(), false).unwrap();
    }

    #[test]
    fn test_ensure_int8_encoder_skip_flag() {
        let tmp = tempfile::tempdir().unwrap();
        ensure_int8_encoder(ModelVariant::Rnnt, tmp.path().to_str().unwrap(), true).unwrap();
    }

    #[test]
    fn test_ensure_int8_encoder_missing_input() {
        let tmp = tempfile::tempdir().unwrap();
        let err = ensure_int8_encoder(ModelVariant::Rnnt, tmp.path().to_str().unwrap(), false)
            .unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("Cannot quantize"), "unexpected error: {msg}");
    }

    #[test]
    fn test_ensure_int8_encoder_e2e_targets_e2e_encoder_name() {
        // With the e2e variant, the FP32 input it looks for is the e2e encoder;
        // an rnnt encoder in the dir must NOT satisfy it.
        let tmp = tempfile::tempdir().unwrap();
        std::fs::write(tmp.path().join("v3_rnnt_encoder.onnx"), b"rnnt").unwrap();
        let err = ensure_int8_encoder(ModelVariant::E2eRnnt, tmp.path().to_str().unwrap(), false)
            .unwrap_err();
        assert!(format!("{err}").contains("Cannot quantize"));
    }

    #[test]
    fn test_log_rss_does_not_panic() {
        // Simply exercise the function on the current platform.
        // On Linux it reads /proc/self/status; on macOS it spawns ps.
        log_rss();
    }

    #[test]
    fn test_build_vad_config_defaults_when_unset() {
        // Both overrides None → library defaults pass through untouched.
        let cfg = build_vad_config(None, None);
        let default = gigastt_core::vad::VadConfig::default();
        assert_eq!(cfg.threshold, default.threshold);
        assert_eq!(cfg.min_silence_ms, default.min_silence_ms);
        assert_eq!(cfg.min_speech_ms, default.min_speech_ms);
        assert_eq!(cfg.speech_pad_ms, default.speech_pad_ms);
    }

    #[test]
    fn test_build_vad_config_applies_overrides() {
        let cfg = build_vad_config(Some(0.75), Some(1200));
        assert_eq!(cfg.threshold, 0.75);
        assert_eq!(cfg.min_silence_ms, 1200);
    }

    #[test]
    fn test_build_vad_config_clamps_threshold() {
        // Out-of-range thresholds clamp into [0, 1].
        assert_eq!(build_vad_config(Some(5.0), None).threshold, 1.0);
        assert_eq!(build_vad_config(Some(-3.0), None).threshold, 0.0);
    }

    #[test]
    fn test_maybe_load_vad_disabled_skips_load() {
        // Disabled → never touches the filesystem, returns None.
        assert!(maybe_load_vad(false, "/nonexistent").is_none());
    }

    #[test]
    fn test_maybe_load_vad_missing_model_falls_back_to_none() {
        // Enabled but model absent → graceful warn + None (no panic).
        let tmp = tempfile::tempdir().unwrap();
        let dir = tmp.path().join("absent");
        assert!(maybe_load_vad(true, dir.to_str().unwrap()).is_none());
    }

    #[test]
    fn test_engine_recipe_offline_defaults() {
        let r = EngineRecipe::offline(
            "/models".into(),
            None,
            PunctuationMode::Auto,
            "/punct".into(),
            ItnMode::Auto,
            None,
            false,
            None,
            false,
            None,
            None,
            "/vad".into(),
            None,
            2,
        );
        assert_eq!(r.pool_size, 2);
        assert_eq!(r.pool_min_size, 1);
        assert_eq!(r.batch_pool_size, 0);
        assert!(!r.quantize);
        assert!(r.endpoint_mode.is_none());
    }
}
