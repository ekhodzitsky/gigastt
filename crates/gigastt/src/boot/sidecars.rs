//! Optional punctuation, VAD, hotword, and INT8 sidecar loaders.

use anyhow::Context;
use gigastt_core::model::{self, ModelVariant};

use super::{PunctuationMode, resolve_punctuation};

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
/// Rust quantization pipeline from a local FP32 encoder when missing.
///
/// Runtime product path is **INT8-only** (`gigastt download`); this helper is
/// for packaging (`gigastt quantize`) and tests that still exercise on-device
/// quantization. `skip` no longer allows FP32 inference — missing INT8 is an
/// error either way when `skip` is true.
pub fn ensure_int8_encoder(
    variant: ModelVariant,
    model_dir: &str,
    skip: bool,
) -> anyhow::Result<()> {
    let dir = std::path::Path::new(model_dir);
    model::reject_versioned_pack(dir)?;
    let int8_path = dir.join(variant.encoder_int8_file());
    if int8_path.exists() {
        return Ok(());
    }
    if skip {
        anyhow::bail!(
            "INT8 encoder not found at {} — gigastt runs INT8 only. \
             Run `gigastt download` (lean INT8 bundle). FP32 encoders are not supported.",
            int8_path.display()
        );
    }
    #[cfg(not(feature = "quantize"))]
    anyhow::bail!(
        "Cannot quantize: this build does not include the quantize feature. \
         Run `gigastt download` for INT8 or enable the quantize feature."
    );
    #[cfg(feature = "quantize")]
    {
        let input = dir.join(variant.encoder_file());
        if !input.exists() {
            anyhow::bail!(
                "Cannot quantize: FP32 encoder not found at {} \
                 (packaging source). For runtime, use `gigastt download` for INT8.",
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
}
