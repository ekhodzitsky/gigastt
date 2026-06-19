use anyhow::Context;
use clap::{Parser, Subcommand};
use gigastt::server;
use gigastt::server::{OriginPolicy, RuntimeLimits, ServerConfig};
use gigastt_core::export::{ExportFormat, RenderOpts};
use gigastt_core::model::ModelVariant;
use gigastt_core::{inference, model};
use std::net::IpAddr;
use std::str::FromStr;
use tracing_subscriber::EnvFilter;

#[derive(Parser)]
#[command(
    name = "gigastt",
    version,
    about = "Local STT server powered by GigaAM v3"
)]
struct Cli {
    /// Log level [default: info]
    #[arg(long, global = true, default_value = "info")]
    log_level: String,

    #[command(subcommand)]
    command: Commands,
}

// `Serve` carries many optional CLI flags, so it is much larger than the other
// variants. The enum is parsed once at startup and never stored in bulk, so
// boxing the fields would only hurt readability.
#[allow(clippy::large_enum_variant)]
#[derive(Subcommand)]
enum Commands {
    /// Start WebSocket STT server (auto-downloads model if missing)
    Serve {
        /// Port to listen on
        #[arg(short, long, default_value_t = 9876)]
        port: u16,

        /// Bind address. Loopback by default; non-loopback requires `--bind-all`.
        #[arg(long, default_value = "127.0.0.1")]
        host: String,

        /// Model directory
        #[arg(long, default_value_t = model::default_model_dir())]
        model_dir: String,

        /// Recognition head to use. Omit to auto-detect from the model
        /// directory: if a model is already installed its variant is used as-is
        /// (no download). Only required when the directory is empty or you want
        /// to switch variants. `rnnt` (lower WER, bare lowercase) or
        /// `e2e_rnnt` (punctuation / casing / ITN). Env: GIGASTT_MODEL_VARIANT.
        #[arg(
            long,
            env = "GIGASTT_MODEL_VARIANT",
            value_parser = parse_model_variant
        )]
        model_variant: Option<ModelVariant>,

        /// Punctuation + capitalization restoration: `on`, `off`, or `auto`.
        /// `auto` (default) enables it for the `rnnt` head (bare output) and
        /// disables it for `e2e_rnnt` (already punctuated). Requires the punct
        /// model in `--punct-model-dir`; missing model → bare text + a warning.
        /// Env: GIGASTT_PUNCTUATION.
        #[arg(
            long,
            env = "GIGASTT_PUNCTUATION",
            default_value = "auto",
            value_parser = parse_punctuation_mode
        )]
        punctuation: PunctuationMode,

        /// Directory holding the optional punctuation model
        /// (`rupunct_small_int8.onnx`, `tokenizer.json`, `config.json`).
        /// Defaults to `~/.gigastt/models/punct/`. Auto-downloaded from
        /// `ekhodzitsky/rupunct-small-onnx` when enabled and absent.
        /// Env: GIGASTT_PUNCT_MODEL_DIR.
        #[arg(
            long,
            env = "GIGASTT_PUNCT_MODEL_DIR",
            default_value_t = model::default_punct_model_dir()
        )]
        punct_model_dir: String,

        /// Inverse text normalization (Russian number-words → digits):
        /// `on`, `off`, or `auto`. `auto` (default) enables it for the `rnnt`
        /// head (spells numbers as words) and disables it for `e2e_rnnt`
        /// (ITN already baked in). Runs before punctuation. Env: GIGASTT_ITN.
        #[arg(
            long,
            env = "GIGASTT_ITN",
            default_value = "auto",
            value_parser = parse_itn_mode
        )]
        itn: ItnMode,

        /// Number of concurrent inference sessions. Each session deserializes
        /// its own encoder copy (~0.4 GB resident for the INT8 encoder), so the
        /// default is 2 to bound the idle footprint; raise it for higher
        /// concurrency when RAM allows. The server auto-caps this by available
        /// RAM at load and logs a warning if it has to clamp.
        #[arg(long, default_value_t = 2)]
        pool_size: usize,

        /// Minimum session triplets that must load for the server to boot. When
        /// `1 <= min < pool_size` and some triplets fail (e.g. low memory), the
        /// server starts on a degraded pool with a warning instead of failing.
        /// Clamped to `1..=pool_size` [default: 1].
        #[arg(long, env = "GIGASTT_POOL_MIN_SIZE", default_value_t = 1)]
        pool_min_size: usize,

        /// Triplets reserved for batch REST file transcription, split off from
        /// `--pool-size` so a long file job can't starve WebSocket / SSE
        /// streaming. `0` disables the split (REST shares the interactive pool);
        /// clamped to leave at least one interactive triplet [default: 0].
        #[arg(long, env = "GIGASTT_BATCH_POOL_SIZE", default_value_t = 0)]
        batch_pool_size: usize,

        /// Explicitly acknowledge binding to a non-loopback address.
        /// Can also be enabled via `GIGASTT_ALLOW_BIND_ANY=1`.
        /// Without this flag the server refuses to listen on anything other than
        /// 127.0.0.1 / ::1 / localhost to prevent accidental public exposure.
        #[arg(long, default_value_t = false)]
        bind_all: bool,

        /// Additional Origin allowed to call the REST / WebSocket API (repeatable).
        /// Loopback origins (localhost, 127.0.0.1, ::1) are always allowed.
        /// Match is exact and case-insensitive, e.g. `https://app.example.com`.
        #[arg(long = "allow-origin", value_name = "URL")]
        allow_origin: Vec<String>,

        /// Echo `Access-Control-Allow-Origin: *` and accept any cross-origin
        /// caller. Disabled by default — every non-loopback Origin must be
        /// listed explicitly via `--allow-origin` unless this flag is set.
        #[arg(long, default_value_t = false)]
        cors_allow_any: bool,

        /// WebSocket idle timeout in seconds [default: 300].
        /// Server closes the connection when no frame arrives within this window.
        #[arg(long, env = "GIGASTT_IDLE_TIMEOUT_SECS")]
        idle_timeout_secs: Option<u64>,

        /// Maximum WebSocket frame / message size in bytes [default: 524288].
        #[arg(long, env = "GIGASTT_WS_FRAME_MAX_BYTES")]
        ws_frame_max_bytes: Option<usize>,

        /// Maximum REST request body size in bytes [default: 52428800].
        #[arg(long, env = "GIGASTT_BODY_LIMIT_BYTES")]
        body_limit_bytes: Option<usize>,

        /// Per-IP rate limit — requests per minute. 0 = off [default: 0].
        #[arg(long, env = "GIGASTT_RATE_LIMIT_PER_MINUTE")]
        rate_limit_per_minute: Option<u32>,

        /// Rate-limit burst size [default: 10].
        #[arg(long, env = "GIGASTT_RATE_LIMIT_BURST")]
        rate_limit_burst: Option<u32>,

        /// Expose Prometheus metrics. Off by default — keeps the server quiet
        /// for single-user installs. When on, `/metrics` is served on a
        /// separate loopback listener (see `--metrics-listen`), never on the
        /// primary port, so it is not gated by the CORS allowlist or limiter.
        #[arg(long, env = "GIGASTT_METRICS", default_value_t = false)]
        metrics: bool,

        /// Bind address for the separate Prometheus `/metrics` listener
        /// [default: 127.0.0.1:9090]. Loopback by default; expose it
        /// deliberately to a scraper. Only used when `--metrics` is set.
        #[arg(long, env = "GIGASTT_METRICS_LISTEN")]
        metrics_listen: Option<std::net::SocketAddr>,

        /// Maximum wall-clock duration of a single WebSocket session in seconds.
        /// 0 disables the cap (not recommended) [default: 3600].
        #[arg(long, env = "GIGASTT_MAX_SESSION_SECS")]
        max_session_secs: Option<u64>,

        /// Grace window in seconds after shutdown during which in-flight
        /// sessions may emit Final frames. 0 is clamped to 1 [default: 10].
        #[arg(long, env = "GIGASTT_SHUTDOWN_DRAIN_SECS")]
        shutdown_drain_secs: Option<u64>,

        /// Pool checkout timeout in seconds. Handlers wait this long for a
        /// free session triplet before returning 503 [default: 30].
        #[arg(long, env = "GIGASTT_POOL_CHECKOUT_TIMEOUT_SECS")]
        pool_checkout_timeout_secs: Option<u64>,

        /// Per-request inference timeout in seconds. A run exceeding this
        /// returns `inference_timeout`; `0` disables [default: 600].
        #[arg(long, env = "GIGASTT_INFERENCE_TIMEOUT_SECS")]
        inference_timeout_secs: Option<u64>,

        /// Skip the automatic INT8 quantization step after download.
        /// Default behaviour is to quantize the encoder (~2 min, one-time)
        /// so the pool loads the 210 MB INT8 encoder instead of the 844 MB
        /// FP32. Opt out when you need the FP32 encoder for debugging.
        #[arg(long, env = "GIGASTT_SKIP_QUANTIZE", default_value_t = false)]
        skip_quantize: bool,

        /// Trust `X-Forwarded-For` and `X-Real-IP` headers for rate-limit IP
        /// extraction. When enabled, the direct peer must be loopback or an
        /// RFC1918 private address; otherwise headers are ignored.
        #[arg(long, env = "GIGASTT_TRUST_PROXY", default_value_t = false)]
        trust_proxy: bool,

        /// Path to TOML config file for runtime limits (reloaded on SIGHUP)
        #[arg(long)]
        config: Option<String>,
    },

    /// Download model without starting server
    Download {
        /// Model directory
        #[arg(long, default_value_t = model::default_model_dir())]
        model_dir: String,

        /// Recognition head to download: `rnnt` (default — lower WER, bare
        /// lowercase) or `e2e_rnnt` (punctuation / casing / ITN).
        #[arg(
            long,
            env = "GIGASTT_MODEL_VARIANT",
            default_value = "rnnt",
            value_parser = parse_model_variant
        )]
        model_variant: ModelVariant,

        /// Skip downloading the speaker diarization model
        #[cfg(feature = "diarization")]
        #[arg(long, default_value_t = false)]
        skip_diarization: bool,

        /// Skip the automatic INT8 quantization step after download.
        /// Default behaviour is to quantize the encoder (~2 min, one-time)
        /// so subsequent `gigastt serve` calls load the 210 MB INT8 encoder.
        /// Opt out when you need the FP32 encoder for debugging.
        #[arg(long, env = "GIGASTT_SKIP_QUANTIZE", default_value_t = false)]
        skip_quantize: bool,
    },

    /// Quantize encoder model to INT8 (replaces scripts/quantize.py)
    Quantize {
        /// Model directory
        #[arg(long, default_value_t = model::default_model_dir())]
        model_dir: String,

        /// Force re-quantization even if INT8 model exists
        #[arg(long)]
        force: bool,
    },

    /// Transcribe an audio file (offline)
    Transcribe {
        /// Path to audio file (WAV, M4A, MP3, OGG, FLAC)
        file: String,

        /// Model directory
        #[arg(long, default_value_t = model::default_model_dir())]
        model_dir: String,

        /// Recognition head to use. Omit to auto-detect from the model
        /// directory (existing install used as-is; only downloads if empty).
        /// `rnnt` (lower WER, bare lowercase) or `e2e_rnnt` (punctuation /
        /// casing / ITN). Env: GIGASTT_MODEL_VARIANT.
        #[arg(
            long,
            env = "GIGASTT_MODEL_VARIANT",
            value_parser = parse_model_variant
        )]
        model_variant: Option<ModelVariant>,

        /// Punctuation + capitalization restoration: `on`, `off`, or `auto`.
        /// `auto` (default) enables it for `rnnt`, disables it for `e2e_rnnt`.
        /// Env: GIGASTT_PUNCTUATION.
        #[arg(
            long,
            env = "GIGASTT_PUNCTUATION",
            default_value = "auto",
            value_parser = parse_punctuation_mode
        )]
        punctuation: PunctuationMode,

        /// Directory holding the optional punctuation model. Defaults to
        /// `~/.gigastt/models/punct/`. Auto-downloaded from
        /// `ekhodzitsky/rupunct-small-onnx` when enabled and absent.
        /// Env: GIGASTT_PUNCT_MODEL_DIR.
        #[arg(
            long,
            env = "GIGASTT_PUNCT_MODEL_DIR",
            default_value_t = model::default_punct_model_dir()
        )]
        punct_model_dir: String,

        /// Inverse text normalization (Russian number-words → digits):
        /// `on`, `off`, or `auto`. `auto` (default) enables it for `rnnt`,
        /// disables it for `e2e_rnnt`. Runs before punctuation. Env: GIGASTT_ITN.
        #[arg(
            long,
            env = "GIGASTT_ITN",
            default_value = "auto",
            value_parser = parse_itn_mode
        )]
        itn: ItnMode,

        /// Export format: json, txt, srt, vtt, md [default: txt]
        #[arg(short, long, env = "GIGASTT_FORMAT", default_value = "txt")]
        format: String,

        /// Output file. When omitted, prints to stdout.
        #[arg(short, long, env = "GIGASTT_OUTPUT")]
        output: Option<String>,

        /// Maximum characters per subtitle/caption line (SRT/VTT) [default: 80]
        #[arg(long, env = "GIGASTT_MAX_CHARS_PER_LINE")]
        max_chars_per_line: Option<usize>,

        /// Maximum words per subtitle/caption line (SRT/VTT) [default: 14]
        #[arg(long, env = "GIGASTT_MAX_WORDS_PER_LINE")]
        max_words_per_line: Option<usize>,

        /// Include per-word timestamps in Markdown output
        #[arg(long, env = "GIGASTT_WORD_TIMESTAMPS", default_value_t = false)]
        word_timestamps: bool,
    },
}

#[allow(clippy::too_many_arguments)]
fn build_limits(
    config_path: Option<&str>,
    idle_timeout_secs: Option<u64>,
    ws_frame_max_bytes: Option<usize>,
    body_limit_bytes: Option<usize>,
    rate_limit_per_minute: Option<u32>,
    rate_limit_burst: Option<u32>,
    max_session_secs: Option<u64>,
    shutdown_drain_secs: Option<u64>,
    pool_checkout_timeout_secs: Option<u64>,
    inference_timeout_secs: Option<u64>,
) -> anyhow::Result<RuntimeLimits> {
    let mut limits = if let Some(path) = config_path {
        server::config::load_config_file(std::path::Path::new(path))?
    } else {
        RuntimeLimits::default()
    };
    if let Some(v) = idle_timeout_secs {
        limits.idle_timeout_secs = v;
    }
    if let Some(v) = ws_frame_max_bytes {
        limits.ws_frame_max_bytes = v;
    }
    if let Some(v) = body_limit_bytes {
        limits.body_limit_bytes = v;
    }
    if let Some(v) = rate_limit_per_minute {
        limits.rate_limit_per_minute = v;
    }
    if let Some(v) = rate_limit_burst {
        limits.rate_limit_burst = v;
    }
    if limits.rate_limit_per_minute > 0 && limits.rate_limit_burst == 0 {
        anyhow::bail!("--rate-limit-burst must be > 0 when --rate-limit-per-minute is enabled");
    }
    if let Some(v) = max_session_secs {
        limits.max_session_secs = v;
    }
    if let Some(v) = shutdown_drain_secs {
        limits.shutdown_drain_secs = v;
    }
    if let Some(v) = pool_checkout_timeout_secs {
        limits.pool_checkout_timeout_secs = v;
    }
    if let Some(v) = inference_timeout_secs {
        limits.inference_timeout_secs = v;
    }
    Ok(limits)
}

#[allow(clippy::too_many_arguments)]
fn build_server_config(
    port: u16,
    host: String,
    allow_origin: Vec<String>,
    cors_allow_any: bool,
    limits: RuntimeLimits,
    metrics: bool,
    metrics_listen: std::net::SocketAddr,
    trust_proxy: bool,
    config: Option<String>,
) -> ServerConfig {
    ServerConfig {
        port,
        host,
        origin_policy: OriginPolicy {
            allow_any: cors_allow_any,
            allowed_origins: allow_origin,
        },
        limits,
        metrics_enabled: metrics,
        metrics_listen,
        trust_proxy,
        config_path: config.map(std::path::PathBuf::from),
    }
}

fn log_rss() {
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

/// Guard non-loopback binds. Privacy-first default: the server will only
/// listen on 127.0.0.1 / ::1 / localhost unless the operator opts in via
/// `--bind-all` or `GIGASTT_ALLOW_BIND_ANY=1`. Mirrors the intent of Docker's
/// `--host 0.0.0.0` — explicit consent to expose a local STT service.
fn ensure_bind_allowed(host: &str, bind_all_flag: bool) -> anyhow::Result<()> {
    if is_loopback_host(host) {
        return Ok(());
    }
    let env_opt_in = std::env::var("GIGASTT_ALLOW_BIND_ANY")
        .map(|v| matches!(v.trim(), "1" | "true" | "TRUE" | "yes" | "YES"))
        .unwrap_or(false);
    if bind_all_flag || env_opt_in {
        tracing::warn!(
            host = %host,
            "binding to non-loopback address — anyone on the network can reach this server"
        );
        return Ok(());
    }
    anyhow::bail!(
        "refusing to bind to '{host}': non-loopback addresses require \
         `--bind-all` (or env GIGASTT_ALLOW_BIND_ANY=1) to prevent accidental \
         public exposure of local transcription"
    )
}

fn is_loopback_host(host: &str) -> bool {
    // Accept the common human forms first.
    let lowered = host.trim().to_ascii_lowercase();
    if lowered == "localhost" || lowered == "::1" {
        return true;
    }
    // Strip optional brackets around IPv6 literals.
    let stripped = lowered.trim_start_matches('[').trim_end_matches(']');
    if let Ok(ip) = stripped.parse::<IpAddr>() {
        return ip.is_loopback();
    }
    false
}

/// clap value parser for `--model-variant`. Accepts `rnnt` / `e2e_rnnt`
/// (case-insensitive); see [`ModelVariant::from_str`].
fn parse_model_variant(s: &str) -> Result<ModelVariant, String> {
    s.parse()
}

/// Whether to run the optional punctuation / casing restoration pass.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PunctuationMode {
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
fn parse_punctuation_mode(s: &str) -> Result<PunctuationMode, String> {
    s.parse()
}

/// Whether to run the optional inverse text normalization pass
/// (Russian number-words → digits). Mirrors [`PunctuationMode`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ItnMode {
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
fn parse_itn_mode(s: &str) -> Result<ItnMode, String> {
    s.parse()
}

/// Resolve `--itn` against the active model variant: `auto` enables ITN only
/// for the bare `rnnt` head (the `e2e_rnnt` head already digitizes numbers).
fn resolve_itn(mode: ItnMode, variant: ModelVariant) -> bool {
    match mode {
        ItnMode::On => true,
        ItnMode::Off => false,
        ItnMode::Auto => variant == ModelVariant::Rnnt,
    }
}

/// Resolve `--punctuation` against the active model variant and, when the pass
/// should run, load the punctuation restorer from `punct_model_dir`.
///
/// Graceful fallback: when the punct model dir / files are absent or the model
/// fails to load, a warning is logged once and `None` is returned so
/// transcription proceeds with bare text — the punct pass is strictly optional
/// and never blocks recognition.
/// Resolve `--punctuation` against the active model variant: `auto` enables the
/// pass only for the bare `rnnt` head (`e2e_rnnt` already punctuates).
fn resolve_punctuation(mode: PunctuationMode, variant: ModelVariant) -> bool {
    match mode {
        PunctuationMode::On => true,
        PunctuationMode::Off => false,
        // e2e_rnnt already emits punctuation/casing, so only the bare rnnt head
        // benefits from the restoration pass.
        PunctuationMode::Auto => variant == ModelVariant::Rnnt,
    }
}

fn maybe_load_punctuator(
    mode: PunctuationMode,
    punct_model_dir: &str,
    variant: ModelVariant,
) -> Option<gigastt_core::punctuation::Punctuator> {
    if !resolve_punctuation(mode, variant) {
        return None;
    }
    match gigastt_core::punctuation::Punctuator::load(std::path::Path::new(punct_model_dir)) {
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
async fn maybe_download_punct_model(
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

/// Ensure the INT8 encoder exists for `variant`, producing it via the native
/// Rust quantization pipeline if missing. Honoured by `serve` and `download`.
/// First-time quantization takes ~2 minutes on the FP32 encoder.
fn ensure_int8_encoder(variant: ModelVariant, model_dir: &str, skip: bool) -> anyhow::Result<()> {
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
    gigastt_core::quantize::quantize_model(&input, &int8_path)?;
    tracing::info!("INT8 encoder saved to {}", int8_path.display());
    Ok(())
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    let directive = format!("gigastt={}", cli.log_level);
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env().add_directive(directive.parse()?))
        .init();

    match cli.command {
        Commands::Serve {
            port,
            host,
            model_dir,
            model_variant,
            punctuation,
            punct_model_dir,
            itn,
            pool_size,
            pool_min_size,
            batch_pool_size,
            bind_all,
            allow_origin,
            cors_allow_any,
            idle_timeout_secs,
            ws_frame_max_bytes,
            body_limit_bytes,
            rate_limit_per_minute,
            rate_limit_burst,
            metrics,
            metrics_listen,
            max_session_secs,
            shutdown_drain_secs,
            pool_checkout_timeout_secs,
            inference_timeout_secs,
            skip_quantize,
            trust_proxy,
            config,
        } => {
            ensure_bind_allowed(&host, bind_all)?;
            let resolved = model::ensure_model_variant(model_variant, &model_dir).await?;
            ensure_int8_encoder(resolved, &model_dir, skip_quantize)?;
            maybe_download_punct_model(punctuation, &punct_model_dir, resolved).await;
            let punctuator = maybe_load_punctuator(punctuation, &punct_model_dir, resolved);
            let engine = inference::Engine::load_with_pools(
                &model_dir,
                pool_size,
                pool_min_size,
                batch_pool_size,
            )?
            .with_punctuator(punctuator)
            .with_itn(resolve_itn(itn, resolved));
            log_rss();
            let limits = build_limits(
                config.as_deref(),
                idle_timeout_secs,
                ws_frame_max_bytes,
                body_limit_bytes,
                rate_limit_per_minute,
                rate_limit_burst,
                max_session_secs,
                shutdown_drain_secs,
                pool_checkout_timeout_secs,
                inference_timeout_secs,
            )?;
            let metrics_listen =
                metrics_listen.unwrap_or_else(server::config::default_metrics_listen);
            // The metrics listener carries no CORS allowlist or rate limiter, so
            // a non-loopback bind requires the same explicit `--bind-all` opt-in
            // the primary port does — keeps the loopback-by-default invariant
            // symmetric instead of letting telemetry leak network-wide silently.
            if metrics {
                ensure_bind_allowed(&metrics_listen.ip().to_string(), bind_all)?;
            }
            let config = build_server_config(
                port,
                host,
                allow_origin,
                cors_allow_any,
                limits,
                metrics,
                metrics_listen,
                trust_proxy,
                config,
            );
            server::run_with_config(engine, config, None).await?;
        }
        Commands::Download {
            model_dir,
            model_variant,
            #[cfg(feature = "diarization")]
            skip_diarization,
            skip_quantize,
        } => {
            // `download` is an explicit action: None here maps to the default
            // (Rnnt) so a bare `gigastt download` fetches something useful.
            let requested = Some(model_variant);
            let resolved = model::ensure_model_variant(requested, &model_dir).await?;
            #[cfg(feature = "diarization")]
            {
                if !skip_diarization {
                    model::ensure_speaker_model(&model_dir).await?;
                }
            }
            ensure_int8_encoder(resolved, &model_dir, skip_quantize)?;
            tracing::info!("Model ready at {model_dir}");
        }
        Commands::Quantize { model_dir, force } => {
            // Quantize an existing model dir: detect the head already on disk
            // (default rnnt when the dir is empty and `ensure_model` must fetch).
            let dir = std::path::Path::new(&model_dir);
            let resolved = model::ensure_model_variant(None, &model_dir).await?;
            let input = dir.join(resolved.encoder_file());
            let output = dir.join(resolved.encoder_int8_file());
            if output.exists() && !force {
                tracing::info!("INT8 model already exists: {}", output.display());
                tracing::info!("Use --force to re-quantize.");
                return Ok(());
            }
            gigastt_core::quantize::quantize_model(&input, &output)?;
            tracing::info!("Quantized model saved to {}", output.display());
        }
        Commands::Transcribe {
            file,
            model_dir,
            model_variant,
            punctuation,
            punct_model_dir,
            itn,
            format,
            output,
            max_chars_per_line,
            max_words_per_line,
            word_timestamps,
        } => {
            let resolved = model::ensure_model_variant(model_variant, &model_dir).await?;
            maybe_download_punct_model(punctuation, &punct_model_dir, resolved).await;
            let punctuator = maybe_load_punctuator(punctuation, &punct_model_dir, resolved);
            let engine = inference::Engine::load_with_pool_size(&model_dir, 1)?
                .with_punctuator(punctuator)
                .with_itn(resolve_itn(itn, resolved));
            log_rss();
            let mut guard = engine.pool.checkout().await?;
            let result = engine.transcribe_file(&file, &mut guard);
            drop(guard);
            let result = result?;

            let format = ExportFormat::from_str(&format).map_err(|e| anyhow::anyhow!("{e}"))?;
            let opts = RenderOpts {
                max_chars_per_line: max_chars_per_line.unwrap_or(80),
                max_words_per_line: max_words_per_line.unwrap_or(14),
                include_word_timestamps: word_timestamps,
            };
            let rendered = format.render(&result, &opts);

            match output {
                Some(path) => {
                    std::fs::write(&path, rendered)
                        .with_context(|| format!("failed to write {path}"))?;
                    tracing::info!("Wrote {} export to {path}", format);
                }
                None => println!("{rendered}"),
            }
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    // Serialize tests that mutate process env vars to avoid races under
    // cargo test's default multi-threaded harness (used by tarpaulin).
    static ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    #[test]
    fn test_is_loopback_host_recognises_common_forms() {
        assert!(is_loopback_host("127.0.0.1"));
        assert!(is_loopback_host("localhost"));
        assert!(is_loopback_host("::1"));
        assert!(is_loopback_host("[::1]"));
        assert!(is_loopback_host("127.0.0.2")); // loopback /8
        assert!(!is_loopback_host("0.0.0.0"));
        assert!(!is_loopback_host("192.168.1.10"));
        assert!(!is_loopback_host("example.com"));
    }

    #[test]
    fn test_ensure_bind_allowed_loopback_ok() {
        ensure_bind_allowed("127.0.0.1", false).expect("loopback must be allowed");
        ensure_bind_allowed("localhost", false).expect("localhost must be allowed");
    }

    #[test]
    fn test_ensure_bind_allowed_non_loopback_requires_flag() {
        let _guard = ENV_LOCK.lock().unwrap();
        let previous = std::env::var("GIGASTT_ALLOW_BIND_ANY").ok();
        unsafe {
            std::env::remove_var("GIGASTT_ALLOW_BIND_ANY");
        }
        let result = ensure_bind_allowed("0.0.0.0", false);
        if let Some(v) = previous {
            unsafe {
                std::env::set_var("GIGASTT_ALLOW_BIND_ANY", v);
            }
        }
        assert!(
            result.is_err(),
            "0.0.0.0 without --bind-all must be rejected"
        );
    }

    #[test]
    fn test_ensure_bind_allowed_explicit_flag_ok() {
        ensure_bind_allowed("0.0.0.0", true).expect("explicit --bind-all must pass");
    }

    #[test]
    fn test_cli_serve_parsing() {
        let cli = Cli::parse_from(["gigastt", "serve", "--port", "1234", "--bind-all"]);
        match cli.command {
            Commands::Serve {
                port,
                bind_all,
                metrics,
                model_variant,
                ..
            } => {
                assert_eq!(port, 1234);
                assert!(bind_all);
                assert!(!metrics);
                // No --model-variant → None (auto-detect from disk).
                assert_eq!(model_variant, None);
            }
            _ => panic!("expected Serve"),
        }
    }

    #[test]
    fn test_cli_serve_model_variant_override() {
        let cli = Cli::parse_from(["gigastt", "serve", "--model-variant", "e2e_rnnt"]);
        match cli.command {
            Commands::Serve { model_variant, .. } => {
                assert_eq!(model_variant, Some(ModelVariant::E2eRnnt));
            }
            _ => panic!("expected Serve"),
        }
    }

    #[test]
    fn test_cli_serve_model_variant_explicit_rnnt() {
        let cli = Cli::parse_from(["gigastt", "serve", "--model-variant", "rnnt"]);
        match cli.command {
            Commands::Serve { model_variant, .. } => {
                assert_eq!(model_variant, Some(ModelVariant::Rnnt));
            }
            _ => panic!("expected Serve"),
        }
    }

    #[test]
    fn test_cli_download_parsing() {
        let cli = Cli::parse_from(["gigastt", "download", "--model-dir", "/tmp/models"]);
        match cli.command {
            Commands::Download {
                model_dir,
                model_variant,
                ..
            } => {
                assert_eq!(model_dir, "/tmp/models");
                assert_eq!(model_variant, ModelVariant::Rnnt);
            }
            _ => panic!("expected Download"),
        }
    }

    #[test]
    fn test_cli_download_model_variant_override() {
        let cli = Cli::parse_from(["gigastt", "download", "--model-variant", "e2e_rnnt"]);
        match cli.command {
            Commands::Download { model_variant, .. } => {
                assert_eq!(model_variant, ModelVariant::E2eRnnt);
            }
            _ => panic!("expected Download"),
        }
    }

    #[test]
    fn test_cli_quantize_parsing() {
        let cli = Cli::parse_from(["gigastt", "quantize", "--force"]);
        match cli.command {
            Commands::Quantize { force, .. } => {
                assert!(force);
            }
            _ => panic!("expected Quantize"),
        }
    }

    #[test]
    fn test_cli_transcribe_parsing() {
        let cli = Cli::parse_from(["gigastt", "transcribe", "audio.wav"]);
        match cli.command {
            Commands::Transcribe {
                file,
                model_variant,
                format,
                output,
                ..
            } => {
                assert_eq!(file, "audio.wav");
                // No --model-variant → None (auto-detect from disk).
                assert_eq!(model_variant, None);
                assert_eq!(format, "txt");
                assert!(output.is_none());
            }
            _ => panic!("expected Transcribe"),
        }
    }

    #[test]
    fn test_cli_transcribe_format_and_output() {
        let cli = Cli::parse_from([
            "gigastt",
            "transcribe",
            "audio.wav",
            "--format",
            "srt",
            "-o",
            "out.srt",
        ]);
        match cli.command {
            Commands::Transcribe {
                file,
                format,
                output,
                ..
            } => {
                assert_eq!(file, "audio.wav");
                assert_eq!(format, "srt");
                assert_eq!(output, Some("out.srt".to_string()));
            }
            _ => panic!("expected Transcribe"),
        }
    }

    #[test]
    fn test_cli_transcribe_subtitle_options() {
        let cli = Cli::parse_from([
            "gigastt",
            "transcribe",
            "audio.wav",
            "--format",
            "vtt",
            "--max-chars-per-line",
            "60",
            "--max-words-per-line",
            "10",
            "--word-timestamps",
        ]);
        match cli.command {
            Commands::Transcribe {
                format,
                max_chars_per_line,
                max_words_per_line,
                word_timestamps,
                ..
            } => {
                assert_eq!(format, "vtt");
                assert_eq!(max_chars_per_line, Some(60));
                assert_eq!(max_words_per_line, Some(10));
                assert!(word_timestamps);
            }
            _ => panic!("expected Transcribe"),
        }
    }

    #[test]
    fn test_cli_serve_rejects_unknown_model_variant() {
        let res = Cli::try_parse_from(["gigastt", "serve", "--model-variant", "whisper"]);
        assert!(res.is_err(), "unknown variant must be rejected by clap");
    }

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
    fn test_cli_serve_punctuation_defaults_auto() {
        let cli = Cli::parse_from(["gigastt", "serve"]);
        match cli.command {
            Commands::Serve {
                punctuation,
                punct_model_dir,
                ..
            } => {
                assert_eq!(punctuation, PunctuationMode::Auto);
                assert!(punct_model_dir.contains("punct"));
            }
            _ => panic!("expected Serve"),
        }
    }

    #[test]
    fn test_cli_serve_punctuation_override() {
        let cli = Cli::parse_from([
            "gigastt",
            "serve",
            "--punctuation",
            "on",
            "--punct-model-dir",
            "/tmp/punct",
        ]);
        match cli.command {
            Commands::Serve {
                punctuation,
                punct_model_dir,
                ..
            } => {
                assert_eq!(punctuation, PunctuationMode::On);
                assert_eq!(punct_model_dir, "/tmp/punct");
            }
            _ => panic!("expected Serve"),
        }
    }

    #[test]
    fn test_cli_transcribe_punctuation_off() {
        let cli = Cli::parse_from(["gigastt", "transcribe", "a.wav", "--punctuation", "off"]);
        match cli.command {
            Commands::Transcribe { punctuation, .. } => {
                assert_eq!(punctuation, PunctuationMode::Off);
            }
            _ => panic!("expected Transcribe"),
        }
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
    fn test_resolve_itn_auto_per_variant() {
        // auto → on for the bare rnnt head, off for the already-ITN e2e head.
        assert!(resolve_itn(ItnMode::Auto, ModelVariant::Rnnt));
        assert!(!resolve_itn(ItnMode::Auto, ModelVariant::E2eRnnt));
        // on/off override the variant.
        assert!(resolve_itn(ItnMode::On, ModelVariant::E2eRnnt));
        assert!(!resolve_itn(ItnMode::Off, ModelVariant::Rnnt));
    }

    #[test]
    fn test_cli_serve_itn_defaults_auto() {
        let cli = Cli::parse_from(["gigastt", "serve"]);
        match cli.command {
            Commands::Serve { itn, .. } => assert_eq!(itn, ItnMode::Auto),
            _ => panic!("expected Serve"),
        }
    }

    #[test]
    fn test_cli_transcribe_itn_override() {
        let cli = Cli::parse_from(["gigastt", "transcribe", "a.wav", "--itn", "on"]);
        match cli.command {
            Commands::Transcribe { itn, .. } => assert_eq!(itn, ItnMode::On),
            _ => panic!("expected Transcribe"),
        }
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
    fn test_cli_serve_with_metrics() {
        let cli = Cli::parse_from(["gigastt", "serve", "--metrics"]);
        match cli.command {
            Commands::Serve {
                metrics,
                metrics_listen,
                ..
            } => {
                assert!(metrics);
                // Unset → resolved to the loopback default downstream.
                assert!(metrics_listen.is_none());
            }
            _ => panic!("expected Serve"),
        }
    }

    #[test]
    fn test_cli_serve_metrics_listen_override() {
        let cli = Cli::parse_from([
            "gigastt",
            "serve",
            "--metrics",
            "--metrics-listen",
            "127.0.0.1:9123",
        ]);
        match cli.command {
            Commands::Serve { metrics_listen, .. } => {
                let addr = metrics_listen.expect("--metrics-listen must parse");
                assert_eq!(addr.port(), 9123);
                assert!(addr.ip().is_loopback());
            }
            _ => panic!("expected Serve"),
        }
        // Default when omitted resolves to 127.0.0.1:9090.
        assert_eq!(server::config::default_metrics_listen().port(), 9090);
    }

    #[test]
    fn test_is_loopback_host_ipv6_bracketed() {
        assert!(is_loopback_host("[::1]"));
        assert!(!is_loopback_host("[2001:db8::1]"));
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
    fn test_ensure_bind_allowed_env_opt_in() {
        let _guard = ENV_LOCK.lock().unwrap();
        let previous = std::env::var("GIGASTT_ALLOW_BIND_ANY").ok();
        unsafe {
            std::env::set_var("GIGASTT_ALLOW_BIND_ANY", "1");
        }
        let result = ensure_bind_allowed("0.0.0.0", false);
        if let Some(v) = previous {
            unsafe {
                std::env::set_var("GIGASTT_ALLOW_BIND_ANY", v);
            }
        } else {
            unsafe {
                std::env::remove_var("GIGASTT_ALLOW_BIND_ANY");
            }
        }
        assert!(result.is_ok(), "env opt-in must allow non-loopback bind");
    }

    #[test]
    fn test_build_limits_defaults_when_no_config() {
        let limits =
            build_limits(None, None, None, None, None, None, None, None, None, None).unwrap();
        assert_eq!(limits.idle_timeout_secs, 300);
        assert_eq!(limits.ws_frame_max_bytes, 512 * 1024);
    }

    #[test]
    fn test_build_limits_applies_overrides() {
        let limits = build_limits(
            None,
            Some(600),
            Some(1024),
            Some(10 * 1024 * 1024),
            Some(60),
            Some(20),
            Some(1800),
            Some(5),
            Some(15),
            Some(45),
        )
        .unwrap();
        assert_eq!(limits.idle_timeout_secs, 600);
        assert_eq!(limits.ws_frame_max_bytes, 1024);
        assert_eq!(limits.body_limit_bytes, 10 * 1024 * 1024);
        assert_eq!(limits.rate_limit_per_minute, 60);
        assert_eq!(limits.rate_limit_burst, 20);
        assert_eq!(limits.max_session_secs, 1800);
        assert_eq!(limits.shutdown_drain_secs, 5);
        assert_eq!(limits.pool_checkout_timeout_secs, 15);
        assert_eq!(limits.inference_timeout_secs, 45);
    }

    #[test]
    fn test_build_limits_with_valid_config_file() {
        let tmp = tempfile::NamedTempFile::new().unwrap();
        std::fs::write(tmp.path(), b"idle_timeout_secs = 123\n").unwrap();
        let limits = build_limits(
            Some(tmp.path().to_str().unwrap()),
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )
        .unwrap();
        assert_eq!(limits.idle_timeout_secs, 123);
    }

    #[test]
    fn test_build_limits_with_invalid_config_file() {
        let tmp = tempfile::NamedTempFile::new().unwrap();
        std::fs::write(tmp.path(), b"not valid toml {{{").unwrap();
        let result = build_limits(
            Some(tmp.path().to_str().unwrap()),
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_build_limits_rejects_zero_burst_with_nonzero_rpm() {
        let result = build_limits(
            None,
            None,
            None,
            None,
            Some(30),
            Some(0),
            None,
            None,
            None,
            None,
        );
        assert!(result.is_err());
        let msg = format!("{}", result.unwrap_err());
        assert!(msg.contains("rate-limit-burst"));
    }

    #[test]
    fn test_build_limits_allows_zero_rpm() {
        let limits = build_limits(
            None,
            None,
            None,
            None,
            Some(0),
            Some(0),
            None,
            None,
            None,
            None,
        )
        .unwrap();
        assert_eq!(limits.rate_limit_per_minute, 0);
        assert_eq!(limits.rate_limit_burst, 0);
    }

    #[test]
    fn test_build_server_config() {
        let limits = RuntimeLimits::default();
        let cfg = build_server_config(
            1234,
            "127.0.0.1".into(),
            vec!["https://app.example.com".into()],
            false,
            limits.clone(),
            true,
            "127.0.0.1:9099".parse().unwrap(),
            true,
            Some("/tmp/config.toml".into()),
        );
        assert_eq!(cfg.port, 1234);
        assert_eq!(cfg.metrics_listen.port(), 9099);
        assert_eq!(cfg.host, "127.0.0.1");
        assert_eq!(cfg.origin_policy.allowed_origins.len(), 1);
        assert!(!cfg.origin_policy.allow_any);
        assert!(cfg.metrics_enabled);
        assert!(cfg.trust_proxy);
        assert_eq!(
            cfg.config_path,
            Some(std::path::PathBuf::from("/tmp/config.toml"))
        );
        assert_eq!(cfg.limits.idle_timeout_secs, limits.idle_timeout_secs);
    }
}
