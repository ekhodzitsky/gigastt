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

        /// Contextual hotword biasing: path to a file of phrases to boost during
        /// recognition (one phrase per line, optional `\t<weight>` suffix; blank
        /// lines and `#` comments ignored). Off when unset. Env:
        /// GIGASTT_HOTWORDS_FILE.
        #[arg(long, env = "GIGASTT_HOTWORDS_FILE")]
        hotwords_file: Option<String>,

        /// Also bias the built-in Russian brand/acronym lexicon. Combined with
        /// any `--hotwords-file` phrases. Env: GIGASTT_HOTWORDS_DEFAULT.
        #[arg(long, env = "GIGASTT_HOTWORDS_DEFAULT", default_value_t = false)]
        hotwords_default: bool,

        /// Additive logit boost applied to hotword continuation tokens during
        /// greedy decode [default: 5.0]. Higher = stronger bias. No effect
        /// unless hotwords are configured. Env: GIGASTT_HOTWORDS_BOOST.
        #[arg(long, env = "GIGASTT_HOTWORDS_BOOST")]
        hotwords_boost: Option<f32>,

        /// Voice activity detection: skip silence in file transcription and
        /// finalize streaming segments on detected trailing silence. Off by
        /// default; downloads the Silero VAD model (MIT) on first use. Env:
        /// GIGASTT_VAD.
        #[arg(long, env = "GIGASTT_VAD", default_value_t = false)]
        vad: bool,

        /// VAD speech-probability threshold in [0,1] [default: 0.5]. Higher =
        /// stricter. No effect unless `--vad`. Env: GIGASTT_VAD_THRESHOLD.
        #[arg(long, env = "GIGASTT_VAD_THRESHOLD")]
        vad_threshold: Option<f32>,

        /// Minimum trailing silence (ms) to close a speech region / finalize a
        /// streaming segment [default: 500]. No effect unless `--vad`. Env:
        /// GIGASTT_VAD_MIN_SILENCE_MS.
        #[arg(long, env = "GIGASTT_VAD_MIN_SILENCE_MS")]
        vad_min_silence_ms: Option<u32>,

        /// Directory holding the Silero VAD model (`silero_vad.onnx`). Defaults
        /// to `~/.gigastt/models/vad/`. Auto-downloaded when `--vad` is set and
        /// the model is absent. Env: GIGASTT_VAD_MODEL_DIR.
        #[arg(long, env = "GIGASTT_VAD_MODEL_DIR", default_value_t = model::default_vad_model_dir())]
        vad_model_dir: String,

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

        /// Intra-op thread count for the encoder session on the CPU build. The
        /// encoder dominates the single-utterance cost, so more threads speed up
        /// weak CPUs / long single-file jobs. When unset, defaults to the logical
        /// CPU count divided across the concurrently-running pool triplets
        /// (`pool_size + batch_pool_size`), so a default install uses every core.
        /// An explicit value (flag or env, including `1`) is honoured as-is. The
        /// resolved value is still auto-clamped so `pool_size * threads` can't
        /// exceed the logical CPU count. No effect on CoreML / CUDA builds.
        #[arg(long, env = "GIGASTT_ENCODER_INTRA_THREADS")]
        encoder_intra_threads: Option<usize>,

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

        /// Fetch the pre-quantized INT8 bundle from the pinned GitHub Release
        /// instead of the FP32 set + on-device quantization. The lean path:
        /// no ~844 MB FP32 download, no ~2-minute quantize, no `protoc`.
        /// Mutually exclusive with `--skip-quantize` (which only applies to the
        /// FP32 download path).
        #[arg(long, default_value_t = false)]
        prequantized: bool,

        /// Also fetch the per-bucket palettized ANE (Core ML) encoder packages
        /// into `~/.gigastt/models/ane/` for the macOS Neural Engine backend.
        /// Requires a published ANE release.
        #[cfg(feature = "ane")]
        #[arg(long, default_value_t = false)]
        ane: bool,
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

        /// Contextual hotword biasing: path to a file of phrases to boost during
        /// recognition (one phrase per line, optional `\t<weight>` suffix; blank
        /// lines and `#` comments ignored). Off when unset. Env:
        /// GIGASTT_HOTWORDS_FILE.
        #[arg(long, env = "GIGASTT_HOTWORDS_FILE")]
        hotwords_file: Option<String>,

        /// Also bias the built-in Russian brand/acronym lexicon. Combined with
        /// any `--hotwords-file` phrases. Env: GIGASTT_HOTWORDS_DEFAULT.
        #[arg(long, env = "GIGASTT_HOTWORDS_DEFAULT", default_value_t = false)]
        hotwords_default: bool,

        /// Additive logit boost applied to hotword continuation tokens during
        /// greedy decode [default: 5.0]. Higher = stronger bias. No effect
        /// unless hotwords are configured. Env: GIGASTT_HOTWORDS_BOOST.
        #[arg(long, env = "GIGASTT_HOTWORDS_BOOST")]
        hotwords_boost: Option<f32>,

        /// Voice activity detection: skip silence before decoding. Off by
        /// default; downloads the Silero VAD model (MIT) on first use. Env:
        /// GIGASTT_VAD.
        #[arg(long, env = "GIGASTT_VAD", default_value_t = false)]
        vad: bool,

        /// VAD speech-probability threshold in [0,1] [default: 0.5]. Higher =
        /// stricter. No effect unless `--vad`. Env: GIGASTT_VAD_THRESHOLD.
        #[arg(long, env = "GIGASTT_VAD_THRESHOLD")]
        vad_threshold: Option<f32>,

        /// Minimum trailing silence (ms) to close a speech region [default: 500].
        /// No effect unless `--vad`. Env: GIGASTT_VAD_MIN_SILENCE_MS.
        #[arg(long, env = "GIGASTT_VAD_MIN_SILENCE_MS")]
        vad_min_silence_ms: Option<u32>,

        /// Directory holding the Silero VAD model (`silero_vad.onnx`). Defaults
        /// to `~/.gigastt/models/vad/`. Auto-downloaded when `--vad` is set and
        /// the model is absent. Env: GIGASTT_VAD_MODEL_DIR.
        #[arg(long, env = "GIGASTT_VAD_MODEL_DIR", default_value_t = model::default_vad_model_dir())]
        vad_model_dir: String,

        /// Intra-op thread count for the encoder session on the CPU build. The
        /// encoder dominates the single-utterance cost, so more threads speed up
        /// long single-file jobs on weak CPUs. When unset, defaults to the logical
        /// CPU count (offline transcription runs a single triplet). An explicit
        /// value (flag or env, including `1`) is honoured as-is. No effect on
        /// CoreML / CUDA builds.
        #[arg(long, env = "GIGASTT_ENCODER_INTRA_THREADS")]
        encoder_intra_threads: Option<usize>,

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
fn resolve_encoder_intra_threads(
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

/// Build a [`gigastt_core::vad::VadConfig`] from CLI overrides, falling back to
/// the library defaults for any option left unset.
fn build_vad_config(
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
fn maybe_load_vad(enabled: bool, vad_model_dir: &str) -> Option<gigastt_core::vad::SileroVad> {
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
async fn maybe_download_vad_model(enabled: bool, vad_model_dir: &str) {
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
const DEFAULT_HOTWORDS_BOOST: f32 = 5.0;

/// Parse a hotwords file: one phrase per line, optional `\t<weight>` suffix.
/// Blank lines and `#`-prefixed comment lines are skipped. A malformed weight
/// falls back to `1.0` (the phrase is still kept). Returns the `(phrase, weight)`
/// pairs, or an error only when the file can't be read.
fn parse_hotwords_file(path: &str) -> anyhow::Result<Vec<(String, f32)>> {
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
fn resolve_hotwords(
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
fn log_ane_backend(resolved: ModelVariant) {
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
            hotwords_file,
            hotwords_default,
            hotwords_boost,
            vad,
            vad_threshold,
            vad_min_silence_ms,
            vad_model_dir,
            pool_size,
            pool_min_size,
            batch_pool_size,
            encoder_intra_threads,
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
            let server_config = build_server_config(
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

            // The reusable engine build recipe, captured so BOTH first-run boot
            // and `POST /v1/admin/reload` produce a byte-for-byte identical
            // engine — including the punctuation / ITN / VAD / hotword chain a
            // fresh `Engine::load_*` starts without. Synchronous (ONNX session
            // load, quantization) so it can run on a blocking thread on either
            // path; it re-detects the on-disk variant so a reload picks up a
            // model swapped on disk between boot and reload.
            let build_engine: server::EngineBuilder = {
                let model_dir = model_dir.clone();
                let punct_model_dir = punct_model_dir.clone();
                let vad_model_dir = vad_model_dir.clone();
                let hotwords_file = hotwords_file.clone();
                std::sync::Arc::new(move || -> anyhow::Result<inference::Engine> {
                    // Honor the explicit --model-variant when set; otherwise
                    // detect what is present on disk. Reload never downloads, so
                    // if the requested variant's files are absent the engine load
                    // will fail with a clear error — the operator asked for a
                    // variant that isn't there.
                    let resolved = model_variant
                        .or_else(|| {
                            model::ModelVariant::detect_in_dir(std::path::Path::new(&model_dir))
                        })
                        .unwrap_or_default();
                    ensure_int8_encoder(resolved, &model_dir, skip_quantize)?;
                    let punctuator = maybe_load_punctuator(punctuation, &punct_model_dir, resolved);
                    let hotwords = resolve_hotwords(hotwords_file.as_deref(), hotwords_default);
                    // Resolve the intra-op default from the effective number of
                    // concurrently-running triplets when the operator didn't set
                    // it. The engine still clamps `pool_size * threads` below the
                    // logical CPU count.
                    let resolved_intra_threads = resolve_encoder_intra_threads(
                        encoder_intra_threads,
                        pool_size + batch_pool_size,
                        std::thread::available_parallelism()
                            .map(|n| n.get())
                            .unwrap_or(1),
                    );
                    let mut engine = inference::Engine::load_with_pools_threads(
                        &model_dir,
                        pool_size,
                        pool_min_size,
                        batch_pool_size,
                        resolved_intra_threads,
                    )?
                    .with_punctuator(punctuator)
                    .with_itn(resolve_itn(itn, resolved))
                    .with_vad(
                        maybe_load_vad(vad, &vad_model_dir),
                        build_vad_config(vad_threshold, vad_min_silence_ms),
                    );
                    if let Some(pairs) = hotwords {
                        engine = engine.with_hotwords(
                            &pairs,
                            hotwords_boost.unwrap_or(DEFAULT_HOTWORDS_BOOST),
                        );
                    }
                    #[cfg(feature = "ane")]
                    log_ane_backend(resolved);
                    log_rss();
                    Ok(engine)
                })
            };

            // Build the engine in the background while a minimal bootstrap
            // responder serves /health (200) and /ready (503 initializing) on the
            // port, so probes / Docker HEALTHCHECK don't see connection-refused
            // during the first-run model download + INT8 quantization. The heavy
            // synchronous work (quantize, ONNX session load, post-processor loads)
            // runs on a blocking thread so the bootstrap responder stays snappy.
            let boot_builder = build_engine.clone();
            let load = async move {
                let resolved = model::ensure_model_variant(model_variant, &model_dir).await?;
                maybe_download_punct_model(punctuation, &punct_model_dir, resolved).await;
                maybe_download_vad_model(vad, &vad_model_dir).await;
                tokio::task::spawn_blocking(move || boot_builder())
                    .await
                    .context("engine load task panicked")?
            };
            server::run_with_config_loading_reloadable(
                server_config,
                None,
                load,
                Some(build_engine),
            )
            .await?;
        }
        Commands::Download {
            model_dir,
            model_variant,
            #[cfg(feature = "diarization")]
            skip_diarization,
            skip_quantize,
            prequantized,
            #[cfg(feature = "ane")]
            ane,
        } => {
            // `download` is an explicit action: the requested variant maps to
            // the default (Rnnt) so a bare `gigastt download` fetches something
            // useful.
            if prequantized {
                // Lean path: fetch the INT8 bundle from the pinned Release — no
                // FP32 download, no on-device quantization, no protoc.
                model::ensure_prequantized_model_variant(Some(model_variant), &model_dir).await?;
            } else {
                let resolved = model::ensure_model_variant(Some(model_variant), &model_dir).await?;
                ensure_int8_encoder(resolved, &model_dir, skip_quantize)?;
            }
            #[cfg(feature = "diarization")]
            {
                if !skip_diarization {
                    model::ensure_speaker_model(&model_dir).await?;
                }
            }
            #[cfg(feature = "ane")]
            if ane {
                let ane_dir = model::default_ane_model_dir();
                model::ensure_ane_packages(&ane_dir).await?;
                tracing::info!("ANE encoder packages ready at {ane_dir}");
            }
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
            hotwords_file,
            hotwords_default,
            hotwords_boost,
            vad,
            vad_threshold,
            vad_min_silence_ms,
            vad_model_dir,
            encoder_intra_threads,
            format,
            output,
            max_chars_per_line,
            max_words_per_line,
            word_timestamps,
        } => {
            let resolved = model::ensure_model_variant(model_variant, &model_dir).await?;
            maybe_download_punct_model(punctuation, &punct_model_dir, resolved).await;
            maybe_download_vad_model(vad, &vad_model_dir).await;
            let punctuator = maybe_load_punctuator(punctuation, &punct_model_dir, resolved);
            let hotwords = resolve_hotwords(hotwords_file.as_deref(), hotwords_default);
            // Single-triplet pool for offline file transcription; when the
            // thread count is unset it defaults to every logical CPU (one
            // running triplet), else the explicit value is used as-is.
            let resolved_intra_threads = resolve_encoder_intra_threads(
                encoder_intra_threads,
                1,
                std::thread::available_parallelism()
                    .map(|n| n.get())
                    .unwrap_or(1),
            );
            let mut engine = inference::Engine::load_with_pools_threads(
                &model_dir,
                1,
                1,
                0,
                resolved_intra_threads,
            )?
            .with_punctuator(punctuator)
            .with_itn(resolve_itn(itn, resolved))
            .with_vad(
                maybe_load_vad(vad, &vad_model_dir),
                build_vad_config(vad_threshold, vad_min_silence_ms),
            );
            if let Some(pairs) = hotwords {
                engine =
                    engine.with_hotwords(&pairs, hotwords_boost.unwrap_or(DEFAULT_HOTWORDS_BOOST));
            }
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

    // Restore a captured env value when dropped, so an env-mutating test never
    // leaks `GIGASTT_ENCODER_INTRA_THREADS` to a sibling test (clap reads the
    // process environment). Paired with `ENV_LOCK` to serialize these tests.
    struct EnvRestore(&'static str, Option<String>);
    impl Drop for EnvRestore {
        fn drop(&mut self) {
            match &self.1 {
                Some(v) => unsafe { std::env::set_var(self.0, v) },
                None => unsafe { std::env::remove_var(self.0) },
            }
        }
    }

    #[test]
    fn test_cli_serve_encoder_intra_threads_default() {
        // Unset → None, so the default resolves from the pool size at load time.
        let _guard = ENV_LOCK.lock().unwrap();
        let _restore = EnvRestore(
            "GIGASTT_ENCODER_INTRA_THREADS",
            std::env::var("GIGASTT_ENCODER_INTRA_THREADS").ok(),
        );
        unsafe {
            std::env::remove_var("GIGASTT_ENCODER_INTRA_THREADS");
        }
        let cli = Cli::parse_from(["gigastt", "serve"]);
        match cli.command {
            Commands::Serve {
                encoder_intra_threads,
                ..
            } => assert_eq!(encoder_intra_threads, None),
            _ => panic!("expected Serve"),
        }
    }

    #[test]
    fn test_cli_serve_encoder_intra_threads_flag() {
        // The explicit flag wins over any inherited env value.
        let _guard = ENV_LOCK.lock().unwrap();
        let _restore = EnvRestore(
            "GIGASTT_ENCODER_INTRA_THREADS",
            std::env::var("GIGASTT_ENCODER_INTRA_THREADS").ok(),
        );
        unsafe {
            std::env::remove_var("GIGASTT_ENCODER_INTRA_THREADS");
        }
        let cli = Cli::parse_from(["gigastt", "serve", "--encoder-intra-threads", "4"]);
        match cli.command {
            Commands::Serve {
                encoder_intra_threads,
                ..
            } => assert_eq!(encoder_intra_threads, Some(4)),
            _ => panic!("expected Serve"),
        }
    }

    #[test]
    fn test_cli_serve_encoder_intra_threads_env() {
        // The flag is wired to GIGASTT_ENCODER_INTRA_THREADS; clap reads the
        // process environment, so serialize against other env-mutating tests.
        let _guard = ENV_LOCK.lock().unwrap();
        let _restore = EnvRestore(
            "GIGASTT_ENCODER_INTRA_THREADS",
            std::env::var("GIGASTT_ENCODER_INTRA_THREADS").ok(),
        );
        unsafe {
            std::env::set_var("GIGASTT_ENCODER_INTRA_THREADS", "6");
        }
        let cli = Cli::parse_from(["gigastt", "serve"]);
        match cli.command {
            Commands::Serve {
                encoder_intra_threads,
                ..
            } => assert_eq!(encoder_intra_threads, Some(6)),
            _ => panic!("expected Serve"),
        }
    }

    #[test]
    fn test_cli_transcribe_encoder_intra_threads_flag() {
        let _guard = ENV_LOCK.lock().unwrap();
        let _restore = EnvRestore(
            "GIGASTT_ENCODER_INTRA_THREADS",
            std::env::var("GIGASTT_ENCODER_INTRA_THREADS").ok(),
        );
        unsafe {
            std::env::remove_var("GIGASTT_ENCODER_INTRA_THREADS");
        }
        let cli = Cli::parse_from([
            "gigastt",
            "transcribe",
            "audio.wav",
            "--encoder-intra-threads",
            "3",
        ]);
        match cli.command {
            Commands::Transcribe {
                encoder_intra_threads,
                ..
            } => assert_eq!(encoder_intra_threads, Some(3)),
            _ => panic!("expected Transcribe"),
        }
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
    fn test_cli_serve_hotwords_flags() {
        let cli = Cli::parse_from([
            "gigastt",
            "serve",
            "--hotwords-file",
            "/tmp/hw.txt",
            "--hotwords-default",
            "--hotwords-boost",
            "8.5",
        ]);
        match cli.command {
            Commands::Serve {
                hotwords_file,
                hotwords_default,
                hotwords_boost,
                ..
            } => {
                assert_eq!(hotwords_file, Some("/tmp/hw.txt".to_string()));
                assert!(hotwords_default);
                assert_eq!(hotwords_boost, Some(8.5));
            }
            _ => panic!("expected Serve"),
        }
    }

    #[test]
    fn test_cli_serve_hotwords_default_off() {
        let cli = Cli::parse_from(["gigastt", "serve"]);
        match cli.command {
            Commands::Serve {
                hotwords_file,
                hotwords_default,
                hotwords_boost,
                ..
            } => {
                assert_eq!(hotwords_file, None);
                assert!(!hotwords_default);
                assert_eq!(hotwords_boost, None);
            }
            _ => panic!("expected Serve"),
        }
    }

    #[test]
    fn test_cli_transcribe_hotwords_flags() {
        let cli = Cli::parse_from([
            "gigastt",
            "transcribe",
            "a.wav",
            "--hotwords-file",
            "hw.txt",
        ]);
        match cli.command {
            Commands::Transcribe {
                hotwords_file,
                hotwords_default,
                ..
            } => {
                assert_eq!(hotwords_file, Some("hw.txt".to_string()));
                assert!(!hotwords_default);
            }
            _ => panic!("expected Transcribe"),
        }
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

    #[test]
    fn test_parse_model_variant_valid_and_invalid() {
        assert_eq!(parse_model_variant("rnnt").unwrap(), ModelVariant::Rnnt);
        assert_eq!(
            parse_model_variant("e2e_rnnt").unwrap(),
            ModelVariant::E2eRnnt
        );
        assert!(parse_model_variant("whisper").is_err());
    }

    #[test]
    fn test_parse_punctuation_mode_value_parser() {
        assert_eq!(parse_punctuation_mode("on").unwrap(), PunctuationMode::On);
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
    fn test_cli_serve_vad_flags() {
        let _guard = ENV_LOCK.lock().unwrap();
        let _restore_vad = EnvRestore("GIGASTT_VAD", std::env::var("GIGASTT_VAD").ok());
        let _restore_threshold = EnvRestore(
            "GIGASTT_VAD_THRESHOLD",
            std::env::var("GIGASTT_VAD_THRESHOLD").ok(),
        );
        let _restore_sil = EnvRestore(
            "GIGASTT_VAD_MIN_SILENCE_MS",
            std::env::var("GIGASTT_VAD_MIN_SILENCE_MS").ok(),
        );
        let _restore_dir = EnvRestore(
            "GIGASTT_VAD_MODEL_DIR",
            std::env::var("GIGASTT_VAD_MODEL_DIR").ok(),
        );
        unsafe {
            std::env::remove_var("GIGASTT_VAD");
            std::env::remove_var("GIGASTT_VAD_THRESHOLD");
            std::env::remove_var("GIGASTT_VAD_MIN_SILENCE_MS");
            std::env::remove_var("GIGASTT_VAD_MODEL_DIR");
        }
        let cli = Cli::parse_from([
            "gigastt",
            "serve",
            "--vad",
            "--vad-threshold",
            "0.8",
            "--vad-min-silence-ms",
            "700",
            "--vad-model-dir",
            "/tmp/vad",
        ]);
        match cli.command {
            Commands::Serve {
                vad,
                vad_threshold,
                vad_min_silence_ms,
                vad_model_dir,
                ..
            } => {
                assert!(vad);
                assert_eq!(vad_threshold, Some(0.8));
                assert_eq!(vad_min_silence_ms, Some(700));
                assert_eq!(vad_model_dir, "/tmp/vad");
            }
            _ => panic!("expected Serve"),
        }
    }

    #[test]
    fn test_cli_serve_vad_defaults_off() {
        let _guard = ENV_LOCK.lock().unwrap();
        let _restore_vad = EnvRestore("GIGASTT_VAD", std::env::var("GIGASTT_VAD").ok());
        let _restore_threshold = EnvRestore(
            "GIGASTT_VAD_THRESHOLD",
            std::env::var("GIGASTT_VAD_THRESHOLD").ok(),
        );
        let _restore_sil = EnvRestore(
            "GIGASTT_VAD_MIN_SILENCE_MS",
            std::env::var("GIGASTT_VAD_MIN_SILENCE_MS").ok(),
        );
        unsafe {
            std::env::remove_var("GIGASTT_VAD");
            std::env::remove_var("GIGASTT_VAD_THRESHOLD");
            std::env::remove_var("GIGASTT_VAD_MIN_SILENCE_MS");
        }
        let cli = Cli::parse_from(["gigastt", "serve"]);
        match cli.command {
            Commands::Serve {
                vad,
                vad_threshold,
                vad_min_silence_ms,
                ..
            } => {
                assert!(!vad);
                assert_eq!(vad_threshold, None);
                assert_eq!(vad_min_silence_ms, None);
            }
            _ => panic!("expected Serve"),
        }
    }

    #[test]
    fn test_cli_serve_pool_and_thread_flags() {
        let _guard = ENV_LOCK.lock().unwrap();
        let _restore_min = EnvRestore(
            "GIGASTT_POOL_MIN_SIZE",
            std::env::var("GIGASTT_POOL_MIN_SIZE").ok(),
        );
        let _restore_batch = EnvRestore(
            "GIGASTT_BATCH_POOL_SIZE",
            std::env::var("GIGASTT_BATCH_POOL_SIZE").ok(),
        );
        let _restore_threads = EnvRestore(
            "GIGASTT_ENCODER_INTRA_THREADS",
            std::env::var("GIGASTT_ENCODER_INTRA_THREADS").ok(),
        );
        unsafe {
            std::env::remove_var("GIGASTT_POOL_MIN_SIZE");
            std::env::remove_var("GIGASTT_BATCH_POOL_SIZE");
            std::env::remove_var("GIGASTT_ENCODER_INTRA_THREADS");
        }
        let cli = Cli::parse_from([
            "gigastt",
            "serve",
            "--pool-size",
            "8",
            "--pool-min-size",
            "3",
            "--batch-pool-size",
            "2",
        ]);
        match cli.command {
            Commands::Serve {
                pool_size,
                pool_min_size,
                batch_pool_size,
                ..
            } => {
                assert_eq!(pool_size, 8);
                assert_eq!(pool_min_size, 3);
                assert_eq!(batch_pool_size, 2);
            }
            _ => panic!("expected Serve"),
        }
    }

    #[test]
    fn test_cli_serve_runtime_limit_flags() {
        let _guard = ENV_LOCK.lock().unwrap();
        // These flags read env vars; clear them so the explicit args win.
        let restores: Vec<EnvRestore> = [
            "GIGASTT_IDLE_TIMEOUT_SECS",
            "GIGASTT_WS_FRAME_MAX_BYTES",
            "GIGASTT_BODY_LIMIT_BYTES",
            "GIGASTT_RATE_LIMIT_PER_MINUTE",
            "GIGASTT_RATE_LIMIT_BURST",
            "GIGASTT_MAX_SESSION_SECS",
            "GIGASTT_SHUTDOWN_DRAIN_SECS",
            "GIGASTT_POOL_CHECKOUT_TIMEOUT_SECS",
            "GIGASTT_INFERENCE_TIMEOUT_SECS",
        ]
        .iter()
        .map(|k| {
            let r = EnvRestore(k, std::env::var(k).ok());
            unsafe {
                std::env::remove_var(k);
            }
            r
        })
        .collect();
        let cli = Cli::parse_from([
            "gigastt",
            "serve",
            "--idle-timeout-secs",
            "120",
            "--ws-frame-max-bytes",
            "4096",
            "--body-limit-bytes",
            "8192",
            "--rate-limit-per-minute",
            "90",
            "--rate-limit-burst",
            "15",
            "--max-session-secs",
            "777",
            "--shutdown-drain-secs",
            "7",
            "--pool-checkout-timeout-secs",
            "11",
            "--inference-timeout-secs",
            "300",
            "--trust-proxy",
        ]);
        match cli.command {
            Commands::Serve {
                idle_timeout_secs,
                ws_frame_max_bytes,
                body_limit_bytes,
                rate_limit_per_minute,
                rate_limit_burst,
                max_session_secs,
                shutdown_drain_secs,
                pool_checkout_timeout_secs,
                inference_timeout_secs,
                trust_proxy,
                ..
            } => {
                assert_eq!(idle_timeout_secs, Some(120));
                assert_eq!(ws_frame_max_bytes, Some(4096));
                assert_eq!(body_limit_bytes, Some(8192));
                assert_eq!(rate_limit_per_minute, Some(90));
                assert_eq!(rate_limit_burst, Some(15));
                assert_eq!(max_session_secs, Some(777));
                assert_eq!(shutdown_drain_secs, Some(7));
                assert_eq!(pool_checkout_timeout_secs, Some(11));
                assert_eq!(inference_timeout_secs, Some(300));
                assert!(trust_proxy);
            }
            _ => panic!("expected Serve"),
        }
        drop(restores);
    }

    #[test]
    fn test_cli_serve_config_and_skip_quantize_flags() {
        let _guard = ENV_LOCK.lock().unwrap();
        let _restore = EnvRestore(
            "GIGASTT_SKIP_QUANTIZE",
            std::env::var("GIGASTT_SKIP_QUANTIZE").ok(),
        );
        unsafe {
            std::env::remove_var("GIGASTT_SKIP_QUANTIZE");
        }
        let cli = Cli::parse_from([
            "gigastt",
            "serve",
            "--config",
            "/tmp/limits.toml",
            "--skip-quantize",
        ]);
        match cli.command {
            Commands::Serve {
                config,
                skip_quantize,
                ..
            } => {
                assert_eq!(config, Some("/tmp/limits.toml".to_string()));
                assert!(skip_quantize);
            }
            _ => panic!("expected Serve"),
        }
    }

    #[test]
    fn test_cli_serve_cors_and_origin_flags() {
        let cli = Cli::parse_from([
            "gigastt",
            "serve",
            "--allow-origin",
            "https://a.example.com",
            "--allow-origin",
            "https://b.example.com",
            "--cors-allow-any",
        ]);
        match cli.command {
            Commands::Serve {
                allow_origin,
                cors_allow_any,
                ..
            } => {
                assert_eq!(allow_origin.len(), 2);
                assert_eq!(allow_origin[0], "https://a.example.com");
                assert!(cors_allow_any);
            }
            _ => panic!("expected Serve"),
        }
    }

    #[test]
    fn test_cli_download_skip_quantize_flag() {
        let _guard = ENV_LOCK.lock().unwrap();
        let _restore = EnvRestore(
            "GIGASTT_SKIP_QUANTIZE",
            std::env::var("GIGASTT_SKIP_QUANTIZE").ok(),
        );
        unsafe {
            std::env::remove_var("GIGASTT_SKIP_QUANTIZE");
        }
        let cli = Cli::parse_from(["gigastt", "download", "--skip-quantize"]);
        match cli.command {
            Commands::Download { skip_quantize, .. } => assert!(skip_quantize),
            _ => panic!("expected Download"),
        }
    }

    #[cfg(feature = "ane")]
    #[test]
    fn test_cli_download_ane_flag() {
        let cli = Cli::parse_from(["gigastt", "download", "--ane"]);
        match cli.command {
            Commands::Download { ane, .. } => assert!(ane),
            _ => panic!("expected Download"),
        }
        // Absent by default.
        let cli = Cli::parse_from(["gigastt", "download"]);
        match cli.command {
            Commands::Download { ane, .. } => assert!(!ane),
            _ => panic!("expected Download"),
        }
    }

    #[test]
    fn test_cli_transcribe_vad_and_itn_flags() {
        let _guard = ENV_LOCK.lock().unwrap();
        let restores: Vec<EnvRestore> = ["GIGASTT_VAD", "GIGASTT_ITN", "GIGASTT_VAD_THRESHOLD"]
            .iter()
            .map(|k| {
                let r = EnvRestore(k, std::env::var(k).ok());
                unsafe {
                    std::env::remove_var(k);
                }
                r
            })
            .collect();
        let cli = Cli::parse_from([
            "gigastt",
            "transcribe",
            "a.wav",
            "--vad",
            "--vad-threshold",
            "0.6",
            "--itn",
            "off",
        ]);
        match cli.command {
            Commands::Transcribe {
                vad,
                vad_threshold,
                itn,
                ..
            } => {
                assert!(vad);
                assert_eq!(vad_threshold, Some(0.6));
                assert_eq!(itn, ItnMode::Off);
            }
            _ => panic!("expected Transcribe"),
        }
        drop(restores);
    }

    #[test]
    fn test_cli_rejects_unknown_subcommand() {
        let res = Cli::try_parse_from(["gigastt", "bogus"]);
        assert!(res.is_err(), "unknown subcommand must be rejected");
    }

    #[test]
    fn test_cli_serve_rejects_bad_punctuation_value() {
        let res = Cli::try_parse_from(["gigastt", "serve", "--punctuation", "sometimes"]);
        assert!(res.is_err(), "invalid punctuation mode must be rejected");
    }

    #[test]
    fn test_cli_serve_rejects_bad_itn_value() {
        let res = Cli::try_parse_from(["gigastt", "serve", "--itn", "sometimes"]);
        assert!(res.is_err(), "invalid itn mode must be rejected");
    }
}
