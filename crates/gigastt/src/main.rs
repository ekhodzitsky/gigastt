use clap::{Parser, Subcommand};
use gigastt::server;
use gigastt::server::{OriginPolicy, RuntimeLimits, ServerConfig};
use gigastt_core::{inference, model};
use std::net::IpAddr;
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

        /// Number of concurrent inference sessions
        #[arg(long, default_value_t = 4)]
        pool_size: usize,

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
        /// Path to WAV file (PCM16 mono)
        file: String,

        /// Model directory
        #[arg(long, default_value_t = model::default_model_dir())]
        model_dir: String,
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

/// Ensure the INT8 encoder exists, producing it via the native Rust
/// quantization pipeline if missing. Honoured by `serve` and `download`.
/// First-time quantization takes ~2 minutes on the 844 MB FP32 encoder.
fn ensure_int8_encoder(model_dir: &str, skip: bool) -> anyhow::Result<()> {
    let int8_path = std::path::Path::new(model_dir).join("v3_e2e_rnnt_encoder_int8.onnx");
    if int8_path.exists() {
        return Ok(());
    }
    if skip {
        tracing::info!(
            "Skipping INT8 quantization (--skip-quantize). Engine will load the FP32 encoder."
        );
        return Ok(());
    }
    let input = std::path::Path::new(model_dir).join("v3_e2e_rnnt_encoder.onnx");
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
            pool_size,
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
            skip_quantize,
            trust_proxy,
            config,
        } => {
            ensure_bind_allowed(&host, bind_all)?;
            model::ensure_model(&model_dir).await?;
            ensure_int8_encoder(&model_dir, skip_quantize)?;
            let engine = inference::Engine::load_with_pool_size(&model_dir, pool_size)?;
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
            )?;
            let metrics_listen =
                metrics_listen.unwrap_or_else(server::config::default_metrics_listen);
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
            #[cfg(feature = "diarization")]
            skip_diarization,
            skip_quantize,
        } => {
            model::ensure_model(&model_dir).await?;
            #[cfg(feature = "diarization")]
            {
                if !skip_diarization {
                    model::ensure_speaker_model(&model_dir).await?;
                }
            }
            ensure_int8_encoder(&model_dir, skip_quantize)?;
            tracing::info!("Model ready at {model_dir}");
        }
        Commands::Quantize { model_dir, force } => {
            model::ensure_model(&model_dir).await?;
            let input = std::path::Path::new(&model_dir).join("v3_e2e_rnnt_encoder.onnx");
            let output = std::path::Path::new(&model_dir).join("v3_e2e_rnnt_encoder_int8.onnx");
            if output.exists() && !force {
                tracing::info!("INT8 model already exists: {}", output.display());
                tracing::info!("Use --force to re-quantize.");
                return Ok(());
            }
            gigastt_core::quantize::quantize_model(&input, &output)?;
            tracing::info!("Quantized model saved to {}", output.display());
        }
        Commands::Transcribe { file, model_dir } => {
            model::ensure_model(&model_dir).await?;
            let engine = inference::Engine::load_with_pool_size(&model_dir, 1)?;
            log_rss();
            let mut guard = engine.pool.checkout().await?;
            let result = engine.transcribe_file(&file, &mut guard);
            drop(guard);
            println!("{}", result?.text);
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
                ..
            } => {
                assert_eq!(port, 1234);
                assert!(bind_all);
                assert!(!metrics);
            }
            _ => panic!("expected Serve"),
        }
    }

    #[test]
    fn test_cli_download_parsing() {
        let cli = Cli::parse_from(["gigastt", "download", "--model-dir", "/tmp/models"]);
        match cli.command {
            Commands::Download { model_dir, .. } => {
                assert_eq!(model_dir, "/tmp/models");
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
            Commands::Transcribe { file, .. } => {
                assert_eq!(file, "audio.wav");
            }
            _ => panic!("expected Transcribe"),
        }
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
        let int8_path = tmp.path().join("v3_e2e_rnnt_encoder_int8.onnx");
        std::fs::write(&int8_path, b"fake").unwrap();
        ensure_int8_encoder(tmp.path().to_str().unwrap(), false).unwrap();
    }

    #[test]
    fn test_ensure_int8_encoder_skip_flag() {
        let tmp = tempfile::tempdir().unwrap();
        ensure_int8_encoder(tmp.path().to_str().unwrap(), true).unwrap();
    }

    #[test]
    fn test_ensure_int8_encoder_missing_input() {
        let tmp = tempfile::tempdir().unwrap();
        let err = ensure_int8_encoder(tmp.path().to_str().unwrap(), false).unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("Cannot quantize"), "unexpected error: {msg}");
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
        let limits = build_limits(None, None, None, None, None, None, None, None, None).unwrap();
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
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_build_limits_rejects_zero_burst_with_nonzero_rpm() {
        let result = build_limits(None, None, None, None, Some(30), Some(0), None, None, None);
        assert!(result.is_err());
        let msg = format!("{}", result.unwrap_err());
        assert!(msg.contains("rate-limit-burst"));
    }

    #[test]
    fn test_build_limits_allows_zero_rpm() {
        let limits =
            build_limits(None, None, None, None, Some(0), Some(0), None, None, None).unwrap();
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
