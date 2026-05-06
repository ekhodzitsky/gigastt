use clap::{Parser, Subcommand};
use gigastt::server::{OriginPolicy, RuntimeLimits, ServerConfig};
use gigastt::{inference, model, server};
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

        /// WebSocket idle timeout (seconds). Server closes the connection
        /// when no frame arrives within this window.
        #[arg(long, env = "GIGASTT_IDLE_TIMEOUT_SECS", default_value_t = 300)]
        idle_timeout_secs: u64,

        /// Maximum WebSocket frame / message size (bytes).
        #[arg(long, env = "GIGASTT_WS_FRAME_MAX_BYTES", default_value_t = 512 * 1024)]
        ws_frame_max_bytes: usize,

        /// Maximum REST request body size (bytes).
        #[arg(long, env = "GIGASTT_BODY_LIMIT_BYTES", default_value_t = 50 * 1024 * 1024)]
        body_limit_bytes: usize,

        /// Per-IP rate limit — requests per minute. 0 = off (default).
        #[arg(long, env = "GIGASTT_RATE_LIMIT_PER_MINUTE", default_value_t = 0)]
        rate_limit_per_minute: u32,

        /// Rate-limit burst size (default 10).
        #[arg(long, env = "GIGASTT_RATE_LIMIT_BURST", default_value_t = 10)]
        rate_limit_burst: u32,

        /// Expose Prometheus metrics at `GET /metrics`. Off by default —
        /// keeps the server quiet for single-user installs. The endpoint is
        /// attached to the protected router so the Origin allowlist applies.
        #[arg(long, env = "GIGASTT_METRICS", default_value_t = false)]
        metrics: bool,

        /// Maximum wall-clock duration of a single WebSocket session (seconds).
        /// `0` disables the cap (not recommended — a silence-streaming client
        /// will hold a triplet forever).
        #[arg(long, env = "GIGASTT_MAX_SESSION_SECS", default_value_t = 3600)]
        max_session_secs: u64,

        /// Grace window (seconds) after shutdown during which in-flight
        /// WebSocket / SSE sessions may emit their Final frames and close.
        /// Values of `0` are clamped to `1`. Should comfortably fit inside
        /// your orchestrator's `terminationGracePeriodSeconds`.
        #[arg(long, env = "GIGASTT_SHUTDOWN_DRAIN_SECS", default_value_t = 10)]
        shutdown_drain_secs: u64,

        /// Pool checkout timeout (seconds). REST and WebSocket handlers wait
        /// this long for a free session triplet before returning 503.
        #[arg(long, env = "GIGASTT_POOL_CHECKOUT_TIMEOUT_SECS", default_value_t = 30)]
        pool_checkout_timeout_secs: u64,

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
    gigastt::quantize::quantize_model(&input, &int8_path)?;
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
            max_session_secs,
            shutdown_drain_secs,
            pool_checkout_timeout_secs,
            skip_quantize,
            trust_proxy,
        } => {
            ensure_bind_allowed(&host, bind_all)?;
            model::ensure_model(&model_dir).await?;
            ensure_int8_encoder(&model_dir, skip_quantize)?;
            let engine = inference::Engine::load_with_pool_size(&model_dir, pool_size)?;
            log_rss();
            let config = ServerConfig {
                port,
                host,
                origin_policy: OriginPolicy {
                    allow_any: cors_allow_any,
                    allowed_origins: allow_origin,
                },
                limits: RuntimeLimits {
                    idle_timeout_secs,
                    ws_frame_max_bytes,
                    body_limit_bytes,
                    rate_limit_per_minute,
                    rate_limit_burst,
                    max_session_secs,
                    shutdown_drain_secs,
                    pool_checkout_timeout_secs,
                },
                metrics_enabled: metrics,
                trust_proxy,
            };
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
            gigastt::quantize::quantize_model(&input, &output)?;
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
        // Temporarily strip any env opt-in that might exist on the runner.
        // SAFETY: single-threaded test harness inside this fn body; env mutation is fine.
        let previous = std::env::var("GIGASTT_ALLOW_BIND_ANY").ok();
        // SAFETY: tests are run sequentially within this module — transient env mutation.
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
}
