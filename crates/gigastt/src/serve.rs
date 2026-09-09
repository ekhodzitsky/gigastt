//! `gigastt serve` flag bag, bind gates, and boot path.
//!
//! Split out of `main.rs` so the clap struct and its runtime mapping stay
//! together without bloating the CLI dispatcher.

use anyhow::Context;
use clap::parser::ValueSource;
use clap::{Args, ValueEnum};
use gigastt::boot::{
    EngineRecipe, ItnMode, PunctuationMode, parse_itn_mode, parse_punctuation_mode,
};
use gigastt::server;
use gigastt::server::{OriginPolicy, RuntimeLimits, ServerConfig};
use gigastt_core::model;
use gigastt_core::model::ModelVariant;

// `Serve` carries many optional CLI flags, so it is much larger than the other
// variants. The enum is parsed once at startup and never stored in bulk, so
// boxing the fields would only hurt readability.

/// Optional deploy profile for `serve`. `Edge` applies weak-host defaults
/// (`--pool-size 1`, `--vad`) only when the operator did not set those flags
/// explicitly.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, ValueEnum)]
pub(crate) enum ServeProfile {
    /// Stock defaults (pool=2, VAD off unless `--vad`).
    #[default]
    Default,
    /// Low-RAM / single-stream hosts: pool-size 1 + VAD on (unless overridden).
    Edge,
}

/// CLI arguments for `gigastt serve`.
///
/// Extracted so `run_serve` takes a typed flag bag and compile-time
/// exhaustiveness catches forgotten mappings.
#[derive(Args)]
pub(crate) struct ServeArgs {
    /// Port to listen on
    #[arg(short, long, default_value_t = 9876)]
    pub(crate) port: u16,

    /// Bind address. Loopback by default; non-loopback requires `--bind-all`.
    #[arg(long, default_value = "127.0.0.1")]
    pub(crate) host: String,

    /// Model directory
    #[arg(long, default_value_t = model::default_model_dir())]
    pub(crate) model_dir: String,

    /// Deploy profile: `default` (stock) or `edge` (pool-size 1 + VAD when
    /// those flags are left at defaults). Explicit `--pool-size` / `--vad`
    /// always win. Env: GIGASTT_PROFILE.
    #[arg(long, env = "GIGASTT_PROFILE", value_enum, default_value_t = ServeProfile::Default)]
    pub(crate) profile: ServeProfile,

    /// Recognition head to use. Omit to auto-detect from the model
    /// directory: if a model is already installed its variant is used as-is
    /// (no download). Only required when the directory is empty or you want
    /// to switch variants. `rnnt` (lower WER, bare lowercase), `e2e_rnnt`
    /// (punctuation / casing / ITN), or `ml_ctc` / `ml_ctc_large` (GigaAM
    /// Multilingual charwise CTC, 220M / 600M — ru/en/kk/ky/uz, bare
    /// lowercase). Env: GIGASTT_MODEL_VARIANT.
    #[arg(
            long,
            env = "GIGASTT_MODEL_VARIANT",
            value_parser = crate::parse_model_variant
        )]
    pub(crate) model_variant: Option<ModelVariant>,

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
    pub(crate) punctuation: PunctuationMode,

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
    pub(crate) punct_model_dir: String,

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
    pub(crate) itn: ItnMode,

    /// Contextual hotword biasing: path to a file of phrases to boost during
    /// recognition (one phrase per line, optional `\t<weight>` suffix; blank
    /// lines and `#` comments ignored). Off when unset. Env:
    /// GIGASTT_HOTWORDS_FILE.
    #[arg(long, env = "GIGASTT_HOTWORDS_FILE")]
    pub(crate) hotwords_file: Option<String>,

    /// Also bias the built-in Russian brand/acronym lexicon. Combined with
    /// any `--hotwords-file` phrases. Env: GIGASTT_HOTWORDS_DEFAULT.
    #[arg(long, env = "GIGASTT_HOTWORDS_DEFAULT", default_value_t = false)]
    pub(crate) hotwords_default: bool,

    /// Additive logit boost applied to hotword continuation tokens during
    /// greedy decode [default: 5.0]. Higher = stronger bias. No effect
    /// unless hotwords are configured. Env: GIGASTT_HOTWORDS_BOOST.
    #[arg(long, env = "GIGASTT_HOTWORDS_BOOST")]
    pub(crate) hotwords_boost: Option<f32>,

    /// Voice activity detection: skip silence in file transcription and
    /// finalize streaming segments on detected trailing silence. Off by
    /// default; downloads the Silero VAD model (MIT) on first use. Env:
    /// GIGASTT_VAD.
    #[arg(long, env = "GIGASTT_VAD", default_value_t = false)]
    pub(crate) vad: bool,

    /// VAD speech-probability threshold in [0,1] [default: 0.5]. Higher =
    /// stricter. No effect unless `--vad`. Env: GIGASTT_VAD_THRESHOLD.
    #[arg(long, env = "GIGASTT_VAD_THRESHOLD")]
    pub(crate) vad_threshold: Option<f32>,

    /// Minimum trailing silence (ms) to close a speech region / finalize a
    /// streaming segment [default: 500]. No effect unless `--vad`. Env:
    /// GIGASTT_VAD_MIN_SILENCE_MS.
    #[arg(long, env = "GIGASTT_VAD_MIN_SILENCE_MS")]
    pub(crate) vad_min_silence_ms: Option<u32>,

    /// Directory holding the Silero VAD model (`silero_vad.onnx`). Defaults
    /// to `~/.gigastt/models/vad/`. Auto-downloaded when `--vad` is set and
    /// the model is absent. Env: GIGASTT_VAD_MODEL_DIR.
    #[arg(long, env = "GIGASTT_VAD_MODEL_DIR", default_value_t = model::default_vad_model_dir())]
    pub(crate) vad_model_dir: String,

    /// Directory for the CPU encoder's ORT optimized-graph cache
    /// (`*_optimized.ort`). Defaults to `<model-dir>/optimized_cache`; set this
    /// (e.g. to a systemd `CacheDirectory`) when the model directory is
    /// read-only. An unwritable cache dir is skipped with a warning instead of
    /// failing boot. Only used by the CPU ort backend. Env:
    /// GIGASTT_OPTIMIZED_CACHE_DIR.
    #[arg(long, env = "GIGASTT_OPTIMIZED_CACHE_DIR", value_name = "PATH")]
    pub(crate) optimized_cache_dir: Option<String>,

    /// Streaming utterance-end policy for WebSocket sessions.
    /// `auto` (default): VAD silence if `--vad`, else decoder blank-run (~0.6 s).
    /// `assistant`: only VAD silence ends utterances (use with `--vad`); blank-run
    /// is ignored — preferred for voice-command clients like Irene.
    /// `manual`: only client `stop` ends utterances.
    /// The encoder window cap never emits `final` under any mode.
    /// Env: GIGASTT_ENDPOINT_MODE. Overridable per session via WS configure.
    #[arg(
            long,
            env = "GIGASTT_ENDPOINT_MODE",
            value_parser = ["auto", "assistant", "manual"],
            default_value = "auto"
        )]
    pub(crate) endpoint_mode: String,

    /// Max retained streaming encoder window in seconds (WebSocket / SSE).
    /// Re-decoding the whole window every 0.8 s stride bounds WER on phrases
    /// longer than the window: raising this (e.g. 7.5) improves long-phrase
    /// streaming WER at a linear per-stride encoder-cost increase. Clamped to
    /// 2.4–30. Env: GIGASTT_STREAM_MAX_WINDOW_SECS.
    #[arg(long, env = "GIGASTT_STREAM_MAX_WINDOW_SECS", default_value_t = 2.5)]
    pub(crate) stream_max_window_secs: f64,

    /// Stable-prefix commits at the streaming window cap: only the prefix two
    /// consecutive window hypotheses agree on becomes stable; the rest stays
    /// revisable. Reduces long-phrase streaming WER loss at slide boundaries
    /// without widening the window. Default on; opt out with
    /// `--stream-stable-prefix=false`. Env: GIGASTT_STREAM_STABLE_PREFIX.
    #[arg(long, env = "GIGASTT_STREAM_STABLE_PREFIX", default_value_t = true)]
    pub(crate) stream_stable_prefix: bool,

    /// Number of concurrent inference sessions. The INT8 encoder is
    /// memory-mapped and shared; budget **resident** footprint (~46 MB at
    /// `--pool-size 1`, ~66 MB at the default 2; ~20 MB per extra slot).
    /// `ps` RSS is higher (~277 / ~510 MB) because it counts the mapping.
    /// Edge / low-RAM: `--pool-size 1` (full cores for one job). Pool > 1
    /// can cost ~10–20% single-job RTF because encoder threads split across
    /// slots. CLI-only (no `GIGASTT_POOL_SIZE`). The server auto-caps by
    /// available RAM at load and logs a warning if it clamps.
    #[arg(long, default_value_t = 2)]
    pub(crate) pool_size: usize,

    /// Minimum session triplets that must load for the server to boot. When
    /// `1 <= min < pool_size` and some triplets fail (e.g. low memory), the
    /// server starts on a degraded pool with a warning instead of failing.
    /// Clamped to `1..=pool_size` [default: 1].
    #[arg(long, env = "GIGASTT_POOL_MIN_SIZE", default_value_t = 1)]
    pub(crate) pool_min_size: usize,

    /// Triplets reserved for batch REST file transcription, split off from
    /// `--pool-size` so a long file job can't starve WebSocket / SSE
    /// streaming. `0` disables the split (REST shares the interactive pool);
    /// clamped to leave at least one interactive triplet [default: 0].
    #[arg(long, env = "GIGASTT_BATCH_POOL_SIZE", default_value_t = 0)]
    pub(crate) batch_pool_size: usize,

    /// Enable the asynchronous `/v1/jobs` API for long-file and batch
    /// transcription. Off by default; when disabled the `/v1/jobs` routes
    /// are not registered and return 404. Env: GIGASTT_ENABLE_JOBS.
    #[arg(long, env = "GIGASTT_ENABLE_JOBS", default_value_t = false)]
    pub(crate) enable_jobs: bool,

    /// TTL in seconds for finished/failed/cancelled jobs before eviction
    /// from the in-memory store [default: 3600]. Env: GIGASTT_JOBS_TTL_SECS.
    #[arg(long, env = "GIGASTT_JOBS_TTL_SECS")]
    pub(crate) jobs_ttl_secs: Option<u64>,

    /// Maximum number of jobs kept in memory (queued + finished). When full,
    /// POST /v1/jobs returns 429 + Retry-After [default: 100].
    /// Env: GIGASTT_JOBS_MAX.
    #[arg(long, env = "GIGASTT_JOBS_MAX")]
    pub(crate) jobs_max: Option<usize>,

    /// Maximum total bytes of buffered job uploads kept in memory across the
    /// queue (queued + processing). Bounds RAM independently of --jobs-max,
    /// which counts jobs but not their size; a submission over budget gets
    /// 429 + Retry-After [default: 536870912 = 512 MiB].
    /// Env: GIGASTT_JOBS_MAX_BYTES.
    #[arg(long, env = "GIGASTT_JOBS_MAX_BYTES")]
    pub(crate) jobs_max_bytes: Option<usize>,

    /// Maximum retry attempts for a job that panics [default: 3].
    /// Env: GIGASTT_JOBS_RETRY.
    #[arg(long, env = "GIGASTT_JOBS_RETRY")]
    pub(crate) jobs_retry: Option<u32>,

    /// Intra-op thread count for the encoder session on the CPU build. The
    /// encoder dominates the single-utterance cost, so more threads speed up
    /// weak CPUs / long single-file jobs. When unset, defaults to the logical
    /// CPU count divided across the concurrently-running pool triplets
    /// (`pool_size + batch_pool_size`), so a default install uses every core.
    /// Do not set `1` on multi-core hosts unless debugging — it is ~3× slower
    /// than auto. An explicit value (flag or env, including `1`) is still
    /// honoured as-is for debug passthrough. The resolved value is auto-
    /// clamped so `pool_size * threads` can't exceed the logical CPU count.
    /// No effect on CoreML / CUDA builds.
    #[arg(long, env = "GIGASTT_ENCODER_INTRA_THREADS")]
    pub(crate) encoder_intra_threads: Option<usize>,

    /// Max pooled triplets one file transcription may hold to decode
    /// overlapping 24 s windows in parallel. Default `1` is the historical
    /// serial loop. `2` with `--pool-size 2` lets a long file use an idle
    /// extra slot (try-checkout, never waits — two long files each holding
    /// one slot stay serial). File path only; WebSocket is unchanged.
    /// Env: GIGASTT_FILE_WINDOW_CONCURRENCY.
    #[arg(long, env = "GIGASTT_FILE_WINDOW_CONCURRENCY", default_value_t = 1)]
    pub(crate) file_window_concurrency: usize,

    /// Explicitly acknowledge binding to a non-loopback address.
    /// Can also be enabled via `GIGASTT_ALLOW_BIND_ANY=1`.
    /// Without this flag the server refuses to listen on anything other than
    /// 127.0.0.1 / ::1 / localhost to prevent accidental public exposure.
    #[arg(long, default_value_t = false)]
    pub(crate) bind_all: bool,

    /// Additional Origin allowed to call the REST / WebSocket API (repeatable).
    /// Loopback origins (localhost, 127.0.0.1, ::1) are always allowed.
    /// Match is exact and case-insensitive, e.g. `https://app.example.com`.
    #[arg(long = "allow-origin", value_name = "URL")]
    pub(crate) allow_origin: Vec<String>,

    /// Echo `Access-Control-Allow-Origin: *` and accept any cross-origin
    /// caller. Disabled by default — every non-loopback Origin must be
    /// listed explicitly via `--allow-origin` unless this flag is set.
    #[arg(long, default_value_t = false)]
    pub(crate) cors_allow_any: bool,

    /// WebSocket idle timeout in seconds [default: 300].
    /// Server closes the connection when no frame arrives within this window.
    #[arg(long, env = "GIGASTT_IDLE_TIMEOUT_SECS")]
    pub(crate) idle_timeout_secs: Option<u64>,

    /// Maximum WebSocket frame / message size in bytes [default: 524288].
    #[arg(long, env = "GIGASTT_WS_FRAME_MAX_BYTES")]
    pub(crate) ws_frame_max_bytes: Option<usize>,

    /// Maximum REST request body size in bytes [default: 52428800].
    #[arg(long, env = "GIGASTT_BODY_LIMIT_BYTES")]
    pub(crate) body_limit_bytes: Option<usize>,

    /// Per-IP rate limit — requests per minute. 0 = off [default: 0].
    #[arg(long, env = "GIGASTT_RATE_LIMIT_PER_MINUTE")]
    pub(crate) rate_limit_per_minute: Option<u32>,

    /// Rate-limit burst size [default: 10].
    #[arg(long, env = "GIGASTT_RATE_LIMIT_BURST")]
    pub(crate) rate_limit_burst: Option<u32>,

    /// Expose Prometheus metrics. Off by default — keeps the server quiet
    /// for single-user installs. When on, `/metrics` is served on a
    /// separate loopback listener (see `--metrics-listen`), never on the
    /// primary port, so it is not gated by the CORS allowlist or limiter.
    #[arg(long, env = "GIGASTT_METRICS", default_value_t = false)]
    pub(crate) metrics: bool,

    /// Bind address for the separate Prometheus `/metrics` listener
    /// [default: 127.0.0.1:9090]. Loopback by default; expose it
    /// deliberately to a scraper. Only used when `--metrics` is set.
    #[arg(long, env = "GIGASTT_METRICS_LISTEN")]
    pub(crate) metrics_listen: Option<std::net::SocketAddr>,

    /// Maximum wall-clock duration of a single WebSocket session in seconds.
    /// 0 disables the cap (not recommended) [default: 3600].
    #[arg(long, env = "GIGASTT_MAX_SESSION_SECS")]
    pub(crate) max_session_secs: Option<u64>,

    /// Grace window in seconds after shutdown during which in-flight
    /// sessions may emit Final frames. 0 is clamped to 1 [default: 10].
    #[arg(long, env = "GIGASTT_SHUTDOWN_DRAIN_SECS")]
    pub(crate) shutdown_drain_secs: Option<u64>,

    /// Pool checkout timeout in seconds. Handlers wait this long for a
    /// free session triplet before returning 503 [default: 30].
    #[arg(long, env = "GIGASTT_POOL_CHECKOUT_TIMEOUT_SECS")]
    pub(crate) pool_checkout_timeout_secs: Option<u64>,

    /// Per-request inference timeout in seconds. A run exceeding this
    /// returns `inference_timeout`; `0` disables [default: 600].
    #[arg(long, env = "GIGASTT_INFERENCE_TIMEOUT_SECS")]
    pub(crate) inference_timeout_secs: Option<u64>,

    /// Maximum decoded audio length in seconds for file transcription.
    /// `0` (default) means no limit — a file of any length transcribes,
    /// since the default path decodes in bounded windows. Audio longer than
    /// a positive value is rejected with HTTP 413 `audio_too_long`. The
    /// whole-buffer paths (diarization, `channels=split`, telephony)
    /// keep a ~30-minute safety ceiling regardless [default: 0].
    #[arg(long, env = "GIGASTT_MAX_AUDIO_SECS")]
    pub(crate) max_audio_secs: Option<u64>,

    /// Trust `X-Forwarded-For` and `X-Real-IP` headers for rate-limit IP
    /// extraction. When enabled, the direct peer must be loopback, RFC1918,
    /// IPv6 unique-local, or IPv6 link-local; otherwise headers are ignored.
    #[arg(long, env = "GIGASTT_TRUST_PROXY", default_value_t = false)]
    pub(crate) trust_proxy: bool,

    /// Path to TOML config file for runtime limits (reloaded on SIGHUP)
    #[arg(long)]
    pub(crate) config: Option<String>,
}

mod bind;
pub(crate) use bind::{ensure_bind_allowed, ensure_metrics_bind_allowed};

/// Apply edge-profile defaults and boot the STT server.
pub(crate) async fn run_serve(
    matches: &clap::ArgMatches,
    mut args: ServeArgs,
) -> anyhow::Result<()> {
    // Edge profile fills weak-host defaults only when the operator left
    // the corresponding flags at clap defaults (explicit flags always win).
    if args.profile == ServeProfile::Edge
        && let Some(serve_m) = matches.subcommand_matches("serve")
    {
        if serve_m.value_source("pool_size") == Some(ValueSource::DefaultValue) {
            args.pool_size = 1;
        }
        if serve_m.value_source("vad") == Some(ValueSource::DefaultValue) {
            args.vad = true;
        }
        tracing::info!(
            pool_size = args.pool_size,
            vad = args.vad,
            "serve profile=edge (pool/vad defaults applied when unset)"
        );
    }
    ensure_bind_allowed(&args.host, args.bind_all)?;
    let mut limits = build_limits(
        args.config.as_deref(),
        args.idle_timeout_secs,
        args.ws_frame_max_bytes,
        args.body_limit_bytes,
        args.rate_limit_per_minute,
        args.rate_limit_burst,
        args.max_session_secs,
        args.shutdown_drain_secs,
        args.pool_checkout_timeout_secs,
        args.inference_timeout_secs,
        Some(args.enable_jobs),
        args.jobs_ttl_secs,
        args.jobs_max,
        args.jobs_max_bytes,
        args.jobs_retry,
    )?;
    // `--max-audio-secs` overrides the config-file value with the same
    // precedence as every other runtime limit; `0` (default) = unlimited.
    if let Some(v) = args.max_audio_secs {
        limits.max_audio_secs = v;
    }
    let metrics_listen = args
        .metrics_listen
        .unwrap_or_else(server::config::default_metrics_listen);
    ensure_metrics_bind_allowed(args.metrics, &metrics_listen, args.bind_all)?;
    let server_config = build_server_config(
        args.port,
        args.host,
        args.allow_origin,
        args.cors_allow_any,
        limits,
        args.metrics,
        metrics_listen,
        args.trust_proxy,
        args.config,
        args.batch_pool_size,
    );

    // Shared recipe: first-run boot and `POST /v1/admin/reload` both
    // build through `EngineRecipe::build_engine` so post-processors
    // (punct / ITN / VAD / hotwords / endpoint mode) stay identical.
    // Synchronous (ONNX session load, quantization) so it can run on a
    // blocking thread; it re-detects the on-disk variant so a reload
    // picks up a model swapped between boot and reload.
    let recipe = EngineRecipe {
        model_dir: args.model_dir,
        model_variant: args.model_variant,
        punctuation: args.punctuation,
        punct_model_dir: args.punct_model_dir,
        itn: args.itn,
        hotwords_file: args.hotwords_file,
        hotwords_default: args.hotwords_default,
        hotwords_boost: args.hotwords_boost,
        vad: args.vad,
        vad_threshold: args.vad_threshold,
        vad_min_silence_ms: args.vad_min_silence_ms,
        vad_model_dir: args.vad_model_dir,
        encoder_intra_threads: args.encoder_intra_threads,
        pool_size: args.pool_size,
        pool_min_size: args.pool_min_size,
        batch_pool_size: args.batch_pool_size,
        // INT8-only: never on-device-quantize from FP32 at serve time.
        quantize: false,
        skip_quantize: true,
        endpoint_mode: Some(args.endpoint_mode),
        stream_max_window_secs: Some(args.stream_max_window_secs),
        stream_stable_prefix: args.stream_stable_prefix,
        file_window_concurrency: args.file_window_concurrency.max(1),
        optimized_cache_dir: args.optimized_cache_dir,
    };
    let build_engine: server::EngineBuilder = {
        let recipe = recipe.clone();
        std::sync::Arc::new(move || recipe.build_engine())
    };

    // Build the engine in the background while a minimal bootstrap
    // responder serves /health (200) and /ready (503 initializing) on the
    // port, so probes / Docker HEALTHCHECK don't see connection-refused
    // during the first-run lean INT8 download. The heavy synchronous
    // work (ONNX session load, post-processor loads) runs on a blocking
    // thread so the bootstrap responder stays snappy.
    let boot_builder = build_engine.clone();
    let load = async move {
        let resolved = model::ensure_model_variant(recipe.model_variant, &recipe.model_dir).await?;
        recipe.ensure_side_assets(resolved).await;
        tokio::task::spawn_blocking(move || boot_builder())
            .await
            .context("engine load task panicked")?
    };
    server::run_with_config_loading_reloadable(server_config, None, load, Some(build_engine)).await
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn build_limits(
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
    jobs_enabled: Option<bool>,
    jobs_ttl_secs: Option<u64>,
    jobs_max: Option<usize>,
    jobs_max_bytes: Option<usize>,
    jobs_retry: Option<u32>,
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
    if let Some(v) = jobs_enabled {
        limits.jobs_enabled = v;
    }
    if let Some(v) = jobs_ttl_secs {
        limits.jobs_ttl_secs = v;
    }
    if let Some(v) = jobs_max {
        limits.jobs_max = v;
    }
    if let Some(v) = jobs_max_bytes {
        limits.jobs_max_bytes = v;
    }
    if let Some(v) = jobs_retry {
        limits.jobs_retry = v;
    }
    Ok(limits)
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn build_server_config(
    port: u16,
    host: String,
    allow_origin: Vec<String>,
    cors_allow_any: bool,
    limits: RuntimeLimits,
    metrics: bool,
    metrics_listen: std::net::SocketAddr,
    trust_proxy: bool,
    config: Option<String>,
    batch_pool_size: usize,
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
        batch_pool_size,
    }
}

#[cfg(test)]
mod tests;
