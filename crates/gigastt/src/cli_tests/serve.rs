use super::*;

#[test]
fn test_cli_serve_parsing() {
    let cli = Cli::parse_from(["gigastt", "serve", "--port", "1234", "--bind-all"]);
    match cli.command {
        Commands::Serve(ServeArgs {
            port,
            bind_all,
            metrics,
            model_variant,
            ..
        }) => {
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
fn test_cli_serve_profile_edge_defaults_pool_and_vad_flags() {
    // Parsing only: profile field is Edge; runtime applies pool/vad in main.
    let cli = Cli::try_parse_from(["gigastt", "serve", "--profile", "edge"]).expect("parse");
    match cli.command {
        Commands::Serve(ServeArgs {
            profile,
            pool_size,
            vad,
            ..
        }) => {
            assert_eq!(profile, ServeProfile::Edge);
            // clap defaults before profile application:
            assert_eq!(pool_size, 2);
            assert!(!vad);
        }
        _ => panic!("expected Serve"),
    }
}

#[test]
fn test_cli_serve_optimized_cache_dir_flag_and_env() {
    let _guard = ENV_LOCK.lock().unwrap();
    let _restore = EnvRestore(
        "GIGASTT_OPTIMIZED_CACHE_DIR",
        std::env::var("GIGASTT_OPTIMIZED_CACHE_DIR").ok(),
    );
    unsafe {
        std::env::remove_var("GIGASTT_OPTIMIZED_CACHE_DIR");
    }
    // Unset → None: the engine keeps `<model-dir>/optimized_cache`.
    let cli = Cli::parse_from(["gigastt", "serve"]);
    match cli.command {
        Commands::Serve(ServeArgs {
            optimized_cache_dir,
            ..
        }) => assert_eq!(optimized_cache_dir, None),
        _ => panic!("expected Serve"),
    }
    // Explicit flag wins.
    let cli = Cli::parse_from([
        "gigastt",
        "serve",
        "--optimized-cache-dir",
        "/var/cache/gigastt",
    ]);
    match cli.command {
        Commands::Serve(ServeArgs {
            optimized_cache_dir,
            ..
        }) => assert_eq!(optimized_cache_dir.as_deref(), Some("/var/cache/gigastt")),
        _ => panic!("expected Serve"),
    }
    // Env var fallback.
    unsafe {
        std::env::set_var("GIGASTT_OPTIMIZED_CACHE_DIR", "/tmp/gigastt-cache");
    }
    let cli = Cli::parse_from(["gigastt", "serve"]);
    match cli.command {
        Commands::Serve(ServeArgs {
            optimized_cache_dir,
            ..
        }) => assert_eq!(optimized_cache_dir.as_deref(), Some("/tmp/gigastt-cache")),
        _ => panic!("expected Serve"),
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
        Commands::Serve(ServeArgs {
            encoder_intra_threads,
            ..
        }) => assert_eq!(encoder_intra_threads, None),
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
        Commands::Serve(ServeArgs {
            encoder_intra_threads,
            ..
        }) => assert_eq!(encoder_intra_threads, Some(4)),
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
        Commands::Serve(ServeArgs {
            encoder_intra_threads,
            ..
        }) => assert_eq!(encoder_intra_threads, Some(6)),
        _ => panic!("expected Serve"),
    }
}

#[test]
fn test_cli_serve_file_window_concurrency_default() {
    let _guard = ENV_LOCK.lock().unwrap();
    let _restore = EnvRestore(
        "GIGASTT_FILE_WINDOW_CONCURRENCY",
        std::env::var("GIGASTT_FILE_WINDOW_CONCURRENCY").ok(),
    );
    unsafe {
        std::env::remove_var("GIGASTT_FILE_WINDOW_CONCURRENCY");
    }
    let cli = Cli::parse_from(["gigastt", "serve"]);
    match cli.command {
        Commands::Serve(ServeArgs {
            file_window_concurrency,
            ..
        }) => assert_eq!(file_window_concurrency, 1),
        _ => panic!("expected Serve"),
    }
}

#[test]
fn test_cli_serve_file_window_concurrency_flag() {
    let _guard = ENV_LOCK.lock().unwrap();
    let _restore = EnvRestore(
        "GIGASTT_FILE_WINDOW_CONCURRENCY",
        std::env::var("GIGASTT_FILE_WINDOW_CONCURRENCY").ok(),
    );
    unsafe {
        std::env::remove_var("GIGASTT_FILE_WINDOW_CONCURRENCY");
    }
    let cli = Cli::parse_from(["gigastt", "serve", "--file-window-concurrency", "2"]);
    match cli.command {
        Commands::Serve(ServeArgs {
            file_window_concurrency,
            ..
        }) => assert_eq!(file_window_concurrency, 2),
        _ => panic!("expected Serve"),
    }
}

#[test]
fn test_cli_serve_model_variant_override() {
    let cli = Cli::parse_from(["gigastt", "serve", "--model-variant", "e2e_rnnt"]);
    match cli.command {
        Commands::Serve(ServeArgs { model_variant, .. }) => {
            assert_eq!(model_variant, Some(ModelVariant::E2eRnnt));
        }
        _ => panic!("expected Serve"),
    }
}

#[test]
fn test_cli_serve_model_variant_explicit_rnnt() {
    let cli = Cli::parse_from(["gigastt", "serve", "--model-variant", "rnnt"]);
    match cli.command {
        Commands::Serve(ServeArgs { model_variant, .. }) => {
            assert_eq!(model_variant, Some(ModelVariant::Rnnt));
        }
        _ => panic!("expected Serve"),
    }
}

#[test]
fn test_cli_serve_rejects_unknown_model_variant() {
    let res = Cli::try_parse_from(["gigastt", "serve", "--model-variant", "whisper"]);
    assert!(res.is_err(), "unknown variant must be rejected by clap");
}

#[test]
fn test_cli_serve_punctuation_defaults_auto() {
    let cli = Cli::parse_from(["gigastt", "serve"]);
    match cli.command {
        Commands::Serve(ServeArgs {
            punctuation,
            punct_model_dir,
            ..
        }) => {
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
        Commands::Serve(ServeArgs {
            punctuation,
            punct_model_dir,
            ..
        }) => {
            assert_eq!(punctuation, PunctuationMode::On);
            assert_eq!(punct_model_dir, "/tmp/punct");
        }
        _ => panic!("expected Serve"),
    }
}

#[test]
fn test_cli_serve_itn_defaults_auto() {
    let cli = Cli::parse_from(["gigastt", "serve"]);
    match cli.command {
        Commands::Serve(ServeArgs { itn, .. }) => assert_eq!(itn, ItnMode::Auto),
        _ => panic!("expected Serve"),
    }
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
        Commands::Serve(ServeArgs {
            hotwords_file,
            hotwords_default,
            hotwords_boost,
            ..
        }) => {
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
        Commands::Serve(ServeArgs {
            hotwords_file,
            hotwords_default,
            hotwords_boost,
            ..
        }) => {
            assert_eq!(hotwords_file, None);
            assert!(!hotwords_default);
            assert_eq!(hotwords_boost, None);
        }
        _ => panic!("expected Serve"),
    }
}

#[test]
fn test_cli_serve_with_metrics() {
    let cli = Cli::parse_from(["gigastt", "serve", "--metrics"]);
    match cli.command {
        Commands::Serve(ServeArgs {
            metrics,
            metrics_listen,
            ..
        }) => {
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
        Commands::Serve(ServeArgs { metrics_listen, .. }) => {
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
fn test_cli_serve_jobs_flags() {
    let cli = Cli::parse_from([
        "gigastt",
        "serve",
        "--enable-jobs",
        "--jobs-ttl-secs",
        "7200",
        "--jobs-max",
        "50",
        "--jobs-max-bytes",
        "1048576",
        "--jobs-retry",
        "5",
    ]);
    match cli.command {
        Commands::Serve(ServeArgs {
            enable_jobs,
            jobs_ttl_secs,
            jobs_max,
            jobs_max_bytes,
            jobs_retry,
            ..
        }) => {
            assert!(enable_jobs);
            assert_eq!(jobs_ttl_secs, Some(7200));
            assert_eq!(jobs_max, Some(50));
            assert_eq!(jobs_max_bytes, Some(1048576));
            assert_eq!(jobs_retry, Some(5));
        }
        _ => panic!("expected Serve"),
    }
}

#[test]
fn test_cli_serve_jobs_defaults_off() {
    let cli = Cli::parse_from(["gigastt", "serve"]);
    match cli.command {
        Commands::Serve(ServeArgs {
            enable_jobs,
            jobs_ttl_secs,
            jobs_max,
            jobs_retry,
            ..
        }) => {
            assert!(!enable_jobs);
            assert_eq!(jobs_ttl_secs, None);
            assert_eq!(jobs_max, None);
            assert_eq!(jobs_retry, None);
        }
        _ => panic!("expected Serve"),
    }
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
        Commands::Serve(ServeArgs {
            vad,
            vad_threshold,
            vad_min_silence_ms,
            vad_model_dir,
            ..
        }) => {
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
        Commands::Serve(ServeArgs {
            vad,
            vad_threshold,
            vad_min_silence_ms,
            endpoint_mode,
            ..
        }) => {
            assert!(!vad);
            assert_eq!(vad_threshold, None);
            assert_eq!(vad_min_silence_ms, None);
            assert_eq!(endpoint_mode, "auto");
        }
        _ => panic!("expected Serve"),
    }
}

#[test]
fn test_cli_serve_endpoint_mode_assistant() {
    let _guard = ENV_LOCK.lock().unwrap();
    let _restore = EnvRestore(
        "GIGASTT_ENDPOINT_MODE",
        std::env::var("GIGASTT_ENDPOINT_MODE").ok(),
    );
    unsafe {
        std::env::remove_var("GIGASTT_ENDPOINT_MODE");
    }
    let cli = Cli::parse_from(["gigastt", "serve", "--endpoint-mode", "assistant"]);
    match cli.command {
        Commands::Serve(ServeArgs { endpoint_mode, .. }) => {
            assert_eq!(endpoint_mode, "assistant");
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
        Commands::Serve(ServeArgs {
            pool_size,
            pool_min_size,
            batch_pool_size,
            ..
        }) => {
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
        Commands::Serve(ServeArgs {
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
        }) => {
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
fn test_cli_serve_config_flag() {
    let cli = Cli::parse_from(["gigastt", "serve", "--config", "/tmp/limits.toml"]);
    match cli.command {
        Commands::Serve(ServeArgs { config, .. }) => {
            assert_eq!(config, Some("/tmp/limits.toml".to_string()));
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
        Commands::Serve(ServeArgs {
            allow_origin,
            cors_allow_any,
            ..
        }) => {
            assert_eq!(allow_origin.len(), 2);
            assert_eq!(allow_origin[0], "https://a.example.com");
            assert!(cors_allow_any);
        }
        _ => panic!("expected Serve"),
    }
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
