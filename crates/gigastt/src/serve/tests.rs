use super::bind::is_loopback_host;
use super::{build_limits, build_server_config, ensure_bind_allowed, ensure_metrics_bind_allowed};
use gigastt::server::RuntimeLimits;

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
fn test_ensure_bind_allowed_explicit_flag_ok() {
    ensure_bind_allowed("0.0.0.0", true).expect("explicit --bind-all must pass");
}

#[test]
fn test_ensure_metrics_bind_allowed_disabled_skips_gate() {
    // Metrics off: no listener is bound, so even a wildcard address needs
    // no consent.
    let addr = "0.0.0.0:9090".parse().unwrap();
    ensure_metrics_bind_allowed(false, &addr, false)
        .expect("disabled metrics listener must skip the gate");
}

#[test]
fn test_is_loopback_host_ipv6_bracketed() {
    assert!(is_loopback_host("[::1]"));
    assert!(!is_loopback_host("[2001:db8::1]"));
}

#[test]
fn test_build_limits_defaults_when_no_config() {
    let limits = build_limits(
        None, None, None, None, None, None, None, None, None, None, None, None, None, None, None,
    )
    .unwrap();
    assert_eq!(limits.idle_timeout_secs, 300);
    assert_eq!(limits.ws_frame_max_bytes, 512 * 1024);
}

#[test]
fn test_build_limits_job_overrides() {
    let limits = build_limits(
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        Some(true),
        Some(7200),
        Some(50),
        Some(2 * 1024 * 1024),
        Some(5),
    )
    .unwrap();
    assert!(limits.jobs_enabled);
    assert_eq!(limits.jobs_ttl_secs, 7200);
    assert_eq!(limits.jobs_max, 50);
    assert_eq!(limits.jobs_max_bytes, 2 * 1024 * 1024);
    assert_eq!(limits.jobs_retry, 5);
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
        None,
        None,
        None,
        None,
        None,
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
        None,
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
        None,
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
        2,
    );
    assert_eq!(cfg.port, 1234);
    assert_eq!(cfg.metrics_listen.port(), 9099);
    assert_eq!(cfg.host, "127.0.0.1");
    assert_eq!(cfg.batch_pool_size, 2);
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

// Serialize tests that mutate process env vars to avoid races under
// cargo test's default multi-threaded harness (used by tarpaulin).
static ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

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
fn test_ensure_metrics_bind_allowed_loopback_ok() {
    let _guard = ENV_LOCK.lock().unwrap();
    let addr = "127.0.0.1:9090".parse().unwrap();
    ensure_metrics_bind_allowed(true, &addr, false).expect("loopback metrics bind must be allowed");
}

#[test]
fn test_ensure_metrics_bind_allowed_non_loopback_requires_flag() {
    let _guard = ENV_LOCK.lock().unwrap();
    let previous = std::env::var("GIGASTT_ALLOW_BIND_ANY").ok();
    unsafe {
        std::env::remove_var("GIGASTT_ALLOW_BIND_ANY");
    }
    let addr = "0.0.0.0:9090".parse().unwrap();
    let result = ensure_metrics_bind_allowed(true, &addr, false);
    if let Some(v) = previous {
        unsafe {
            std::env::set_var("GIGASTT_ALLOW_BIND_ANY", v);
        }
    }
    assert!(
        result.is_err(),
        "0.0.0.0 metrics bind without --bind-all must be rejected"
    );
}

#[test]
fn test_ensure_metrics_bind_allowed_explicit_flag_ok() {
    let _guard = ENV_LOCK.lock().unwrap();
    let addr = "0.0.0.0:9090".parse().unwrap();
    ensure_metrics_bind_allowed(true, &addr, true)
        .expect("explicit --bind-all must allow the metrics bind");
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

fn ctc_recipe() -> gigastt::boot::EngineRecipe {
    use gigastt::boot::{EngineRecipe, ItnMode, PunctuationMode};
    let mut r = EngineRecipe::offline(
        "/primary".into(),
        Some(gigastt_core::model::ModelVariant::Rnnt),
        PunctuationMode::Off,
        "/punct".into(),
        ItnMode::Off,
        None,
        false,
        None,
        false,
        None,
        None,
        "/vad".into(),
        Some(1),
        1,
    );
    r.execution_provider = gigastt_core::ExecutionProviderChoice::Cpu;
    r.endpoint_mode = Some("auto".into());
    r.stream_stable_prefix = true;
    r
}

#[test]
fn test_stream_ctc_envelope_and_default_off() {
    let r = ctc_recipe();
    assert!(super::validate_stream_ctc_recipe(&r, Some("/ctc")).is_ok());
    let mut invalid = Vec::new();
    let mut v = r.clone();
    v.pool_size = 2;
    invalid.push(v);
    let mut v = r.clone();
    v.pool_min_size = 0;
    invalid.push(v);
    let mut v = r.clone();
    v.batch_pool_size = 1;
    invalid.push(v);
    let mut v = r.clone();
    v.encoder_intra_threads = None;
    invalid.push(v);
    let mut v = r.clone();
    v.encoder_intra_threads = Some(2);
    invalid.push(v);
    let mut v = r.clone();
    v.model_variant = None;
    invalid.push(v);
    let mut v = r.clone();
    v.model_variant = Some(gigastt_core::model::ModelVariant::MlCtcLarge);
    invalid.push(v);
    let mut v = r.clone();
    v.execution_provider = gigastt_core::ExecutionProviderChoice::Auto;
    invalid.push(v);
    let mut v = r.clone();
    v.vad = true;
    invalid.push(v);
    let mut v = r.clone();
    v.stream_stable_prefix = false;
    invalid.push(v);
    let mut v = r.clone();
    v.endpoint_mode = Some("assistant".into());
    invalid.push(v);
    let mut v = r.clone();
    v.file_window_concurrency = 2;
    invalid.push(v);
    for r in invalid {
        assert!(super::validate_stream_ctc_recipe(&r, Some("/ctc")).is_err());
        assert!(super::validate_stream_ctc_recipe(&r, None).is_ok());
    }
    assert!(super::validate_stream_ctc_recipe(&r, Some("")).is_err());
}

#[test]
fn test_stream_ctc_boot_and_reload_fail_closed_on_missing_sidecar() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("not-downloaded");
    let path = path.to_str().unwrap();
    let builder = super::make_engine_builder(ctc_recipe(), Some(path.into()));
    let reload = builder.clone();
    for build in [builder, reload] {
        let error = match build() {
            Ok(_) => panic!("missing model accepted"),
            Err(e) => e,
        };
        assert!(matches!(
            error.downcast_ref::<gigastt_core::error::GigasttError>(),
            Some(gigastt_core::error::GigasttError::ModelLoad { .. })
        ));
        assert!(
            !std::path::Path::new(path).exists(),
            "must not download/create missing model directory"
        );
    }
}
