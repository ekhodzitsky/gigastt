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
        maybe_load_punctuator(PunctuationMode::Off, "/nonexistent", ModelVariant::Rnnt).is_none()
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
fn test_ensure_int8_encoder_skip_flag_rejects_missing_int8() {
    let tmp = tempfile::tempdir().unwrap();
    let err =
        ensure_int8_encoder(ModelVariant::Rnnt, tmp.path().to_str().unwrap(), true).unwrap_err();
    let msg = format!("{err}");
    assert!(
        msg.contains("INT8") || msg.contains("int8"),
        "skip without INT8 must error (no FP32 fallback): {msg}"
    );
}

#[test]
fn test_ensure_int8_encoder_missing_input() {
    let tmp = tempfile::tempdir().unwrap();
    let err =
        ensure_int8_encoder(ModelVariant::Rnnt, tmp.path().to_str().unwrap(), false).unwrap_err();
    let msg = format!("{err}");
    assert!(msg.contains("Cannot quantize"), "unexpected error: {msg}");
}

#[cfg(feature = "quantize")]
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

#[cfg(not(feature = "quantize"))]
#[test]
fn test_ensure_int8_encoder_without_quantizer_preserves_fp32() {
    let tmp = tempfile::tempdir().unwrap();
    let fp32 = tmp.path().join("v3_rnnt_encoder.onnx");
    std::fs::write(&fp32, b"source").unwrap();
    let err =
        ensure_int8_encoder(ModelVariant::Rnnt, tmp.path().to_str().unwrap(), false).unwrap_err();
    assert!(
        err.to_string()
            .contains("does not include the quantize feature")
    );
    assert_eq!(std::fs::read(fp32).unwrap(), b"source");
    assert!(!tmp.path().join("v3_rnnt_encoder_int8.onnx").exists());
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
    assert_eq!(r.file_window_concurrency, 1);
    assert!(!r.quantize);
    assert!(r.endpoint_mode.is_none());
}

#[test]
fn test_engine_recipe_file_window_concurrency_clamps_zero() {
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
        1,
    )
    .with_file_window_concurrency(0);
    assert_eq!(r.file_window_concurrency, 1);
}
