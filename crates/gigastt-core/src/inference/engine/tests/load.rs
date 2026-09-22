use super::*;
use crate::inference::EndpointMode;

#[test]
fn test_split_pool_routes_items_to_two_pools() {
    // Exercises the real split underlying `split_triplets` with a synthetic
    // `Pool<u32>` (no model). 4 items, batch 1 → interactive 3, batch 1.
    let (pool, batch) = Engine::split_pool(vec![1u32, 2, 3, 4], 1);
    assert_eq!(pool.total(), 3);
    assert_eq!(batch.as_ref().map(|b| b.total()), Some(1));

    // batch_pool_size 0 → split disabled, no batch pool.
    let (pool, batch) = Engine::split_pool(vec![1u32, 2, 3, 4], 0);
    assert_eq!(pool.total(), 4);
    assert!(batch.is_none());

    // Over-request clamps so at least one triplet stays interactive.
    let (pool, batch) = Engine::split_pool(vec![1u32, 2, 3], 9);
    assert_eq!(pool.total(), 1);
    assert_eq!(batch.as_ref().map(|b| b.total()), Some(2));

    // A single item can't be split.
    let (pool, batch) = Engine::split_pool(vec![1u32], 1);
    assert_eq!(pool.total(), 1);
    assert!(batch.is_none());
}

#[test]
fn test_engine_load_missing_dir() {
    let result = Engine::load_with_pool_size("/nonexistent/path/for/tests", 1);
    assert!(matches!(result, Err(GigasttError::ModelLoad { .. })));
}

#[test]
fn test_exact_execution_provider_missing_from_build_fails_load() {
    let dir = tempfile::tempdir().unwrap();
    crate::test_support::write_rnnt_layout(dir.path()).unwrap();
    let err = Engine::load_with_execution_provider(
        dir.path().to_str().unwrap(),
        None,
        1,
        1,
        0,
        1,
        None,
        crate::ExecutionProviderChoice::Coreml,
    );
    if cfg!(feature = "coreml") {
        // The provider is compiled in. This fixture is not a real CoreML
        // model, so load may fail later for a different reason. The
        // selection itself must not reject the request.
        if let Err(e) = &err {
            let msg = std::error::Error::source(e)
                .map(|s| s.to_string())
                .unwrap_or_default();
            assert!(
                !msg.contains("does not include it"),
                "compiled coreml must not be reported missing: {msg}"
            );
        }
    } else {
        let e = match err {
            Ok(_) => panic!("exact coreml on a CPU build must fail"),
            Err(e) => e,
        };
        let msg = std::error::Error::source(&e)
            .map(|s| s.to_string())
            .unwrap_or_default();
        assert!(msg.contains("coreml"), "{msg}");
    }
}

#[test]
fn test_mock_engine_reports_cpu_provider() {
    let (engine, _tmp) = crate::test_support::rnnt_engine();
    assert_eq!(engine.execution_provider(), "cpu");
}

#[test]
fn test_engine_load_empty_dir() {
    let dir = tempfile::tempdir().unwrap();
    let result = Engine::load_with_pool_size(dir.path().to_str().unwrap(), 1);
    assert!(matches!(result, Err(GigasttError::ModelLoad { .. })));
}

#[test]
fn test_speech_endpoint_policy_matrix() {
    // Auto: blank without VAD, or VAD fire.
    assert!(Engine::speech_endpoint(
        EndpointMode::Auto,
        true,
        false,
        false
    ));
    assert!(Engine::speech_endpoint(
        EndpointMode::Auto,
        false,
        true,
        true
    ));
    assert!(!Engine::speech_endpoint(
        EndpointMode::Auto,
        true,
        false,
        true
    )); // blank ignored w/ VAD
    // Assistant: only VAD.
    assert!(!Engine::speech_endpoint(
        EndpointMode::Assistant,
        true,
        false,
        false
    ));
    assert!(Engine::speech_endpoint(
        EndpointMode::Assistant,
        false,
        true,
        true
    ));
    // Manual: never auto.
    assert!(!Engine::speech_endpoint(
        EndpointMode::Manual,
        true,
        true,
        true
    ));
}

#[test]
#[ignore = "requires model"]
fn test_warmup_runs_silent_inference_on_every_triplet() {
    let engine = Engine::load_with_pool_size(&crate::model::default_model_dir(), 2)
        .expect("engine should load");
    engine
        .warmup()
        .expect("warmup must succeed on a working engine");
    assert_eq!(
        engine.pool.available(),
        engine.pool.total(),
        "every triplet must be returned to the pool after warmup"
    );
}

// ---- Model-backed coverage (process_chunk / transcribe / state) --------
//
// These need the GigaAM model on disk; CI / coverage runs them with
// `--include-ignored`. They exercise the real streaming + file-decode
// branches (empty input, sub-stride, sub-N_FFT, full decode + slide,
