// Installed unchanged into each revision's engine tests. No production behavior changes.
use super::*;
use std::time::Instant;

#[test]
#[ignore = "run with scripts/asr-performance.py"]
#[allow(clippy::assertions_on_constants)]
fn test_asr_revision_probe() {
    assert!(!cfg!(debug_assertions), "ASR timing requires release mode");
    let model = std::env::var("GIGASTT_ASR_PROBE_MODEL").unwrap();
    let head: ModelVariant = std::env::var("GIGASTT_ASR_PROBE_HEAD")
        .unwrap()
        .parse()
        .unwrap();
    let fixtures = PathBuf::from(std::env::var("GIGASTT_ASR_PROBE_AUDIO").unwrap());
    let cache = PathBuf::from(std::env::var("GIGASTT_ASR_PROBE_CACHE").unwrap());
    assert!(!cache.exists(), "each process needs an empty graph cache");
    let threads = std::thread::available_parallelism().unwrap().get().min(2);
    let started = Instant::now();
    let engine = Engine::load_with_execution_provider(
        &model,
        Some(head),
        1,
        1,
        0,
        threads,
        Some(cache),
        crate::runtime::ExecutionProviderChoice::Cpu,
    )
    .unwrap();
    let load_seconds = started.elapsed().as_secs_f64();
    assert_eq!(engine.pool.total(), 1);
    let mut guard = engine.pool.checkout_blocking().unwrap();
    let started = Instant::now();
    let first = engine
        .transcribe_file(fixtures.join("golos_00.wav").to_str().unwrap(), &mut guard)
        .unwrap();
    let first_seconds = started.elapsed().as_secs_f64();
    assert!(!first.text.trim().is_empty());
    let mut cases = serde_json::Map::new();
    for name in ["golos_00.wav", "golos_01.wav", "golos_02.wav", "long.wav"] {
        let path = fixtures.join(name);
        let reference = engine
            .transcribe_file(path.to_str().unwrap(), &mut guard)
            .unwrap();
        assert!(
            !reference.text.trim().is_empty(),
            "empty transcript: {name}"
        );
        // Amortize scheduler jitter on the sub-100ms clips without relaxing the gate.
        let iterations = if name == "long.wav" { 1 } else { 10 };
        let started = Instant::now();
        for _ in 0..iterations {
            let result = engine
                .transcribe_file(path.to_str().unwrap(), &mut guard)
                .unwrap();
            assert_eq!(result.text, reference.text, "unstable transcript: {name}");
            assert_eq!(result.duration_s, reference.duration_s);
        }
        let seconds = started.elapsed().as_secs_f64() / f64::from(iterations);
        cases.insert(
            name.into(),
            serde_json::json!({
                "seconds": seconds, "iterations": iterations,
                "duration": reference.duration_s, "text": reference.text,
            }),
        );
    }
    let status = std::fs::read_to_string("/proc/self/status").unwrap();
    let kib = |key: &str| -> u64 {
        status
            .lines()
            .find_map(|s| s.strip_prefix(key))
            .unwrap()
            .split_whitespace()
            .next()
            .unwrap()
            .parse()
            .unwrap()
    };
    eprintln!(
        "asr revision {}",
        serde_json::json!({
            "load_seconds": load_seconds, "first_seconds": first_seconds,
            "first_text": first.text, "cases": cases, "provider": engine.execution_provider(),
            "pool_size": engine.pool.total(), "threads": threads,
            "cpu_affinity": status.lines().find_map(|s| s.strip_prefix("Cpus_allowed_list:")).unwrap().trim(),
            "rss_kib": kib("VmRSS:"), "peak_rss_kib": kib("VmHWM:"),
        })
    );
}
