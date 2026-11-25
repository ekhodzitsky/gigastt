// Installed only in the temporary worktrees of the optional cached-load pilot.
use super::*;
use std::path::PathBuf;
use std::time::Instant;

const REPEATED_LOADS: usize = 5;

fn cache_hit(package: &std::path::Path) -> bool {
    super::cache::cached_model_path(package).is_dir()
        && super::cache::meta_matches(
            &super::cache::cached_meta_path(package),
            &super::cache::current_source_key(package).unwrap(),
        )
}

#[test]
#[ignore = "run with scripts/compare-objc-cached-load.py on macOS"]
#[allow(clippy::assertions_on_constants)]
fn test_cached_load_pilot() {
    assert!(!cfg!(debug_assertions), "release probe required");
    let package = PathBuf::from(std::env::var_os("GIGASTT_OBJC_PROBE_MODEL").unwrap());
    assert!(package.is_dir(), "verified model package missing");
    let phase = std::env::var("GIGASTT_OBJC_PILOT_PHASE").unwrap();
    let result = match phase.as_str() {
        "prime" => {
            let started = Instant::now();
            let model = compile_and_load(&package, false).unwrap();
            let seconds = started.elapsed().as_secs_f64();
            drop(model);
            assert!(cache_hit(&package), "prime did not populate the cache");
            serde_json::json!({"prime_seconds": seconds})
        }
        "cpu" | "cpu_and_ne" => {
            // Do not call current_source_key before the first timed load: that
            // would warm the key path and change what this sample measures.
            assert!(super::cache::cached_model_path(&package).is_dir());
            let mut timings = Vec::new();
            for _ in 0..=REPEATED_LOADS {
                let seconds = objc2::rc::autoreleasepool(|_| {
                    let started = Instant::now();
                    let model = compile_and_load(&package, phase == "cpu_and_ne").unwrap();
                    let seconds = started.elapsed().as_secs_f64();
                    drop(model);
                    seconds
                });
                timings.push(seconds);
            }
            assert!(cache_hit(&package), "cache key changed");
            serde_json::json!({"load_seconds": timings})
        }
        "key" => {
            // Separate process; do not subtract these from the load samples.
            let started = Instant::now();
            let key = super::cache::current_source_key(&package).unwrap();
            let key_seconds = started.elapsed().as_secs_f64();
            let started = Instant::now();
            assert!(super::cache::meta_matches(
                &super::cache::cached_meta_path(&package),
                &key,
            ));
            let meta_seconds = started.elapsed().as_secs_f64();
            serde_json::json!({"key_seconds": key_seconds, "meta_seconds": meta_seconds})
        }
        phase => panic!("unknown pilot phase: {phase}"),
    };
    println!("objc cached load {result}");
}
