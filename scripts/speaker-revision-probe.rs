// Installed unchanged as a test-only child of diarization in both revisions.
// Exercise the production loader and pipelines, including their dependencies.
use super::*;
use sha2::Digest;
use std::time::Instant;

#[test]
#[ignore = "run with scripts/compare-speaker-revisions.py"]
#[allow(clippy::assertions_on_constants)]
fn test_speaker_revision_probe() {
    assert!(
        !cfg!(debug_assertions),
        "revision probe requires release mode"
    );
    for name in ["POLYVOICE_SESSION_POOL_SIZE", "POLYVOICE_INTRA_THREADS"] {
        assert!(std::env::var_os(name).is_none(), "unset {name}");
    }
    let hash = |path: &Path| -> String {
        sha2::Sha256::digest(std::fs::read(path).unwrap())
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect()
    };
    let model = PathBuf::from(crate::model::default_model_dir()).join("wespeaker_resnet34.onnx");
    // Reading the weights warms the filesystem cache equally for both versions.
    let model_sha256 = hash(&model);
    assert_eq!(
        model_sha256,
        "3955447b0499dc9e0a4541a895df08b03c69098eba4e56c02b5603e9f7f4fcbb"
    );
    let started = Instant::now();
    let encoder = Arc::new(load_speaker_encoder(&model, 4).unwrap());
    let load_seconds = started.elapsed().as_secs_f64();
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../gigastt/tests/fixtures");
    let mut fixture_sha256 = Vec::new();
    let mut clips = Vec::new();
    for name in ["golos_00.wav", "golos_01.wav", "golos_02.wav"] {
        let path = root.join(name);
        fixture_sha256.push(hash(&path));
        let samples = crate::inference::audio::decode_audio_file(path.to_str().unwrap()).unwrap();
        for len in [8000, 24000, 48000] {
            clips.push(samples[..len].to_vec());
        }
    }
    for clip in &clips {
        std::hint::black_box(encoder.embed(clip).unwrap());
    }
    let started = Instant::now();
    for clip in &clips {
        std::hint::black_box(encoder.embed(clip).unwrap());
    }
    let embedding_seconds = started.elapsed().as_secs_f64();
    let embeddings: Vec<_> = [1, 4, 7]
        .iter()
        .map(|&i| encoder.embed(&clips[i]).unwrap())
        .collect();
    let samples: Vec<_> = clips
        .iter()
        .flat_map(|c| c.iter().copied().chain(vec![0.0; 8000]))
        .collect();
    std::hint::black_box(run_offline(&encoder, &samples).unwrap());
    let started = Instant::now();
    let offline = run_offline(&encoder, &samples).unwrap();
    let offline_seconds = started.elapsed().as_secs_f64();
    let offline_turns: Vec<_> = offline
        .iter()
        .map(|t| (t.start, t.end, t.speaker))
        .collect();
    let stream = || {
        let mut pipeline = open_streaming(&encoder).unwrap();
        for chunk in samples.chunks(1600) {
            pipeline.feed(chunk).unwrap();
        }
        pipeline.flush().unwrap();
        pipeline.turns().to_vec()
    };
    std::hint::black_box(stream());
    let started = Instant::now();
    let streaming = stream();
    let streaming_seconds = started.elapsed().as_secs_f64();
    let streaming_turns: Vec<_> = streaming
        .iter()
        .map(|t| (t.time.start, t.time.end, t.speaker.0))
        .collect();
    let status = std::fs::read_to_string("/proc/self/status").unwrap();
    let kib = |name: &str| -> u64 {
        status
            .lines()
            .find_map(|line| line.strip_prefix(name))
            .unwrap()
            .split_whitespace()
            .next()
            .unwrap()
            .parse()
            .unwrap()
    };
    eprintln!(
        "speaker revision {}",
        serde_json::json!({
            "model_sha256": model_sha256, "fixture_sha256": fixture_sha256,
            "pool_size": 4, "logical_cpus": std::thread::available_parallelism().unwrap().get(),
            "load_seconds": load_seconds, "embedding_seconds": embedding_seconds,
            "offline_seconds": offline_seconds, "streaming_seconds": streaming_seconds,
            "rss_kib": kib("VmRSS:"), "peak_rss_kib": kib("VmHWM:"),
            "embeddings": embeddings, "offline_turns": offline_turns,
            "streaming_turns": streaming_turns,
        })
    );
}
