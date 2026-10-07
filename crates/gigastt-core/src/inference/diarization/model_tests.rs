//! Real-model oracle and latency guard. Missing assets must fail, never skip.
use super::*;
use ort::session::{Session, builder::GraphOptimizationLevel};
use ort::value::Tensor;
use polyvoice::features::{FbankConfig, FbankExtractor, apply_cmvn};
use sha2::Digest;
use std::time::Instant;

struct CpuOracle(Mutex<Session>);

impl CpuOracle {
    fn new(model: &Path, intra_threads: usize) -> Self {
        Self(Mutex::new(
            Session::builder()
                .unwrap()
                .with_execution_providers([ort::ep::CPU::default().build()])
                .unwrap()
                .with_optimization_level(GraphOptimizationLevel::Level3)
                .unwrap()
                .with_intra_threads(intra_threads)
                .unwrap()
                .with_inter_threads(1)
                .unwrap()
                .with_intra_op_spinning(false)
                .unwrap()
                .commit_from_file(model)
                .unwrap(),
        ))
    }
}

impl Embedder for CpuOracle {
    fn dim(&self) -> usize {
        SPEAKER_EMBEDDING_DIM
    }
    fn embed(&self, samples: &[f32]) -> Result<Vec<f32>, EmbedderError> {
        let fbank = FbankExtractor::new(FbankConfig::default());
        let padded;
        let samples = if samples.len() < 400 {
            padded = {
                let mut p = vec![0.0; 400];
                p[..samples.len()].copy_from_slice(samples);
                p
            };
            &padded
        } else {
            samples
        };
        let frames = apply_cmvn(&fbank.extract(samples).unwrap());
        let input = Tensor::from_array((
            [1, frames.len(), 80],
            frames.into_iter().flatten().collect::<Vec<_>>(),
        ))
        .unwrap();
        let mut session = self.0.lock();
        let outputs = session.run(ort::inputs![input]).unwrap();
        let mut embedding = outputs[0].try_extract_tensor::<f32>().unwrap().1.to_vec();
        polyvoice::utils::l2_normalize(&mut embedding);
        Ok(embedding)
    }
}

fn median(values: &mut [f64]) -> f64 {
    values.sort_by(f64::total_cmp);
    values[values.len() / 2]
}

#[test]
#[ignore = "requires pinned WeSpeaker model; run with --release --ignored --exact"]
#[allow(clippy::assertions_on_constants)] // Reject accidental debug invocation at runtime.
fn test_speaker_quality_and_latency_against_cpu_oracle() {
    assert!(!cfg!(debug_assertions), "latency guard requires --release");
    for name in ["POLYVOICE_SESSION_POOL_SIZE", "POLYVOICE_INTRA_THREADS"] {
        assert!(
            std::env::var_os(name).is_none(),
            "unset {name} for the reference CPU budget"
        );
    }
    let reference_threads = std::thread::available_parallelism()
        .map(|n| (n.get() / SPEAKER_POOL_SIZE).max(1))
        .unwrap_or(1);
    let model_dir = PathBuf::from(crate::model::default_model_dir());
    let model = model_dir.join("wespeaker_resnet34.onnx");
    assert!(
        model.is_file(),
        "download WeSpeaker before running this test"
    );
    let hash: String = sha2::Sha256::digest(std::fs::read(&model).unwrap())
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect();
    assert_eq!(
        hash, "3955447b0499dc9e0a4541a895df08b03c69098eba4e56c02b5603e9f7f4fcbb",
        "unexpected WeSpeaker weights"
    );
    let started = Instant::now();
    let encoder = Arc::new(load_speaker_encoder(&model, SPEAKER_POOL_SIZE).unwrap());
    let load_seconds = started.elapsed().as_secs_f64();
    let oracle = CpuOracle::new(&model, reference_threads);
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../gigastt/tests/fixtures");
    let mut clips = Vec::new();
    for name in ["golos_00.wav", "golos_01.wav", "golos_02.wav"] {
        let samples =
            crate::inference::audio::decode_audio_file(root.join(name).to_str().unwrap()).unwrap();
        for seconds in [0.5, 1.5, 3.0] {
            let len = (seconds * 16000.0) as usize;
            assert!(samples.len() >= len, "fixture too short: {name}");
            clips.push(samples[..len].to_vec());
        }
    }
    // Frozen CPU outputs catch frontend changes shared by candidate and oracle.
    // Three 1.5 s Golos prefixes, 256 little-endian f32 values each.
    let golden: Vec<f32> = include_bytes!("speaker-embeddings.f32le")
        .as_chunks::<4>()
        .0
        .iter()
        .copied()
        .map(f32::from_le_bytes)
        .collect();
    assert_eq!(golden.len(), 3 * SPEAKER_EMBEDDING_DIM);
    for (clip, expected) in [1, 4, 7]
        .into_iter()
        .zip(golden.as_chunks::<SPEAKER_EMBEDDING_DIM>().0.iter())
    {
        let actual = encoder.embed(&clips[clip]).unwrap();
        assert_eq!(actual.len(), expected.len());
        assert!(
            actual
                .iter()
                .zip(expected)
                .all(|(a, b)| a.is_finite() && (a - b).abs() < 0.001),
            "frozen speaker oracle drift"
        );
    }
    let expected = encoder.embed(&clips[1]).unwrap();
    std::thread::scope(|scope| {
        let workers: Vec<_> = (0..6)
            .map(|_| scope.spawn(|| encoder.embed(&clips[1]).unwrap()))
            .collect();
        for worker in workers {
            assert_eq!(worker.join().unwrap(), expected);
        }
    });
    // Silence and sub-window inputs also exercise padding and normalization.
    for samples in clips
        .iter()
        .chain([vec![], vec![0.0; 80], vec![0.0; 16000]].iter())
    {
        let actual = encoder.embed(samples).unwrap();
        let expected = oracle.embed(samples).unwrap();
        assert_eq!(actual.len(), SPEAKER_EMBEDDING_DIM);
        assert!(actual.iter().all(|v| v.is_finite()));
        let max_error = actual
            .iter()
            .zip(&expected)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f32, f32::max);
        assert!(max_error < 0.001, "speaker embedding drift: {max_error}");
    }
    // Compare whole offline turns and streaming labels, not just vector shape.
    let samples: Vec<_> = clips
        .iter()
        .flat_map(|c| c.iter().copied().chain(vec![0.0; 8000]))
        .collect();
    let config = DiaConfig::default();
    let vad_config = VadConfig::default();
    let reference = LegacyPipeline::new(config, vad_config)
        .run(
            &samples,
            &oracle,
            &mut EnergyVad::new(-40.0, 16000, vad_config.frame_size),
        )
        .unwrap();
    let actual = run_offline(&encoder, &samples).unwrap();
    assert!(!reference.turns.is_empty());
    assert_eq!(actual.len(), reference.turns.len());
    for (a, b) in actual.iter().zip(&reference.turns) {
        assert_eq!(
            (a.start, a.end, a.speaker),
            (b.time.start, b.time.end, b.speaker.0)
        );
    }
    let mut actual_stream = open_streaming(&encoder).unwrap();
    let mut reference_stream = StreamingPipeline::new(
        EnergyVad::new(-40.0, 16000, vad_config.frame_size),
        oracle,
        DiaConfig {
            cluster: ClusterConfig {
                threshold: 0.5,
                ..ClusterConfig::default()
            },
            ..DiaConfig::default()
        },
        vad_config,
    )
    .unwrap();
    for chunk in samples.chunks(1600) {
        actual_stream.feed(chunk).unwrap();
        reference_stream.feed(chunk).unwrap();
        assert_eq!(actual_stream.turns(), reference_stream.turns());
    }
    assert_eq!(
        actual_stream.flush().unwrap(),
        reference_stream.flush().unwrap()
    );
    // A second oracle avoids borrowing it back out of StreamingPipeline.
    let oracle = CpuOracle::new(&model, reference_threads);
    let mut ratios = Vec::new();
    let mut candidate = Vec::new();
    let mut reference = Vec::new();
    for round in 0..7 {
        let mut times = [0.0; 2];
        // Alternate order to avoid consistently favoring the warmed CPU.
        for which in [round % 2, 1 - round % 2] {
            let start = Instant::now();
            for clip in &clips {
                let result = if which == 0 {
                    encoder.embed(clip)
                } else {
                    oracle.embed(clip)
                };
                std::hint::black_box(result.unwrap());
            }
            times[which] = start.elapsed().as_secs_f64();
        }
        candidate.push(times[0]);
        reference.push(times[1]);
        ratios.push(times[0] / times[1]);
    }
    // Retain a measured control for the backend this repair replaced. Its
    // timing is diagnostic; only the production/oracle ratio gates CI.
    let legacy = FbankOnnxExtractor::new(
        &model,
        SPEAKER_EMBEDDING_DIM,
        SPEAKER_POOL_SIZE,
        polyvoice::onnx::ExecutionProvider::Cpu,
    )
    .unwrap();
    let mut legacy_times = Vec::new();
    let mut control_times = Vec::new();
    let mut legacy_ratios = Vec::new();
    for round in 0..7 {
        let mut times = [0.0; 2];
        for which in [round % 2, 1 - round % 2] {
            let started = Instant::now();
            for clip in &clips {
                let embedding = if which == 0 {
                    legacy.embed(clip)
                } else {
                    oracle.embed(clip)
                };
                std::hint::black_box(embedding.unwrap());
            }
            times[which] = started.elapsed().as_secs_f64();
        }
        legacy_times.push(times[0]);
        control_times.push(times[1]);
        legacy_ratios.push(times[0] / times[1]);
    }
    eprintln!(
        "legacy tract control {}",
        serde_json::json!({
            "legacy_seconds": median(&mut legacy_times.clone()),
            "oracle_seconds": median(&mut control_times.clone()),
            "ratio": median(&mut legacy_ratios), "legacy_round_seconds": legacy_times,
            "oracle_round_seconds": control_times,
        })
    );
    let threaded = CpuOracle::new(&model, 1);
    let mut threaded_times = Vec::new();
    for _ in 0..7 {
        let started = Instant::now();
        for clip in &clips {
            std::hint::black_box(threaded.embed(clip).unwrap());
        }
        threaded_times.push(started.elapsed().as_secs_f64());
    }
    eprintln!(
        "single-thread CPU control {}",
        serde_json::json!({"intra_threads": 1, "seconds": median(&mut threaded_times)})
    );
    let ratio = median(&mut ratios);
    eprintln!(
        "speaker performance {}",
        serde_json::json!({"load_seconds": load_seconds,
        "candidate_seconds": median(&mut candidate.clone()), "oracle_seconds": median(&mut reference.clone()),
        "ratio": ratio, "clips": clips.len(), "intra_threads": reference_threads, "onnx_runtime": ort::info(), "rounds": ratios.len(), "candidate_round_seconds": candidate, "oracle_round_seconds": reference})
    );
    // Broad enough for hosted-runner noise, tight enough to catch a backend swap.
    assert!(
        ratio <= 1.2,
        "speaker inference is {ratio:.2}x slower than the CPU oracle (limit 1.2x)"
    );
}

/// Run each backend in its own process so retained allocations do not bias RSS.
#[cfg(target_os = "linux")]
#[test]
#[ignore = "requires WeSpeaker; set GIGASTT_SPEAKER_RESOURCE_BACKEND=production or legacy-tract"]
#[allow(clippy::assertions_on_constants)]
fn test_speaker_resource_probe() {
    assert!(!cfg!(debug_assertions), "resource probe requires --release");
    let backend = std::env::var("GIGASTT_SPEAKER_RESOURCE_BACKEND").unwrap();
    let model = PathBuf::from(crate::model::default_model_dir()).join("wespeaker_resnet34.onnx");
    let started = Instant::now();
    let encoder: Box<dyn Embedder> = match backend.as_str() {
        "production" => Box::new(load_speaker_encoder(&model, SPEAKER_POOL_SIZE).unwrap()),
        "legacy-tract" => Box::new(
            FbankOnnxExtractor::new(
                &model,
                SPEAKER_EMBEDDING_DIM,
                SPEAKER_POOL_SIZE,
                polyvoice::onnx::ExecutionProvider::Cpu,
            )
            .unwrap(),
        ),
        _ => panic!("unknown resource backend"),
    };
    let load_seconds = started.elapsed().as_secs_f64();
    let fixture =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../gigastt/tests/fixtures/golos_00.wav");
    let samples = crate::inference::audio::decode_audio_file(fixture.to_str().unwrap()).unwrap();
    for _ in 0..9 {
        let embedding = encoder.embed(&samples[..48000]).unwrap();
        assert_eq!(embedding.len(), SPEAKER_EMBEDDING_DIM);
        assert!(embedding.iter().all(|v| v.is_finite()));
        std::hint::black_box(embedding);
    }
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
        "speaker resources {}",
        serde_json::json!({
            "backend": backend, "load_seconds": load_seconds,
            "rss_kib": kib("VmRSS:"), "peak_rss_kib": kib("VmHWM:"),
            "pool_size": SPEAKER_POOL_SIZE,
        })
    );
}
