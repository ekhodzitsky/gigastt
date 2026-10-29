use super::*;
use crate::inference::{Engine, TranscriptSegment, audio::decode_audio_file};
use serde_json::{Value, json};
use std::path::Path;
use std::sync::{Arc, Barrier};
use std::time::Instant;

#[derive(Clone)]
struct Clip {
    name: String,
    reference: String,
    samples: Vec<f32>,
}
fn environment<T: std::str::FromStr>(name: &str, default: T) -> T {
    std::env::var(name)
        .ok()
        .map(|s| s.parse().ok().expect("invalid research parameter"))
        .unwrap_or(default)
}
fn proc_cpu_ticks() -> u64 {
    let stat = std::fs::read_to_string("/proc/self/stat").unwrap();
    let fields: Vec<_> = stat
        .rsplit_once(") ")
        .unwrap()
        .1
        .split_whitespace()
        .collect();
    fields[11].parse::<u64>().unwrap() + fields[12].parse::<u64>().unwrap()
}
fn rss_kib() -> u64 {
    let status = std::fs::read_to_string("/proc/self/status").unwrap();
    status
        .lines()
        .find(|line| line.starts_with("VmRSS:"))
        .unwrap()
        .split_whitespace()
        .nth(1)
        .unwrap()
        .parse()
        .unwrap()
}
fn receive(
    segment: TranscriptSegment,
    elapsed_ms: u128,
    events: &mut Vec<Value>,
    finals: &mut Vec<String>,
) {
    if segment.is_final && !segment.text.trim().is_empty() {
        finals.push(segment.text.clone());
    }
    events.push(json!({"at_ms":elapsed_ms,"final":segment.is_final,"text":segment.text,"committed":segment.committed,"tentative":segment.tentative,"endpoint":segment.endpoint_reason}));
}
fn stream(engine: &Engine, clip: &Clip, stride_ms: usize, barrier: &Barrier) -> Value {
    let mut triplet = engine.pool.checkout_blocking().unwrap();
    let mut state = engine.create_state(false);
    PROBE.with(|probe| {
        *probe.borrow_mut() = Some(Probe {
            stride_samples: stride_ms * 16,
            cpu_enabled: environment("GIGASTT_LIVE_STAGE_CPU", 0_u8) != 0,
            ..Probe::default()
        })
    });
    let mut events = Vec::new();
    let mut finals = Vec::new();
    let mut peak_rss = rss_kib();
    let mut work_ns = 0;
    let mut max_audio_samples = 0;
    barrier.wait();
    let start = Instant::now();
    for (index, chunk) in clip.samples.chunks(1600).enumerate() {
        let due = Duration::from_millis(index as u64 * 100);
        if let Some(delay) = due.checked_sub(start.elapsed()) {
            std::thread::sleep(delay);
        }
        let work = Instant::now();
        for segment in engine
            .process_chunk(chunk, &mut state, &mut triplet)
            .unwrap()
        {
            receive(
                segment,
                start.elapsed().as_millis(),
                &mut events,
                &mut finals,
            );
        }
        work_ns += work.elapsed().as_nanos();
        max_audio_samples = max_audio_samples.max(state.audio_buffer.len());
        peak_rss = peak_rss.max(rss_kib());
    }
    let duration = Duration::from_secs_f64(clip.samples.len() as f64 / 16000.0);
    if let Some(delay) = duration.checked_sub(start.elapsed()) {
        std::thread::sleep(delay);
    }
    let stop = Instant::now();
    if let Some(segment) = engine.finish_stream(&mut state, &mut triplet) {
        receive(
            segment,
            start.elapsed().as_millis(),
            &mut events,
            &mut finals,
        );
    }
    let finalize_ns = stop.elapsed().as_nanos();
    work_ns += finalize_ns;
    let probe = PROBE.with(|probe| probe.borrow_mut().take().unwrap());
    json!({"file":clip.name,"reference":clip.reference,"audio_samples":clip.samples.len(),"text":finals.join(" "),"events":events,"final_after_stop_ms":finalize_ns as f64 / 1e6,"elapsed_ms":start.elapsed().as_millis(),"work_ns":work_ns,"peak_rss_kib":peak_rss,"max_audio_samples":max_audio_samples,"probe":probe})
}

#[test]
#[ignore = "long local live-window experiment; requires GIGASTT_LIVE_PROBE output path"]
fn benchmark_live_window_workload() {
    let Ok(output) = std::env::var("GIGASTT_LIVE_PROBE") else {
        eprintln!("skip live-window experiment: set GIGASTT_LIVE_PROBE to an output JSON path");
        return;
    };
    let concurrent = environment("GIGASTT_LIVE_CONCURRENT", 1_usize);
    assert!((1..=2).contains(&concurrent));
    assert!(environment("GIGASTT_LIVE_STAGE_CPU", 0_u8) == 0 || concurrent == 1);
    let stride_ms = environment("GIGASTT_LIVE_STRIDE_MS", 800_usize);
    assert!(stride_ms > 0);
    let window_secs = environment("GIGASTT_LIVE_WINDOW_SECS", 2.5_f64);
    let vad = environment("GIGASTT_LIVE_VAD", 0_u8) != 0;
    let model_dir = crate::model::default_model_dir();
    let mut engine = Engine::load_with_factory(
        Path::new(&model_dir),
        Some(crate::model::ModelVariant::Rnnt),
        concurrent,
        concurrent,
        0,
        crate::runtime::cpu_factory(),
        4,
        true,
        "cpu",
    )
    .unwrap()
    .with_stream_max_window_secs(window_secs)
    .with_stream_stable_prefix(true)
    .with_itn(true)
    .with_punctuator(Some(
        crate::punctuation::Punctuator::load(Path::new(&crate::model::default_punct_model_dir()))
            .unwrap(),
    ));
    if vad {
        engine = engine.with_vad(
            Some(
                crate::vad::SileroVad::load(
                    &Path::new(&crate::model::default_vad_model_dir())
                        .join(crate::vad::VAD_MODEL_FILE),
                )
                .unwrap(),
            ),
            crate::vad::VadConfig::default(),
        );
    }
    engine.warmup().unwrap();
    let fixtures = Path::new(env!("CARGO_MANIFEST_DIR")).join("../gigastt/tests/fixtures");
    let manifest: Vec<Value> =
        serde_json::from_slice(&std::fs::read(fixtures.join("manifest.json")).unwrap()).unwrap();
    let mut clips: Vec<_> = manifest
        .iter()
        .map(|row| Clip {
            name: row["filename"].as_str().unwrap().into(),
            reference: row["reference"].as_str().unwrap().into(),
            samples: decode_audio_file(
                fixtures
                    .join(row["filename"].as_str().unwrap())
                    .to_str()
                    .unwrap(),
            )
            .unwrap(),
        })
        .collect();
    let long = Clip {
        name: "concatenated_00_01_02".into(),
        reference: clips[..3]
            .iter()
            .map(|c| c.reference.as_str())
            .collect::<Vec<_>>()
            .join(" "),
        samples: clips[..3]
            .iter()
            .flat_map(|c| c.samples.iter().copied())
            .collect(),
    };
    let mut padded = vec![0.0; 32000];
    padded.extend_from_slice(&clips[2].samples);
    padded.extend_from_slice(&[0.0; 32000]);
    clips.push(Clip {
        name: "silence_padded_02".into(),
        reference: clips[2].reference.clone(),
        samples: padded,
    });
    clips.push(Clip {
        name: "silence_only".into(),
        reference: String::new(),
        samples: vec![0.0; 48000],
    });
    clips.push(long);
    // Warm the punctuation sidecar before CPU/time sampling.
    engine
        .punctuator
        .as_ref()
        .unwrap()
        .restore("заказать яблоки зеленые");
    let engine = Arc::new(engine);
    let barrier = Arc::new(Barrier::new(concurrent));
    let cpu_start = proc_cpu_ticks();
    let start = Instant::now();
    let rows = std::thread::scope(|scope| {
        let workers: Vec<_> = (0..concurrent)
            .map(|worker| {
                let engine = engine.clone();
                let barrier = barrier.clone();
                let clips = &clips;
                scope.spawn(move || {
                    clips
                        .iter()
                        .map(|clip| {
                            let mut row = stream(&engine, clip, stride_ms, &barrier);
                            row["worker"] = json!(worker);
                            row
                        })
                        .collect::<Vec<_>>()
                })
            })
            .collect();
        workers
            .into_iter()
            .flat_map(|worker| worker.join().unwrap())
            .collect::<Vec<_>>()
    });
    let artifact = json!({"schema":1,"config":{"concurrent":concurrent,"stride_ms":stride_ms,"window_secs":window_secs,"vad":vad,"punctuation":true,"itn":true,"encoder_threads_per_slot":4,"chunk_ms":100},"process_cpu_ticks":proc_cpu_ticks()-cpu_start,"elapsed_ms":start.elapsed().as_millis(),"rows":rows});
    std::fs::write(output, serde_json::to_vec_pretty(&artifact).unwrap()).unwrap();
}

/// Fast local replay: input-audio time, not wall-clock/network final latency.
#[test]
#[ignore = "requires models and GIGASTT_ENDPOINT_PROBE output path"]
fn benchmark_endpoint_duration() {
    let output = std::env::var("GIGASTT_ENDPOINT_PROBE").expect("output path required");
    assert!(
        !Path::new(&output).exists(),
        "preserve previous measurements"
    );
    let model_dir = std::env::var("GIGASTT_TEST_MODEL_DIR").expect("explicit model directory");
    let variant = match std::env::var("GIGASTT_ENDPOINT_VARIANT").as_deref() {
        Ok("e2e_rnnt") => crate::model::ModelVariant::E2eRnnt,
        _ => crate::model::ModelVariant::Rnnt,
    };
    let engine = Engine::load_with_factory(
        Path::new(&model_dir),
        Some(variant),
        1,
        1,
        0,
        crate::runtime::cpu_factory(),
        4,
        true,
        "cpu",
    )
    .unwrap()
    .with_itn(false);
    engine.warmup().unwrap();
    let fixtures = Path::new(env!("CARGO_MANIFEST_DIR")).join("../gigastt/tests/fixtures");
    let manifest: Vec<Value> =
        serde_json::from_slice(&std::fs::read(fixtures.join("manifest.json")).unwrap()).unwrap();
    let mut clips: Vec<_> = manifest
        .iter()
        .take(3)
        .map(|row| Clip {
            name: row["filename"].as_str().unwrap().into(),
            reference: row["reference"].as_str().unwrap().into(),
            samples: decode_audio_file(
                fixtures
                    .join(row["filename"].as_str().unwrap())
                    .to_str()
                    .unwrap(),
            )
            .unwrap(),
        })
        .collect();
    let mut paused = clips[0].samples.clone();
    let midpoint = paused.len() / 2;
    paused.splice(midpoint..midpoint, std::iter::repeat_n(0.0, 12800));
    clips.push(Clip {
        name: "synthetic_midpoint_pause_800ms".into(),
        reference: clips[0].reference.clone(),
        samples: paused,
    });
    // Pitch-changing slowdown is a stress probe, not natural slow speech.
    let slowed = clips[0].samples.iter().flat_map(|&s| [s, s]).collect();
    clips.push(Clip {
        name: "synthetic_half_speed".into(),
        reference: clips[0].reference.clone(),
        samples: slowed,
    });
    let mut rows = Vec::new();
    for ms in [600, 1000, 1500] {
        for clip in &clips {
            let mut state = engine.create_state(false);
            state.endpoint_silence = crate::inference::EndpointSilence::new(ms).unwrap();
            let mut guard = engine.pool.checkout_blocking().unwrap();
            let mut samples = clip.samples.clone();
            samples.extend(std::iter::repeat_n(0.0, 48000));
            let mut events = Vec::new();
            let mut finals = Vec::new();
            let started = Instant::now();
            for (index, chunk) in samples.chunks(1600).enumerate() {
                for segment in engine.process_chunk(chunk, &mut state, &mut guard).unwrap() {
                    receive(
                        segment,
                        ((index + 1) * 100) as u128,
                        &mut events,
                        &mut finals,
                    );
                }
            }
            if let Some(segment) = engine.finish_stream(&mut state, &mut guard) {
                receive(
                    segment,
                    (samples.len() / 16) as u128,
                    &mut events,
                    &mut finals,
                );
            }
            rows.push(json!({"file":clip.name,"reference":clip.reference,"text":finals.join(" "),"endpoint_silence_ms":ms,"audio_samples":clip.samples.len(),"events":events,"work_ms":started.elapsed().as_secs_f64()*1000.0}));
        }
    }
    std::fs::write(output, serde_json::to_vec_pretty(&json!({"schema":1,"variant":format!("{variant:?}"),"clock":"input_audio_ms_not_wall_time","chunk_ms":100,"tail_silence_ms":3000,"rows":rows})).unwrap()).unwrap();
}
