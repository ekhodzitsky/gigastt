//! Local production-path punctuation comparison. See docs/rupunct-evaluation.md.
use anyhow::{Context, Result};
use gigastt_core::{
    inference::{Engine, TranscribeRequest, TranscribeSource},
    punctuation::{NativePunctuator, Punctuator},
};
use serde_json::{Value, json};
use std::{
    path::Path,
    sync::{Arc, Barrier},
    time::Instant,
};

enum Model {
    Onnx(Punctuator),
    Native(Box<NativePunctuator>),
}
impl Model {
    fn restore(&self, text: &str) -> String {
        match self {
            Self::Onnx(p) => p.restore(text),
            Self::Native(p) => p.restore(text),
        }
    }
    fn failures(&self) -> u64 {
        match self {
            Self::Onnx(p) => p.failed_windows(),
            Self::Native(p) => p.failed_restores(),
        }
    }
    fn attach(self, engine: Engine) -> Engine {
        match self {
            Self::Onnx(p) => engine.with_punctuator(Some(p)),
            Self::Native(p) => engine.with_native_punctuator(Some(*p)),
        }
    }
}
fn memory() -> Value {
    let status = std::fs::read_to_string("/proc/self/status").unwrap_or_default();
    let read = |name: &str| {
        status
            .lines()
            .find(|l| l.starts_with(name))
            .and_then(|l| l.split_whitespace().nth(1))
            .and_then(|v| v.parse::<u64>().ok())
    };
    json!({"rss_kib":read("VmRSS:"),"peak_rss_kib":read("VmHWM:"),"threads":read("Threads:")})
}
fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    anyhow::ensure!(
        args.len() >= 6,
        "MODE BACKEND MODEL_DIR CASES_JSON OUTPUT_JSON [CONCURRENCY|ASR_MODEL_DIR]"
    );
    let cases: Vec<Value> = serde_json::from_slice(&std::fs::read(&args[4])?)?;
    let before = memory();
    let start = Instant::now();
    let model = match args[2].as_str() {
        "onnx" => Model::Onnx(Punctuator::load(Path::new(&args[3]))?),
        "native" => Model::Native(Box::new(NativePunctuator::load(Path::new(&args[3]))?)),
        _ => anyhow::bail!("backend must be onnx or native"),
    };
    let load_seconds = start.elapsed().as_secs_f64();
    let loaded = memory();
    let mut report = json!({"backend":args[2],"mode":args[1],"load_seconds":load_seconds,"before":before,"loaded":loaded});
    match args[1].as_str() {
        "quality" => {
            let mut rows = Vec::new();
            for case in &cases {
                let text = case["input"].as_str().context("input")?;
                let start = Instant::now();
                let output = model.restore(text);
                rows.push(
                    json!({"id":case["id"],"text":output,"seconds":start.elapsed().as_secs_f64()}),
                );
            }
            anyhow::ensure!(
                model.failures() == 0,
                "punctuation failures: {}",
                model.failures()
            );
            report["rows"] = json!(rows);
            report["active"] = memory();
        }
        "timing" => {
            let concurrency: usize = args.get(6).context("concurrency")?.parse()?;
            anyhow::ensure!(concurrency > 0, "concurrency must be positive");
            let model = Arc::new(model);
            let mut rows = Vec::new();
            for case in &cases {
                let text = case["input"].as_str().context("input")?;
                let cold = Instant::now();
                let expected = model.restore(text);
                let first = cold.elapsed().as_secs_f64();
                model.restore(text);
                let mut rounds = Vec::new();
                for _ in 0..7 {
                    let barrier = Barrier::new(concurrency + 1);
                    let (wall, requests) = std::thread::scope(|scope| -> Result<_> {
                        let handles: Vec<_> = (0..concurrency)
                            .map(|_| {
                                let model = &model;
                                let barrier = &barrier;
                                scope.spawn(move || {
                                    barrier.wait();
                                    let start = Instant::now();
                                    let output = model.restore(text);
                                    (start.elapsed().as_secs_f64(), output)
                                })
                            })
                            .collect();
                        let start = Instant::now();
                        barrier.wait();
                        let mut requests = Vec::new();
                        for handle in handles {
                            let (seconds, output) = handle
                                .join()
                                .map_err(|_| anyhow::anyhow!("restore worker panicked"))?;
                            anyhow::ensure!(
                                output == expected,
                                "nondeterministic concurrent output"
                            );
                            requests.push(seconds);
                        }
                        Ok((start.elapsed().as_secs_f64(), requests))
                    })?;
                    rounds.push(json!({"wall_seconds":wall,"request_seconds":requests}));
                }
                rows.push(json!({"id":case["id"],"first_seconds":first,"rounds":rounds,"output":expected}));
            }
            anyhow::ensure!(
                model.failures() == 0,
                "punctuation failures: {}",
                model.failures()
            );
            report["rows"] = json!(rows);
            report["active"] = memory();
            report["concurrency"] = json!(concurrency);
        }
        "asr" => {
            let engine = model.attach(Engine::load_with_pool_size(
                args.get(6).context("ASR model directory")?,
                1,
            )?);
            let mut triplet = engine.pool.checkout_blocking()?;
            let mut rows = Vec::new();
            for case in &cases {
                let path = case["audio"].as_str().context("audio path")?;
                let start = Instant::now();
                let result = engine.transcribe_request(
                    TranscribeRequest::new(TranscribeSource::Path(path)),
                    &mut triplet,
                )?;
                rows.push(json!({"id":case["id"],"result":result,"seconds":start.elapsed().as_secs_f64()}));
            }
            report["rows"] = json!(rows);
            report["active"] = memory();
        }
        _ => anyhow::bail!("invalid mode"),
    }
    report["finished"] = memory();
    std::fs::write(&args[5], serde_json::to_vec_pretty(&report)?)?;
    Ok(())
}
