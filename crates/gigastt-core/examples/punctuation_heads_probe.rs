//! Decode frozen audio once per explicit head for punctuation evaluation.
use anyhow::{Context, Result};
use gigastt_core::{
    inference::{Engine, TranscribeRequest, TranscribeSource},
    model::ModelVariant,
};
use serde_json::{Value, json};
use std::{io::Write, time::Instant};
fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    anyhow::ensure!(args.len() == 5, "HEAD MODEL_DIR CASES_JSON OUTPUT_JSONL");
    let variant = match args[1].as_str() {
        "rnnt" => ModelVariant::Rnnt,
        "e2e_rnnt" => ModelVariant::E2eRnnt,
        _ => anyhow::bail!("unsupported head"),
    };
    let cases: Vec<Value> = serde_json::from_slice(&std::fs::read(&args[3])?)?;
    let start = Instant::now();
    let engine = Engine::load_with_pools_threads_variant(&args[2], Some(variant), 1, 1, 0, 4)?;
    let mut triplet = engine.pool.checkout_blocking()?;
    let mut output = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&args[4])?;
    writeln!(
        output,
        "{}",
        json!({"head":args[1], "load_seconds":start.elapsed().as_secs_f64(), "encoder_threads":4, "pool_size":1})
    )?;
    for (index, case) in cases.iter().enumerate() {
        let start = Instant::now();
        let result = engine.transcribe_request(
            TranscribeRequest::new(TranscribeSource::Path(
                case["audio"].as_str().context("audio")?,
            )),
            &mut triplet,
        )?;
        writeln!(
            output,
            "{}",
            json!({"id":case["id"], "reference_itn":gigastt_core::itn::apply_itn(case["reference"].as_str().context("reference")?), "itn_text":gigastt_core::itn::apply_itn(&result.text), "text":result.text, "seconds":start.elapsed().as_secs_f64()})
        )?;
        output.flush()?;
        if (index + 1) % 25 == 0 {
            eprintln!("{}: {}/{}", args[1], index + 1, cases.len());
        }
    }
    writeln!(
        output,
        "{}",
        json!({"memory":std::fs::read_to_string("/proc/self/status").unwrap_or_default()})
    )?;
    Ok(())
}
