//! Frozen-file CPU comparison, including explicitly provisioned CTC model packs.
use anyhow::{Context, Result};
use gigastt_core::{
    inference::{Engine, TranscribeRequest, TranscribeSource},
    model::ModelVariant,
};
use serde_json::{Value, json};
use std::{io::Write, time::Instant};

fn memory() -> Value {
    let status = std::fs::read_to_string("/proc/self/status").unwrap_or_default();
    let read = |name: &str| {
        status
            .lines()
            .find(|line| line.starts_with(name))
            .and_then(|line| line.split_whitespace().nth(1))
            .and_then(|v| v.parse::<u64>().ok())
    };
    json!({"rss_kib":read("VmRSS:"), "peak_rss_kib":read("VmHWM:"), "threads":read("Threads:")})
}
fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() == 2 && args[1] == "catalog" {
        let catalog = ModelVariant::ALL.map(|variant| {
            let assets: std::collections::BTreeMap<_, _> = variant
                .prequantized_files()
                .into_iter()
                .map(|name| (name, variant.prequantized_checksum(name)))
                .collect();
            json!({"head":variant.as_str(),"model_id":variant.model_id(),"assets":assets})
        });
        serde_json::to_writer(std::io::stdout(), &catalog)?;
        return Ok(());
    }
    anyhow::ensure!(
        args.len() == 6,
        "HEAD ARCHITECTURE MODEL_DIR CASES_JSON OUTPUT_JSONL"
    );
    let variant: ModelVariant = args[2].parse().map_err(anyhow::Error::msg)?;
    let cases: Vec<Value> = serde_json::from_slice(&std::fs::read(&args[4])?)?;
    anyhow::ensure!(!cases.is_empty(), "empty comparison");
    let mut output = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&args[5])?;
    let start = Instant::now();
    let engine = Engine::load_with_pools_threads_variant(&args[3], Some(variant), 1, 1, 0, 4)?;
    let load_seconds = start.elapsed().as_secs_f64();
    let loaded = memory();
    let mut triplet = engine.pool.checkout_blocking()?;
    let start = Instant::now();
    for case in cases.iter().take(2) {
        engine.transcribe_request(
            TranscribeRequest::new(TranscribeSource::Path(
                case["audio"].as_str().context("warmup audio")?,
            )),
            &mut triplet,
        )?;
    }
    writeln!(
        output,
        "{}",
        json!({"head":args[1],"architecture":args[2],"encoder_threads":4,"pool_size":1,"load_seconds":load_seconds,"warmup_seconds":start.elapsed().as_secs_f64(),"loaded_memory":loaded,"ready_memory":memory()})
    )?;
    for (index, case) in cases.iter().enumerate() {
        let start = Instant::now();
        let result = engine.transcribe_request(
            TranscribeRequest::new(TranscribeSource::Path(
                case["audio"].as_str().context("audio")?,
            )),
            &mut triplet,
        );
        let seconds = start.elapsed().as_secs_f64();
        let (text, error) = match result {
            Ok(result) => (result.text, None),
            Err(error) => (String::new(), Some(error.to_string())),
        };
        writeln!(
            output,
            "{}",
            json!({"id":case["id"],"seconds":seconds,"itn_text":gigastt_core::itn::apply_itn(&text),"reference_itn":gigastt_core::itn::apply_itn(case["reference"].as_str().context("reference")?),"text":text,"error":error})
        )?;
        output.flush()?;
        if (index + 1) % 100 == 0 {
            eprintln!("{}: {}/{}", args[1], index + 1, cases.len());
        }
    }
    writeln!(
        output,
        "{}",
        json!({"complete":true,"count":cases.len(),"memory":memory()})
    )?;
    Ok(())
}
