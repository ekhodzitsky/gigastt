//! Production-path codec timing, Linux peak RSS and optional float PCM dumps.
//! Usage: aac_probe file|bytes|split|scan|probe INPUT REPEATS [OUTPUT_PREFIX]
//! Two warmups precede timed iterations; errors are JSON and must be checked.

use anyhow::{Context, Result};
use bytes::Bytes;
use gigastt_core::inference::audio;
use serde_json::json;
use std::{hint::black_box, io::Write, time::Instant};
fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let mode = args.get(1).context("mode")?;
    let path = args.get(2).context("path")?;
    let reps: usize = args.get(3).map(|s| s.parse()).transpose()?.unwrap_or(5);
    let data = Bytes::from(std::fs::read(path)?);
    let mut samples = Vec::new();
    let mut detail = json!(null);
    let mut times = Vec::new();
    for i in 0..reps + 2 {
        let start = Instant::now();
        let result: Result<Vec<Vec<f32>>> = match mode.as_str() {
            "file" => audio::decode_audio_file(path).map(|x| vec![x]),
            "bytes" => audio::decode_audio_bytes_shared(data.clone()).map(|x| vec![x]),
            "split" => audio::decode_audio_bytes_shared_channels(data.clone()),
            "scan" => audio::scan_channels(data.clone(), None).map(|s| {
                detail = json!({"channels":s.channels,"dual_mono":s.dual_mono});
                vec![]
            }),
            "probe" => audio::probe_duration_file(path).and_then(|f| {
                audio::probe_duration_bytes(data.clone()).map(|b| {
                    detail = json!({"file":f,"bytes":b});
                    vec![]
                })
            }),
            _ => anyhow::bail!("invalid mode"),
        };
        let elapsed = start.elapsed().as_secs_f64();
        match result {
            Ok(value) => samples = black_box(value),
            Err(error) => {
                println!("{}", json!({"mode":mode,"error":format!("{error:#}")}));
                return Ok(());
            }
        }
        if i >= 2 {
            times.push(elapsed)
        }
    }
    let status = std::fs::read_to_string("/proc/self/status").unwrap_or_default();
    let peak = status
        .lines()
        .find(|s| s.starts_with("VmHWM:"))
        .and_then(|s| s.split_whitespace().nth(1))
        .and_then(|s| s.parse::<u64>().ok());
    let summary:Vec<_>=samples.iter().map(|s|json!({"samples":s.len(),"energy":s.iter().map(|x|f64::from(*x).powi(2)).sum::<f64>(),"finite":s.iter().all(|x|x.is_finite())})).collect();
    if let Some(prefix) = args.get(4) {
        for (i, s) in samples.iter().enumerate() {
            let mut f =
                std::io::BufWriter::new(std::fs::File::create(format!("{prefix}.{i}.f32"))?);
            for x in s {
                f.write_all(&x.to_le_bytes())?
            }
            f.flush()?;
        }
    }
    println!(
        "{}",
        json!({"mode":mode,"seconds":times,"peak_rss_kib":peak,"planes":summary,"detail":detail})
    );
    Ok(())
}
