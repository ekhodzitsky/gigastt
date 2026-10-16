//! Compare decoder PCM through one fixed ASR engine (CPU, pool=1, ITN off).
//! Usage: aac_asr_probe MODEL_DIR PCM_F32LE_16KHZ...

use anyhow::{Context, Result};
use gigastt_core::inference::{Engine, TranscribeRequest, TranscribeSource};
fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let engine =
        Engine::load_with_pool_size(args.get(1).context("model directory")?, 1)?.with_itn(false);
    let mut triplet = engine.pool.checkout_blocking()?;
    for path in args.iter().skip(2) {
        let raw = std::fs::read(path)?;
        let (floats, tail) = raw.as_chunks::<4>();
        anyhow::ensure!(tail.is_empty(), "invalid float PCM");
        let pcm: Vec<f32> = floats.iter().map(|b| f32::from_le_bytes(*b)).collect();
        let result = engine.transcribe_request(
            TranscribeRequest::new(TranscribeSource::Samples(&pcm)),
            &mut triplet,
        )?;
        println!("{}", serde_json::json!({"path":path,"text":result.text}));
    }
    Ok(())
}
