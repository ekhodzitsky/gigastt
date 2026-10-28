use std::{env, fs, io::Write, time::Instant};
use gigastt_core::inference::audio::decode_audio_bytes;
fn main() -> anyhow::Result<()> {
    let args: Vec<_> = env::args().collect();
    let data = fs::read(&args[1])?;
    let pcm = decode_audio_bytes(&data)?;
    let mut out = fs::File::create(&args[2])?;
    for value in &pcm { out.write_all(&value.to_le_bytes())?; }
    let start = Instant::now();
    for _ in 0..100 { std::hint::black_box(decode_audio_bytes(&data)?); }
    println!("{{\"samples\":{},\"seconds_100\":{}}}", pcm.len(), start.elapsed().as_secs_f64());
    Ok(())
}
