//! Laboratory 8 -> 16 kHz sinc-filter control; does not change runtime defaults.
//! Arguments: INPUT_F32 OUTPUT_F32 SINC_LENGTH CUTOFF

use anyhow::{Context, Result, ensure};
use rubato::audioadapter_buffers::direct::SequentialSliceOfVecs;
use rubato::{
    Async, FixedAsync, Resampler, SincInterpolationParameters, SincInterpolationType,
    WindowFunction,
};
use std::io::{BufWriter, Write};

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let source = args.next().context("missing float32 input")?;
    let destination = args.next().context("missing float32 output")?;
    let sinc_len = args
        .next()
        .context("missing sinc length")?
        .parse::<usize>()?;
    let cutoff = args.next().context("missing cutoff")?.parse::<f32>()?;
    ensure!((8..=512).contains(&sinc_len), "sinc length outside 8..512");
    ensure!(cutoff > 0.0 && cutoff <= 1.0, "cutoff outside (0,1]");
    let bytes = std::fs::read(source)?;
    let (chunks, rest) = bytes.as_chunks::<4>();
    ensure!(
        rest.is_empty() && !chunks.is_empty(),
        "expected nonempty float32 samples"
    );
    let samples: Vec<f32> = chunks.iter().copied().map(f32::from_le_bytes).collect();
    ensure!(samples.iter().all(|s| s.is_finite()), "non-finite samples");
    let frames = samples.len();
    let parameters = SincInterpolationParameters {
        sinc_len,
        f_cutoff: Some(cutoff),
        interpolation: SincInterpolationType::Linear,
        oversampling_factor: 256,
        window: WindowFunction::BlackmanHarris2,
    };
    let mut resampler =
        Async::<f32>::new_sinc(2.0, 2.0, &parameters, frames, 1, FixedAsync::Input)?;
    let input_data = [samples];
    let capacity = resampler.output_frames_next();
    let mut output_data = [vec![0.0; capacity]];
    let input = SequentialSliceOfVecs::new(&input_data, 1, frames)?;
    let mut output = SequentialSliceOfVecs::new_mut(&mut output_data, 1, capacity)?;
    let (_, written) = resampler.process_into_buffer(&input, &mut output, None)?;
    let mut file = BufWriter::new(std::fs::File::create(destination)?);
    for value in &output_data[0][..written] {
        file.write_all(&value.to_le_bytes())?;
    }
    file.flush()?;
    Ok(())
}
