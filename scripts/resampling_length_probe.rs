use rubato::audioadapter_buffers::direct::SequentialSliceOfVecs;
use rubato::{
    Async, FixedAsync, Resampler, Resizable, SincInterpolationParameters, SincInterpolationType,
    WindowFunction,
};
#[allow(dead_code)]
#[path = "production.rs"]
mod production;
fn make(n: usize) -> anyhow::Result<Async<f32>> {
    Ok(Async::new_sinc(
        2.,
        2.,
        &SincInterpolationParameters {
            sinc_len: 256,
            f_cutoff: Some(0.95),
            interpolation: SincInterpolationType::Linear,
            oversampling_factor: 256,
            window: WindowFunction::BlackmanHarris2,
        },
        n,
        1,
        FixedAsync::Input,
    )?)
}
fn process(r: &mut Async<f32>, x: &[f32]) -> anyhow::Result<Vec<f32>> {
    r.set_chunk_size(x.len())?;
    let src = [x.to_vec()];
    let mut dst = [vec![0.; r.output_frames_next()]];
    let out_n = dst[0].len();
    r.process_into_buffer(
        &SequentialSliceOfVecs::new(&src, 1, x.len())?,
        &mut SequentialSliceOfVecs::new_mut(&mut dst, 1, out_n)?,
        None,
    )?;
    Ok(dst[0].clone())
}
fn peak(x: &[f32]) -> usize {
    x.iter()
        .enumerate()
        .max_by(|a, b| a.1.abs().total_cmp(&b.1.abs()))
        .map_or(0, |x| x.0)
}
fn main() -> anyhow::Result<()> {
    let mut rows = vec![];
    for n in [160usize, 8000, 48001] {
        for pos in [0, n / 2, n - 1] {
            let mut x = vec![0.; n];
            x[pos] = 0.5;
            let current = production::resample(
                &x,
                production::SampleRate(8000),
                production::SampleRate(16000),
            )?;
            let mut r = make(n.max(512))?;
            let delay = r.output_delay();
            let mut raw = process(&mut r, &x)?;
            raw.extend(process(&mut r, &vec![0.; 512])?);
            let corrected = &raw[delay..delay + 2 * n];
            let src = [x.clone()];
            let mut allr = make(n)?;
            let need = allr.process_all_needed_output_len(n);
            let mut allo = [vec![0.; need]];
            let (_, alln) = allr.process_all_into_buffer(
                &SequentialSliceOfVecs::new(&src, 1, n)?,
                &mut SequentialSliceOfVecs::new_mut(&mut allo, 1, need)?,
                n,
                None,
            )?;
            let all = &allo[0][..alln];
            let mut cache = None;
            let mut staged = vec![];
            let mut scratch = vec![];
            for p in x.chunks(127) {
                production::resample_with_cache(
                    p.to_vec(),
                    production::SampleRate(8000),
                    production::SampleRate(16000),
                    &mut cache,
                    &mut scratch,
                )?;
                staged.extend_from_slice(&scratch);
            }
            rows.push(serde_json::json!({"process_all_peak":peak(all),"process_all_max":all.iter().copied().map(f32::abs).fold(0.,f32::max),"input_len":n,"impulse_position":pos,"output_delay":delay,"current_len":current.len(),"current_peak":peak(&current),"current_max":current.iter().copied().map(f32::abs).fold(0.,f32::max),"corrected_len":corrected.len(),"corrected_peak":peak(corrected),"corrected_max":corrected.iter().copied().map(f32::abs).fold(0.,f32::max),"cached_len":staged.len(),"cached_identical":current==staged}));
        }
    }
    println!("{}", serde_json::to_string_pretty(&rows)?);
    Ok(())
}
