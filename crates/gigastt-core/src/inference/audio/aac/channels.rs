//! Resample each AAC plane incrementally; scans retain only correlation state.
use super::super::resample::{ResampleTo16k, SampleRate};
use super::super::{
    ChannelScan, DualMonoDetector, MAX_SAMPLE_RATE, audio_too_long_err, check_decode_abort,
    resolve_budget, whole_buffer_limit_secs,
};
use super::{
    input::{Reader, decode_reader},
    map_aac,
    probe::{sniff_file, sniff_slice},
    stream_options,
};
use anyhow::{Context, Result};
use bytes::Bytes;
use syom::{AacError, Frame};

pub(crate) fn try_decode_planes_bytes(
    data: Bytes,
    max_audio_secs: Option<f64>,
    abort: Option<&(dyn Fn() -> bool + Sync)>,
) -> Result<Option<Vec<Vec<f32>>>> {
    if !sniff_slice(&data) {
        return Ok(None);
    }
    process(
        Reader::bytes(data),
        Some(whole_buffer_limit_secs(max_audio_secs)),
        abort,
        true,
    )
    .map(|r| r.map(|(_, pcm)| pcm))
}
pub(crate) fn try_decode_planes_path(
    path: &str,
    max_audio_secs: Option<f64>,
) -> Result<Option<Vec<Vec<f32>>>> {
    let mut file =
        std::fs::File::open(path).with_context(|| format!("Failed to open audio file: {path}"))?;
    if !sniff_file(&mut file)? {
        return Ok(None);
    }
    process(
        Reader::File(file),
        Some(whole_buffer_limit_secs(max_audio_secs)),
        None,
        true,
    )
    .map(|r| r.map(|(_, pcm)| pcm))
}
pub(crate) fn try_scan_bytes(
    data: Bytes,
    max_audio_secs: Option<f64>,
    abort: Option<&(dyn Fn() -> bool + Sync)>,
) -> Result<Option<ChannelScan>> {
    if !sniff_slice(&data) {
        return Ok(None);
    }
    process(Reader::bytes(data), max_audio_secs, abort, false).map(|r| r.map(|(scan, _)| scan))
}

type ChannelResult = (ChannelScan, Vec<Vec<f32>>);
fn process(
    mut input: Reader,
    limit: Option<f64>,
    abort: Option<&(dyn Fn() -> bool + Sync)>,
    retain: bool,
) -> Result<Option<ChannelResult>> {
    let mut state: Option<Channels> = None;
    let mut failure = None;
    let mut header_only = false;
    let result = decode_reader(&mut input, stream_options(limit), |frame| {
        let consumed = (|| -> Result<()> {
            check_decode_abort(abort)?;
            if state.is_none() {
                state = Some(Channels::new(&frame, limit, retain)?);
            }
            if let Some(state) = &mut state {
                state.push(&frame, abort)?;
            }
            Ok(())
        })();
        if let Err(error) = consumed {
            failure = Some(error);
            return Err(AacError::decode("AAC channel consumer failed"));
        }
        // Inspect the actual decoded layout (implicit HE-AAC v2 can advertise
        // mono in ADTS but produce stereo). Non-stereo needs no correlation.
        if !retain && frame.planar.len() != 2 {
            header_only = true;
            return Err(AacError::decode("AAC non-stereo scan complete"));
        }
        Ok(())
    });
    if let Some(error) = failure {
        return Err(error);
    }
    match result {
        Err(AacError::NotAac) => return Ok(None),
        Err(error) if !header_only => return Err(map_aac(error)),
        _ => {}
    }
    let mut state = state.context("No audio track found")?;
    for (resampler, ready) in state.resamplers.iter_mut().zip(&mut state.ready) {
        check_decode_abort(abort)?;
        resampler.finish_into(ready)?;
    }
    state.consume();
    check_decode_abort(abort)?;
    Ok(Some((
        ChannelScan {
            channels: state.resamplers.len(),
            dual_mono: state.detector.is_dual_mono(),
        },
        state.output,
    )))
}

struct Channels {
    rate: u32,
    frames: usize,
    max_samples: usize,
    limit_secs: f64,
    resamplers: Vec<ResampleTo16k>,
    ready: Vec<Vec<f32>>,
    output: Vec<Vec<f32>>,
    detector: DualMonoDetector,
    retain: bool,
}
impl Channels {
    fn new(frame: &Frame<'_>, limit: Option<f64>, retain: bool) -> Result<Self> {
        anyhow::ensure!(
            frame.sample_rate > 0 && frame.sample_rate <= MAX_SAMPLE_RATE,
            "Unsupported sample rate: {}Hz",
            frame.sample_rate
        );
        anyhow::ensure!(!frame.planar.is_empty(), "No AAC channels");
        let (max_samples, limit_secs) = resolve_budget(limit, frame.sample_rate);
        Ok(Self {
            rate: frame.sample_rate,
            frames: 0,
            max_samples,
            limit_secs,
            resamplers: (0..frame.planar.len())
                .map(|_| ResampleTo16k::new(SampleRate(frame.sample_rate), None))
                .collect(),
            ready: vec![Vec::new(); frame.planar.len()],
            output: vec![Vec::new(); frame.planar.len()],
            detector: DualMonoDetector::new(),
            retain,
        })
    }
    fn push(&mut self, frame: &Frame<'_>, abort: Option<&(dyn Fn() -> bool + Sync)>) -> Result<()> {
        anyhow::ensure!(
            frame.sample_rate == self.rate && frame.planar.len() == self.resamplers.len(),
            "AAC rate or channel count changed"
        );
        self.frames = self
            .frames
            .checked_add(frame.samples)
            .context("AAC sample count overflow")?;
        if self.frames > self.max_samples {
            return Err(audio_too_long_err(self.frames, self.rate, self.limit_secs));
        }
        for ((resampler, ready), plane) in self
            .resamplers
            .iter_mut()
            .zip(&mut self.ready)
            .zip(frame.planar)
        {
            check_decode_abort(abort)?;
            resampler.stage().extend_from_slice(plane);
            resampler.flush_full()?;
            resampler.drain_ready_into(ready);
        }
        self.consume();
        Ok(())
    }
    fn consume(&mut self) {
        if !self.retain && self.ready.len() == 2 {
            self.detector.push(&self.ready[0], &self.ready[1]);
        }
        for (ready, output) in self.ready.iter_mut().zip(&mut self.output) {
            if self.retain {
                output.append(ready);
            } else {
                ready.clear();
            }
        }
    }
}
