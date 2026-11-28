//! Explicit v4 CPU MKL frontend; no legacy inference dispatch.

use anyhow::{Context, Result, ensure};
use std::path::Path;

use super::V4FrameGrid;
use crate::model::{PackAbi, QualifiedV4Pack, QualifiedV4StreamingPack};

mod mkl;

const WINDOW: usize = 320;
const HOP: usize = 160;
const BINS: usize = 161;
const MELS: usize = 64;
const MAX_SAMPLES: usize = 480_000;
const COEFFICIENTS: &[u8; 42496] = include_bytes!("frontend/gigaam_v4_frontend.f32le");

const BUFFER_GUARD: f32 = f32::from_bits(0x7fc0_a5a5);

struct AlignedFloats {
    storage: Vec<f32>,
    offset: usize,
    len: usize,
}
impl AlignedFloats {
    fn new(len: usize, value: f32) -> Result<Self> {
        // 16-float guards on both sides plus at most 15 floats for alignment.
        let capacity = len.checked_add(47).context("FFT buffer length overflow")?;
        let mut storage = Vec::new();
        storage
            .try_reserve_exact(capacity)
            .context("cannot allocate FFT buffer")?;
        storage.resize(capacity, BUFFER_GUARD);
        let alignment = storage[16..].as_ptr().align_offset(64);
        ensure!(alignment <= 15, "cannot align FFT buffer");
        let offset = 16 + alignment;
        let mut result = Self {
            storage,
            offset,
            len,
        };
        result.as_mut_slice().fill(value);
        Ok(result)
    }
    fn as_mut_slice(&mut self) -> &mut [f32] {
        &mut self.storage[self.offset..self.offset + self.len]
    }
    fn as_slice(&self) -> &[f32] {
        &self.storage[self.offset..self.offset + self.len]
    }
    fn guards_intact(&self) -> bool {
        self.storage[..self.offset]
            .iter()
            .chain(&self.storage[self.offset + self.len..])
            .all(|value| value.to_bits() == BUFFER_GUARD.to_bits())
    }
}

fn log_energy(energy: f32) -> f32 {
    energy.clamp(1e-9, 1e9).ln()
}

pub(super) fn validate_pcm(samples: &[f32], valid_samples: usize) -> Result<V4FrameGrid> {
    ensure!(
        samples.len() <= MAX_SAMPLES,
        "v4 MKL frontend supports at most 30 seconds per full-context buffer"
    );
    let grid = V4FrameGrid::new(samples.len(), valid_samples)?;
    ensure!(
        samples.iter().all(|x| x.is_finite()),
        "v4 PCM must be finite"
    );
    Ok(grid)
}

/// Log-mel tensor and explicit physical/valid geometry.
#[derive(Debug)]
pub struct V4Features {
    /// Float32 `[1,64,M]`, channel-major.
    pub values: Vec<f32>,
    /// Physical and valid lengths and full-context timestamps.
    pub grid: V4FrameGrid,
    #[cfg(test)]
    trace: FeatureTrace,
}

#[cfg(test)]
#[derive(Debug)]
struct FeatureTrace {
    windowed: Vec<f32>,
    spectrum: Vec<f32>,
    power: Vec<f32>,
    mel: Vec<f32>,
}

struct MelBand {
    start: usize,
    weights: Vec<f32>,
}

/// Explicitly side-loaded MKL frontend for the verified v4 split pack.
///
/// This is a synchronous full-context feature extractor, not inference or
/// a streaming frontend. The runtime is one process-lifetime isolated group.
pub struct V4Frontend {
    runtime: &'static mkl::MklRuntime,
    window: Vec<f32>,
    bands: Vec<MelBand>,
}

impl V4Frontend {
    /// Select the acoustic frontend by the verified v4 ABI and an explicit
    /// trusted local oneMKL runtime directory. No download or fallback.
    pub fn from_pack(pack: &QualifiedV4Pack, runtime_dir: &Path) -> Result<Self> {
        match pack.descriptor().abi {
            PackAbi::V4RnntSplitV1 => {}
        }
        Self::open(runtime_dir)
    }

    /// Select the same verified author coefficient profile explicitly from
    /// a distinct qualified cached-streaming pack.
    pub fn from_streaming_pack(
        _pack: &QualifiedV4StreamingPack,
        runtime_dir: &Path,
    ) -> Result<Self> {
        Self::open(runtime_dir)
    }

    fn open(runtime_dir: &Path) -> Result<Self> {
        let runtime = mkl::MklRuntime::open(runtime_dir)?;
        let coeff: Vec<f32> = COEFFICIENTS
            .as_chunks::<4>()
            .0
            .iter()
            .map(|x| f32::from_le_bytes(*x))
            .collect();
        let window = coeff[..WINDOW].to_vec();
        let mut bands = Vec::with_capacity(MELS);
        for row in coeff[WINDOW..].as_chunks::<BINS>().0.iter() {
            let first = row
                .iter()
                .position(|&x| x != 0.0)
                .context("empty v4 mel band")?;
            let last = row
                .iter()
                .rposition(|&x| x != 0.0)
                .context("empty v4 mel band")?;
            bands.push(MelBand {
                start: first,
                weights: row[first..=last].to_vec(),
            });
        }
        ensure!(
            bands.len() == MELS && coeff.iter().all(|x| x.is_finite()),
            "invalid built-in v4 frontend coefficients"
        );
        Ok(Self {
            runtime,
            window,
            bands,
        })
    }

    /// Compute full-context features from finite 16 kHz mono float PCM.
    pub fn compute(&self, samples: &[f32]) -> Result<V4Features> {
        self.compute_padded(samples, samples.len())
    }

    /// Compute features on an explicitly padded buffer and preserve its
    /// separate valid prefix. No hidden resampling or padding occurs.
    pub fn compute_padded(&self, samples: &[f32], valid_samples: usize) -> Result<V4Features> {
        let grid = validate_pcm(samples, valid_samples)?;
        let frames = grid.mel_frames();
        let mut input = AlignedFloats::new(frames * WINDOW, 0.0)?;
        let mut spectrum = AlignedFloats::new(frames * BINS * 2, f32::NAN)?;
        for (frame, row) in input
            .as_mut_slice()
            .as_chunks_mut::<WINDOW>()
            .0
            .iter_mut()
            .enumerate()
        {
            let start = frame * HOP;
            for (i, value) in row.iter_mut().enumerate() {
                *value = samples[start + i] * self.window[i];
            }
        }
        self.runtime.forward(&mut input, &mut spectrum, frames)?;
        // This is the qualified Rust post-FFT arithmetic, independent of the
        // original v3 MelSpectrogram's coefficient and FFT policies.
        let power: Vec<f32> = spectrum
            .as_slice()
            .as_chunks::<2>()
            .0
            .iter()
            .map(|z| z[0] * z[0] + z[1] * z[1])
            .collect();
        let mut values = vec![0.0_f32; MELS * frames];
        #[cfg(test)]
        let mut mel = values.clone();
        for (m, band) in self.bands.iter().enumerate() {
            for f in 0..frames {
                let mut energy = 0.0_f32;
                for (i, &weight) in band.weights.iter().enumerate() {
                    energy += weight * power[f * BINS + band.start + i];
                }
                #[cfg(test)]
                {
                    mel[m * frames + f] = energy;
                }
                values[m * frames + f] = log_energy(energy);
            }
        }
        ensure!(
            power.iter().all(|x| x.is_finite()) && values.iter().all(|x| x.is_finite()),
            "non-finite v4 log-mel features"
        );
        Ok(V4Features {
            values,
            grid,
            #[cfg(test)]
            trace: FeatureTrace {
                windowed: input.as_slice().to_vec(),
                spectrum: spectrum.as_slice().to_vec(),
                power,
                mel,
            },
        })
    }
}

#[cfg(all(test, not(miri)))]
mod tests;
