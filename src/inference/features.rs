//! Log-mel spectrogram feature extraction for GigaAM v3.

use rustfft::num_complex::Complex;
use rustfft::{Fft, FftPlanner};
use std::f32::consts::PI;
use std::sync::Arc;

pub struct MelSpectrogram {
    n_fft: usize,
    hop_length: usize,
    window: Vec<f32>,
    mel_filterbank: Vec<Vec<f32>>, // [n_mels][n_fft/2 + 1]
    fft: Arc<dyn Fft<f32>>,
}

impl MelSpectrogram {
    pub fn new() -> Self {
        let n_fft = 320;
        let hop_length = 160;
        let n_mels = 64;
        let sample_rate = 16000.0_f32;
        let fmin = 0.0_f32;
        let fmax = sample_rate / 2.0;

        // Hann window
        let window: Vec<f32> = (0..n_fft)
            .map(|n| 0.5 * (1.0 - (2.0 * PI * n as f32 / (n_fft - 1) as f32).cos()))
            .collect();

        // HTK mel filterbank
        let mel_filterbank = Self::create_mel_filterbank(n_fft, n_mels, sample_rate, fmin, fmax);

        // Pre-plan FFT
        let mut planner = FftPlanner::<f32>::new();
        let fft = planner.plan_fft_forward(n_fft);

        Self {
            n_fft,
            hop_length,
            window,
            mel_filterbank,
            fft,
        }
    }

    fn hz_to_mel(hz: f32) -> f32 {
        2595.0 * (1.0 + hz / 700.0).log10()
    }

    fn mel_to_hz(mel: f32) -> f32 {
        700.0 * (10.0_f32.powf(mel / 2595.0) - 1.0)
    }

    fn create_mel_filterbank(
        n_fft: usize,
        n_mels: usize,
        sample_rate: f32,
        fmin: f32,
        fmax: f32,
    ) -> Vec<Vec<f32>> {
        let n_freqs = n_fft / 2 + 1; // 161

        let mel_min = Self::hz_to_mel(fmin);
        let mel_max = Self::hz_to_mel(fmax);

        // n_mels + 2 equally spaced points in mel space
        let mel_points: Vec<f32> = (0..=(n_mels + 1))
            .map(|i| mel_min + (mel_max - mel_min) * i as f32 / (n_mels + 1) as f32)
            .collect();

        let hz_points: Vec<f32> = mel_points.iter().map(|&m| Self::mel_to_hz(m)).collect();

        // Convert Hz to FFT bin indices (float for interpolation)
        let bin_points: Vec<f32> = hz_points
            .iter()
            .map(|&hz| hz * n_fft as f32 / sample_rate)
            .collect();

        let mut filterbank = vec![vec![0.0_f32; n_freqs]; n_mels];

        for m in 0..n_mels {
            let f_left = bin_points[m];
            let f_center = bin_points[m + 1];
            let f_right = bin_points[m + 2];

            for (k, bin) in filterbank[m].iter_mut().enumerate() {
                let freq = k as f32;
                if freq >= f_left && freq <= f_center && f_center > f_left {
                    *bin = (freq - f_left) / (f_center - f_left);
                } else if freq > f_center && freq <= f_right && f_right > f_center {
                    *bin = (f_right - freq) / (f_right - f_center);
                }
            }
        }

        filterbank
    }

    /// Compute log-mel spectrogram from f32 audio samples.
    /// Returns features in shape [n_mels, num_frames] as a flat Vec.
    pub fn compute(&self, samples: &[f32]) -> (Vec<f32>, usize) {
        let n_freqs = self.n_fft / 2 + 1;

        // Number of frames (center=false)
        if samples.len() < self.n_fft {
            return (vec![0.0; self.mel_filterbank.len()], 1);
        }
        let num_frames = (samples.len() - self.n_fft) / self.hop_length + 1;

        let n_mels = self.mel_filterbank.len();
        let mut output = vec![0.0_f32; n_mels * num_frames];

        // Pre-allocate buffers reused across frames
        let mut fft_input = vec![Complex::new(0.0_f32, 0.0); self.n_fft];
        let mut power = vec![0.0_f32; n_freqs];

        for frame_idx in 0..num_frames {
            let start = frame_idx * self.hop_length;

            // Apply window and fill FFT input in-place
            for i in 0..self.n_fft {
                let sample = if start + i < samples.len() {
                    samples[start + i]
                } else {
                    0.0
                };
                fft_input[i] = Complex::new(sample * self.window[i], 0.0);
            }

            // FFT
            self.fft.process(&mut fft_input);

            // Power spectrum (first n_fft/2 + 1 bins)
            for k in 0..n_freqs {
                power[k] = fft_input[k].norm_sqr();
            }

            // Apply mel filterbank + log
            for (m, filter) in self.mel_filterbank.iter().enumerate() {
                let mut mel_energy: f32 = 0.0;
                for (k, &p) in power.iter().enumerate() {
                    mel_energy += filter[k] * p;
                }
                // Log with floor
                output[m * num_frames + frame_idx] = (mel_energy.max(1e-10)).ln();
            }
        }

        (output, num_frames)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_silence() {
        let mel = MelSpectrogram::new();
        let silence = vec![0.0_f32; 16000]; // 1 second of silence
        let (features, num_frames) = mel.compute(&silence);
        assert!(num_frames > 0);
        assert_eq!(features.len(), 64 * num_frames);
        // All mel energies should be at the floor value ln(1e-10)
        let floor = (1e-10_f32).ln();
        for &v in &features {
            assert!((v - floor).abs() < 0.01, "Expected ~{floor}, got {v}");
        }
    }

    #[test]
    fn test_output_shape() {
        let mel = MelSpectrogram::new();
        let samples = vec![0.0_f32; 3200]; // 200ms at 16kHz
        let (features, num_frames) = mel.compute(&samples);
        // center=false: (3200 - 320) / 160 + 1 = 19 frames
        assert_eq!(num_frames, 19);
        assert_eq!(features.len(), 64 * 19);
    }

    #[test]
    fn test_too_short() {
        let mel = MelSpectrogram::new();
        let samples = vec![0.0_f32; 100]; // Less than n_fft=320
        let (features, num_frames) = mel.compute(&samples);
        assert_eq!(num_frames, 1);
        assert_eq!(features.len(), 64);
    }

    #[test]
    fn test_sine_wave() {
        let mel = MelSpectrogram::new();
        // 440Hz sine wave, 1 second at 16kHz
        let samples: Vec<f32> = (0..16000)
            .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 16000.0).sin())
            .collect();
        let (features, num_frames) = mel.compute(&samples);
        assert!(num_frames > 0);
        // Sine wave should produce non-floor values in some mel bins
        let floor = (1e-10_f32).ln();
        let non_floor = features.iter().filter(|&&v| (v - floor).abs() > 1.0).count();
        assert!(non_floor > 0, "Expected some non-floor values for sine wave");
    }
}
