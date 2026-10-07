//! WeSpeaker's established fbank/CMVN frontend with gigastt's CPU runtime.
use std::path::Path;

use polyvoice::features::{FbankConfig, FbankExtractor, apply_cmvn};
use polyvoice::{Embedder, EmbedderError};

use crate::inference::pool::Pool;
use crate::runtime::{RuntimeSession, Shape, Tensor, TensorData, speaker_runtime};

use super::SPEAKER_EMBEDDING_DIM;

pub(crate) struct SpeakerEmbedder {
    sessions: Pool<Box<dyn RuntimeSession>>,
    fbank: FbankExtractor,
}

fn failed(error: impl std::fmt::Display) -> EmbedderError {
    EmbedderError::InferenceFailed {
        detail: error.to_string(),
    }
}

impl SpeakerEmbedder {
    pub(super) fn load(path: &Path, pool_size: usize) -> anyhow::Result<Self> {
        anyhow::ensure!(pool_size > 0, "speaker pool size must be positive");
        // Preserve the pre-tract CPU budget and its existing environment knobs.
        // CPU selection and idle-worker policy live at the runtime seam.
        let pool_size = polyvoice::onnx::resolve_session_pool_size(pool_size);
        let runtime = speaker_runtime(polyvoice::onnx::resolve_intra_threads(pool_size));
        let sessions = (0..pool_size)
            .map(|_| runtime.load_session(path, true))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self::from_sessions(sessions))
    }

    fn from_sessions(sessions: Vec<Box<dyn RuntimeSession>>) -> Self {
        Self {
            sessions: Pool::new(sessions),
            // Freeze the model's frontend contract across dependency updates.
            fbank: FbankExtractor::new(FbankConfig {
                sample_rate: 16000,
                n_fft: 512,
                win_length: 400,
                hop_length: 160,
                n_mels: 80,
                f_min: 20.0,
                f_max: 7600.0,
                pre_emphasis: 0.97,
            }),
        }
    }
}

impl Embedder for SpeakerEmbedder {
    fn dim(&self) -> usize {
        SPEAKER_EMBEDDING_DIM
    }

    fn embed(&self, samples: &[f32]) -> Result<Vec<f32>, EmbedderError> {
        // Hold the slot through preprocessing too, bounding concurrent work.
        let session = self.sessions.checkout_blocking().map_err(failed)?;
        let padded;
        let samples = if samples.len() < self.fbank.config.win_length {
            padded = {
                let mut data = vec![0.0; self.fbank.config.win_length];
                data[..samples.len()].copy_from_slice(samples);
                data
            };
            &padded
        } else {
            samples
        };
        let frames = self.fbank.extract(samples).map_err(failed)?;
        if frames.is_empty() {
            return Err(EmbedderError::AudioTooShort {
                actual_secs: samples.len() as f32 / 16000.0,
                min_secs: self.fbank.config.win_length as f32 / 16000.0,
            });
        }
        let frames = apply_cmvn(&frames);
        let input = Tensor::new(
            Shape::new(vec![1, frames.len(), self.fbank.config.n_mels]),
            TensorData::F32(frames.into_iter().flatten().collect()),
        )
        .map_err(failed)?;
        let outputs = session.run(&[input]).map_err(failed)?;
        let output = outputs
            .into_iter()
            .next()
            .ok_or_else(|| failed("speaker model produced no outputs"))?;
        let TensorData::F32(mut embedding) = output.into_data() else {
            return Err(failed("speaker model output must be f32"));
        };
        if embedding.len() != self.dim() {
            return Err(EmbedderError::DimMismatch {
                expected: self.dim(),
                actual: embedding.len(),
            });
        }
        // Preserve polyvoice's silence behavior: a non-finite raw norm is
        // normalized to the zero vector before clustering.
        polyvoice::utils::l2_normalize(&mut embedding);
        Ok(embedding)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::RuntimeError;
    use std::sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    };

    struct Session {
        outputs: Vec<Tensor>,
        calls: Arc<AtomicUsize>,
    }
    impl RuntimeSession for Session {
        fn run(&self, inputs: &[Tensor]) -> Result<Vec<Tensor>, RuntimeError> {
            self.calls.fetch_add(1, Ordering::Relaxed);
            assert_eq!(inputs.len(), 1);
            assert_eq!(inputs[0].shape().dims(), &[1, 1, 80]);
            assert!(
                inputs[0]
                    .view()
                    .data()
                    .as_f32()
                    .unwrap()
                    .iter()
                    .all(|v| v.is_finite())
            );
            Ok(self.outputs.clone())
        }
    }

    #[test]
    fn test_speaker_padding_normalization_and_pool_return() {
        let calls = Arc::new(AtomicUsize::new(0));
        let tensor =
            Tensor::new(Shape::new(vec![1, 256]), TensorData::F32(vec![1.0; 256])).unwrap();
        let encoder = SpeakerEmbedder::from_sessions(vec![Box::new(Session {
            outputs: vec![tensor],
            calls: calls.clone(),
        })]);
        for len in [0, 80, 400] {
            assert_eq!(
                encoder.embed(&vec![0.0; len]).unwrap(),
                vec![1.0 / 16.0; 256]
            );
        }
        assert_eq!(calls.load(Ordering::Relaxed), 3);
    }

    #[test]
    fn test_speaker_invalid_outputs_return_errors_and_release_pool_slot() {
        for outputs in [
            vec![],
            vec![Tensor::new(Shape::new(vec![1]), TensorData::F32(vec![1.0])).unwrap()],
            vec![Tensor::new(Shape::new(vec![256]), TensorData::I64(vec![1; 256])).unwrap()],
        ] {
            let encoder = SpeakerEmbedder::from_sessions(vec![Box::new(Session {
                outputs,
                calls: Arc::new(AtomicUsize::new(0)),
            })]);
            assert!(encoder.embed(&[]).is_err());
            assert_eq!(encoder.sessions.available(), 1);
        }
    }

    #[test]
    fn test_speaker_nonfinite_raw_output_preserves_silence_normalization() {
        let tensor =
            Tensor::new(Shape::new(vec![256]), TensorData::F32(vec![f32::NAN; 256])).unwrap();
        let encoder = SpeakerEmbedder::from_sessions(vec![Box::new(Session {
            outputs: vec![tensor],
            calls: Arc::new(AtomicUsize::new(0)),
        })]);
        assert_eq!(encoder.embed(&[]).unwrap(), vec![0.0; 256]);
    }

    #[test]
    fn test_speaker_rejects_empty_pool_before_opening_model() {
        assert!(SpeakerEmbedder::load(Path::new("missing.onnx"), 0).is_err());
    }
}
