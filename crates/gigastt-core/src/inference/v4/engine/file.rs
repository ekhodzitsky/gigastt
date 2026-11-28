//! Bounded native mono file decoding for explicit v4 requests.

use super::{V4Engine, V4Pcm, V4Transcription, decode};
use crate::{error::GigasttError, inference::audio, model::TranscriptStyle};
use bytes::Bytes;
use std::{
    path::Path,
    sync::atomic::{AtomicBool, Ordering},
};

const FULL_CONTEXT_SECONDS: f64 = 30.0;

fn effective_limit(requested: Option<f64>) -> Result<f64, GigasttError> {
    match requested {
        None => Ok(FULL_CONTEXT_SECONDS),
        Some(limit) if limit.is_finite() && limit > 0.0 => Ok(limit.min(FULL_CONTEXT_SECONDS)),
        Some(_) => Err(GigasttError::InvalidInput {
            message: "v4 audio duration limit must be finite and positive".into(),
        }),
    }
}

fn finish_decoded(
    pcm: Vec<f32>,
    limit: f64,
    cancel: Option<&AtomicBool>,
) -> Result<Vec<f32>, GigasttError> {
    decode::check_cancel(cancel)?;
    if pcm.len() > audio::max_samples_for_secs(Some(limit), 16_000) {
        return Err(GigasttError::AudioTooLong {
            observed_secs: pcm.len() as f64 / 16_000.0,
            limit_secs: limit,
        });
    }
    if pcm.len() < 320 {
        return Err(GigasttError::InvalidAudio {
            reason: "v4 requires at least one complete 320-sample window".into(),
        });
    }
    Ok(pcm)
}

fn decode_file(
    path: &Path,
    max_audio_secs: Option<f64>,
    cancel: Option<&AtomicBool>,
) -> Result<Vec<f32>, GigasttError> {
    decode::check_cancel(cancel)?;
    let limit = effective_limit(max_audio_secs)?;
    let path = path.to_str().ok_or_else(|| GigasttError::InvalidInput {
        message: "audio path must be valid UTF-8".into(),
    })?;
    let abort = cancel.map(|flag| move || flag.load(Ordering::Relaxed));
    let pcm = audio::decode_audio_file_with_abort(
        path,
        Some(limit),
        abort
            .as_ref()
            .map(|abort| abort as &(dyn Fn() -> bool + Sync)),
    )
    .map_err(audio::decode_error)?;
    finish_decoded(pcm, limit, cancel)
}

fn decode_bytes(
    bytes: Bytes,
    max_audio_secs: Option<f64>,
    cancel: Option<&AtomicBool>,
) -> Result<Vec<f32>, GigasttError> {
    decode::check_cancel(cancel)?;
    let limit = effective_limit(max_audio_secs)?;
    let abort = cancel.map(|flag| move || flag.load(Ordering::Relaxed));
    let pcm = audio::decode_audio_bytes_with_abort(
        bytes,
        Some(limit),
        abort
            .as_ref()
            .map(|abort| abort as &(dyn Fn() -> bool + Sync)),
    )
    .map_err(audio::decode_error)?;
    finish_decoded(pcm, limit, cancel)
}

impl V4Engine {
    /// Decode a supported container to native mono 16 kHz PCM and recognize
    /// one explicit style. Whole-context v4 is limited to 30 seconds, with
    /// no implicit long-form splitting, raw codec or channel-separation mode.
    pub fn transcribe_file(
        &self,
        path: &Path,
        style: TranscriptStyle,
        cancel: Option<&AtomicBool>,
    ) -> Result<V4Transcription, GigasttError> {
        self.transcribe_file_with_limit(path, style, cancel, None)
    }

    /// Like [Self::transcribe_file], with an additional positive finite duration
    /// limit. The requested limit can lower but never raise the 30-second cap.
    pub fn transcribe_file_with_limit(
        &self,
        path: &Path,
        style: TranscriptStyle,
        cancel: Option<&AtomicBool>,
        max_audio_secs: Option<f64>,
    ) -> Result<V4Transcription, GigasttError> {
        let pcm = decode_file(path, max_audio_secs, cancel)?;
        self.transcribe(V4Pcm::full(&pcm), style, cancel)
    }

    /// Decode a shared container buffer without cloning encoded bytes, then
    /// recognize one explicit style under the same whole-context limit.
    pub fn transcribe_bytes(
        &self,
        bytes: Bytes,
        style: TranscriptStyle,
        cancel: Option<&AtomicBool>,
    ) -> Result<V4Transcription, GigasttError> {
        self.transcribe_bytes_with_limit(bytes, style, cancel, None)
    }

    /// Like [Self::transcribe_bytes], with an additional lower duration limit.
    /// The source-frame decode budget and actual 16 kHz PCM length are checked
    /// before model inference. Cancellation is cooperative at decode/native
    /// inference boundaries; no partial transcript is returned on error.
    pub fn transcribe_bytes_with_limit(
        &self,
        bytes: Bytes,
        style: TranscriptStyle,
        cancel: Option<&AtomicBool>,
        max_audio_secs: Option<f64>,
    ) -> Result<V4Transcription, GigasttError> {
        let pcm = decode_bytes(bytes, max_audio_secs, cancel)?;
        self.transcribe(V4Pcm::full(&pcm), style, cancel)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_v4_file_limit_never_exceeds_full_context() {
        assert_eq!(effective_limit(None).unwrap(), 30.0);
        assert_eq!(effective_limit(Some(50.0)).unwrap(), 30.0);
        assert_eq!(effective_limit(Some(1.25)).unwrap(), 1.25);
        for limit in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            assert!(matches!(
                effective_limit(Some(limit)),
                Err(GigasttError::InvalidInput { .. })
            ));
        }
    }

    #[test]
    fn test_v4_native_file_decode_exact_pcm_and_typed_budget() {
        let samples = vec![0.25; 320];
        let wav = audio::encode_wav_pcm16(&samples, 16_000);
        let actual = decode_bytes(Bytes::from(wav.clone()), None, None).unwrap();
        assert_eq!(actual, samples);
        assert!(matches!(
            decode_bytes(Bytes::from(wav), Some(0.01), None),
            Err(GigasttError::AudioTooLong { limit_secs, .. }) if limit_secs == 0.01
        ));
        let over = audio::encode_wav_pcm16(&vec![0.0; 480_001], 16_000);
        assert!(matches!(
            decode_bytes(Bytes::from(over), None, None),
            Err(GigasttError::AudioTooLong { limit_secs, .. }) if limit_secs == 30.0
        ));
        assert!(matches!(
            decode_bytes(Bytes::from_static(b"invalid audio"), None, None),
            Err(GigasttError::InvalidAudio { .. })
        ));
    }

    #[test]
    fn test_v4_file_post_resample_limit_and_cancel() {
        assert_eq!(
            finish_decoded(vec![0.0; 480_000], 30.0, None)
                .unwrap()
                .len(),
            480_000
        );
        assert!(matches!(
            finish_decoded(vec![0.0; 480_001], 30.0, None),
            Err(GigasttError::AudioTooLong { limit_secs, .. }) if limit_secs == 30.0
        ));
        assert!(matches!(
            finish_decoded(vec![0.0; 319], 30.0, None),
            Err(GigasttError::InvalidAudio { .. })
        ));
        let cancelled = AtomicBool::new(true);
        assert!(matches!(
            decode_bytes(Bytes::from_static(b"invalid"), None, Some(&cancelled)),
            Err(GigasttError::Cancelled)
        ));
        assert!(matches!(
            decode_file(Path::new("/not/a/real/audio/file"), None, Some(&cancelled)),
            Err(GigasttError::Cancelled)
        ));
        assert!(matches!(
            finish_decoded(vec![0.0; 320], 30.0, Some(&cancelled)),
            Err(GigasttError::Cancelled)
        ));
    }

    #[test]
    fn test_v4_native_file_and_bytes_decode_match() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("audio.wav");
        let wav = audio::encode_wav_pcm16(&[0.25; 320], 16_000);
        std::fs::write(&path, &wav).unwrap();
        assert_eq!(
            decode_file(&path, None, None).unwrap(),
            decode_bytes(Bytes::from(wav), None, None).unwrap()
        );
    }
}
