//! Shared file-transcription core for REST, OpenAI, and jobs.
//!
//! Surfaces keep their own HTTP / progress envelopes; the blocking
//! channels / diarization / hotwords routing lives here so those paths
//! cannot drift.

use axum::body::Bytes;
use gigastt_core::error::GigasttError;
use gigastt_core::inference::{
    Engine, HotwordOverride, OwnedReservation, SessionTriplet, TranscribeOverrides,
    TranscribeRequest, TranscribeResult, TranscribeSource,
};

/// Options for a single file-transcription run after request validation.
#[derive(Clone, Default)]
pub(crate) struct FileTranscribeOpts {
    pub overrides: TranscribeOverrides,
    pub hotwords: Option<HotwordOverride>,
    pub split_channels: bool,
    pub diarization: bool,
    /// When set, decode the body as a raw telephony stream and re-wrap as WAV
    /// before the engine path. REST-only today; jobs do not expose `?codec=`.
    pub raw_codec: Option<(gigastt_core::inference::audio::TelephonyCodec, u32)>,
}

/// Decode raw telephony bytes to an in-memory PCM16 WAV for engine paths.
pub(crate) fn raw_codec_to_wav(
    body: &[u8],
    codec: gigastt_core::inference::audio::TelephonyCodec,
    sample_rate: u32,
) -> Result<Bytes, GigasttError> {
    let samples = gigastt_core::inference::audio::decode_telephony_raw(body, codec, sample_rate)
        .map_err(|e| GigasttError::InvalidAudio {
            reason: format!("{e:#}"),
        })?;
    Ok(Bytes::from(
        gigastt_core::inference::audio::encode_wav_pcm16(&samples, 16000),
    ))
}

/// Blocking file transcription against a reserved triplet.
///
/// Handles raw-codec rewrap, `channels=split` with mono fallback, diarization,
/// and the default mono path. Callers own pool checkout, panic wrapping, and
/// timeout policy.
pub(crate) fn run_file_transcribe_blocking(
    engine: &Engine,
    body: Bytes,
    reservation: &mut OwnedReservation<SessionTriplet>,
    opts: &FileTranscribeOpts,
) -> Result<TranscribeResult, GigasttError> {
    let body = match opts.raw_codec {
        Some((codec, rate)) => raw_codec_to_wav(&body, codec, rate)?,
        None => body,
    };

    if opts.split_channels {
        let channels =
            gigastt_core::inference::audio::decode_audio_bytes_shared_channels(body.clone())
                .map_err(|e| GigasttError::InvalidAudio {
                    reason: format!("{e:#}"),
                })?;
        let fallback_reason = match channels.len() {
            0 => Some("no channels"),
            1 => Some("mono audio"),
            2 if gigastt_core::inference::audio::is_dual_mono(&channels) => Some("dual-mono audio"),
            n if n > 2 => Some("more than two channels"),
            _ => None,
        };
        if let Some(reason) = fallback_reason {
            // The mono path re-decodes from `body`; release the split channels
            // first so both copies are never resident at once.
            drop(channels);
            tracing::warn!(
                "channels=split requested but {reason} detected; falling back to mono transcription"
            );
            engine.transcribe_request(
                TranscribeRequest::new(TranscribeSource::Bytes(body)),
                reservation,
            )
        } else {
            engine.transcribe_request(
                TranscribeRequest::new(TranscribeSource::Channels(&channels)),
                reservation,
            )
        }
    } else {
        // Mono path (optional diarization) via unified Engine request API.
        engine.transcribe_request(
            TranscribeRequest::new(TranscribeSource::Bytes(body))
                .with_overrides(opts.overrides)
                .with_hotwords(opts.hotwords.as_ref())
                .with_diarization(opts.diarization),
            reservation,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_file_transcribe_opts_default() {
        let opts = FileTranscribeOpts::default();
        assert!(!opts.split_channels);
        assert!(!opts.diarization);
        assert!(opts.hotwords.is_none());
        assert!(opts.raw_codec.is_none());
        assert!(opts.overrides.punctuation.is_none());
        assert!(opts.overrides.itn.is_none());
        assert!(opts.overrides.vad.is_none());
    }

    #[test]
    fn test_raw_codec_to_wav_rejects_empty_pcmu() {
        // Empty PCMU body is invalid for telephony decode.
        let err = raw_codec_to_wav(
            &[],
            gigastt_core::inference::audio::TelephonyCodec::Pcmu,
            8000,
        )
        .unwrap_err();
        match err {
            GigasttError::InvalidAudio { .. } => {}
            other => panic!("expected InvalidAudio, got {other:?}"),
        }
    }
}
