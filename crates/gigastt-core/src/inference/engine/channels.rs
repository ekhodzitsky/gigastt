//! Stereo / `channels=split` transcription.

use super::*;

fn publish_channel_partial(
    completed: &[TranscribeResult],
    words: &[WordInfo],
    ctl: DecodeControls<'_>,
) {
    let mut channels = completed.to_vec();
    channels.push(TranscribeResult {
        text: String::new(),
        words: words.to_vec(),
        duration_s: 0.0,
        confidence: None,
    });
    ctl.publish(&merge_channel_results(channels).words);
}

impl Engine {
    /// Transcribe a multi-channel recording with one speaker label per channel.
    ///
    /// Runs the engine once per channel sequentially on the supplied triplet. The
    /// caller is responsible for deciding whether to use this mode (e.g. after
    /// checking for dual-mono). Channel 0 becomes `speaker_0`, channel 1
    /// `speaker_1`, and so on. Results are merged into a single chronologically
    /// ordered transcript.
    ///
    /// Per-channel `text` fields are ignored: the merged transcript's text is
    /// rebuilt from the merged words after the final ITN/punctuation pass.
    ///
    /// Thin wrapper over [`Engine::transcribe_request`] with default overrides
    /// and no hotwords.
    #[cfg(feature = "file-decode")]
    pub fn transcribe_channels(
        &self,
        channels: &[Vec<f32>],
        triplet: &mut SessionTriplet,
    ) -> Result<TranscribeResult, GigasttError> {
        self.transcribe_request(
            TranscribeRequest::new(TranscribeSource::Channels(channels)),
            triplet,
        )
    }

    /// Per-channel decode + merge used by [`TranscribeSource::Channels`].
    pub(crate) fn transcribe_channels_inner(
        &self,
        channels: &[Vec<f32>],
        triplet: &mut SessionTriplet,
        overrides: &TranscribeOverrides,
        hotwords: Option<&HotwordOverride>,
        ctl: DecodeControls,
    ) -> Result<TranscribeResult, GigasttError> {
        if channels.is_empty() {
            return Ok(TranscribeResult {
                text: String::new(),
                words: Vec::new(),
                duration_s: 0.0,
                confidence: None,
            });
        }

        let mut per_channel = Vec::with_capacity(channels.len());
        for channel_samples in channels {
            let publish = |words: &[WordInfo]| publish_channel_partial(&per_channel, words, ctl);
            let channel_ctl = DecodeControls {
                on_partial: ctl.on_partial.map(|_| &publish as &dyn Fn(&[WordInfo])),
                ..ctl
            };
            let words = self.decode_words_for_samples(
                channel_samples,
                triplet,
                overrides,
                hotwords,
                channel_ctl,
            )?;
            let duration_s = channel_samples.len() as f64 / 16000.0;
            per_channel.push(TranscribeResult {
                confidence: aggregate_confidence(&words),
                text: String::new(),
                words,
                duration_s,
            });
        }

        let merged = merge_channel_results(per_channel);
        Ok(self.finish_transcribe_result(merged.words, merged.duration_s, overrides))
    }

    /// Streaming twin of the `channels=split` decode.
    ///
    /// Each channel of `data` runs through the windowed loop the mono file path
    /// uses, so
    /// peak audio memory is one window instead of every channel of the whole
    /// file — which is what kept channel splitting on a duration ceiling. The
    /// per-channel results are merged exactly as the whole-buffer twin merges
    /// them.
    ///
    /// The caller decides *whether* to split; use
    /// [`scan_channels`](crate::inference::audio::scan_channels), which answers
    /// that in one pass without materializing anything.
    ///
    /// No progress sink is threaded through: each channel restarts the sample
    /// clock, so a shared monotonic counter would go backwards — the same
    /// reason the whole-buffer twin passes `abort_only`.
    // Bundling these behind `&TranscribeRequest` is what one would want, but the
    // caller's match moves `data` out of `req.source`, so the request cannot be
    // borrowed whole afterwards. Same shape as `run_inference` above.
    #[allow(clippy::too_many_arguments)]
    #[cfg(feature = "file-decode")]
    pub(crate) fn transcribe_channel_streams(
        &self,
        data: bytes::Bytes,
        channels: usize,
        max_audio_secs: Option<f64>,
        triplet: &mut SessionTriplet,
        overrides: &TranscribeOverrides,
        hotwords: Option<&HotwordOverride>,
        ctl: DecodeControls,
    ) -> Result<TranscribeResult, GigasttError> {
        let request_biaser = hotwords.and_then(|hw| self.build_request_biaser(hw));
        let biaser = self.select_biaser(hotwords, &request_biaser);

        let spec = window_spec(self.ane_encoder, self.variant.is_ctc());
        let mut per_channel = Vec::with_capacity(channels);
        for k in 0..channels {
            let publish = |words: &[WordInfo]| publish_channel_partial(&per_channel, words, ctl);
            let channel_ctl = DecodeControls {
                on_partial: ctl.on_partial.map(|_| &publish as &dyn Fn(&[WordInfo])),
                ..ctl
            };
            let mut windows =
                audio::FileWindows::from_bytes_channel(data.clone(), spec, max_audio_secs, k)
                    .map_err(audio::decode_error)?;
            let words = self.decode_words_streaming(&mut windows, triplet, biaser, channel_ctl)?;
            per_channel.push(TranscribeResult {
                confidence: aggregate_confidence(&words),
                text: String::new(),
                words,
                duration_s: windows.total_16k_samples() as f64 / 16000.0,
            });
        }

        let merged = merge_channel_results(per_channel);
        Ok(self.finish_transcribe_result(merged.words, merged.duration_s, overrides))
    }
}
