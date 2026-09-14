//! Windowed file-transcription path (`transcribe_stream_*`).

use super::*;

impl Engine {
    /// True when a `Path` / `Bytes` request can be served by the windowed
    /// streaming decode, whose peak audio memory is O(one window) rather than
    /// O(file).
    ///
    /// Diarization embeds the whole clip in a second pass, so it cannot run
    /// against a stream that is never fully resident: it still forces the
    /// whole-buffer decode, and with it the duration ceiling. VAD does not — it
    /// is causal, and [`VadWindows`](audio::VadWindows) runs it inside the
    /// stream. `diarize` only matters with the `diarization` feature compiled
    /// in.
    #[cfg(feature = "file-decode")]
    pub(crate) fn stream_eligible(&self, diarize: bool) -> bool {
        !(cfg!(feature = "diarization") && diarize)
    }

    /// Streaming mono file transcription, with the VAD stage in front when one
    /// is attached and the request did not opt out.
    ///
    /// `open` builds a fresh window source for a given geometry. It is called
    /// once, or twice when the VAD stage declines — the model failed mid-stream,
    /// or the scan found no speech at all in a non-empty clip — in which case
    /// the clip is decoded whole, exactly as the batch path did.
    #[cfg(feature = "file-decode")]
    pub(crate) fn transcribe_stream_file(
        &self,
        open: impl Fn(WindowSpec) -> Result<audio::FileWindows, GigasttError>,
        triplet: &mut SessionTriplet,
        overrides: &TranscribeOverrides,
        hotwords: Option<&HotwordOverride>,
        ctl: DecodeControls,
    ) -> Result<TranscribeResult, GigasttError> {
        let use_vad = self.vad.is_some() && overrides.vad.unwrap_or(true);
        if let (true, Some(vad)) = (use_vad, &self.vad) {
            let wall_start = std::time::Instant::now();
            let request_biaser = hotwords.and_then(|hw| self.build_request_biaser(hw));
            let biaser = self.select_biaser(hotwords, &request_biaser);
            let mut windows = audio::VadWindows::new(
                open(audio::VadWindows::pull_spec())?,
                vad,
                &self.vad_config,
                window_spec(self.ane_encoder, self.variant.is_ctc()),
                ctl.abort.map(|a| a as &dyn Fn() -> bool),
            );
            let mut words = self.decode_words_streaming(&mut windows, triplet, biaser, ctl)?;
            if !windows.needs_fallback() {
                // Words are decoded on the compressed (silence-removed)
                // timeline; put them back on the clip's own.
                let regions = windows.regions();
                for w in &mut words {
                    w.start = crate::vad::remap_compressed_seconds(w.start, regions, 16000.0);
                    w.end = crate::vad::remap_compressed_seconds(w.end, regions, 16000.0);
                }
                let duration_s = windows.total_16k_samples() as f64 / 16000.0;
                let wall_s = wall_start.elapsed().as_secs_f64();
                tracing::info!(
                    audio_s = format_args!("{duration_s:.2}"),
                    wall_s = format_args!("{wall_s:.2}"),
                    rtf = format_args!(
                        "{:.3}",
                        if duration_s > 0.0 {
                            wall_s / duration_s
                        } else {
                            0.0
                        }
                    ),
                    regions = regions.len(),
                    "transcribe complete (streaming windows, vad)"
                );
                return Ok(self.finish_transcribe_result(words, duration_s, overrides));
            }
            // Either the VAD found no speech at all — tone or continuous speech
            // against a bad threshold — or the model failed mid-stream (already
            // logged with the cause). Both re-read the clip and decode it whole
            // rather than returning an empty transcript.
            tracing::warn!("VAD produced no usable speech regions; decoding full audio");
        }
        self.transcribe_stream_mono(
            open(window_spec(self.ane_encoder, self.variant.is_ctc()))?,
            triplet,
            overrides,
            hotwords,
            ctl,
        )
    }

    /// Mono file-transcription tail that pulls windows straight from the
    /// container instead of decoding the whole file first.
    ///
    /// Equivalent to [`Engine::transcribe_samples_with_overrides`] on the
    /// no-VAD, no-diarization path: [`FileWindows`](audio::FileWindows) yields
    /// exactly the geometry [`SliceWindows`] would over the fully-decoded buffer
    /// (one window for a single-pass-length stream, overlapping windows beyond),
    /// so [`Engine::decode_words_streaming`] produces the same words — but peak
    /// audio memory no longer scales with duration. The duration cap and its
    /// exact error string are enforced inside the decode as before.
    #[cfg(feature = "file-decode")]
    pub(crate) fn transcribe_stream_mono(
        &self,
        mut windows: audio::FileWindows,
        triplet: &mut SessionTriplet,
        overrides: &TranscribeOverrides,
        hotwords: Option<&HotwordOverride>,
        ctl: DecodeControls,
    ) -> Result<TranscribeResult, GigasttError> {
        let wall_start = std::time::Instant::now();

        // Hotword biaser: engine boot biaser, temporary per-request, or off.
        let request_biaser = hotwords.and_then(|hw| self.build_request_biaser(hw));
        let biaser = self.select_biaser(hotwords, &request_biaser);

        let words = self.decode_words_streaming(&mut windows, triplet, biaser, ctl)?;
        // Exact once every window is consumed (the loop above drains to EOF).
        let duration_s = windows.total_16k_samples() as f64 / 16000.0;
        let result = self.finish_transcribe_result(words, duration_s, overrides);

        let wall_s = wall_start.elapsed().as_secs_f64();
        let rtf = if duration_s > 0.0 {
            wall_s / duration_s
        } else {
            0.0
        };
        tracing::info!(
            audio_s = format_args!("{duration_s:.2}"),
            wall_s = format_args!("{wall_s:.2}"),
            rtf = format_args!("{rtf:.3}"),
            "transcribe complete (streaming windows)"
        );

        Ok(result)
    }
}
