//! Windowed / VAD-region word decode for [`Engine`].

use super::*;

/// One long-form window copied out of a [`PcmWindows`] source so overlapping
/// windows can be decoded on separate triplets without holding a borrow into
/// the rolling file buffer.
struct OwnedPcmWindow {
    start_sample: usize,
    samples: Vec<f32>,
}

struct WindowDecode {
    words: Vec<WordInfo>,
    completed: bool,
}

fn publish_window(
    windows: &dyn PcmWindows,
    ctl: DecodeControls<'_>,
    words: &[WordInfo],
) -> Result<(), GigasttError> {
    if ctl.on_partial.is_some() {
        let mut original = words.to_vec();
        windows.remap_words(&mut original);
        ctl.publish(&original);
    }
    ctl.check_abort()
}

impl From<&PcmWindow<'_>> for OwnedPcmWindow {
    fn from(window: &PcmWindow<'_>) -> Self {
        Self {
            start_sample: window.start_sample,
            samples: window.samples.to_vec(),
        }
    }
}

impl OwnedPcmWindow {
    fn span(&self) -> PcmSpan<'_> {
        PcmSpan {
            start_sample: self.start_sample,
            samples: &self.samples,
        }
    }
}

fn next_owned_window(windows: &mut dyn PcmWindows) -> Result<Option<OwnedPcmWindow>, GigasttError> {
    Ok(windows.next_window()?.map(|w| OwnedPcmWindow::from(&w)))
}

/// Borrowed PCM window (origin + samples) shared by the serial iterator
/// path and owned copies so finish/stitch take one argument.
struct PcmSpan<'a> {
    start_sample: usize,
    samples: &'a [f32],
}

impl<'a> From<&'a PcmWindow<'_>> for PcmSpan<'a> {
    fn from(window: &'a PcmWindow<'_>) -> Self {
        Self {
            start_sample: window.start_sample,
            samples: window.samples,
        }
    }
}

fn stitch_report(
    merged: Vec<WordInfo>,
    start_sample: usize,
    n_samples: usize,
    words: Vec<WordInfo>,
    overlap: usize,
    ctl: DecodeControls<'_>,
) -> Vec<WordInfo> {
    // An abort before any new word must not trim the preceding window's tail.
    let extends_previous = words
        .last()
        .is_some_and(|word| merged.last().is_none_or(|previous| word.end > previous.end));
    let merged = if ctl.aborted() && !extends_previous {
        merged
    } else {
        stitch_chunk_words(merged, words, overlap_mid_seconds(start_sample, overlap))
    };
    if !ctl.aborted() {
        ctl.report((start_sample + n_samples) as u64);
    }
    merged
}

impl Engine {
    /// Decode a 16 kHz f32 buffer to words: single-pass for short inputs (one
    /// encoder Run over the whole buffer), chunked overlapping windows for long
    /// inputs so encoder activation memory stays O(chunk), not O(file). Both
    /// paths produce the same `Vec<WordInfo>` shape. This is the no-VAD path and
    /// the per-region decode used by [`Engine::decode_speech_regions`].
    ///
    /// `biaser` is the effective hotword biaser for this call (engine boot
    /// biaser, a temporary per-request biaser, or `None` when forced off).
    pub(crate) fn decode_words(
        &self,
        samples: &[f32],
        triplet: &mut SessionTriplet,
        biaser: Option<&bias::Biaser>,
        ctl: DecodeControls,
    ) -> Result<Vec<WordInfo>, GigasttError> {
        let spec = window_spec(self.ane_encoder, self.variant.is_ctc());
        if spec.is_single_pass(samples.len()) {
            // A single-pass decode is one encoder Run with no window loop to
            // check between, so honour cancellation before starting it. The
            // caller's watchdog treats this whole clip as one "window".
            if ctl.aborted() {
                return Err(GigasttError::Cancelled);
            }
            let (features, num_frames) = self.features.compute(samples);
            tracing::info!("Extracted {} mel frames", num_frames);
            let mut decoder_state = DecoderState::new(self.tokenizer.blank_id());
            let words = self
                .run_inference(
                    triplet,
                    &features,
                    num_frames,
                    &mut decoder_state,
                    0,
                    false, // file-mode fill floor
                    biaser,
                    ctl.abort,
                )
                .map_err(|e| GigasttError::Inference { source: e.into() })?
                .0;
            ctl.publish(&words);
            ctl.check_abort()?;
            ctl.report(samples.len() as u64);
            Ok(words)
        } else {
            tracing::info!(
                "Long-form chunked decode: {:.1}s in ~{}s windows ({}s overlap, ane={})",
                samples.len() as f64 / 16000.0,
                spec.window() / 16000,
                spec.overlap() / 16000,
                self.ane_encoder,
            );
            self.decode_words_streaming(&mut SliceWindows::new(samples, spec), triplet, biaser, ctl)
        }
    }

    /// Decode only the VAD-detected speech `regions` of `float_samples`: copy the
    /// speech spans into one silence-free buffer, decode it, then remap each
    /// word's start/end from the compressed (silence-removed) timeline back to
    /// the original timeline via [`crate::vad::remap_compressed_seconds`]. Empty
    /// `regions` (no speech) yields no words.
    pub(crate) fn decode_speech_regions(
        &self,
        float_samples: &[f32],
        regions: &[(usize, usize)],
        triplet: &mut SessionTriplet,
        biaser: Option<&bias::Biaser>,
        ctl: DecodeControls,
    ) -> Result<Vec<WordInfo>, GigasttError> {
        if regions.is_empty() {
            tracing::info!("VAD found no speech; skipping decode");
            return Ok(Vec::new());
        }
        let speech_len: usize = regions.iter().map(|(s, e)| e - s).sum();
        let mut speech = Vec::with_capacity(speech_len);
        for &(s, e) in regions {
            speech.extend_from_slice(&float_samples[s..e]);
        }
        tracing::info!(
            "VAD kept {}/{} samples ({} speech region(s))",
            speech_len,
            float_samples.len(),
            regions.len()
        );
        let publish = |words: &[WordInfo]| {
            let mut original = words.to_vec();
            for w in &mut original {
                w.start = crate::vad::remap_compressed_seconds(w.start, regions, 16000.0);
                w.end = crate::vad::remap_compressed_seconds(w.end, regions, 16000.0);
            }
            ctl.publish(&original);
        };
        let mapped = DecodeControls {
            on_partial: ctl.on_partial.map(|_| &publish as &dyn Fn(&[WordInfo])),
            ..ctl
        };
        let mut words = self.decode_words(&speech, triplet, biaser, mapped)?;
        for w in &mut words {
            w.start = crate::vad::remap_compressed_seconds(w.start, regions, 16000.0);
            w.end = crate::vad::remap_compressed_seconds(w.end, regions, 16000.0);
        }
        Ok(words)
    }

    /// Long-form decode: pull overlapping windows from `windows`, encode and
    /// decode each independently with a fresh [`DecoderState`], offset each
    /// chunk's word timestamps by the chunk's absolute start, then stitch the
    /// per-chunk word lists with overlap de-dup via [`stitch_chunk_words`].
    ///
    /// Peak encoder activation memory is bounded by the source's window length
    /// (24s ort / 30s ANE, see [`super::windows::chunk_window_samples`]) rather than the full
    /// file length. Window starts are aligned to encoder frame boundaries
    /// (multiples of `HOP_LENGTH * ENCODER_SUBSAMPLING`) so the per-chunk frame
    /// offset is exact, matching the streaming path's math.
    ///
    /// Takes the PCM as a [`PcmWindows`] source rather than a `&[f32]`, so the
    /// loop makes no assumption that the whole file is in memory.
    ///
    /// When [`Engine::file_window_concurrency`] is `> 1` and the clip has at
    /// least two windows, idle extra triplets are `try_checkout_n`'d from the
    /// batch pool (never waited on) so independent windows can encode in
    /// parallel. A closed pool cancels rather than serial-decoding the rest.
    /// Short files and a saturated pool stay on the serial loop.
    pub(crate) fn decode_words_streaming(
        &self,
        windows: &mut dyn PcmWindows,
        triplet: &mut SessionTriplet,
        biaser: Option<&bias::Biaser>,
        ctl: DecodeControls,
    ) -> Result<Vec<WordInfo>, GigasttError> {
        let overlap = windows.spec().overlap();
        let cap = self.file_window_concurrency();
        if cap <= 1 {
            return self.decode_windows_serial(windows, triplet, biaser, ctl, overlap, Vec::new());
        }
        self.decode_windows_maybe_parallel(windows, triplet, biaser, ctl, overlap, cap)
    }

    fn decode_windows_maybe_parallel(
        &self,
        windows: &mut dyn PcmWindows,
        triplet: &mut SessionTriplet,
        biaser: Option<&bias::Biaser>,
        ctl: DecodeControls,
        overlap: usize,
        cap: usize,
    ) -> Result<Vec<WordInfo>, GigasttError> {
        // Copy the first window so we can peek a second without holding a
        // borrow into `FileWindows`. A single-window file never steals extras.
        let Some(first) = next_owned_window(windows)? else {
            return Ok(Vec::new());
        };
        if ctl.aborted() {
            return Err(GigasttError::Cancelled);
        }
        let Some(second) = next_owned_window(windows)? else {
            let words =
                self.finish_window(Vec::new(), first.span(), overlap, triplet, biaser, ctl)?;
            publish_window(windows, ctl, &words)?;
            return Ok(words);
        };

        let mut extras = match self.pool_for_batch().try_checkout_n(cap.saturating_sub(1)) {
            Ok(guards) => guards,
            // Shutdown closed the pool: do not serial-decode the rest of the
            // file on the slot we already hold. The watchdog also flips abort,
            // but Closed is the stronger signal and wins the race.
            Err(PoolError::Closed) => return Err(GigasttError::Cancelled),
        };
        if extras.is_empty() {
            tracing::debug!(
                cap,
                "long-form window-parallel requested; no idle extra slot, serial"
            );
            let mut merged =
                self.finish_window(Vec::new(), first.span(), overlap, triplet, biaser, ctl)?;
            publish_window(windows, ctl, &merged)?;
            merged = self.finish_window(merged, second.span(), overlap, triplet, biaser, ctl)?;
            publish_window(windows, ctl, &merged)?;
            return self.decode_windows_serial(windows, triplet, biaser, ctl, overlap, merged);
        }

        tracing::info!(slots = 1 + extras.len(), "long-form window-parallel decode");
        let n_slots = 1 + extras.len();
        let mut pending = vec![first, second];
        let mut merged = Vec::new();
        loop {
            if ctl.aborted() {
                return Err(GigasttError::Cancelled);
            }
            while pending.len() < n_slots {
                match next_owned_window(windows)? {
                    Some(w) => pending.push(w),
                    None => break,
                }
            }
            if pending.is_empty() {
                break;
            }
            // Last wave is often shorter than `n_slots`. Drop idle extras
            // before the encoder Run so another request can take them.
            extras.truncate(pending.len().saturating_sub(1));
            // Fill can take a while (container decode). Re-check before the
            // encoder wave so a cancel during pull does not start another Run.
            if ctl.aborted() {
                return Err(GigasttError::Cancelled);
            }
            let decoded =
                self.decode_wave_parallel(&pending, triplet, &mut extras, biaser, ctl.abort)?;
            for (win, decoded) in pending.iter().zip(decoded) {
                merged = stitch_report(
                    merged,
                    win.start_sample,
                    win.samples.len(),
                    decoded.words,
                    overlap,
                    ctl,
                );
                if !decoded.completed {
                    break;
                }
            }
            publish_window(windows, ctl, &merged)?;
            pending.clear();
        }
        Ok(merged)
    }

    fn decode_windows_serial(
        &self,
        windows: &mut dyn PcmWindows,
        triplet: &mut SessionTriplet,
        biaser: Option<&bias::Biaser>,
        ctl: DecodeControls,
        overlap: usize,
        mut merged: Vec<WordInfo>,
    ) -> Result<Vec<WordInfo>, GigasttError> {
        while let Some(window) = windows.next_window()? {
            merged = self.finish_window(
                merged,
                PcmSpan::from(&window),
                overlap,
                triplet,
                biaser,
                ctl,
            )?;
            publish_window(windows, ctl, &merged)?;
        }
        ctl.check_abort()?;
        Ok(merged)
    }

    fn finish_window(
        &self,
        merged: Vec<WordInfo>,
        window: PcmSpan<'_>,
        overlap: usize,
        triplet: &mut SessionTriplet,
        biaser: Option<&bias::Biaser>,
        ctl: DecodeControls,
    ) -> Result<Vec<WordInfo>, GigasttError> {
        if ctl.aborted() {
            return Err(GigasttError::Cancelled);
        }
        let words = self.decode_samples_window(
            window.samples,
            window.start_sample,
            triplet,
            biaser,
            ctl.abort,
        )?;
        let merged = stitch_report(
            merged,
            window.start_sample,
            window.samples.len(),
            words.words,
            overlap,
            ctl,
        );
        Ok(merged)
    }

    fn decode_samples_window(
        &self,
        samples: &[f32],
        start_sample: usize,
        triplet: &mut SessionTriplet,
        biaser: Option<&bias::Biaser>,
        abort: Option<&(dyn Fn() -> bool + Sync)>,
    ) -> Result<WindowDecode, GigasttError> {
        let frame_samples = HOP_LENGTH * ENCODER_SUBSAMPLING;
        let (features, num_frames) = self.features.compute(samples);
        let frame_offset = start_sample / frame_samples;
        let mut decoder_state = DecoderState::new(self.tokenizer.blank_id());
        self.run_inference(
            triplet,
            &features,
            num_frames,
            &mut decoder_state,
            frame_offset,
            false, // file-mode fill floor
            biaser,
            abort,
        )
        .map(|r| WindowDecode {
            words: r.0,
            completed: !abort.is_some_and(|abort| abort()),
        })
        .map_err(|e| GigasttError::Inference { source: e.into() })
    }

    fn decode_wave_parallel(
        &self,
        wave: &[OwnedPcmWindow],
        primary: &mut SessionTriplet,
        extras: &mut [PoolGuard<SessionTriplet>],
        biaser: Option<&bias::Biaser>,
        abort: Option<&(dyn Fn() -> bool + Sync)>,
    ) -> Result<Vec<WindowDecode>, GigasttError> {
        debug_assert!(!wave.is_empty());
        // `zip` would silently drop windows if extras ran short — that is a
        // lost-audio bug, so fail loud instead of stitching a hole.
        if wave.len() > 1 + extras.len() {
            return Err(GigasttError::Inference {
                source: anyhow::anyhow!("window-parallel decode ran out of session triplets")
                    .into(),
            });
        }
        if wave.len() == 1 {
            return Ok(vec![self.decode_samples_window(
                &wave[0].samples,
                wave[0].start_sample,
                primary,
                biaser,
                abort,
            )?]);
        }

        std::thread::scope(|s| {
            let mut handles = Vec::with_capacity(wave.len());
            handles.push(s.spawn(|| {
                self.decode_samples_window(
                    &wave[0].samples,
                    wave[0].start_sample,
                    primary,
                    biaser,
                    abort,
                )
            }));
            for (win, guard) in wave[1..].iter().zip(extras.iter_mut()) {
                handles.push(s.spawn(move || {
                    self.decode_samples_window(&win.samples, win.start_sample, guard, biaser, abort)
                }));
            }
            // Join every handle before returning: a leftover panicking sibling
            // would otherwise make `thread::scope` panic on the way out and
            // swallow the first worker's `Result`.
            let mut out = Vec::with_capacity(handles.len());
            let mut first_err: Option<GigasttError> = None;
            for handle in handles {
                match handle.join() {
                    Ok(Ok(words)) => {
                        if first_err.is_none() {
                            out.push(words);
                        }
                    }
                    Ok(Err(e)) => {
                        if first_err.is_none() {
                            first_err = Some(e);
                        }
                    }
                    Err(_) => {
                        if first_err.is_none() {
                            first_err = Some(GigasttError::Inference {
                                source: anyhow::anyhow!("window decode thread panicked").into(),
                            });
                        }
                    }
                }
            }
            match first_err {
                Some(e) => Err(e),
                None => Ok(out),
            }
        })
    }

    /// Decode a 16 kHz f32 buffer to words, applying VAD if configured.
    ///
    /// The VAD path is taken only when a VAD is attached AND the caller hasn't
    /// opted out via `overrides.vad`. `?vad=false` forces whole-buffer decode
    /// even on a VAD-enabled engine; `None` uses the engine default (VAD path
    /// iff a VAD is attached). `vad = Some(true)` on a VAD-less engine can't
    /// reach here — callers should validate overrides first — but the
    /// `self.vad.is_some()` guard keeps this correct regardless.
    ///
    /// Hotword selection via `hotwords`:
    /// - `None` → engine boot biaser
    /// - `Some(empty)` → force biasing off
    /// - `Some(phrases)` → temporary biaser built for this request only
    pub(crate) fn decode_words_for_samples(
        &self,
        float_samples: &[f32],
        triplet: &mut SessionTriplet,
        overrides: &TranscribeOverrides,
        hotwords: Option<&HotwordOverride>,
        ctl: DecodeControls,
    ) -> Result<Vec<WordInfo>, GigasttError> {
        // Temporary biaser only when the request supplies hotwords. Owned
        // here so the `Option<&Biaser>` passed into decode stays valid for
        // the whole call without cloning the engine's boot biaser.
        let request_biaser = hotwords.and_then(|hw| self.build_request_biaser(hw));
        let biaser = self.select_biaser(hotwords, &request_biaser);

        let use_vad = self.vad.is_some() && overrides.vad.unwrap_or(true);
        match (use_vad, &self.vad) {
            (true, Some(vad)) => {
                match vad.speech_regions_with_abort(
                    float_samples,
                    &self.vad_config,
                    ctl.abort.map(|a| a as &dyn Fn() -> bool),
                ) {
                    Ok(regions) if regions.is_empty() => {
                        // Tone / continuous speech can yield zero regions on a bad
                        // threshold; fall back to fixed-window / full decode rather
                        // than returning an empty transcript.
                        tracing::warn!(
                            "VAD found no speech regions; falling back to full/chunked decode"
                        );
                        self.decode_words(float_samples, triplet, biaser, ctl)
                    }
                    Ok(regions) => {
                        self.decode_speech_regions(float_samples, &regions, triplet, biaser, ctl)
                    }
                    Err(e) => {
                        // A flipped abort flag stops the VAD scan too; surface it
                        // as cancellation instead of silently re-decoding the whole
                        // file, which would ignore the cancel and keep the triplet.
                        if ctl.aborted() {
                            return Err(GigasttError::Cancelled);
                        }
                        tracing::warn!("VAD failed, decoding full audio: {e:#}");
                        self.decode_words(float_samples, triplet, biaser, ctl)
                    }
                }
            }
            _ => self.decode_words(float_samples, triplet, biaser, ctl),
        }
    }
}
