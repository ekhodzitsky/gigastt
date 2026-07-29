//! Windowed PCM source for long-form file decode.
//!
//! Long-form decode used to own its `while start < total { … }` loop directly
//! over one `&[f32]` holding the whole file. This module puts that window
//! geometry behind a small source trait so the decode loop no longer assumes
//! the PCM is fully materialized: [`SliceWindows`] is the buffer-backed source
//! used today, and a decoder-backed source can be added later without touching
//! the loop.
//!
//! [`PcmWindows::next_window`] **lends** its window — the returned
//! [`PcmWindow`] borrows the source — so a source can hand out a view into a
//! decoder's own scratch buffer instead of copying. That borrow shape is why
//! this is not an [`Iterator`].

use crate::error::GigasttError;
use crate::inference::{ENCODER_SUBSAMPLING, HOP_LENGTH};

#[cfg(feature = "file-decode")]
use anyhow::{Context, Result};
#[cfg(feature = "file-decode")]
use bytes::Bytes;
#[cfg(feature = "file-decode")]
use symphonia::core::codecs::audio::well_known::CODEC_ID_OPUS;
#[cfg(feature = "file-decode")]
use symphonia::core::codecs::audio::{AudioDecoder, AudioDecoderOptions};
#[cfg(feature = "file-decode")]
use symphonia::core::formats::probe::Hint;
#[cfg(feature = "file-decode")]
use symphonia::core::formats::{FormatOptions, FormatReader, TrackType};
#[cfg(feature = "file-decode")]
use symphonia::core::io::MediaSourceStream;
#[cfg(feature = "file-decode")]
use symphonia::core::meta::MetadataOptions;

#[cfg(feature = "file-decode")]
use super::decode::{BytesMediaSource, mix_channels_to_mono};
#[cfg(feature = "file-decode")]
use super::opus::{decode_opus_channels, next_demux_packet};
#[cfg(feature = "file-decode")]
use super::resample::{RESAMPLE_STAGING_FRAMES, ResampleTo16k, SampleRate};
#[cfg(feature = "file-decode")]
use super::telephony::{sniffs_as_g722_wav, try_decode_g722_wav};
#[cfg(feature = "file-decode")]
use super::{
    MAX_SAMPLE_RATE, audio_too_long_err, decode_error, resolve_budget, whole_buffer_limit_secs,
};

/// Samples per encoder output frame (`HOP_LENGTH * ENCODER_SUBSAMPLING`,
/// 640 @16 kHz). Window starts are multiples of this so each window's frame
/// offset is integral.
const FRAME_SAMPLES: usize = HOP_LENGTH * ENCODER_SUBSAMPLING;

/// Long-form window geometry, all in samples @16 kHz.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct WindowSpec {
    single_pass_max: usize,
    window: usize,
    stride: usize,
}

impl WindowSpec {
    /// Build a spec from the single-pass ceiling, the window length, and the
    /// overlap retained between consecutive windows.
    ///
    /// The stride (`window - overlap`) is aligned **down** to an encoder-frame
    /// boundary so every window start maps to an integral frame offset;
    /// otherwise the offset would drift by a sub-frame each hop. It is clamped
    /// to one frame so a mis-specified `overlap >= window` cannot produce a
    /// zero-stride (non-advancing) source.
    pub(crate) fn new(single_pass_max: usize, window: usize, overlap: usize) -> Self {
        let stride =
            (window.saturating_sub(overlap) / FRAME_SAMPLES * FRAME_SAMPLES).max(FRAME_SAMPLES);
        Self {
            single_pass_max,
            window,
            stride,
        }
    }

    /// Window length in samples.
    pub(crate) fn window(&self) -> usize {
        self.window
    }

    /// Frame-aligned distance between consecutive window starts.
    pub(crate) fn stride(&self) -> usize {
        self.stride
    }

    /// Largest total (samples @16 kHz) that stays on the single-pass branch — one
    /// encoder Run over the whole buffer rather than overlapping windows.
    pub(crate) fn single_pass_max(&self) -> usize {
        self.single_pass_max
    }

    /// Samples shared by two consecutive windows (`window - stride`). Equals the
    /// requested overlap whenever that overlap is already frame-aligned.
    pub(crate) fn overlap(&self) -> usize {
        self.window.saturating_sub(self.stride)
    }

    /// True when `total` samples are short enough for the single-pass (one
    /// encoder Run over the whole buffer) branch.
    pub(crate) fn is_single_pass(&self, total: usize) -> bool {
        total <= self.single_pass_max
    }

    /// Sentinel geometry for flat (`drain_to_vec`) decode, which never windows.
    /// Its window/stride are never read; only [`FileWindows::drain_to_vec`] uses
    /// a `FileWindows` built with it.
    #[cfg(feature = "file-decode")]
    pub(crate) fn flat() -> Self {
        Self::new(usize::MAX, usize::MAX, 0)
    }
}

/// One decode window lent by a [`PcmWindows`] source.
pub(crate) struct PcmWindow<'a> {
    /// Absolute offset of `samples[0]` in the stream, in samples @16 kHz.
    pub(crate) start_sample: usize,
    /// The window's PCM.
    pub(crate) samples: &'a [f32],
}

/// A source of overlapping decode windows.
pub(crate) trait PcmWindows {
    /// The window geometry this source yields.
    fn spec(&self) -> WindowSpec;

    /// Lend the next window, or `Ok(None)` once the stream is exhausted.
    fn next_window(&mut self) -> Result<Option<PcmWindow<'_>>, GigasttError>;
}

/// [`PcmWindows`] over a fully materialized buffer.
///
/// Yields exactly the `(start, end)` sequence of the loop it replaced:
/// `while start < total { end = min(start + window, total); …; if end == total
/// { break } start += stride }`.
pub(crate) struct SliceWindows<'a> {
    samples: &'a [f32],
    spec: WindowSpec,
    next_start: usize,
    done: bool,
}

impl<'a> SliceWindows<'a> {
    pub(crate) fn new(samples: &'a [f32], spec: WindowSpec) -> Self {
        Self {
            samples,
            spec,
            next_start: 0,
            done: false,
        }
    }
}

impl PcmWindows for SliceWindows<'_> {
    fn spec(&self) -> WindowSpec {
        self.spec
    }

    fn next_window(&mut self) -> Result<Option<PcmWindow<'_>>, GigasttError> {
        let total = self.samples.len();
        if self.done || self.next_start >= total {
            return Ok(None);
        }
        let start = self.next_start;
        let end = (start + self.spec.window()).min(total);
        // The replaced loop stopped on `end == total` rather than on the next
        // start passing `total`, so a window that reaches the end is the last
        // one even when `start + stride` is still short of `total`.
        if end == total {
            self.done = true;
        } else {
            self.next_start = start + self.spec.stride();
        }
        Ok(Some(PcmWindow {
            start_sample: start,
            samples: &self.samples[start..end],
        }))
    }
}

/// The decode engine behind [`FileWindows`]: a streaming symphonia loop, or an
/// already-materialized buffer for the formats that cannot stream.
#[cfg(feature = "file-decode")]
enum Source {
    /// Container decoded packet-by-packet, resampled to 16 kHz as it goes.
    Streaming {
        format: Box<dyn FormatReader>,
        decoder: Box<dyn AudioDecoder>,
        track_id: u32,
        /// Source (container) sample rate — the units the length budget counts.
        sample_rate: u32,
        /// Running SOURCE-rate frame count, tracked separately from the 16 kHz
        /// accumulator because the length budget is expressed in source frames.
        source_frames: usize,
        /// Source-rate frame budget. `usize::MAX` when the caller imposed no
        /// limit — the streaming path is O(one window), so length is unbounded
        /// by default; the flat drain and the whole-buffer callers pass a finite
        /// budget (see [`max_samples_for_secs`]).
        max_samples: usize,
        /// The seconds limit `max_samples` was derived from, echoed verbatim in
        /// [`AudioTooLong`](crate::error::GigasttError::AudioTooLong) on a trip.
        limit_secs: f64,
        /// Boxed: it carries the heavyweight rubato FIR state, so keeping it
        /// behind a pointer keeps the `Streaming` variant small.
        resampler: Box<ResampleTo16k>,
        /// Per-packet interleaved scratch, hoisted out of the decode loop.
        interleaved: Vec<f32>,
    },
    /// The whole 16 kHz stream is already in [`FileWindows::buf`]. Used by the
    /// Opus fallback (it still accumulates per channel — streaming it is out of
    /// scope) and the G.722-in-WAV telephony path (no symphonia decoder).
    Eager,
}

/// A [`PcmWindows`] source that pulls overlapping decode windows straight from an
/// audio container, so peak audio memory is O(one window) rather than O(file).
///
/// It holds only a rolling 16 kHz buffer — one window plus a packet of
/// look-ahead — and drops each window's consumed prefix before decoding the
/// next, so a three-hour file costs the same resident audio memory as a
/// thirty-second one. The window sequence is byte-identical to
/// [`SliceWindows`] over the same fully-decoded buffer:
///
/// - a stream that fits the single-pass ceiling yields exactly one window over
///   the whole buffer, matching `Engine::decode_words`' non-windowed branch;
/// - a longer stream yields the standard overlapping geometry.
///
/// [`FileWindows::drain_to_vec`] runs the same decode flat (no windowing) and is
/// byte-identical to the whole-buffer decoder the public `decode_audio_*`
/// wrappers used to call.
#[cfg(feature = "file-decode")]
pub(crate) struct FileWindows {
    src: Source,
    /// True once the container is exhausted (or eagerly materialized).
    eof: bool,
    /// True once the resampler's staged remainder has been flushed at EOF.
    finished: bool,
    /// Rolling 16 kHz buffer holding `[buf_start_abs, decoded_16k_total)`.
    buf: Vec<f32>,
    /// Absolute sample index (@16 kHz) of `buf[0]`.
    buf_start_abs: usize,
    /// Total 16 kHz samples decoded so far (== `buf_start_abs + buf.len()`).
    decoded_16k_total: usize,
    spec: WindowSpec,
    /// Absolute start (@16 kHz) of the next window to yield.
    next_start: usize,
    /// True until the first window's single-pass-vs-windowed decision is made.
    first: bool,
    done: bool,
}

#[cfg(feature = "file-decode")]
impl FileWindows {
    /// Open a file for windowed decode. Mirrors `decode_audio_file`'s probe/hint
    /// setup, including the G.722-in-WAV telephony sniff.
    pub(crate) fn open(path: &str, spec: WindowSpec, max_audio_secs: Option<f64>) -> Result<Self> {
        if sniffs_as_g722_wav(path)? {
            let bytes = std::fs::read(path)
                .with_context(|| format!("Failed to read audio file: {path}"))?;
            if let Some(result) = try_decode_g722_wav(&bytes, max_audio_secs) {
                return Ok(Self::eager(result?, spec));
            }
        }
        let file = std::fs::File::open(path)
            .with_context(|| format!("Failed to open audio file: {path}"))?;
        let mss = MediaSourceStream::new(Box::new(file), Default::default());
        let mut hint = Hint::new();
        if let Some(ext) = std::path::Path::new(path)
            .extension()
            .and_then(|e| e.to_str())
        {
            hint.with_extension(ext);
        }
        Self::from_mss(mss, hint, spec, max_audio_secs)
    }

    /// Open a shared [`Bytes`] buffer for windowed decode. `BytesMediaSource` is
    /// seekable, so the isomp4 demuxer's trailing-`moov` seek works; no spool.
    pub(crate) fn from_bytes(
        data: Bytes,
        spec: WindowSpec,
        max_audio_secs: Option<f64>,
    ) -> Result<Self> {
        if let Some(result) = try_decode_g722_wav(&data, max_audio_secs) {
            return Ok(Self::eager(result?, spec));
        }
        let source = BytesMediaSource::new(data);
        let mss = MediaSourceStream::new(Box::new(source), Default::default());
        Self::from_mss(mss, Hint::new(), spec, max_audio_secs)
    }

    /// Probe the container and either set up the streaming decoder or, for Opus /
    /// G.722, eagerly materialize the whole 16 kHz buffer. Scalar params are
    /// copied out (and the non-Opus decoder built) inside one borrow scope so the
    /// `FormatReader` is free to move or be driven afterwards — the same shape as
    /// the whole-buffer `decode_audio_inner`.
    fn from_mss(
        mss: MediaSourceStream<'static>,
        hint: Hint,
        spec: WindowSpec,
        max_audio_secs: Option<f64>,
    ) -> Result<Self> {
        let mut format = symphonia::default::get_probe()
            .probe(
                &hint,
                mss,
                FormatOptions::default(),
                MetadataOptions::default(),
            )
            .context("Unsupported audio format")?;

        let (track_id, sample_rate, channels, decoder_opt) = {
            let track = format
                .default_track(TrackType::Audio)
                .context("No audio track found")?;
            let track_id = track.id;
            let audio_params = track
                .codec_params
                .as_ref()
                .and_then(|p| p.audio())
                .context("No audio codec parameters")?;
            let sample_rate = audio_params.sample_rate.context("Unknown sample rate")?;
            if sample_rate == 0 || sample_rate > MAX_SAMPLE_RATE {
                anyhow::bail!("Unsupported sample rate: {sample_rate}Hz");
            }
            let channels = audio_params
                .channels
                .as_ref()
                .map(|c| c.count())
                .unwrap_or(1);
            // Opus is demuxed by symphonia but has no symphonia decoder; the
            // `opus-rs` fallback needs the `FormatReader`, so build no decoder
            // here and take the eager branch below.
            let decoder_opt = if audio_params.codec == CODEC_ID_OPUS {
                None
            } else {
                Some(
                    symphonia::default::get_codecs()
                        .make_audio_decoder(audio_params, &AudioDecoderOptions::default())
                        .context("Unsupported audio codec")?,
                )
            };
            (track_id, sample_rate, channels, decoder_opt)
        };

        let (max_samples, limit_secs) = resolve_budget(max_audio_secs, sample_rate);

        match decoder_opt {
            None => {
                // Opus is accumulated whole-buffer (the fallback decoder does not
                // stream), so it must stay under the whole-buffer safety ceiling
                // even when the caller left the streaming budget unbounded.
                let (opus_max, opus_limit) =
                    resolve_budget(Some(whole_buffer_limit_secs(max_audio_secs)), sample_rate);
                let mono = mix_channels_to_mono(&decode_opus_channels(
                    &mut *format,
                    track_id,
                    channels,
                    opus_max,
                    opus_limit,
                )?);
                let mut resampler = ResampleTo16k::new(SampleRate(sample_rate), None);
                for piece in mono.chunks(RESAMPLE_STAGING_FRAMES) {
                    resampler.stage().extend_from_slice(piece);
                    resampler.flush_full()?;
                }
                let mut buf = Vec::new();
                resampler.finish_into(&mut buf)?;
                tracing::info!("Audio (opus): {sample_rate}Hz, {channels}ch (eager)");
                Ok(Self::eager(buf, spec))
            }
            Some(decoder) => {
                tracing::info!("Audio: {sample_rate}Hz, {channels}ch (streaming windows)");
                Ok(Self {
                    src: Source::Streaming {
                        format,
                        decoder,
                        track_id,
                        sample_rate,
                        source_frames: 0,
                        max_samples,
                        limit_secs,
                        // No length hint: the windowed path never materializes the
                        // whole 16 kHz stream, so it must not reserve for it.
                        resampler: Box::new(ResampleTo16k::new(SampleRate(sample_rate), None)),
                        interleaved: Vec::new(),
                    },
                    eof: false,
                    finished: false,
                    buf: Vec::new(),
                    buf_start_abs: 0,
                    decoded_16k_total: 0,
                    spec,
                    next_start: 0,
                    first: true,
                    done: false,
                })
            }
        }
    }

    /// Build an eager source over an already-decoded 16 kHz buffer.
    fn eager(buf: Vec<f32>, spec: WindowSpec) -> Self {
        let total = buf.len();
        Self {
            src: Source::Eager,
            eof: true,
            finished: true,
            buf,
            buf_start_abs: 0,
            decoded_16k_total: total,
            spec,
            next_start: 0,
            first: true,
            done: false,
        }
    }

    /// Total 16 kHz samples decoded. Exact once the stream is drained (every
    /// window consumed), matching the whole-buffer decoder's sample count.
    pub(crate) fn total_16k_samples(&self) -> usize {
        self.decoded_16k_total
    }

    /// Decode the whole stream flat (no windowing) to one 16 kHz mono buffer.
    ///
    /// Byte-identical to the whole-buffer `decode_audio_inner`: the same packet
    /// loop, the same per-packet resampler flush cadence, the same final drain.
    pub(crate) fn drain_to_vec(mut self) -> Result<Vec<f32>> {
        self.fill_to(usize::MAX)?;
        Ok(std::mem::take(&mut self.buf))
    }

    /// Flat-decode a file to 16 kHz mono. The window geometry is irrelevant to
    /// `drain_to_vec`, so a sentinel spec is used. `max_audio_secs` is the
    /// whole-buffer length budget (this drain materializes the entire stream).
    pub(crate) fn decode_file(path: &str, max_audio_secs: Option<f64>) -> Result<Vec<f32>> {
        Self::open(path, WindowSpec::flat(), max_audio_secs)?.drain_to_vec()
    }

    /// Flat-decode a shared byte buffer to 16 kHz mono.
    pub(crate) fn decode_bytes(data: Bytes, max_audio_secs: Option<f64>) -> Result<Vec<f32>> {
        Self::from_bytes(data, WindowSpec::flat(), max_audio_secs)?.drain_to_vec()
    }

    /// Decode until `decoded_16k_total >= target` (or EOF), appending 16 kHz
    /// samples to `buf`. Enforces the source-rate length budget incrementally with
    /// the exact same error string as the whole-buffer path.
    fn fill_to(&mut self, target: usize) -> Result<()> {
        let Source::Streaming {
            format,
            decoder,
            track_id,
            sample_rate,
            source_frames,
            max_samples,
            limit_secs,
            resampler,
            interleaved,
        } = &mut self.src
        else {
            return Ok(()); // Eager: the whole buffer is already resident.
        };

        while !self.eof && self.decoded_16k_total < target {
            let have_pcm = *source_frames > 0;
            let packet = match next_demux_packet(&mut **format, have_pcm)? {
                Some(p) => p,
                None => {
                    self.eof = true;
                    break;
                }
            };
            if packet.track_id != *track_id {
                continue;
            }

            let decoded = decoder.decode(&packet).context("Decode error")?;
            let num_frames = decoded.frames();
            let ch = decoded.spec().channels().count();

            if ch > 1 {
                interleaved.clear();
                decoded.copy_to_vec_interleaved(interleaved);
                let stage = resampler.stage();
                for frame in 0..num_frames {
                    let mut sum = 0.0_f32;
                    for c in 0..ch {
                        sum += interleaved[frame * ch + c];
                    }
                    stage.push(sum / ch as f32);
                }
            } else {
                let stage = resampler.stage();
                let offset = stage.len();
                stage.resize(offset + num_frames, 0.0);
                decoded.copy_to_slice_interleaved(&mut stage[offset..]);
            }
            *source_frames += num_frames;

            if *source_frames > *max_samples {
                return Err(audio_too_long_err(
                    *source_frames,
                    *sample_rate,
                    *limit_secs,
                ));
            }

            resampler.flush_full()?;
            let before = self.buf.len();
            resampler.drain_ready_into(&mut self.buf);
            self.decoded_16k_total += self.buf.len() - before;
        }

        if self.eof && !self.finished {
            let before = self.buf.len();
            resampler.finish_into(&mut self.buf)?;
            self.decoded_16k_total += self.buf.len() - before;
            self.finished = true;
        }
        Ok(())
    }
}

#[cfg(feature = "file-decode")]
impl PcmWindows for FileWindows {
    fn spec(&self) -> WindowSpec {
        self.spec
    }

    fn next_window(&mut self) -> Result<Option<PcmWindow<'_>>, GigasttError> {
        if self.done {
            return Ok(None);
        }

        // Reclaim the previous window's consumed prefix: windows only move
        // forward, so everything before `next_start` is dead. This is what keeps
        // the resident buffer at one window plus look-ahead.
        let drop = self
            .next_start
            .saturating_sub(self.buf_start_abs)
            .min(self.buf.len());
        if drop > 0 {
            self.buf.drain(0..drop);
            self.buf_start_abs += drop;
        }

        // Decode one sample past the window end so `end == total` is
        // distinguishable from a mid-stream boundary; the first window also needs
        // enough to decide single-pass vs windowed.
        let target = if self.first {
            (self.spec.single_pass_max() + 1).max(self.spec.window() + 1)
        } else {
            self.next_start + self.spec.window() + 1
        };
        self.fill_to(target).map_err(decode_error)?;

        let avail_end = self.decoded_16k_total;
        let start = self.next_start;

        if self.first {
            self.first = false;
            if self.eof && avail_end <= self.spec.single_pass_max() {
                // The whole stream fits the single-pass ceiling: yield exactly one
                // window over all of it. `decode_words_streaming` then runs one
                // encoder pass with frame offset 0 and stitches onto an empty list
                // — byte-identical to `decode_words`' non-windowed branch.
                self.done = true;
                let e = avail_end - self.buf_start_abs;
                return Ok(Some(PcmWindow {
                    start_sample: start,
                    samples: &self.buf[..e],
                }));
            }
        }

        if start >= avail_end {
            // Reachable only at EOF, once the last window has been yielded.
            self.done = true;
            return Ok(None);
        }

        // Standard overlapping geometry, matching `SliceWindows` exactly. Because
        // `fill_to` decoded one sample past `start + window` (or hit EOF),
        // `end == avail_end` holds only when this window reaches the true end.
        let end = (start + self.spec.window()).min(avail_end);
        if self.eof && end == avail_end {
            self.done = true;
        } else {
            self.next_start = start + self.spec.stride();
        }
        let s = start - self.buf_start_abs;
        let e = end - self.buf_start_abs;
        Ok(Some(PcmWindow {
            start_sample: start,
            samples: &self.buf[s..e],
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The loop `SliceWindows` replaced, verbatim, as the reference oracle.
    fn legacy_windows(total: usize, window: usize, stride: usize) -> Vec<(usize, usize)> {
        let mut out = Vec::new();
        let mut start = 0usize;
        while start < total {
            let end = (start + window).min(total);
            out.push((start, end - start));
            if end == total {
                break;
            }
            start += stride;
        }
        out
    }

    fn observed(total: usize, spec: WindowSpec) -> Vec<(usize, usize)> {
        let samples = vec![0.0f32; total];
        let mut src = SliceWindows::new(&samples, spec);
        let mut out = Vec::new();
        while let Some(w) = src.next_window().expect("slice source never fails") {
            out.push((w.start_sample, w.samples.len()));
        }
        out
    }

    /// The ort long-form geometry, spelled out so this test pins the numbers the
    /// engine feeds rather than following it.
    fn ort_spec() -> WindowSpec {
        WindowSpec::new(16000 * 30, 16000 * 24, 16000 * 2)
    }

    #[test]
    fn test_window_spec_stride_is_frame_aligned() {
        let spec = ort_spec();
        assert_eq!(spec.window(), 384_000);
        assert_eq!(spec.stride(), 352_000);
        assert_eq!(spec.stride() % FRAME_SAMPLES, 0);
        // 2 s overlap is already frame-aligned, so it survives the alignment.
        assert_eq!(spec.overlap(), 32_000);
        // ANE geometry: 30 s window, same 2 s overlap.
        let ane = WindowSpec::new(16000 * 30, 16000 * 30, 16000 * 2);
        assert_eq!(ane.stride(), 448_000);
        assert_eq!(ane.stride() % FRAME_SAMPLES, 0);
        assert_eq!(ane.overlap(), 32_000);
    }

    #[test]
    fn test_window_spec_single_pass_boundary() {
        let spec = ort_spec();
        assert!(spec.is_single_pass(0));
        assert!(spec.is_single_pass(479_999));
        assert!(spec.is_single_pass(480_000)); // exactly 30 s stays single-pass
        assert!(!spec.is_single_pass(480_001));
    }

    #[test]
    fn test_window_spec_degenerate_overlap_still_advances() {
        // overlap >= window would give a zero stride and a non-advancing source.
        let spec = WindowSpec::new(0, 1000, 4000);
        assert_eq!(spec.stride(), FRAME_SAMPLES);
        assert_eq!(observed(5000, spec).len(), 8);
    }

    #[test]
    fn test_slice_windows_matches_legacy_loop_swept() {
        let spec = ort_spec();
        let (window, stride) = (spec.window(), spec.stride());
        let mut lengths: Vec<usize> = Vec::new();
        // Coarse sweep across 0..3x window.
        let mut n = 0usize;
        while n <= 3 * window {
            lengths.push(n);
            n += 4_001; // deliberately coprime with the stride/frame grid
        }
        // Exact boundaries: window/stride multiples ± 1, the single-pass
        // threshold, and the degenerate sub-frame tail band.
        for anchor in [
            0,
            1,
            FRAME_SAMPLES,
            window,
            stride,
            stride + window,
            2 * stride,
            2 * stride + window,
            480_000, // single-pass branch boundary
        ] {
            for d in [-1isize, 0, 1] {
                let v = anchor as isize + d;
                if v >= 0 {
                    lengths.push(v as usize);
                }
            }
        }
        lengths.extend(704_000..=704_320); // degenerate band, every length
        lengths.sort_unstable();
        lengths.dedup();

        for total in lengths {
            assert_eq!(
                observed(total, spec),
                legacy_windows(total, window, stride),
                "window sequence diverged at total={total}"
            );
        }
    }

    #[test]
    fn test_slice_windows_empty_yields_nothing() {
        assert!(observed(0, ort_spec()).is_empty());
    }

    #[test]
    fn test_slice_windows_stop_exactly_at_the_end() {
        let spec = ort_spec();
        let total = 1_440_000; // 90 s
        let seq = observed(total, spec);
        assert!(!seq.is_empty());
        // Exactly one window reaches the end, and it is the last one emitted.
        assert_eq!(
            seq.iter()
                .filter(|(start, len)| start + len == total)
                .count(),
            1
        );
        let (start, len) = seq[seq.len() - 1];
        assert_eq!(start + len, total);
        // Every start is frame-aligned, so the frame offset is integral.
        for (start, _) in &seq {
            assert_eq!(start % FRAME_SAMPLES, 0);
        }
    }
}

/// [`FileWindows`] streaming-decode tests: prove that pulling windows from the
/// container yields byte-identical geometry to [`SliceWindows`] over the same
/// fully-decoded buffer — and, below the single-pass ceiling, exactly one window
/// over the whole buffer (matching `Engine::decode_words`' non-windowed branch).
/// No model is required.
#[cfg(all(test, feature = "file-decode"))]
mod file_windows_tests {
    use super::*;
    use crate::inference::audio::encode_wav_pcm16;
    use bytes::Bytes;

    /// The ort long-form geometry, matching `Engine::window_spec` (CPU backend).
    fn ort_spec() -> WindowSpec {
        WindowSpec::new(16000 * 30, 16000 * 24, 16000 * 2)
    }

    /// Deterministic, PCM16-quantization-exercising signal in [-1, 1).
    fn signal(n: usize, seed: f32) -> Vec<f32> {
        (0..n)
            .map(|i| {
                let t = i as f32;
                0.4 * ((t * 0.017 + seed).sin() + 0.5 * (t * 0.0031 + seed).sin())
            })
            .collect()
    }

    /// Minimal interleaved-stereo PCM16 WAV (symphonia decodes this to two
    /// channels, exercising the mono-mix branch of the streaming decode).
    fn stereo_wav_pcm16(left: &[f32], right: &[f32], rate: u32) -> Vec<u8> {
        let frames = left.len().min(right.len());
        let data_bytes = (frames * 2 * 2) as u32;
        let byte_rate = rate * 2 * 2;
        let mut w = Vec::with_capacity(44 + data_bytes as usize);
        w.extend_from_slice(b"RIFF");
        w.extend_from_slice(&(36 + data_bytes).to_le_bytes());
        w.extend_from_slice(b"WAVE");
        w.extend_from_slice(b"fmt ");
        w.extend_from_slice(&16u32.to_le_bytes());
        w.extend_from_slice(&1u16.to_le_bytes()); // PCM
        w.extend_from_slice(&2u16.to_le_bytes()); // channels
        w.extend_from_slice(&rate.to_le_bytes());
        w.extend_from_slice(&byte_rate.to_le_bytes());
        w.extend_from_slice(&4u16.to_le_bytes()); // block align
        w.extend_from_slice(&16u16.to_le_bytes()); // bits
        w.extend_from_slice(b"data");
        w.extend_from_slice(&data_bytes.to_le_bytes());
        let q = |s: f32| (s.clamp(-1.0, 1.0) * i16::MAX as f32) as i16;
        for i in 0..frames {
            w.extend_from_slice(&q(left[i]).to_le_bytes());
            w.extend_from_slice(&q(right[i]).to_le_bytes());
        }
        w
    }

    /// Drain the whole windowed source into owned `(start, samples)` pairs.
    fn window_seq(mut fw: FileWindows) -> Vec<(usize, Vec<f32>)> {
        let mut out = Vec::new();
        while let Some(w) = fw.next_window().expect("window") {
            out.push((w.start_sample, w.samples.to_vec()));
        }
        out
    }

    /// The [`SliceWindows`] sequence over a materialized buffer — the oracle for
    /// the windowed (`> single_pass_max`) regime.
    fn slice_seq(buf: &[f32], spec: WindowSpec) -> Vec<(usize, Vec<f32>)> {
        let mut sw = SliceWindows::new(buf, spec);
        let mut out = Vec::new();
        while let Some(w) = sw.next_window().expect("slice window") {
            out.push((w.start_sample, w.samples.to_vec()));
        }
        out
    }

    /// What `Engine::decode_words` would feed for a fully-decoded buffer: one
    /// whole-buffer window at/under the single-pass ceiling, else the standard
    /// overlapping geometry.
    fn expected_seq(flat: &[f32], spec: WindowSpec) -> Vec<(usize, Vec<f32>)> {
        if flat.len() <= spec.single_pass_max() {
            vec![(0, flat.to_vec())]
        } else {
            slice_seq(flat, spec)
        }
    }

    #[test]
    fn test_file_windows_16k_geometry_matches_decode_words() {
        let spec = ort_spec();
        // Lengths straddling the single-pass ceiling (480_000 @16 kHz) and the
        // window/stride grid.
        for &n in &[1usize, 8_000, 480_000, 480_001, 560_000, 900_000] {
            let src = signal(n, 1.0);
            let wav = encode_wav_pcm16(&src, 16000);
            let flat = FileWindows::from_bytes(Bytes::copy_from_slice(&wav), spec, None)
                .expect("open flat")
                .drain_to_vec()
                .expect("drain");
            // 16 kHz is the passthrough path: no resampler, so the decoded length
            // is exact.
            assert_eq!(flat.len(), n, "passthrough length changed at n={n}");
            let got = window_seq(
                FileWindows::from_bytes(Bytes::copy_from_slice(&wav), spec, None)
                    .expect("open windows"),
            );
            assert_eq!(got, expected_seq(&flat, spec), "geometry mismatch at n={n}");
        }
    }

    #[test]
    fn test_file_windows_48k_stereo_matches_slice_over_drain() {
        let spec = ort_spec();
        // 40 s @48 kHz stereo → ~40 s @16 kHz mono, above the single-pass ceiling,
        // so the windowed geometry (not a single window) is exercised, through the
        // resampler and the mono-mix branch.
        let n = 48_000 * 40;
        let left = signal(n, 0.3);
        let right = signal(n, 2.1);
        let wav = stereo_wav_pcm16(&left, &right, 48_000);
        let flat = FileWindows::from_bytes(Bytes::copy_from_slice(&wav), spec, None)
            .expect("open flat")
            .drain_to_vec()
            .expect("drain");
        assert!(
            flat.len() > spec.single_pass_max(),
            "expected the chunked regime, got {} samples",
            flat.len()
        );
        let got = window_seq(
            FileWindows::from_bytes(Bytes::copy_from_slice(&wav), spec, None)
                .expect("open windows"),
        );
        // Incremental resample + windowing is byte-identical to whole-buffer
        // resample + SliceWindows: the staged chunk sequence is the same either
        // way, so every resampled sample matches.
        assert_eq!(got, slice_seq(&flat, spec));
    }

    #[test]
    fn test_file_windows_single_pass_yields_one_window() {
        let spec = ort_spec();
        let src = signal(10_000, 0.7);
        let wav = encode_wav_pcm16(&src, 16000);
        let got = window_seq(
            FileWindows::from_bytes(Bytes::copy_from_slice(&wav), spec, None).expect("open"),
        );
        assert_eq!(got.len(), 1);
        assert_eq!(got[0].0, 0);
        assert_eq!(got[0].1.len(), 10_000);
    }

    #[test]
    fn test_file_windows_total_16k_samples_is_exact_at_16k() {
        let spec = ort_spec();
        let n = 700_000; // chunked
        let src = signal(n, 1.3);
        let wav = encode_wav_pcm16(&src, 16000);
        let mut fw =
            FileWindows::from_bytes(Bytes::copy_from_slice(&wav), spec, None).expect("open");
        while fw.next_window().expect("window").is_some() {}
        assert_eq!(fw.total_16k_samples(), n);
    }

    /// Peak-RSS instrument (decode-only, no model). Writes an `N`-second 48 kHz
    /// stereo WAV to a temp file, streams every window discarding the samples,
    /// and asserts the total is right. Run it under a peak-RSS meter at several
    /// `GIGASTT_PEAK_SECONDS` values — the slope of peak RSS per audio-second is
    /// the memory claim:
    ///
    /// ```sh
    /// for s in 60 300 1200; do GIGASTT_PEAK_SECONDS=$s /usr/bin/time -l \
    ///   cargo test -p gigastt-core --lib \
    ///   file_windows_tests::zzz_streaming_decode_peak_instrument \
    ///   -- --ignored --exact --nocapture 2>&1 | grep -E 'maximum resident'; done
    /// ```
    #[test]
    #[ignore = "decode-only peak-RSS instrument; drive with GIGASTT_PEAK_SECONDS under /usr/bin/time"]
    fn zzz_streaming_decode_peak_instrument() {
        let secs: usize = std::env::var("GIGASTT_PEAK_SECONDS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(5);
        let path = std::env::temp_dir().join(format!("gigastt_peak_{secs}s.wav"));

        // Stream the synthetic WAV to disk one second at a time so generating it
        // never holds more than a second of audio in RAM — the decode is what we
        // are measuring, not the fixture.
        {
            use std::io::Write;
            let rate = 48_000u32;
            let frames = secs * rate as usize;
            let data_bytes = (frames * 2 * 2) as u32;
            let f = std::fs::File::create(&path).expect("create temp wav");
            let mut w = std::io::BufWriter::new(f);
            w.write_all(b"RIFF").unwrap();
            w.write_all(&(36 + data_bytes).to_le_bytes()).unwrap();
            w.write_all(b"WAVE").unwrap();
            w.write_all(b"fmt ").unwrap();
            w.write_all(&16u32.to_le_bytes()).unwrap();
            w.write_all(&1u16.to_le_bytes()).unwrap();
            w.write_all(&2u16.to_le_bytes()).unwrap();
            w.write_all(&rate.to_le_bytes()).unwrap();
            w.write_all(&(rate * 4).to_le_bytes()).unwrap();
            w.write_all(&4u16.to_le_bytes()).unwrap();
            w.write_all(&16u16.to_le_bytes()).unwrap();
            w.write_all(b"data").unwrap();
            w.write_all(&data_bytes.to_le_bytes()).unwrap();
            for sec in 0..secs {
                let base = (sec * rate as usize) as f32;
                for i in 0..rate as usize {
                    let t = base + i as f32;
                    let s = (0.4 * (t * 0.02).sin() * i16::MAX as f32) as i16;
                    w.write_all(&s.to_le_bytes()).unwrap();
                    w.write_all(&s.to_le_bytes()).unwrap();
                }
            }
            w.flush().unwrap();
        }

        let p = path.to_str().unwrap();
        // `GIGASTT_PEAK_MODE=drain` measures the old whole-buffer decode (peak
        // grows with duration) for a same-build A/B against the default windowed
        // path (peak bounded by one window).
        let total = if std::env::var("GIGASTT_PEAK_MODE").as_deref() == Ok("drain") {
            FileWindows::decode_file(p, None).expect("drain").len()
        } else {
            let mut fw = FileWindows::open(p, ort_spec(), None).expect("open temp wav");
            let mut counted = 0usize;
            while let Some(win) = fw.next_window().expect("window") {
                counted += win.samples.len();
            }
            // Overlapping windows re-count their overlap, so the summed window
            // length exceeds the (bounded) true total.
            assert!(counted >= fw.total_16k_samples());
            fw.total_16k_samples()
        };
        let _ = std::fs::remove_file(&path);

        let expected = secs * 16_000;
        assert!(
            (total as i64 - expected as i64).unsigned_abs() < 16_000,
            "total {total} not within 1 s of {expected}"
        );
    }
}
