//! Container-backed [`FileWindows`] source.
//!
//! Pulls overlapping decode windows straight from an audio container so peak
//! audio memory is O(one window) rather than O(file).

use anyhow::{Context, Result};
use bytes::Bytes;
use symphonia::core::codecs::audio::well_known::CODEC_ID_OPUS;
use symphonia::core::codecs::audio::{AudioDecoder, AudioDecoderOptions};
use symphonia::core::formats::probe::Hint;
use symphonia::core::formats::{FormatOptions, FormatReader, TrackType};
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::MetadataOptions;

use super::super::decode::BytesMediaSource;
use super::super::opus::{OPUS_DECODE_RATE, OpusStream, next_demux_packet};
use super::super::resample::{RESAMPLE_STAGING_FRAMES, ResampleTo16k, SampleRate};
use super::super::wave::{WaveRecv, WaveSource, check_budget, try_open_bytes, try_open_path};
use super::super::{MAX_SAMPLE_RATE, audio_too_long_err, decode_error, resolve_budget};
use super::{ChannelSelect, PcmWindow, PcmWindows, WindowCursor, WindowSpec};

use crate::error::GigasttError;

/// The decode engine behind [`FileWindows`]: a streaming symphonia loop, a
/// WAVE-family ryf pull, or Opus via `opus-rs`.
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
    /// OGG/Opus: symphonia demuxes it but ships no decoder, so packets go
    /// through the `opus-rs` fallback ([`OpusStream`]) and are mixed to mono
    /// and resampled as they arrive — same shape as `Streaming`, different
    /// decoder.
    Opus {
        format: Box<dyn FormatReader>,
        track_id: u32,
        /// Boxed: it carries libopus decoder state.
        stream: Box<OpusStream>,
        /// Running decoded frame count at the Opus decode rate (48 kHz), which
        /// is what the budget below counts and what a trip reports.
        decoded_48k: usize,
        max_samples: usize,
        limit_secs: f64,
        resampler: Box<ResampleTo16k>,
        /// Decoded mono samples not yet staged. The whole-buffer path fed the
        /// resampler in exact [`RESAMPLE_STAGING_FRAMES`] chunks; holding the
        /// remainder here reproduces that chunk sequence packet-by-packet, so
        /// the flush boundaries — and with them the resampled output — do not
        /// move. Bounded by one chunk plus one packet.
        pending: Vec<f32>,
    },
    /// WAVE family (PCM/IEEE, G.711, G.722, ADPCM, RF64/RIFX/Wave64) via ryf.
    Wave(Box<WaveSource>),
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
    /// Which channel the packet loop keeps.
    channel: ChannelSelect,
    cursor: WindowCursor,
}

#[cfg(feature = "file-decode")]
impl FileWindows {
    /// Open a file for windowed decode. WAVE containers go through ryf; everything
    /// else is probed by symphonia.
    pub(crate) fn open(path: &str, spec: WindowSpec, max_audio_secs: Option<f64>) -> Result<Self> {
        if let Some(wave) = try_open_path(path, ChannelSelect::Mono, max_audio_secs)? {
            return Ok(Self::from_wave(wave, spec, ChannelSelect::Mono));
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
        Self::from_mss(mss, hint, spec, max_audio_secs, ChannelSelect::Mono)
    }

    /// Open a shared [`Bytes`] buffer for windowed decode. `BytesMediaSource` is
    /// seekable, so the isomp4 demuxer's trailing-`moov` seek works; no spool.
    pub(crate) fn from_bytes(
        data: Bytes,
        spec: WindowSpec,
        max_audio_secs: Option<f64>,
    ) -> Result<Self> {
        if let Some(wave) = try_open_bytes(data.clone(), ChannelSelect::Mono, max_audio_secs)? {
            return Ok(Self::from_wave(wave, spec, ChannelSelect::Mono));
        }
        let source = BytesMediaSource::new(data);
        let mss = MediaSourceStream::new(Box::new(source), Default::default());
        Self::from_mss(mss, Hint::new(), spec, max_audio_secs, ChannelSelect::Mono)
    }

    /// Probe a non-WAVE container and set up the streaming decoder (or the
    /// Opus `opus-rs` fallback). Scalar params are copied out (and the non-Opus
    /// decoder built) inside one borrow scope so the `FormatReader` is free to
    /// move or be driven afterwards — the same shape as the whole-buffer
    /// `decode_audio_inner`.
    fn from_mss(
        mss: MediaSourceStream<'static>,
        hint: Hint,
        spec: WindowSpec,
        max_audio_secs: Option<f64>,
        channel: ChannelSelect,
    ) -> Result<Self> {
        let format = symphonia::default::get_probe()
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
                // The budget and the resampler both count at the Opus decode
                // rate (48 kHz). That is what the decoder emits. The container
                // rate is the capture rate (OpusHead input rate, or a WebM
                // SamplingFrequency a browser sets to the AudioContext rate)
                // and is not the PCM rate (RFC 7845). Feeding it to the
                // resampler leaves 48 kHz audio unresampled.
                let (max_samples, limit_secs) = resolve_budget(max_audio_secs, OPUS_DECODE_RATE);
                tracing::info!(
                    "Audio (opus): container {sample_rate}Hz, decode {OPUS_DECODE_RATE}Hz, {channels}ch (streaming windows)"
                );
                Ok(Self {
                    src: Source::Opus {
                        format,
                        track_id,
                        stream: Box::new(OpusStream::new(channels, channel)?),
                        decoded_48k: 0,
                        max_samples,
                        limit_secs,
                        resampler: Box::new(ResampleTo16k::new(SampleRate(OPUS_DECODE_RATE), None)),
                        pending: Vec::with_capacity(RESAMPLE_STAGING_FRAMES),
                    },
                    eof: false,
                    finished: false,
                    buf: Vec::new(),
                    buf_start_abs: 0,
                    decoded_16k_total: 0,
                    channel,
                    cursor: WindowCursor::new(spec),
                })
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
                    channel,
                    cursor: WindowCursor::new(spec),
                })
            }
        }
    }

    /// Open a shared [`Bytes`] buffer for windowed decode of **one** channel.
    ///
    /// The per-channel twin of [`FileWindows::from_bytes`], for the
    /// `channels=split` path: channel `k` is extracted in the packet loop and
    /// resampled on its own, so peak memory is one window rather than every
    /// channel of the whole file.
    ///
    /// The samples are the ones the whole-buffer per-channel decode produced —
    /// same packet cadence, same resampler, same staging — so the windows the
    /// decoder sees are unchanged.
    pub(crate) fn from_bytes_channel(
        data: Bytes,
        spec: WindowSpec,
        max_audio_secs: Option<f64>,
        channel: usize,
    ) -> Result<Self> {
        if let Some(wave) =
            try_open_bytes(data.clone(), ChannelSelect::One(channel), max_audio_secs)?
        {
            return Ok(Self::from_wave(wave, spec, ChannelSelect::One(channel)));
        }
        let source = BytesMediaSource::new(data);
        let mss = MediaSourceStream::new(Box::new(source), Default::default());
        Self::from_mss(
            mss,
            Hint::new(),
            spec,
            max_audio_secs,
            ChannelSelect::One(channel),
        )
    }

    fn from_wave(wave: WaveSource, spec: WindowSpec, channel: ChannelSelect) -> Self {
        Self {
            src: Source::Wave(Box::new(wave)),
            eof: false,
            finished: false,
            buf: Vec::new(),
            buf_start_abs: 0,
            decoded_16k_total: 0,
            channel,
            cursor: WindowCursor::new(spec),
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
        match self.src {
            Source::Streaming { .. } => self.fill_streaming(target),
            Source::Opus { .. } => self.fill_opus(target),
            Source::Wave(_) => self.fill_wave(target),
        }
    }

    /// [`FileWindows::fill_to`] for a container symphonia can decode itself.
    fn fill_streaming(&mut self, target: usize) -> Result<()> {
        let channel = self.channel;
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
            return Ok(());
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
                match channel {
                    ChannelSelect::Mono => {
                        for frame in 0..num_frames {
                            let mut sum = 0.0_f32;
                            for c in 0..ch {
                                sum += interleaved[frame * ch + c];
                            }
                            stage.push(sum / ch as f32);
                        }
                    }
                    // A packet short of the requested channel contributes
                    // nothing rather than shifting the timeline.
                    ChannelSelect::One(k) if k < ch => {
                        for frame in 0..num_frames {
                            stage.push(interleaved[frame * ch + k]);
                        }
                    }
                    ChannelSelect::One(_) => {}
                }
            } else if matches!(channel, ChannelSelect::Mono | ChannelSelect::One(0)) {
                // Single-channel packet: it *is* channel 0. Asking for a higher
                // index contributes nothing, rather than silently handing back
                // channel 0's audio under another channel's name.
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

    /// [`FileWindows::fill_to`] for OGG/Opus, decoded through the `opus-rs`
    /// fallback. Same loop as [`FileWindows::fill_streaming`]: one packet at a
    /// time, budget checked incrementally, resampler drained into `buf`.
    fn fill_opus(&mut self, target: usize) -> Result<()> {
        let Source::Opus {
            format,
            track_id,
            stream,
            decoded_48k,
            max_samples,
            limit_secs,
            resampler,
            pending,
        } = &mut self.src
        else {
            return Ok(());
        };

        while !self.eof && self.decoded_16k_total < target {
            let have_pcm = *decoded_48k > 0;
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

            *decoded_48k += stream.decode_packet(&packet.data, pending)?;
            if *decoded_48k > *max_samples {
                return Err(audio_too_long_err(
                    *decoded_48k,
                    OPUS_DECODE_RATE,
                    *limit_secs,
                ));
            }

            // Stage in exact `RESAMPLE_STAGING_FRAMES` chunks, keeping the
            // remainder in `pending`: that is the chunk sequence the
            // whole-buffer path produced, and the resampler drains whatever is
            // staged, so any other cadence would move the flush boundaries and
            // with them the output samples.
            while pending.len() >= RESAMPLE_STAGING_FRAMES {
                resampler
                    .stage()
                    .extend_from_slice(&pending[..RESAMPLE_STAGING_FRAMES]);
                pending.drain(..RESAMPLE_STAGING_FRAMES);
                resampler.flush_full()?;
            }
            let before = self.buf.len();
            resampler.drain_ready_into(&mut self.buf);
            self.decoded_16k_total += self.buf.len() - before;
        }

        if self.eof && !self.finished {
            resampler.stage().extend_from_slice(pending);
            pending.clear();
            let before = self.buf.len();
            resampler.finish_into(&mut self.buf)?;
            self.decoded_16k_total += self.buf.len() - before;
            self.finished = true;
        }
        Ok(())
    }

    /// [`FileWindows::fill_to`] for WAVE family files decoded by ryf.
    fn fill_wave(&mut self, target: usize) -> Result<()> {
        let Source::Wave(wave) = &mut self.src else {
            return Ok(());
        };

        while !self.eof && self.decoded_16k_total < target {
            match wave.recv_block()? {
                WaveRecv::Block(block) => {
                    wave.source_frames += block.frames;
                    check_budget(
                        wave.source_frames,
                        wave.sample_rate,
                        wave.max_samples,
                        wave.limit_secs,
                    )?;
                    if !block.samples.is_empty() {
                        wave.resampler.stage().extend_from_slice(&block.samples);
                        wave.resampler.flush_full()?;
                    }
                    let before = self.buf.len();
                    wave.resampler.drain_ready_into(&mut self.buf);
                    self.decoded_16k_total += self.buf.len() - before;
                }
                WaveRecv::Eof => {
                    self.eof = true;
                    break;
                }
            }
        }

        if self.eof && !self.finished {
            let before = self.buf.len();
            wave.resampler.finish_into(&mut self.buf)?;
            self.decoded_16k_total += self.buf.len() - before;
            self.finished = true;
        }
        Ok(())
    }
}

#[cfg(feature = "file-decode")]
impl PcmWindows for FileWindows {
    fn next_window(&mut self) -> Result<Option<PcmWindow<'_>>, GigasttError> {
        if self.cursor.is_done() {
            return Ok(None);
        }

        // Reclaim the consumed prefix: windows only move forward, so everything
        // before `retain_from` is dead. This is what keeps the resident buffer
        // at one single-pass ceiling of audio plus one stride of look-behind.
        let drop = self
            .cursor
            .retain_from()
            .saturating_sub(self.buf_start_abs)
            .min(self.buf.len());
        if drop > 0 {
            self.buf.drain(0..drop);
            self.buf_start_abs += drop;
        }

        self.fill_to(self.cursor.fill_target())
            .map_err(decode_error)?;

        let Some((start, end)) = self.cursor.take(self.decoded_16k_total, self.eof) else {
            return Ok(None);
        };
        let s = start - self.buf_start_abs;
        let e = end - self.buf_start_abs;
        Ok(Some(PcmWindow {
            start_sample: start,
            samples: &self.buf[s..e],
        }))
    }
}
