use anyhow::{Context, Result};

use super::super::opus::{OPUS_DECODE_RATE, next_demux_packet};
use super::super::resample::RESAMPLE_STAGING_FRAMES;
use super::super::wave::{WaveRecv, check_budget};
use super::super::{audio_too_long_err, check_decode_abort};
use super::{ChannelSelect, PcmSource, Source};

impl PcmSource {
    /// Decode until `position() >= target` or EOF, appending PCM to the buffer.
    /// The caller must discard the source after cancellation or decode failure.
    pub(in super::super) fn fill_to(
        &mut self,
        target: usize,
        abort: Option<&(dyn Fn() -> bool + Sync)>,
    ) -> Result<()> {
        check_decode_abort(abort)?;
        match self.src {
            Source::Streaming { .. } => self.fill_streaming(target, abort),
            Source::Opus { .. } => self.fill_opus(target, abort),
            Source::Wave(_) => self.fill_wave(target, abort),
        }
    }

    /// [`PcmSource::fill_to`] for a container symphonia can decode itself.
    fn fill_streaming(
        &mut self,
        target: usize,
        abort: Option<&(dyn Fn() -> bool + Sync)>,
    ) -> Result<()> {
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
            check_decode_abort(abort)?;
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
            check_decode_abort(abort)?;
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

        check_decode_abort(abort)?;
        if self.eof && !self.finished {
            let before = self.buf.len();
            resampler.finish_into(&mut self.buf)?;
            self.decoded_16k_total += self.buf.len() - before;
            self.finished = true;
        }
        Ok(())
    }

    /// [`PcmSource::fill_to`] for OGG/Opus, decoded through the `opus-rs`
    /// fallback. Same loop as [`PcmSource::fill_streaming`]: one packet at a
    /// time, budget checked incrementally, resampler drained into `buf`.
    fn fill_opus(
        &mut self,
        target: usize,
        abort: Option<&(dyn Fn() -> bool + Sync)>,
    ) -> Result<()> {
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
            check_decode_abort(abort)?;
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
            check_decode_abort(abort)?;
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

        check_decode_abort(abort)?;
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

    /// [`PcmSource::fill_to`] for WAVE family files decoded by ryf.
    fn fill_wave(
        &mut self,
        target: usize,
        abort: Option<&(dyn Fn() -> bool + Sync)>,
    ) -> Result<()> {
        let Source::Wave(wave) = &mut self.src else {
            return Ok(());
        };

        while !self.eof && self.decoded_16k_total < target {
            check_decode_abort(abort)?;
            match wave.recv_block()? {
                WaveRecv::Block(block) => {
                    check_decode_abort(abort)?;
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

        check_decode_abort(abort)?;
        if self.eof && !self.finished {
            let before = self.buf.len();
            wave.resampler.finish_into(&mut self.buf)?;
            self.decoded_16k_total += self.buf.len() - before;
            self.finished = true;
        }
        Ok(())
    }
}
