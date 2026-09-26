use super::*;

fn cfg(threshold: f32, min_silence_ms: u32, min_speech_ms: u32, speech_pad_ms: u32) -> VadConfig {
    VadConfig {
        threshold,
        min_silence_ms,
        min_speech_ms,
        speech_pad_ms,
    }
}

#[test]
fn test_ms_to_samples_16khz() {
    assert_eq!(VadConfig::ms_to_samples(1000), 16000);
    assert_eq!(VadConfig::ms_to_samples(500), 8000);
    assert_eq!(VadConfig::ms_to_samples(0), 0);
}

#[test]
fn test_regions_empty_probs_is_empty() {
    let c = VadConfig::default();
    assert!(regions_from_probs(&[], 512, 0, &c).is_empty());
    assert!(regions_from_probs(&[0.9, 0.9], 512, 0, &c).is_empty());
}

#[test]
fn test_regions_all_silence_is_empty() {
    let c = cfg(0.5, 0, 0, 0);
    let probs = vec![0.1f32; 10];
    assert!(regions_from_probs(&probs, 512, 10 * 512, &c).is_empty());
}

#[test]
fn test_regions_single_block_no_pad_no_mins() {
    let c = cfg(0.5, 0, 0, 0);
    // frames: silence, speech, speech, silence
    let probs = [0.1, 0.9, 0.9, 0.1];
    let r = regions_from_probs(&probs, 100, 400, &c);
    assert_eq!(r, vec![(100, 300)]);
}

#[test]
fn test_regions_trailing_speech_clamps_to_total() {
    let c = cfg(0.5, 0, 0, 0);
    let probs = [0.1, 0.9, 0.9];
    // last speech run never closes → clamp to total_samples (not 3*100).
    let r = regions_from_probs(&probs, 100, 250, &c);
    assert_eq!(r, vec![(100, 250)]);
}

#[test]
fn test_regions_min_silence_merges_short_gap() {
    // gap of one 100-sample frame = 100 samples; min_silence 1000 samples
    // (≈ wide) so the two speech blocks merge into one.
    let c = cfg(0.5, /*min_silence_ms*/ 100, 0, 0); // 100ms = 1600 samples
    let probs = [0.9, 0.1, 0.9];
    let r = regions_from_probs(&probs, 100, 300, &c);
    assert_eq!(r, vec![(0, 300)]);
}

#[test]
fn test_regions_long_gap_keeps_two_regions() {
    // min_silence small (0) so any gap splits.
    let c = cfg(0.5, 0, 0, 0);
    let probs = [0.9, 0.1, 0.1, 0.9];
    let r = regions_from_probs(&probs, 100, 400, &c);
    assert_eq!(r, vec![(0, 100), (300, 400)]);
}

#[test]
fn test_regions_min_speech_drops_short_blip() {
    // One 100-sample speech frame, min_speech 1600 samples → dropped.
    let c = cfg(0.5, 0, /*min_speech_ms*/ 100, 0);
    let probs = [0.1, 0.9, 0.1];
    assert!(regions_from_probs(&probs, 100, 300, &c).is_empty());
}

#[test]
fn test_regions_padding_extends_and_clamps() {
    let c = cfg(0.5, 0, 0, /*speech_pad_ms*/ 10); // 10ms = 160 samples
    let probs = [0.1, 0.9, 0.1];
    // raw region (100, 200); pad ±160 → (0 clamped, 360).
    let r = regions_from_probs(&probs, 100, 1000, &c);
    assert_eq!(r, vec![(0, 360)]);
}

#[test]
fn test_regions_padding_merges_overlapping_neighbours() {
    let c = cfg(0.5, 0, 0, 50); // 50ms = 800 samples pad
    // raw regions (0,100) and (300,400) — the trailing silence frame closes
    // the second run at 400; pad ±800 makes them overlap → merge to (0,1200).
    let probs = [0.9, 0.1, 0.1, 0.9, 0.1];
    let r = regions_from_probs(&probs, 100, 2000, &c);
    assert_eq!(r, vec![(0, 1200)]);
}

#[test]
fn test_hangover_fires_once_after_min_silence() {
    let c = cfg(0.5, /*min_silence_ms*/ 100, 0, 0); // 1600 samples = ~3.125 frames @512
    let mut h = Hangover::new(&c);
    // speech
    assert!(!h.update(0.9, 512));
    // silence accumulates: need >=1600 samples → 4 frames (2048) to cross.
    assert!(!h.update(0.1, 512)); // 512
    assert!(!h.update(0.1, 512)); // 1024
    assert!(!h.update(0.1, 512)); // 1536
    assert!(h.update(0.1, 512)); // 2048 >= 1600 → fire
    // does not fire again on continued silence
    assert!(!h.update(0.1, 512));
}

#[test]
fn test_hangover_no_fire_before_any_speech() {
    let c = cfg(0.5, 0, 0, 0);
    let mut h = Hangover::new(&c);
    // leading silence must never fire (no speech seen yet).
    for _ in 0..10 {
        assert!(!h.update(0.1, 512));
    }
}

#[test]
fn test_hangover_rearms_for_next_utterance() {
    let c = cfg(0.5, 50, 0, 0); // 800 samples → 2 frames @512 (1024) to cross
    let mut h = Hangover::new(&c);
    h.update(0.9, 512); // speech
    assert!(!h.update(0.1, 512)); // 512
    assert!(h.update(0.1, 512)); // 1024 >= 800 → fire #1
    // new speech re-arms
    assert!(!h.update(0.9, 512));
    assert!(!h.update(0.1, 512)); // 512
    assert!(h.update(0.1, 512)); // 1024 → fire #2
}

/// Streaming-segmenter equivalence. Gated with the segmenter itself:
/// a lean build has no file VAD to compare against.
#[cfg(feature = "file-decode")]
mod segmenter {
    use super::*;

    /// Drive a [`VadSegmenter`] from a probability sequence instead of the
    /// model, in deliberately irregular chunks so the frame-buffering seam is
    /// exercised. Returns the regions it settled on, the samples it released,
    /// and the high-water mark of retained PCM.
    ///
    /// Sample `i` carries the value `i as f32`, so a released sample names its
    /// own absolute index and the concatenation can be compared element-wise.
    fn stream(
        probs: &[f32],
        total: usize,
        cfg: &VadConfig,
    ) -> (Vec<(usize, usize)>, Vec<f32>, usize) {
        let raw: Vec<f32> = (0..total).map(|i| i as f32).collect();
        let mut seg = VadSegmenter::new(cfg);
        let mut out = Vec::new();
        let mut it = probs.iter().copied();
        let mut peak = 0usize;
        let mut i = 0usize;
        let mut chunk = 1usize;
        while i < total {
            let end = (i + chunk).min(total);
            seg.push_with(&raw[i..end], &mut out, |_, _| Ok(it.next().unwrap_or(0.0)))
                .expect("push");
            peak = peak.max(seg.retained());
            i = end;
            chunk = chunk % 977 + 1;
        }
        seg.finish_with(total, &mut out, |_, _| Ok(it.next().unwrap_or(0.0)))
            .expect("finish");
        (seg.regions().to_vec(), out, peak)
    }

    /// The batch pair the streamer must reproduce: `regions_from_probs` plus the
    /// silence-free concatenation `Engine::decode_speech_regions` builds.
    fn batch(probs: &[f32], total: usize, cfg: &VadConfig) -> (Vec<(usize, usize)>, Vec<f32>) {
        let regions = regions_from_probs(probs, VAD_FRAME_SAMPLES, total, cfg);
        let out = regions
            .iter()
            .flat_map(|&(s, e)| (s..e).map(|i| i as f32))
            .collect();
        (regions, out)
    }

    fn assert_stream_matches_batch(probs: &[f32], total: usize, cfg: &VadConfig) {
        let (got_regions, got_samples, _) = stream(probs, total, cfg);
        let (want_regions, want_samples) = batch(probs, total, cfg);
        assert_eq!(
            got_regions, want_regions,
            "regions diverged (total={total})"
        );
        assert_eq!(
            got_samples, want_samples,
            "compressed buffer diverged (total={total})"
        );
    }

    /// Probability sequence covering `total` samples at the production frame size.
    fn probs_for(total: usize, f: impl Fn(usize) -> f32) -> Vec<f32> {
        (0..total.div_ceil(VAD_FRAME_SAMPLES)).map(f).collect()
    }

    #[test]
    fn test_segmenter_matches_batch_on_shaped_sequences() {
        let c = VadConfig::default();
        let fs = VAD_FRAME_SAMPLES;
        // Alternating speech/silence blocks of many different periods, plus the
        // degenerate all-speech / all-silence ends. The long sweep does not
        // finish under Miri; a few periods on a shorter clip still compare
        // the streamer to the batch oracle.
        let periods: &[usize] = if cfg!(miri) {
            &[1, 8, 64]
        } else {
            &[1, 2, 3, 5, 8, 16, 20, 31, 64]
        };
        let frames = if cfg!(miri) { 48 } else { 200 };
        for &period in periods {
            let total = frames * fs + 137; // deliberately not frame-aligned
            let probs = probs_for(total, |i| if (i / period) % 2 == 0 { 0.9 } else { 0.1 });
            assert_stream_matches_batch(&probs, total, &c);
        }
        let flat_frames = if cfg!(miri) { 24 } else { 97 };
        for level in [0.1f32, 0.9] {
            let total = flat_frames * fs;
            let probs = probs_for(total, |_| level);
            assert_stream_matches_batch(&probs, total, &c);
        }
    }

    #[test]
    fn test_segmenter_matches_batch_on_degenerate_configs() {
        let fs = VAD_FRAME_SAMPLES;
        let total = 120 * fs + 11;
        let probs = probs_for(total, |i| if (i / 7) % 3 == 0 { 0.9 } else { 0.1 });
        // Padding wider than the silence gap is the config where step 2 and
        // step 4 of `regions_from_probs` can both merge the same pair.
        for c in [
            cfg(0.5, 0, 0, 0),
            cfg(0.5, 0, 0, 200),
            cfg(0.5, 10, 0, 500),
            cfg(0.5, 1000, 2000, 100),
            cfg(0.5, 40, 40, 40),
        ] {
            assert_stream_matches_batch(&probs, total, &c);
        }
    }

    #[test]
    fn test_segmenter_matches_batch_on_short_and_empty_inputs() {
        let c = VadConfig::default();
        for total in [
            0usize,
            1,
            2,
            VAD_FRAME_SAMPLES - 1,
            VAD_FRAME_SAMPLES,
            VAD_FRAME_SAMPLES + 1,
        ] {
            for level in [0.1f32, 0.9] {
                assert_stream_matches_batch(&probs_for(total, |_| level), total, &c);
            }
        }
    }

    // Excluded under Miri: each case drives thousands of frames through the
    // segmenter and the batch oracle, orders of magnitude too slow for the
    // interpreter. The same property runs natively on every `cargo test`.
    #[cfg(not(miri))]
    proptest::proptest! {
        #![proptest_config(proptest::prelude::ProptestConfig::with_cases(256))]
        /// The load-bearing claim: for *any* probability sequence and *any*
        /// config, the causal segmenter settles on the same spans and releases
        /// the same samples as the batch pipeline it replaces.
        #[test]
        fn prop_segmenter_matches_batch(
            probs in proptest::collection::vec(0.0f32..=1.0, 1..60),
            tail in 1usize..=VAD_FRAME_SAMPLES,
            threshold in 0.1f32..0.9,
            min_silence_ms in 0u32..800,
            min_speech_ms in 0u32..500,
            speech_pad_ms in 0u32..400,
        ) {
            let total = (probs.len() - 1) * VAD_FRAME_SAMPLES + tail;
            let c = cfg(threshold, min_silence_ms, min_speech_ms, speech_pad_ms);
            let (got_regions, got_samples, _) = stream(&probs, total, &c);
            let (want_regions, want_samples) = batch(&probs, total, &c);
            proptest::prop_assert_eq!(got_regions, want_regions);
            proptest::prop_assert_eq!(got_samples, want_samples);
        }
    }

    #[test]
    fn test_segmenter_retains_bounded_pcm_on_unbroken_speech() {
        // An hour of unbroken speech: the region never closes, so a segmenter
        // that waited for it would hold the whole hour. Released early, the
        // retained PCM stays inside the look-ahead the config implies.
        let c = VadConfig::default();
        // The bound is about a second of look-ahead. Three seconds already
        // exceeds it, so a segmenter that kept every sample fails the check.
        // An hour of samples does not finish under Miri.
        let total = if cfg!(miri) { 16000 * 3 } else { 16000 * 3600 };
        let probs = probs_for(total, |_| 0.9);
        let (regions, out, peak) = stream(&probs, total, &c);
        assert_eq!(regions, vec![(0, total)]);
        assert_eq!(out.len(), total);
        let bound = VadConfig::ms_to_samples(c.min_speech_ms + c.min_silence_ms + c.speech_pad_ms)
            + VAD_FRAME_SAMPLES
            + 977; // + the largest test chunk
        assert!(
            peak <= bound,
            "retained {peak} samples, expected at most {bound}"
        );
    }

    #[test]
    fn test_segmenter_retains_bounded_pcm_on_sparse_speech() {
        // Three hours of mostly silence with periodic speech: the same bound
        // must hold when regions open and close throughout.
        let c = VadConfig::default();
        // Eight seconds covers a full speech/silence cycle of this pattern.
        // Three hours of samples does not finish under Miri.
        let total = if cfg!(miri) {
            16000 * 8
        } else {
            16000 * 3600 * 3
        };
        let probs = probs_for(total, |i| if (i / 40) % 5 == 0 { 0.9 } else { 0.1 });
        let (regions, out, peak) = stream(&probs, total, &c);
        assert!(!regions.is_empty());
        assert_eq!(out.len(), regions.iter().map(|(s, e)| e - s).sum::<usize>());
        let bound = VadConfig::ms_to_samples(c.min_speech_ms + c.min_silence_ms + c.speech_pad_ms)
            + VAD_FRAME_SAMPLES
            + 977;
        assert!(
            peak <= bound,
            "retained {peak} samples, expected at most {bound}"
        );
    }
}

#[test]
fn test_remap_no_regions_is_identity() {
    assert_eq!(remap_compressed_seconds(1.5, &[], 16000.0), 1.5);
}

#[test]
fn test_remap_single_region_offsets_by_start() {
    // One region [16000, 32000) = original [1.0s, 2.0s). Compressed time 0
    // maps to 1.0s; compressed 0.5s maps to 1.5s.
    let regions = [(16000usize, 32000usize)];
    assert_eq!(remap_compressed_seconds(0.0, &regions, 16000.0), 1.0);
    assert_eq!(remap_compressed_seconds(0.5, &regions, 16000.0), 1.5);
}

#[test]
fn test_remap_second_region_skips_silence_gap() {
    // Regions: [0, 16000) then [48000, 64000) — a 2 s silence gap was cut.
    // Compressed timeline: [0,1s) then [1s,2s). A compressed time of 1.5s
    // falls in the second region 0.5s in → original 48000/16000 + 0.5 = 3.5s.
    let regions = [(0usize, 16000usize), (48000usize, 64000usize)];
    assert_eq!(remap_compressed_seconds(0.5, &regions, 16000.0), 0.5);
    assert_eq!(remap_compressed_seconds(1.5, &regions, 16000.0), 3.5);
}

#[test]
fn test_remap_past_end_clamps_to_last_region_end() {
    let regions = [(0usize, 16000usize), (48000usize, 64000usize)];
    // Compressed 10s is well past total speech (2s) → clamp to 64000/16000 = 4.0s.
    assert_eq!(remap_compressed_seconds(10.0, &regions, 16000.0), 4.0);
}

/// Model-gated: exercises the real Silero ONNX session through `ort` to
/// confirm the I/O plumbing (scalar `sr`, `[2,1,128]` recurrent state).
/// Run with the model present at `~/.gigastt/models/vad/silero_vad.onnx`:
/// `cargo test -p gigastt-core --lib vad::tests::test_silero -- --ignored`.
#[test]
#[ignore = "requires the Silero VAD model at ~/.gigastt/models/vad/silero_vad.onnx"]
fn test_silero_silence_low_prob_and_runs() {
    let home = std::env::var("HOME").expect("HOME");
    let path = std::path::PathBuf::from(home).join(".gigastt/models/vad/silero_vad.onnx");
    // The Silero VAD model is a separate, optional download (not part of the
    // GigaAM model cache). Skip gracefully when it is absent so the
    // `--ignored` coverage run doesn't fail where only GigaAM is present.
    if !path.exists() {
        eprintln!("skipping {}: Silero VAD model not present", path.display());
        return;
    }
    let vad = SileroVad::load(&path).expect("load silero");

    // 1 s of pure silence → several frames, all low probability.
    let silence = vec![0.0f32; 16000];
    let probs = vad.frame_probs(&silence).expect("frame_probs");
    assert!(!probs.is_empty(), "expected at least one frame");
    for p in &probs {
        assert!((0.0..=1.0).contains(p), "prob {p} out of range");
    }
    let max_silence = probs.iter().cloned().fold(0.0f32, f32::max);
    assert!(
        max_silence < 0.5,
        "silence should be below threshold, got {max_silence}"
    );

    // A loud 200 Hz tone is not speech either, but it must run cleanly and
    // stay in range (the point is to exercise the session, not classify).
    let tone: Vec<f32> = (0..16000)
        .map(|i| 0.5 * (2.0 * std::f32::consts::PI * 200.0 * i as f32 / 16000.0).sin())
        .collect();
    let probs2 = vad.frame_probs(&tone).expect("frame_probs tone");
    for p in &probs2 {
        assert!((0.0..=1.0).contains(p), "tone prob {p} out of range");
    }

    // No speech anywhere → no regions.
    assert!(
        vad.speech_regions(&silence, &VadConfig::default())
            .expect("regions")
            .is_empty()
    );
}

fn silero_model_path() -> std::path::PathBuf {
    let home = std::env::var("HOME").expect("HOME");
    std::path::PathBuf::from(home).join(".gigastt/models/vad/silero_vad.onnx")
}

/// Model-gated: drives [`VadEndpointer::push`] with sub-frame chunks to
/// exercise the leftover-buffer accumulation + drain across `push` calls
/// (the model is required because `push` runs every full frame through the
/// real Silero session). Verifies the chunk-accumulation mechanics, not
/// classification: chunks that individually fall short of one 512-sample
/// frame must not error and must not endpoint (no frame processed yet); once
/// a full frame's worth of samples accumulates, the frame is consumed and
/// the remainder retained for the next push.
#[test]
#[ignore = "requires the Silero VAD model at ~/.gigastt/models/vad/silero_vad.onnx"]
fn test_endpointer_buffers_subframe_chunks_across_pushes() {
    let path = silero_model_path();
    if !path.exists() {
        eprintln!("skipping {}: Silero VAD model not present", path.display());
        return;
    }
    let vad = SileroVad::load(&path).expect("load silero");
    let c = VadConfig::default();
    let mut ep = VadEndpointer::new(&c);

    // Two sub-frame silence chunks that together fall short of one frame:
    // no frame is processed, so no endpoint.
    let part = vec![0.0f32; 200];
    assert!(!ep.push(&vad, &part).expect("push part 1"));
    assert!(!ep.push(&vad, &part).expect("push part 2")); // 400 < 512 buffered

    // A third chunk crosses the frame boundary (600 buffered) → exactly one
    // full frame is consumed and the remainder retained; still no endpoint
    // on silence alone.
    let rest = vec![0.0f32; 200];
    assert!(!ep.push(&vad, &rest).expect("push part 3")); // 600 buffered, 1 frame
}

/// Model-gated: a single large silence chunk processes many frames in one
/// `push` (the inner accumulation loop) and must never endpoint before any
/// speech is seen; a following empty push processes no frames and stays
/// non-endpointing.
#[test]
#[ignore = "requires the Silero VAD model at ~/.gigastt/models/vad/silero_vad.onnx"]
fn test_endpointer_no_endpoint_on_leading_silence() {
    let path = silero_model_path();
    if !path.exists() {
        eprintln!("skipping {}: Silero VAD model not present", path.display());
        return;
    }
    let vad = SileroVad::load(&path).expect("load silero");
    let c = VadConfig::default();
    let mut ep = VadEndpointer::new(&c);

    // 1 s of silence = ~31 frames in a single push; leading silence (no
    // speech yet) must never report an endpoint.
    let silence = vec![0.0f32; 16000];
    assert!(
        !ep.push(&vad, &silence).expect("push silence"),
        "leading silence must not endpoint"
    );
    // A follow-up empty push processes no frames and stays non-endpointing.
    assert!(!ep.push(&vad, &[]).expect("push empty"));
}
