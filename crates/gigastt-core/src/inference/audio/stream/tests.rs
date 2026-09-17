use super::*;

/// Reference oracle for the window sequence: the fixed-stride loop the source
/// replaced, with the trailing-window rules on top. A window that can reach
/// the end within the single-pass ceiling absorbs the remainder (the first
/// window included: a stream within the ceiling is one window); otherwise a
/// trailing window that would fall short of the full window length is
/// anchored to the end of the stream (start rounded up to the frame grid) so
/// it overlaps its predecessor by more than the nominal overlap.
fn expected_windows(total: usize, spec: WindowSpec) -> Vec<(usize, usize)> {
    let (window, stride) = (spec.window(), spec.stride());
    let last_max = spec.single_pass_max().max(window);
    let mut out = Vec::new();
    let mut start = 0usize;
    while start < total {
        let remaining = total - start;
        if remaining <= last_max && (remaining >= window || out.is_empty()) {
            out.push((start, remaining));
            break;
        }
        if remaining < window {
            if spec.overlap() > 0 {
                let anchored = (total - window).div_ceil(FRAME_SAMPLES) * FRAME_SAMPLES;
                assert!(anchored > out[out.len() - 1].0 && anchored <= start);
                out.push((anchored, total - anchored));
            } else {
                out.push((start, remaining));
            }
            break;
        }
        out.push((start, window));
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
fn test_slice_windows_match_anchored_geometry_swept() {
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
        let seq = observed(total, spec);
        assert_eq!(
            seq,
            expected_windows(total, spec),
            "window sequence diverged at total={total}"
        );
        // Structural invariants: contiguous coverage of [0, total), strictly
        // increasing frame-aligned starts, no window longer than the
        // single-pass ceiling, and — beyond that ceiling — no window shorter
        // than the full window minus one frame.
        let mut covered = 0usize;
        let mut prev_start = None;
        for &(start, len) in &seq {
            assert!(start <= covered, "gap before {start} at total={total}");
            assert_eq!(start % FRAME_SAMPLES, 0, "unaligned start at total={total}");
            assert!(
                prev_start.is_none_or(|p| start > p),
                "non-advancing at total={total}"
            );
            assert!(
                len <= spec.single_pass_max(),
                "over-long window at total={total}"
            );
            if total > spec.single_pass_max() {
                assert!(
                    len > window - FRAME_SAMPLES,
                    "short window {len} at total={total}"
                );
            }
            covered = covered.max(start + len);
            prev_start = Some(start);
        }
        assert_eq!(covered, total, "coverage at total={total}");
    }
}

#[test]
fn test_short_remainder_is_absorbed_by_the_last_window() {
    // The 71.25 s official long-form example: two stride windows, then the
    // third reaches the end within the 30 s single-pass ceiling (27.25 s), so
    // the 5.25 s remainder never becomes a window of its own.
    let total = 1_140_000;
    assert_eq!(
        observed(total, ort_spec()),
        vec![(0, 384_000), (352_000, 384_000), (704_000, total - 704_000)]
    );
}

#[test]
fn test_long_remainder_is_anchored_to_the_end() {
    // 80 s: after [44, 68] the remainder (12 s) exceeds what the last window
    // may absorb (6 s), so the trailing window is a full 24 s ending at the
    // end, [56, 80], instead of a 14 s window at the 66 s stride.
    let total = 16_000 * 80;
    assert_eq!(
        observed(total, ort_spec()),
        vec![
            (0, 384_000),
            (352_000, 384_000),
            (704_000, 384_000),
            (16_000 * 56, 384_000),
        ]
    );
    // A remainder that is not frame-aligned rounds the anchored start up so
    // the window stays at most one window long: 80.01 s → start 56.04 s.
    let total = 16_000 * 80 + 160;
    let seq = observed(total, ort_spec());
    assert_eq!(
        seq.last(),
        Some(&(16_000 * 56 + 640, total - 16_000 * 56 - 640))
    );
}

#[test]
fn test_trailing_window_that_fits_exactly_is_unchanged() {
    // One stride plus one window: the second window already reaches the end.
    let total = 352_000 + 384_000;
    assert_eq!(
        observed(total, ort_spec()),
        vec![(0, 384_000), (352_000, 384_000)]
    );
}

#[test]
fn test_remainder_past_the_ceiling_gets_a_full_window() {
    // Two seconds past the single-pass ceiling: the first window cannot absorb
    // 32 s, and the 10 s remainder after it exceeds the 6 s absorb margin, so
    // the trailing window is re-anchored to [8 s, 32 s].
    let total = 16_000 * 32;
    assert_eq!(
        observed(total, ort_spec()),
        vec![(0, 384_000), (16_000 * 8, 384_000)]
    );
    // Just past the first window plus the absorb margin (30.01 s) the same
    // applies: [6.04 s, 30.01 s] rather than a 8 s tail at 22 s.
    let total = 16_000 * 30 + 160;
    let seq = observed(total, ort_spec());
    assert_eq!(seq.len(), 2);
    assert_eq!(seq[1], (16_000 * 6 + 640, total - 16_000 * 6 - 640));
}

#[test]
fn test_first_window_is_never_anchored() {
    // A stream shorter than one window (forced past the single-pass ceiling)
    // is a single partial window starting at 0: anchoring needs a predecessor.
    let spec = WindowSpec::new(0, 384_000, 32_000);
    assert_eq!(observed(16_000 * 20, spec), vec![(0, 16_000 * 20)]);
}

#[test]
fn test_partition_spec_never_anchors() {
    // Zero overlap is a partition (the VAD's flat container pulls): the tail
    // stays a short block, never a re-read of audio already handed out.
    let spec = WindowSpec::new(0, 32_000, 0);
    assert_eq!(
        observed(70_000, spec),
        vec![(0, 32_000), (32_000, 32_000), (64_000, 6_000)]
    );
}

#[test]
fn test_cursor_retains_one_stride_of_look_behind() {
    let spec = ort_spec();
    let mut cursor = WindowCursor::new(spec);
    assert_eq!(cursor.retain_from(), 0);
    let total = 16_000 * 90;
    cursor.take(total, true);
    // Next nominal start is one stride in; look-behind reaches back to 0.
    assert_eq!(cursor.next_start(), spec.stride());
    assert_eq!(cursor.retain_from(), 0);
    cursor.take(total, true);
    assert_eq!(cursor.retain_from(), spec.stride());
    // 90 s: [0, 24], [22, 46], [44, 68], then a 22 s remainder that is too
    // long to absorb, so the anchored trailing window [66, 90] starts after
    // the retained origin.
    let mut starts = vec![];
    while let Some((start, _)) = cursor.take(total, true) {
        assert!(start >= cursor.retain_from().saturating_sub(spec.stride()));
        starts.push(start);
    }
    assert_eq!(starts, vec![704_000, 16_000 * 66]);
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
