use super::*;

// --- prepare_audio_buffer tests ---

#[test]
fn test_buffer_short_input_returns_none() {
    // Less than N_FFT (320) samples → buffer everything
    let new_samples = vec![0.0; 100];
    let mut buffer = Vec::new();
    let result = prepare_audio_buffer(&new_samples, &mut buffer);
    assert!(result.is_none());
    assert_eq!(buffer.len(), 100);
}

#[test]
fn test_buffer_exact_frame() {
    // Exactly N_FFT (320) samples → one frame, no leftover
    let new_samples = vec![1.0; N_FFT];
    let mut buffer = Vec::new();
    let result = prepare_audio_buffer(&new_samples, &mut buffer);
    assert!(result.is_some());
    let usable = result.unwrap();
    assert_eq!(usable, N_FFT);
    consume_audio_buffer(&mut buffer, usable);
    assert!(buffer.is_empty());
}

#[test]
fn test_buffer_leftover_correct() {
    // N_FFT + 50 samples → one frame usable, 50 leftover
    let new_samples = vec![1.0; N_FFT + 50];
    let mut buffer = Vec::new();
    let result = prepare_audio_buffer(&new_samples, &mut buffer);
    assert!(result.is_some());
    let usable = result.unwrap();
    assert_eq!(usable, N_FFT); // one frame
    consume_audio_buffer(&mut buffer, usable);
    assert_eq!(buffer.len(), 50);
}

#[test]
fn test_buffer_accumulates_across_calls() {
    let mut buffer = Vec::new();
    // First call: 200 samples (< 320) → buffered
    let result = prepare_audio_buffer(&vec![1.0; 200], &mut buffer);
    assert!(result.is_none());
    assert_eq!(buffer.len(), 200);

    // Second call: 200 more → total 400, enough for 1 frame (320), leftover 80
    let result = prepare_audio_buffer(&vec![2.0; 200], &mut buffer);
    assert!(result.is_some());
    let usable = result.unwrap();
    assert_eq!(usable, 320);
    consume_audio_buffer(&mut buffer, usable);
    assert_eq!(buffer.len(), 80);
}

#[test]
fn test_buffer_truncation_at_5s() {
    // More than 80000 samples (5s at 16kHz) → truncate to last 80000
    let mut buffer = vec![0.0; 90000];
    let new_samples = vec![1.0; 1000];
    let result = prepare_audio_buffer(&new_samples, &mut buffer);
    // Total was 91000, truncated to 80000, then split into usable + leftover
    assert!(result.is_some());
    let usable = result.unwrap();
    consume_audio_buffer(&mut buffer, usable);
    assert!(usable + buffer.len() <= MAX_BUFFER_SAMPLES);
}

#[test]
fn test_buffer_multi_frame() {
    // N_FFT + HOP_LENGTH = 480 → 2 frames, no leftover
    let new_samples = vec![1.0; N_FFT + HOP_LENGTH];
    let mut buffer = Vec::new();
    let result = prepare_audio_buffer(&new_samples, &mut buffer);
    assert!(result.is_some());
    // 2 frames: usable = (2-1)*160 + 320 = 480
    let usable = result.unwrap();
    assert_eq!(usable, N_FFT + HOP_LENGTH);
    consume_audio_buffer(&mut buffer, usable);
    assert!(buffer.is_empty());
}

#[test]
fn test_prepare_buffer_empty_input() {
    let mut buffer = vec![1.0; 100];
    let result = prepare_audio_buffer(&[], &mut buffer);
    // Empty new samples: buffer should retain its contents
    assert!(result.is_none());
    assert_eq!(buffer.len(), 100);
}

#[test]
fn test_prepare_buffer_exactly_max() {
    // Exactly MAX_BUFFER_SAMPLES — should not trigger truncation warning
    let new_samples = vec![1.0; MAX_BUFFER_SAMPLES];
    let mut buffer = Vec::new();
    let result = prepare_audio_buffer(&new_samples, &mut buffer);
    assert!(result.is_some());
    let usable = result.unwrap();
    consume_audio_buffer(&mut buffer, usable);
    assert!(usable + buffer.len() <= MAX_BUFFER_SAMPLES);
}

#[test]
fn test_prepare_buffer_one_over_max() {
    // MAX_BUFFER_SAMPLES + 1 — triggers truncation
    let new_samples = vec![1.0; MAX_BUFFER_SAMPLES + 1];
    let mut buffer = Vec::new();
    let result = prepare_audio_buffer(&new_samples, &mut buffer);
    assert!(result.is_some());
    let usable = result.unwrap();
    consume_audio_buffer(&mut buffer, usable);
    assert!(usable + buffer.len() <= MAX_BUFFER_SAMPLES);
}

// --- parse_pcm16_with_carry tests ---

#[test]
fn test_parse_pcm16_basic() {
    let data: &[u8] = &[0x00, 0x40, 0x00, 0xC0]; // two i16 samples: 16384, -16384
    let mut pending: Option<u8> = None;
    let samples = parse_pcm16_with_carry(data, &mut pending);
    assert_eq!(samples.len(), 2);
    assert!(pending.is_none());
    assert!((samples[0] - 0.5).abs() < 0.001);
    assert!((samples[1] + 0.5).abs() < 0.001);
}

#[test]
fn test_parse_pcm16_odd_length_carry() {
    let mut pending: Option<u8> = None;
    let samples = parse_pcm16_with_carry(&[0x00, 0x00, 0xFF], &mut pending);
    assert_eq!(samples.len(), 1);
    assert_eq!(pending, Some(0xFF));

    let samples = parse_pcm16_with_carry(&[0x7F], &mut pending);
    assert_eq!(samples.len(), 1);
    assert!(pending.is_none());
}

#[test]
fn test_parse_pcm16_empty() {
    let mut pending: Option<u8> = None;
    let samples = parse_pcm16_with_carry(&[], &mut pending);
    assert!(samples.is_empty());
    assert!(pending.is_none());
}

#[cfg(feature = "file-decode")]
#[test]
fn test_sample_budget_pure() {
    // No caller limit means unbounded — the streaming path is O(one window),
    // so a file of any length transcribes.
    assert_eq!(max_samples_for_secs(None, 16000), usize::MAX);
    assert_eq!(max_samples_for_secs(Some(0.0), 16000), usize::MAX);
    assert_eq!(max_samples_for_secs(Some(-5.0), 16000), usize::MAX);

    // A finite limit scales linearly with the UNCLAMPED source rate — the fix
    // for the old 48 kHz clamp that made a 96 kHz file expire at half its
    // stated seconds. 1800 s at 96 kHz is twice the 48 kHz budget, not equal.
    assert_eq!(max_samples_for_secs(Some(1800.0), 16000), 1800 * 16000);
    assert_eq!(max_samples_for_secs(Some(1800.0), 96_000), 1800 * 96_000);
    assert_eq!(
        max_samples_for_secs(Some(1800.0), 192_000),
        4 * max_samples_for_secs(Some(1800.0), 48_000),
    );
}

#[cfg(feature = "file-decode")]
#[test]
fn test_whole_buffer_limit_clamps_but_only_downward() {
    // The whole-buffer ceiling is the default when the operator sets nothing.
    assert_eq!(whole_buffer_limit_secs(None), WHOLE_BUFFER_MAX_AUDIO_SECS);
    assert_eq!(
        whole_buffer_limit_secs(Some(0.0)),
        WHOLE_BUFFER_MAX_AUDIO_SECS
    );
    // A smaller operator limit lowers it.
    assert_eq!(whole_buffer_limit_secs(Some(300.0)), 300.0);
    // A larger operator limit cannot raise it above the ceiling.
    assert_eq!(
        whole_buffer_limit_secs(Some(10_000.0)),
        WHOLE_BUFFER_MAX_AUDIO_SECS
    );
}
