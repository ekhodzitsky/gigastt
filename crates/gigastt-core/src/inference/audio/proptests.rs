use super::tests::make_wav_bytes;
use super::*;
use proptest::prelude::*;

proptest! {
    #[test]
    fn proptest_pcm16_carry_invariant(
        chunks in proptest::collection::vec(
            proptest::collection::vec(any::<u8>(), 0..1000),
            1..20
        )
    ) {
        let mut pending: Option<u8> = None;
        let mut total_samples = 0usize;
        let mut total_bytes = 0usize;

        for chunk in &chunks {
            total_bytes += chunk.len();
            let samples = parse_pcm16_with_carry(chunk, &mut pending);
            total_samples += samples.len();
        }

        let expected = total_bytes / 2;
        prop_assert_eq!(total_samples, expected,
            "samples ({}) must equal total_bytes/2 ({})", total_samples, expected);

        if total_bytes % 2 == 1 {
            prop_assert!(pending.is_some());
        } else {
            prop_assert!(pending.is_none());
        }
    }

    #[test]
    fn proptest_resample_no_panic(
        samples in proptest::collection::vec(-1.0f32..1.0f32, 1..5_000),
        rate_idx in 0..5usize,
    ) {
        let rates = [8000u32, 16000, 24000, 44100, 48000];
        let from_rate = SampleRate(rates[rate_idx]);
        if from_rate.0 == 16000 {
            return Ok(());
        }
        let result = resample(&samples, from_rate, SampleRate(16000));
        prop_assert!(result.is_ok(), "resample failed: {:?}", result.err());
    }

    #[test]
    fn proptest_decode_header_sample_rate_never_panics(rate in 0u32..=300_000u32) {
        // Decoding a WAV with an arbitrary header sample rate must never panic;
        // any rate above the ceiling must be rejected, never accepted.
        let silence: Vec<i16> = vec![0; 8];
        let result = decode_audio_bytes(&make_wav_bytes(&silence, rate));
        if rate > MAX_SAMPLE_RATE {
            prop_assert!(result.is_err(), "rate {} above ceiling must be rejected", rate);
        }
    }
}
