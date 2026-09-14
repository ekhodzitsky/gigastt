use super::*;

#[test]
fn test_abort_between_tokens_keeps_decoded_prefix() {
    use std::sync::atomic::{AtomicUsize, Ordering};
    let checks = AtomicUsize::new(0);
    let abort = || checks.fetch_add(1, Ordering::Relaxed) >= 3;
    let mut backend = FakeBackend::new(vec![0; 100], 2, 1);
    let encoded = Tensor::new_checked(
        Shape::new(vec![1, ENC_DIM, 2]),
        TensorData::F32(vec![0.0; ENC_DIM * 2]),
    );
    let result = greedy_decode_impl_with_abort(
        &mut backend,
        &encoded.view(),
        2,
        1,
        &mut DecoderState::new(1),
        None,
        Some(&abort),
    )
    .unwrap();
    assert_eq!(result.tokens.len(), 3);
    assert_eq!(backend.joiner_calls, 3);
    assert!(result.tokens.iter().all(|t| t.frame_index == 0));
    assert!(!result.endpoint_detected);
}

#[test]
fn test_abort_during_blank_run_stops_cached_decode() {
    use std::sync::atomic::{AtomicUsize, Ordering};
    let checks = AtomicUsize::new(0);
    let abort = || checks.fetch_add(1, Ordering::Relaxed) >= 3;
    let mut backend = FakeBackend::new(vec![], 2, 1);
    let encoded = Tensor::new_checked(
        Shape::new(vec![1, ENC_DIM, 50]),
        TensorData::F32(vec![0.0; ENC_DIM * 50]),
    );
    let result = greedy_decode_impl_with_abort(
        &mut backend,
        &encoded.view(),
        50,
        1,
        &mut DecoderState::new(1),
        None,
        Some(&abort),
    )
    .unwrap();
    assert!(result.tokens.is_empty());
    assert_eq!(backend.joiner_calls, 3);
    assert_eq!(backend.decoder_calls, 1);
}

// --- extract_encoder_frame tests ---

#[test]
fn test_extract_encoder_frame_first() {
    // 2 channels, 3 time steps: [ch0: 1,2,3, ch1: 4,5,6]
    let encoded = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let mut frame = vec![0.0; 2];
    extract_encoder_frame(&encoded, 3, 0, &mut frame);
    assert_eq!(frame, vec![1.0, 4.0]);
}

#[test]
fn test_extract_encoder_frame_last() {
    let encoded = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let mut frame = vec![0.0; 2];
    extract_encoder_frame(&encoded, 3, 2, &mut frame);
    assert_eq!(frame, vec![3.0, 6.0]);
}

#[test]
fn test_extract_encoder_frame_middle() {
    let encoded = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let mut frame = vec![0.0; 2];
    extract_encoder_frame(&encoded, 3, 1, &mut frame);
    assert_eq!(frame, vec![2.0, 5.0]);
}

// --- argmax tests ---

#[test]
fn test_argmax_clear_winner() {
    let logits = vec![0.1, 0.5, 0.9, 0.2];
    assert_eq!(argmax(&logits, 999), 2);
}

#[test]
fn test_argmax_tie_returns_last() {
    // Rust's Iterator::max_by returns the last element on ties
    let logits = vec![1.0, 1.0, 0.5];
    assert_eq!(argmax(&logits, 999), 1);
}

#[test]
fn test_argmax_single_element() {
    let logits = vec![42.0];
    assert_eq!(argmax(&logits, 999), 0);
}

#[test]
fn test_argmax_negative_values() {
    let logits = vec![-3.0, -1.0, -2.0];
    assert_eq!(argmax(&logits, 999), 1);
}

#[test]
fn test_argmax_empty_returns_blank() {
    let logits: Vec<f32> = vec![];
    assert_eq!(argmax(&logits, 1024), 1024);
}

#[test]
fn test_argmax_blank_id_selected() {
    // If blank_id is the argmax, it should be returned
    let logits = vec![0.1, 0.2, 0.9]; // index 2 is max
    assert_eq!(argmax(&logits, 2), 2); // blank_id matches argmax
}

// --- greedy_decode tests via a deterministic stub backend (no model) ---

/// Stub backend: a scripted sequence of token ids drives the joiner's argmax;
/// decoder/joiner call counts are recorded so the blank-run cache can be checked.
struct FakeBackend {
    script: std::collections::VecDeque<usize>,
    vocab: usize,
    blank_id: usize,
    decoder_calls: u32,
    joiner_calls: u32,
}

impl FakeBackend {
    fn new(script: Vec<usize>, vocab: usize, blank_id: usize) -> Self {
        Self {
            script: script.into(),
            vocab,
            blank_id,
            decoder_calls: 0,
            joiner_calls: 0,
        }
    }
}

impl DecodeBackend for FakeBackend {
    fn decode_step(
        &mut self,
        _state: &DecoderState,
        out: &mut DecoderOutput,
        _bufs: &mut DecodeBuffers,
    ) -> Result<()> {
        self.decoder_calls += 1;
        DecoderOutput::fill(&mut out.dec_data, &[0.0; PRED_HIDDEN]);
        DecoderOutput::fill(&mut out.new_h, &[0.0; PRED_HIDDEN]);
        DecoderOutput::fill(&mut out.new_c, &[0.0; PRED_HIDDEN]);
        Ok(())
    }

    fn joiner_step(
        &mut self,
        _enc_frame: &[f32],
        _dec_data: &[f32],
        logits_buf: &mut Vec<f32>,
        _bufs: &mut DecodeBuffers,
    ) -> Result<()> {
        self.joiner_calls += 1;
        // Once the script runs out, return blank so the loop terminates.
        let tok = self.script.pop_front().unwrap_or(self.blank_id);
        logits_buf.clear();
        logits_buf.resize(self.vocab, 0.0);
        logits_buf[tok] = 10.0; // argmax → tok
        Ok(())
    }
}

/// Stub backend returning the same joiner logits on every call — the shape
/// where the model prefers blank but a hotword token sits just below it.
struct FlatLogitsBackend {
    logits: Vec<f32>,
}

impl DecodeBackend for FlatLogitsBackend {
    fn decode_step(
        &mut self,
        _state: &DecoderState,
        out: &mut DecoderOutput,
        _bufs: &mut DecodeBuffers,
    ) -> Result<()> {
        DecoderOutput::fill(&mut out.dec_data, &[0.0; PRED_HIDDEN]);
        DecoderOutput::fill(&mut out.new_h, &[0.0; PRED_HIDDEN]);
        DecoderOutput::fill(&mut out.new_c, &[0.0; PRED_HIDDEN]);
        Ok(())
    }

    fn joiner_step(
        &mut self,
        _enc_frame: &[f32],
        _dec_data: &[f32],
        logits_buf: &mut Vec<f32>,
        _bufs: &mut DecodeBuffers,
    ) -> Result<()> {
        logits_buf.clear();
        logits_buf.extend_from_slice(&self.logits);
        Ok(())
    }
}

/// vocab=5, blank=4: blank leads, and the hotword phrase [1, 2] trails it by
/// less than the boost, so biasing decides every pick.
fn blank_leading_logits() -> Vec<f32> {
    let mut logits = vec![0.0; 5];
    logits[4] = 3.0; // blank
    logits[1] = 1.0; // first token of the hotword
    logits[2] = 1.5; // its continuation
    logits
}

#[test]
fn test_biasing_emits_at_most_one_token_per_frame() {
    // The reported failure: with the model wanting blank on every call, the
    // boost kept re-winning the *same* encoder frame, and the decoder
    // emptied MAX_TOKENS_PER_STEP into one 40 ms slot — a hotword rendered
    // as a stutter of its own prefix. Biasing may flip the greedy pick only
    // once per frame now, so the phrase advances at the pace of speech.
    let biaser = Biaser::from_sequences(vec![vec![1, 2]], 5.0).expect("biaser compiles");
    let mut backend = FlatLogitsBackend {
        logits: blank_leading_logits(),
    };
    let mut state = DecoderState::new(4);
    let enc = fake_enc_tensor(3);
    let result =
        greedy_decode_impl(&mut backend, &enc.view(), 3, 4, &mut state, Some(&biaser)).unwrap();

    let frames: Vec<usize> = result.tokens.iter().map(|t| t.frame_index).collect();
    assert_eq!(
        frames,
        vec![0, 1, 2],
        "each frame may contribute one biased token, not a burst"
    );
    // The hotword still completes: its prefix is emitted in order across
    // frames rather than being stuttered inside one.
    let ids: Vec<usize> = result.tokens.iter().map(|t| t.token_id).collect();
    assert_eq!(&ids[..2], &[1, 2], "hotword advances across frames");
}

#[test]
fn test_biasing_reports_the_models_own_confidence() {
    // The boost exists to steer the pick, not to inflate what we report
    // about it: confidence must come from the model's logits.
    let logits = blank_leading_logits();
    let biaser = Biaser::from_sequences(vec![vec![1, 2]], 5.0).expect("biaser compiles");
    let mut backend = FlatLogitsBackend {
        logits: logits.clone(),
    };
    let mut state = DecoderState::new(4);
    let enc = fake_enc_tensor(1);
    let result =
        greedy_decode_impl(&mut backend, &enc.view(), 1, 4, &mut state, Some(&biaser)).unwrap();

    assert_eq!(result.tokens.len(), 1);
    let expected = token_confidence(&logits, 1);
    assert!(
        (result.tokens[0].confidence - expected).abs() < 1e-6,
        "confidence {} should be the un-boosted {expected}",
        result.tokens[0].confidence
    );
}

#[test]
fn test_biasing_leaves_a_confident_model_pick_alone() {
    // When the model already outranks the boost, biasing changes nothing and
    // spends no per-frame budget.
    let mut logits = vec![0.0; 5];
    logits[3] = 10.0; // a non-hotword token the model is sure about
    logits[1] = 1.0;
    let biaser = Biaser::from_sequences(vec![vec![1, 2]], 5.0).expect("biaser compiles");
    let mut backend = FlatLogitsBackend { logits };
    let mut state = DecoderState::new(4);
    let enc = fake_enc_tensor(1);
    let result =
        greedy_decode_impl(&mut backend, &enc.view(), 1, 4, &mut state, Some(&biaser)).unwrap();

    assert_eq!(
        result.tokens.len(),
        MAX_TOKENS_PER_STEP,
        "an unbiased pick is not rationed by the biasing budget"
    );
    assert!(result.tokens.iter().all(|t| t.token_id == 3));
}

/// Encoder buffer of `frames` zeroed frames (content is irrelevant to the stub).
fn fake_enc(frames: usize) -> Vec<f32> {
    vec![0.0_f32; ENC_DIM * frames]
}

/// Wrapped encoder tensor for tests driving `greedy_decode_impl`.
fn fake_enc_tensor(frames: usize) -> Tensor {
    Tensor::new(
        Shape::new(vec![1, ENC_DIM, frames]),
        TensorData::F32(fake_enc(frames)),
    )
    .unwrap()
}

#[test]
fn test_greedy_decode_happy_path() {
    // vocab=5, blank=4. Frame 0 emits token 1 then blank; frame 1 emits token 2.
    let mut backend = FakeBackend::new(vec![1, 4, 2, 4], 5, 4);
    let mut state = DecoderState::new(4);
    let enc = fake_enc_tensor(2);
    let result = greedy_decode_impl(&mut backend, &enc.view(), 2, 4, &mut state, None).unwrap();

    assert_eq!(result.tokens.len(), 2);
    assert_eq!(result.tokens[0].token_id, 1);
    assert_eq!(result.tokens[0].frame_index, 0);
    assert_eq!(result.tokens[1].token_id, 2);
    assert_eq!(result.tokens[1].frame_index, 1);
    // Last committed token updates prev_token and the LSTM state buffers.
    assert_eq!(state.prev_token, 2);
    assert_eq!(state.h.len(), PRED_HIDDEN);
    assert!(!result.endpoint_detected);
}

#[test]
fn test_greedy_decode_blank_run_skips_decoder() {
    // Frame 0: token then blank (2 decoder calls). Frames 1-3: blank only.
    // The decoder must NOT be called again during the blank run (cache reuse).
    let mut backend = FakeBackend::new(vec![1, 4, 4, 4, 4], 5, 4);
    let mut state = DecoderState::new(4);
    let enc = fake_enc_tensor(4);
    let result = greedy_decode_impl(&mut backend, &enc.view(), 4, 4, &mut state, None).unwrap();

    assert_eq!(result.tokens.len(), 1);
    assert_eq!(
        backend.decoder_calls, 2,
        "decoder must not run during the blank run"
    );
    assert!(backend.joiner_calls >= 5);
}

#[test]
fn test_greedy_decode_endpoint_after_threshold_blanks() {
    // One token, then ENDPOINT_BLANK_THRESHOLD+ blanks → endpoint detected.
    let mut script = vec![1usize];
    script.extend(std::iter::repeat_n(4usize, ENDPOINT_BLANK_THRESHOLD + 1));
    let frames = ENDPOINT_BLANK_THRESHOLD + 2;
    let mut backend = FakeBackend::new(script, 5, 4);
    let mut state = DecoderState::new(4);
    let enc = fake_enc_tensor(frames);
    let result =
        greedy_decode_impl(&mut backend, &enc.view(), frames, 4, &mut state, None).unwrap();

    assert!(
        result.endpoint_detected,
        "{ENDPOINT_BLANK_THRESHOLD}+ blanks after a token must endpoint"
    );
}

#[test]
fn test_greedy_decode_no_endpoint_before_first_token() {
    // All blanks, no token emitted → the !tokens.is_empty() guard blocks endpoint.
    let frames = ENDPOINT_BLANK_THRESHOLD + 5;
    let mut backend = FakeBackend::new(vec![4usize; frames], 5, 4);
    let mut state = DecoderState::new(4);
    let enc = fake_enc_tensor(frames);
    let result =
        greedy_decode_impl(&mut backend, &enc.view(), frames, 4, &mut state, None).unwrap();

    assert!(result.tokens.is_empty());
    assert!(
        !result.endpoint_detected,
        "blanks before any token must not endpoint"
    );
}

#[test]
fn test_greedy_decode_token_cap_does_not_inflate_blanks() {
    // One frame; the joiner returns a non-blank token on every call past the cap.
    // Exactly MAX_TOKENS_PER_STEP tokens are emitted, and the token cap must NOT
    // bump the blank counter or fire an endpoint.
    let mut backend = FakeBackend::new(vec![1usize; MAX_TOKENS_PER_STEP + 1], 5, 4);
    let mut state = DecoderState::new(4);
    let enc = fake_enc_tensor(1);
    let result = greedy_decode_impl(&mut backend, &enc.view(), 1, 4, &mut state, None).unwrap();

    assert_eq!(result.tokens.len(), MAX_TOKENS_PER_STEP);
    assert_eq!(
        state.consecutive_blanks, 0,
        "token cap must not inflate the blank counter"
    );
    assert!(!result.endpoint_detected);
}

#[test]
fn test_argmax_with_confidence_clear_winner() {
    let (tok, conf) = argmax_with_confidence(&[0.1, 5.0, 0.2], 99);
    assert_eq!(tok, 1);
    assert!(
        conf > 0.5 && conf <= 1.0,
        "confidence should be a softmax prob in (0.5, 1], got {conf}"
    );
}

#[test]
fn test_argmax_with_confidence_empty_returns_blank_zero() {
    let (tok, conf) = argmax_with_confidence(&[], 1024);
    assert_eq!(tok, 1024);
    assert_eq!(conf, 0.0);
}

// --- contextual hotword biasing gate tests (no model) ---

/// Stub backend that returns a fixed per-call logit vector, so a test can
/// set a small margin between two tokens and check whether the bias boost
/// flips the argmax. Each `joiner_step` pops the next scripted logit vector;
/// once exhausted it emits all-blank so the loop terminates.
struct LogitBackend {
    script: std::collections::VecDeque<Vec<f32>>,
    vocab: usize,
    blank_id: usize,
}

impl LogitBackend {
    fn new(script: Vec<Vec<f32>>, vocab: usize, blank_id: usize) -> Self {
        Self {
            script: script.into(),
            vocab,
            blank_id,
        }
    }
}

impl DecodeBackend for LogitBackend {
    fn decode_step(
        &mut self,
        _state: &DecoderState,
        out: &mut DecoderOutput,
        _bufs: &mut DecodeBuffers,
    ) -> Result<()> {
        DecoderOutput::fill(&mut out.dec_data, &[0.0; PRED_HIDDEN]);
        DecoderOutput::fill(&mut out.new_h, &[0.0; PRED_HIDDEN]);
        DecoderOutput::fill(&mut out.new_c, &[0.0; PRED_HIDDEN]);
        Ok(())
    }

    fn joiner_step(
        &mut self,
        _enc_frame: &[f32],
        _dec_data: &[f32],
        logits_buf: &mut Vec<f32>,
        _bufs: &mut DecodeBuffers,
    ) -> Result<()> {
        logits_buf.clear();
        match self.script.pop_front() {
            Some(v) => logits_buf.extend_from_slice(&v),
            None => {
                // Exhausted → blank wins so the frame loop ends.
                logits_buf.resize(self.vocab, 0.0);
                logits_buf[self.blank_id] = 10.0;
            }
        }
        Ok(())
    }
}

/// vocab = 4: ids 0,1,2 are real tokens, id 3 is blank. Token A=1 leads B=2
/// by a small margin on the first frame; the second frame is blank.
fn ab_script() -> Vec<Vec<f32>> {
    vec![
        // frame 0: A(1)=2.0 beats B(2)=1.0 with no bias.
        vec![0.0, 2.0, 1.0, 0.0],
        // frame 0 continuation after a token emit: blank dominates (large so
        // a bias boost can't overtake it) → next frame.
        vec![0.0, 0.0, 0.0, 100.0],
    ]
}

#[test]
fn test_bias_steers_argmax_to_boosted_token() {
    // Without bias the model picks A=1. With a hotword [2] and a boost large
    // enough to clear A's 1.0 lead, the loop must instead emit B=2.
    // Baseline (no bias): emits token 1.
    let mut backend = LogitBackend::new(ab_script(), 4, 3);
    let mut state = DecoderState::new(3);
    let enc = fake_enc_tensor(2);
    let unbiased = greedy_decode_impl(&mut backend, &enc.view(), 2, 3, &mut state, None).unwrap();
    assert_eq!(unbiased.tokens.len(), 1);
    assert_eq!(unbiased.tokens[0].token_id, 1, "no bias → model picks A");

    // Biased: hotword whose first token is B=2, boost 5.0 > the 1.0 gap.
    let biaser = Biaser::from_sequences(vec![vec![2]], 5.0).unwrap();
    let mut backend = LogitBackend::new(ab_script(), 4, 3);
    let mut state = DecoderState::new(3);
    let enc = fake_enc_tensor(2);
    let biased =
        greedy_decode_impl(&mut backend, &enc.view(), 2, 3, &mut state, Some(&biaser)).unwrap();
    assert_eq!(biased.tokens.len(), 1);
    assert_eq!(
        biased.tokens[0].token_id, 2,
        "boost must steer the argmax from A to the hotword token B"
    );
}

#[test]
fn test_bias_prefix_advances_then_boosts_continuation() {
    // vocab = 6: 0,1,2,4,5 real, 3 = blank. Hotword is the two-token
    // sequence [5,2], where the prefix token 5 is distinct from the
    // competitor A=1 so the continuation boost is unambiguous. Frame 0
    // emits 5 (wins outright), advancing the prefix to expect 2; the boost
    // on 2 then steers frame 1 where A=1 would otherwise win.
    // Continuation (blank) frames use a large blank logit so the bias boost
    // can never overtake the blank and spuriously emit another token —
    // these frames only exist to terminate the per-frame inner loop.
    let script = vec![
        // frame 0: token 5 wins outright (start of the hotword).
        vec![0.0, 0.0, 0.0, 0.0, 0.0, 3.0],
        // frame 0 continuation: blank dominates → advance to frame 1.
        vec![0.0, 0.0, 0.0, 100.0, 0.0, 0.0],
        // frame 1: A(1)=2.0 vs B(2)=1.0 — without the prefix boost A wins.
        vec![0.0, 2.0, 1.0, 0.0, 0.0, 0.0],
        // frame 1 continuation: blank dominates → end.
        vec![0.0, 0.0, 0.0, 100.0, 0.0, 0.0],
    ];
    let biaser = Biaser::from_sequences(vec![vec![5, 2]], 5.0).unwrap();
    let mut backend = LogitBackend::new(script, 6, 3);
    let mut state = DecoderState::new(3);
    let enc = fake_enc_tensor(2);
    let result =
        greedy_decode_impl(&mut backend, &enc.view(), 2, 3, &mut state, Some(&biaser)).unwrap();
    assert_eq!(
        result.tokens.iter().map(|t| t.token_id).collect::<Vec<_>>(),
        vec![5, 2],
        "prefix [5] must advance so the boost on the continuation 2 steers frame 1"
    );
}

#[test]
fn test_bias_none_is_byte_for_byte_unchanged() {
    // No-op safety: a hotword that can never apply (Some biaser) vs None must
    // produce identical tokens to the un-biased decode on the same script.
    // Here we compare None against a biaser whose only hotword token (id 0)
    // never wins, so selection is unchanged but the bias code path runs.
    let base_script = || {
        vec![
            vec![0.0, 2.0, 1.0, 0.0],
            vec![0.0, 0.0, 0.0, 100.0],
            vec![0.0, 1.5, 2.5, 0.0],
            vec![0.0, 0.0, 0.0, 100.0],
        ]
    };
    let mut b_none = LogitBackend::new(base_script(), 4, 3);
    let mut s_none = DecoderState::new(3);
    let enc_none = fake_enc_tensor(2);
    let none = greedy_decode_impl(&mut b_none, &enc_none.view(), 2, 3, &mut s_none, None).unwrap();

    // A biaser for token id 0, which has a -inf-equivalent (0.0) logit and is
    // dominated on every frame, so it can never change the argmax.
    let biaser = Biaser::from_sequences(vec![vec![0]], 0.5).unwrap();
    let mut b_some = LogitBackend::new(base_script(), 4, 3);
    let mut s_some = DecoderState::new(3);
    let enc_some = fake_enc_tensor(2);
    let some = greedy_decode_impl(
        &mut b_some,
        &enc_some.view(),
        2,
        3,
        &mut s_some,
        Some(&biaser),
    )
    .unwrap();

    assert_eq!(
        none.tokens.iter().map(|t| t.token_id).collect::<Vec<_>>(),
        some.tokens.iter().map(|t| t.token_id).collect::<Vec<_>>(),
        "a non-winning hotword must not change the decoded tokens"
    );
    assert_eq!(none.tokens.len(), 2);
}
