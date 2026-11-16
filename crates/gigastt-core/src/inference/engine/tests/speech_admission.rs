use super::*;
use crate::runtime::mock::MockSession;
use crate::runtime::tensor::{Shape, Tensor, TensorData};

fn outputs(blank: bool) -> Vec<Tensor> {
    let mut logits = vec![-10.0; 2 * 71];
    logits[70] = 0.0;
    logits[71 + if blank { 70 } else { 0 }] = 0.0;
    vec![
        Tensor::new_checked(Shape::new(vec![1, 2, 71]), TensorData::F32(logits)),
        Tensor::new_checked(Shape::new(vec![1]), TensorData::I64(vec![2])),
    ]
}
fn attach(engine: &mut Engine, blank: bool) {
    let (mut sidecar, _tmp) = crate::test_support::rnnt_engine();
    sidecar.variant = ModelVariant::MlCtcLarge;
    sidecar.tokenizer = Tokenizer::from_tokens(
        (0..71)
            .map(|i| {
                if i == 70 {
                    "<blk>".into()
                } else {
                    format!("t{i}")
                }
            })
            .collect(),
    );
    sidecar.pool.checkout_blocking().unwrap().encoder =
        Box::new(MockSession::unconstrained(outputs(blank)));
    engine.attach_ctc_speech_admission(sidecar).unwrap();
}

#[test]
fn test_speech_admission_veto_consumes_stride_without_recognition_or_coverage() {
    let (mut engine, _tmp) = crate::test_support::rnnt_engine();
    attach(&mut engine, true);
    let mut triplet = engine.pool.checkout_blocking().unwrap();
    triplet.encoder = Box::new(MockSession::new(vec![], vec![]));
    let mut state = engine.create_state(false);
    assert!(
        engine
            .process_chunk(&vec![0.0; 12800], &mut state, &mut triplet)
            .unwrap()
            .is_empty()
    );
    assert!(state.assembler.is_empty());
    assert_eq!(state.pending_samples, 0);
    assert_eq!(state.audio_buffer.len(), 12800);
    assert_eq!(state.context_samples, 0);
}

#[test]
fn test_speech_admission_idle_cap_and_repeated_stop_do_not_commit_pcm() {
    let (mut engine, _tmp) = crate::test_support::rnnt_engine();
    attach(&mut engine, true);
    let mut triplet = engine.pool.checkout_blocking().unwrap();
    let mut state = engine.create_state(false);
    state.context_samples = 20000;
    engine
        .process_chunk(&vec![0.0; 51200], &mut state, &mut triplet)
        .unwrap();
    assert_eq!(state.audio_buffer.len(), 24000);
    assert_eq!(state.window_start_samples, 27200);
    assert_eq!(state.context_samples, 0);
    engine
        .process_chunk(&[0.0; 1600], &mut state, &mut triplet)
        .unwrap();
    assert!(
        engine
            .try_finish_stream(&mut state, &mut triplet)
            .unwrap()
            .is_none()
    );
    assert_eq!(state.window_start_samples, 28800);
    assert_eq!(state.context_samples, 0);
    assert!(
        engine
            .try_finish_stream(&mut state, &mut triplet)
            .unwrap()
            .is_none()
    );
    assert_eq!(state.context_samples, 0);
    assert_eq!(state.pending_samples, 0);
}

#[test]
fn test_speech_admission_nonempty_assembler_bypasses_sidecar() {
    let (mut engine, _tmp) = crate::test_support::rnnt_engine();
    attach(&mut engine, true);
    // A malformed sidecar must never run for either tentative or committed text.
    engine
        .speech_admission
        .as_ref()
        .unwrap()
        .pool
        .checkout_blocking()
        .unwrap()
        .encoder = Box::new(MockSession::unconstrained(vec![]));
    let mut triplet = engine.pool.checkout_blocking().unwrap();
    for committed in [false, true] {
        let mut state = engine.create_state(false);
        state.assembler.append(vec![word("не", 0.0, 0.1)]);
        if committed {
            state.assembler.commit_live();
        }
        engine
            .process_chunk(&vec![0.0; 12800], &mut state, &mut triplet)
            .unwrap();
    }
}

struct CaptureAdmission {
    seen: std::sync::Arc<std::sync::Mutex<Vec<Vec<Tensor>>>>,
    abort: Option<std::sync::Arc<std::sync::atomic::AtomicBool>>,
}
impl crate::runtime::session::RuntimeSession for CaptureAdmission {
    fn run(&self, inputs: &[Tensor]) -> Result<Vec<Tensor>, crate::runtime::RuntimeError> {
        self.seen.lock().unwrap().push(inputs.to_vec());
        if let Some(abort) = &self.abort {
            abort.store(true, std::sync::atomic::Ordering::Relaxed);
        }
        Ok(outputs(true))
    }
}

#[test]
fn test_speech_admission_uses_exact_mel_and_rechecks_pending_stop_only() {
    let (mut engine, _tmp) = crate::test_support::rnnt_engine();
    attach(&mut engine, true);
    let seen = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
    engine
        .speech_admission
        .as_ref()
        .unwrap()
        .pool
        .checkout_blocking()
        .unwrap()
        .encoder = Box::new(CaptureAdmission {
        seen: seen.clone(),
        abort: None,
    });
    let mut triplet = engine.pool.checkout_blocking().unwrap();
    triplet.encoder = Box::new(MockSession::new(vec![], vec![]));
    let mut state = engine.create_state(false);
    let pcm: Vec<_> = (0..12800).map(|i| (i as f32 * 0.01).sin()).collect();
    engine
        .process_chunk(&pcm, &mut state, &mut triplet)
        .unwrap();
    let captured = seen.lock().unwrap();
    assert_eq!(captured.len(), 1);
    assert_eq!(
        captured[0][0].view().data().as_f32().unwrap(),
        state.mel_output
    );
    drop(captured);
    engine
        .process_chunk(&[0.5; 1600], &mut state, &mut triplet)
        .unwrap();
    engine.try_finish_stream(&mut state, &mut triplet).unwrap();
    engine.try_finish_stream(&mut state, &mut triplet).unwrap();
    assert_eq!(seen.lock().unwrap().len(), 2);
    assert_eq!(state.context_samples, 0);
}

#[test]
fn test_speech_admission_nonblank_space_allows_native_decode_and_errors_propagate() {
    let (mut engine, _tmp) = crate::test_support::rnnt_engine();
    attach(&mut engine, false); // Nonblank class 0: even a rendered space admits RNNT.
    let mut triplet = engine.pool.checkout_blocking().unwrap();
    triplet.encoder = Box::new(MockSession::new(vec![], vec![]));
    let mut state = engine.create_state(false);
    assert!(
        engine
            .process_chunk(&vec![0.0; 12800], &mut state, &mut triplet)
            .is_err()
    );
    assert_eq!(state.pending_samples, 12800);
    engine
        .speech_admission
        .as_ref()
        .unwrap()
        .pool
        .checkout_blocking()
        .unwrap()
        .encoder = Box::new(MockSession::unconstrained(vec![]));
    let mut state = engine.create_state(false);
    state.audio_buffer = vec![0.0; 1000];
    state.pending_samples = 1000;
    assert!(engine.try_finish_stream(&mut state, &mut triplet).is_err());
    assert!(state.failed);
    assert_eq!(state.pending_samples, 1000);
    assert_eq!(state.context_samples, 0);
}

#[test]
fn test_speech_admission_abort_during_sidecar_is_not_successful_silence() {
    for finish in [false, true] {
        let (mut engine, _tmp) = crate::test_support::rnnt_engine();
        attach(&mut engine, true);
        let flag = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        engine
            .speech_admission
            .as_ref()
            .unwrap()
            .pool
            .checkout_blocking()
            .unwrap()
            .encoder = Box::new(CaptureAdmission {
            seen: Default::default(),
            abort: Some(flag.clone()),
        });
        let mut state = engine.create_state(false);
        state.abort = Some(flag);
        let mut triplet = engine.pool.checkout_blocking().unwrap();
        let result = if finish {
            state.audio_buffer = vec![0.0; 1000];
            state.pending_samples = 1000;
            engine
                .try_finish_stream(&mut state, &mut triplet)
                .map(|_| ())
        } else {
            engine
                .process_chunk(&vec![0.0; 12800], &mut state, &mut triplet)
                .map(|_| ())
        };
        assert!(matches!(result, Err(GigasttError::Cancelled)));
        assert!(state.failed);
        assert_eq!(state.context_samples, 0);
    }
}

#[test]
fn test_speech_admission_short_stop_preserves_pending_and_existing_coverage() {
    let (mut engine, _tmp) = crate::test_support::rnnt_engine();
    attach(&mut engine, true);
    engine
        .speech_admission
        .as_ref()
        .unwrap()
        .pool
        .checkout_blocking()
        .unwrap()
        .encoder = Box::new(MockSession::unconstrained(vec![]));
    let mut state = engine.create_state(false);
    let mut triplet = engine.pool.checkout_blocking().unwrap();
    engine
        .process_chunk(&[0.0; 319], &mut state, &mut triplet)
        .unwrap();
    engine.try_finish_stream(&mut state, &mut triplet).unwrap();
    assert_eq!(state.pending_samples, 319);
    assert_eq!(state.audio_buffer.len(), 319);
    state.audio_buffer = vec![0.0; 25000];
    state.context_samples = 2000;
    Engine::slide_uncommitted_speech_window(&mut state);
    assert_eq!(state.context_samples, 1000);
    assert_eq!(state.window_start_samples, 1000);
}

#[test]
fn test_speech_admission_later_word_after_idle_cap_is_not_marked_committed() {
    let (mut engine, _tmp) = crate::test_support::rnnt_engine();
    attach(&mut engine, true);
    let mut state = engine.create_state(false);
    let mut triplet = engine.pool.checkout_blocking().unwrap();
    engine
        .process_chunk(&vec![0.0; 51200], &mut state, &mut triplet)
        .unwrap();
    engine
        .speech_admission
        .as_ref()
        .unwrap()
        .pool
        .checkout_blocking()
        .unwrap()
        .encoder = Box::new(MockSession::unconstrained(outputs(false)));
    triplet.encoder = Box::new(MockSession::unconstrained(vec![
        Tensor::new_checked(
            Shape::new(vec![1, 768, 2]),
            TensorData::F32(vec![0.0; 768 * 2]),
        ),
        Tensor::new_checked(Shape::new(vec![1]), TensorData::I64(vec![2])),
    ]));
    let joint = |values| {
        vec![Tensor::new_checked(
            Shape::new(vec![1, 1, 2]),
            TensorData::F32(values),
        )]
    };
    triplet.joiner = Some(Box::new(
        MockSession::unconstrained(joint(vec![0.0, 10.0])).with_script(vec![
            joint(vec![0.0, 10.0]),
            joint(vec![10.0, 0.0]),
            joint(vec![0.0, 10.0]),
        ]),
    ));
    engine
        .process_chunk(&vec![0.0; 12800], &mut state, &mut triplet)
        .unwrap();
    assert!(!state.assembler.is_empty());
    assert_eq!(state.assembler.live_word_count(), 1);
    assert_eq!(state.context_samples, 0);
}

#[test]
fn test_speech_admission_frame_zero_word_after_idle_cap_is_not_marked_committed() {
    let (mut engine, _tmp) = crate::test_support::rnnt_engine();
    attach(&mut engine, true);
    let mut state = engine.create_state(false);
    let mut triplet = engine.pool.checkout_blocking().unwrap();
    engine
        .process_chunk(&vec![0.0; 51200], &mut state, &mut triplet)
        .unwrap();
    engine
        .speech_admission
        .as_ref()
        .unwrap()
        .pool
        .checkout_blocking()
        .unwrap()
        .encoder = Box::new(MockSession::unconstrained(outputs(false)));
    triplet.encoder = Box::new(MockSession::unconstrained(vec![
        Tensor::new_checked(
            Shape::new(vec![1, 768, 2]),
            TensorData::F32(vec![0.0; 768 * 2]),
        ),
        Tensor::new_checked(Shape::new(vec![1]), TensorData::I64(vec![2])),
    ]));
    let joint = |values| {
        vec![Tensor::new_checked(
            Shape::new(vec![1, 1, 2]),
            TensorData::F32(values),
        )]
    };
    triplet.joiner = Some(Box::new(
        MockSession::unconstrained(joint(vec![0.0, 10.0]))
            .with_script(vec![joint(vec![10.0, 0.0]), joint(vec![0.0, 10.0])]),
    ));
    engine
        .process_chunk(&vec![0.0; 12800], &mut state, &mut triplet)
        .unwrap();
    assert!(!state.assembler.is_empty());
    assert_eq!(state.assembler.live_word_count(), 1);
    assert_eq!(state.context_samples, 0);
}

#[test]
fn test_speech_admission_nonempty_frame_zero_preserves_native_filter() {
    let (mut engine, _tmp) = crate::test_support::rnnt_engine();
    attach(&mut engine, true);
    engine
        .speech_admission
        .as_ref()
        .unwrap()
        .pool
        .checkout_blocking()
        .unwrap()
        .encoder = Box::new(MockSession::unconstrained(vec![]));
    let mut state = engine.create_state(false);
    state.window_start_samples = 27200;
    state.assembler.append(vec![word("не", 0.0, 0.1)]);
    state.assembler.commit_live();
    let mut triplet = engine.pool.checkout_blocking().unwrap();
    triplet.encoder = Box::new(MockSession::unconstrained(vec![
        Tensor::new_checked(
            Shape::new(vec![1, 768, 2]),
            TensorData::F32(vec![0.0; 768 * 2]),
        ),
        Tensor::new_checked(Shape::new(vec![1]), TensorData::I64(vec![2])),
    ]));
    let joint = |values| {
        vec![Tensor::new_checked(
            Shape::new(vec![1, 1, 2]),
            TensorData::F32(values),
        )]
    };
    triplet.joiner = Some(Box::new(
        MockSession::unconstrained(joint(vec![0.0, 10.0]))
            .with_script(vec![joint(vec![10.0, 0.0]), joint(vec![0.0, 10.0])]),
    ));
    engine
        .process_chunk(&vec![0.0; 12800], &mut state, &mut triplet)
        .unwrap();
    assert_eq!(state.assembler.live_word_count(), 0);
    assert_eq!(state.assembler.partial(0.0).text, "не");
}

struct TailAdmission {
    trailing: usize,
    calls: std::sync::Arc<std::sync::atomic::AtomicUsize>,
}
impl crate::runtime::session::RuntimeSession for TailAdmission {
    fn run(&self, _: &[Tensor]) -> Result<Vec<Tensor>, crate::runtime::RuntimeError> {
        self.calls
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let frames = 22;
        let mut logits = vec![-10.0; frames * 71];
        for row in logits.as_chunks_mut::<71>().0 {
            row[70] = 0.0;
        }
        logits[(frames - self.trailing - 1) * 71] = 1.0;
        Ok(vec![
            Tensor::new_checked(Shape::new(vec![1, frames, 71]), TensorData::F32(logits)),
            Tensor::new_checked(Shape::new(vec![1]), TensorData::I64(vec![frames as i64])),
        ])
    }
}
fn endpoint_rnnt(triplet: &mut SessionTriplet, emit: bool, frames: usize) {
    triplet.encoder = Box::new(MockSession::unconstrained(vec![
        Tensor::new_checked(
            Shape::new(vec![1, 768, frames]),
            TensorData::F32(vec![0.0; 768 * frames]),
        ),
        Tensor::new_checked(Shape::new(vec![1]), TensorData::I64(vec![frames as i64])),
    ]));
    let joint = |v| {
        vec![Tensor::new_checked(
            Shape::new(vec![1, 1, 2]),
            TensorData::F32(v),
        )]
    };
    let mut script = vec![joint(vec![0.0, 10.0])];
    if emit {
        script.push(joint(vec![10.0, 0.0]));
    }
    script.push(joint(vec![0.0, 10.0]));
    triplet.joiner = Some(Box::new(
        MockSession::unconstrained(joint(vec![0.0, 10.0])).with_script(script),
    ));
}
fn attach_tail(
    engine: &mut Engine,
    trailing: usize,
) -> std::sync::Arc<std::sync::atomic::AtomicUsize> {
    attach(engine, false);
    let calls = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
    engine
        .speech_admission
        .as_ref()
        .unwrap()
        .pool
        .checkout_blocking()
        .unwrap()
        .encoder = Box::new(TailAdmission {
        trailing,
        calls: calls.clone(),
    });
    calls
}

#[test]
fn test_speech_admission_endpoint_uses_existing_threshold_and_reuses_observation() {
    for trailing in [6, 15] {
        let (mut engine, _tmp) = crate::test_support::rnnt_engine();
        let calls = attach_tail(&mut engine, trailing);
        let mut triplet = engine.pool.checkout_blocking().unwrap();
        endpoint_rnnt(&mut triplet, true, 22);
        let mut state = engine.create_state(false);
        let segments = engine
            .process_chunk(&vec![0.0; 12800], &mut state, &mut triplet)
            .unwrap();
        assert_eq!(segments.len(), 1);
        assert_eq!(segments[0].is_final, trailing == 15);
        assert_eq!(calls.load(std::sync::atomic::Ordering::Relaxed), 1);
        assert_eq!(state.assembler.is_empty(), trailing == 15);
        if trailing == 6 {
            assert!(
                engine
                    .try_finish_stream(&mut state, &mut triplet)
                    .unwrap()
                    .unwrap()
                    .is_final
            );
            assert_eq!(calls.load(std::sync::atomic::Ordering::Relaxed), 1);
        }
    }
}

#[test]
fn test_speech_admission_confirms_trailing_blank_endpoint_without_new_tokens() {
    let (mut engine, _tmp) = crate::test_support::rnnt_engine();
    let calls = attach_tail(&mut engine, 6);
    let mut triplet = engine.pool.checkout_blocking().unwrap();
    endpoint_rnnt(&mut triplet, false, 22);
    let mut state = engine.create_state(false);
    state.assembler.append(vec![word("не", 0.0, 0.1)]);
    state.assembler.commit_live();
    let segments = engine
        .process_chunk(&vec![0.0; 12800], &mut state, &mut triplet)
        .unwrap();
    assert!(!segments[0].is_final);
    assert_eq!(segments[0].text, "не");
    assert_eq!(calls.load(std::sync::atomic::Ordering::Relaxed), 1);
}

#[test]
fn test_speech_admission_nonendpoint_and_explicit_stop_do_not_query_confirmation() {
    for stop in [false, true] {
        let (mut engine, _tmp) = crate::test_support::rnnt_engine();
        let calls = attach_tail(&mut engine, 6);
        let mut triplet = engine.pool.checkout_blocking().unwrap();
        endpoint_rnnt(&mut triplet, true, if stop { 22 } else { 3 });
        let mut state = engine.create_state(false);
        state.assembler.append(vec![word("не", 0.0, 0.1)]);
        state.assembler.commit_live();
        if stop {
            state.audio_buffer = vec![0.0; 12800];
            state.pending_samples = 12800;
            assert!(
                engine
                    .try_finish_stream(&mut state, &mut triplet)
                    .unwrap()
                    .unwrap()
                    .is_final
            );
        } else {
            assert!(
                !engine
                    .process_chunk(&vec![0.0; 12800], &mut state, &mut triplet)
                    .unwrap()[0]
                    .is_final
            );
        }
        assert_eq!(calls.load(std::sync::atomic::Ordering::Relaxed), 0);
    }
}

#[test]
fn test_speech_admission_confirmation_failure_preserves_prior_text_and_fails_state() {
    let (mut engine, _tmp) = crate::test_support::rnnt_engine();
    attach(&mut engine, true);
    engine
        .speech_admission
        .as_ref()
        .unwrap()
        .pool
        .checkout_blocking()
        .unwrap()
        .encoder = Box::new(MockSession::unconstrained(vec![]));
    let mut triplet = engine.pool.checkout_blocking().unwrap();
    endpoint_rnnt(&mut triplet, true, 22);
    let mut state = engine.create_state(false);
    state.assembler.append(vec![word("не", 0.0, 0.1)]);
    state.agreed_prefix = 7;
    assert!(
        engine
            .process_chunk(&vec![0.0; 12800], &mut state, &mut triplet)
            .is_err()
    );
    assert!(state.failed);
    assert_eq!(state.assembler.partial(0.0).text, "не");
    assert_eq!(state.agreed_prefix, 7);
}

#[test]
fn test_speech_admission_guard_off_keeps_native_endpoint() {
    let (engine, _tmp) = crate::test_support::rnnt_engine();
    let mut triplet = engine.pool.checkout_blocking().unwrap();
    endpoint_rnnt(&mut triplet, true, 22);
    let mut state = engine.create_state(false);
    assert!(
        engine
            .process_chunk(&vec![0.0; 12800], &mut state, &mut triplet)
            .unwrap()[0]
            .is_final
    );
}

#[test]
fn test_speech_admission_endpoint_cancellation_keeps_prior_text() {
    let (mut engine, _tmp) = crate::test_support::rnnt_engine();
    attach(&mut engine, true);
    let flag = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    engine
        .speech_admission
        .as_ref()
        .unwrap()
        .pool
        .checkout_blocking()
        .unwrap()
        .encoder = Box::new(CaptureAdmission {
        seen: Default::default(),
        abort: Some(flag.clone()),
    });
    let mut state = engine.create_state(false);
    state.abort = Some(flag);
    state.assembler.append(vec![word("не", 0.0, 0.1)]);
    state.agreed_prefix = 7;
    let mut triplet = engine.pool.checkout_blocking().unwrap();
    endpoint_rnnt(&mut triplet, true, 22);
    assert!(matches!(
        engine.process_chunk(&vec![0.0; 12800], &mut state, &mut triplet),
        Err(GigasttError::Cancelled)
    ));
    assert!(state.failed);
    assert_eq!(state.assembler.partial(0.0).text, "не");
    assert_eq!(state.agreed_prefix, 7);
}

#[test]
fn test_speech_admission_rejects_duplicate_attachment_atomically() {
    let (mut engine, _tmp) = crate::test_support::rnnt_engine();
    attach(&mut engine, true);
    let (mut other, _tmp2) = crate::test_support::rnnt_engine();
    attach(&mut other, false);
    let sidecar = *other.speech_admission.take().unwrap();
    assert!(engine.attach_ctc_speech_admission(sidecar).is_err());
}

#[test]
fn test_speech_admission_attachment_rejects_incompatible_sidecars() {
    for bad in 0..5 {
        let (mut engine, _tmp) = crate::test_support::rnnt_engine();
        let (mut holder, _tmp2) = crate::test_support::rnnt_engine();
        attach(&mut holder, true);
        let mut sidecar = *holder.speech_admission.take().unwrap();
        match bad {
            0 => sidecar.variant = ModelVariant::MlCtc,
            1 => sidecar.execution_provider = "cuda".into(),
            2 => sidecar.int8 = false,
            3 => engine.variant = ModelVariant::E2eRnnt,
            _ => engine.batch_pool = Some(crate::inference::SessionPool::new(vec![])),
        }
        assert!(matches!(
            engine.attach_ctc_speech_admission(sidecar),
            Err(GigasttError::ModelLoad { .. })
        ));
        assert!(engine.speech_admission.is_none());
    }
}
