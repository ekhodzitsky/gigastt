use super::*;
use crate::inference::TranscriptSnapshot;
use crate::runtime::{
    RuntimeError,
    session::RuntimeSession,
    tensor::{Shape, Tensor, TensorData},
};
use std::sync::{
    Arc,
    atomic::{AtomicBool, AtomicUsize, Ordering},
};

/// Trigger cancellation from inside a runtime call, after real tokens have
/// already reached the engine. No sleeps or scheduler timing assumptions.
struct AbortingJoiner {
    abort: Arc<AtomicBool>,
    calls: AtomicUsize,
    abort_after: usize,
}

impl RuntimeSession for AbortingJoiner {
    fn run(&self, _: &[Tensor]) -> Result<Vec<Tensor>, RuntimeError> {
        if self.calls.fetch_add(1, Ordering::Relaxed) + 1 == self.abort_after {
            self.abort.store(true, Ordering::Relaxed);
        }
        Ok(vec![Tensor::new_checked(
            Shape::new(vec![1, 1, 2]),
            TensorData::F32(vec![10.0, 0.0]),
        )])
    }
}

fn install_abort(triplet: &mut SessionTriplet, flag: &Arc<AtomicBool>, after: usize) {
    triplet.joiner = Some(Box::new(AbortingJoiner {
        abort: flag.clone(),
        calls: AtomicUsize::new(0),
        abort_after: after,
    }));
}

#[test]
fn test_cancelled_request_keeps_words_from_interrupted_decode() {
    for (samples, after) in [(8000, 3), (800_000, 14)] {
        let (engine, _tmp) = crate::test_support::rnnt_engine();
        let flag = Arc::new(AtomicBool::new(false));
        let partial = Arc::new(TranscriptSnapshot::default());
        let mut guard = engine.pool.checkout_blocking().unwrap();
        install_abort(&mut guard, &flag, after);
        let audio = vec![0.0; samples];
        let request = TranscribeRequest::new(TranscribeSource::Samples(&audio))
            .with_abort(Some(flag))
            .with_partial(Some(partial.clone()));
        assert!(matches!(
            engine.transcribe_request(request, &mut guard),
            Err(GigasttError::Cancelled)
        ));
        let snapshot = partial.get().expect("readable partial after cancellation");
        assert!(!snapshot.text.is_empty());
        assert!(snapshot.words.len() >= 3);
        if samples > 8000 {
            assert!(
                snapshot.words.first().unwrap().start < 1.0,
                "previous window was lost"
            );
        }
        assert!(!snapshot.is_final);
        drop(guard);
        assert!(engine.pool.try_checkout().unwrap().is_some());
    }
}

#[test]
fn test_cancelled_stream_is_terminal_and_partial_stays_readable() {
    let (engine, _tmp) = crate::test_support::rnnt_engine();
    let flag = Arc::new(AtomicBool::new(false));
    let partial = Arc::new(TranscriptSnapshot::default());
    let mut guard = engine.pool.checkout_blocking().unwrap();
    install_abort(&mut guard, &flag, 3);
    let mut state = engine.create_state(false);
    state.abort = Some(flag.clone());
    state.partial = Some(partial.clone());
    assert!(matches!(
        engine.process_chunk(&[0.0; 16000], &mut state, &mut guard),
        Err(GigasttError::Cancelled)
    ));
    assert!(state.is_failed());
    let text = partial.get().unwrap().text;
    assert!(!text.is_empty());
    assert_eq!(state.assembler.partial(0.0).text, text);
    flag.store(false, Ordering::Relaxed);
    assert!(matches!(
        engine.process_chunk(&[0.0; 16000], &mut state, &mut guard),
        Err(GigasttError::Cancelled)
    ));
    assert_eq!(
        engine.finish_stream(&mut state, &mut guard).unwrap().text,
        text
    );
    assert_eq!(engine.flush_state(&mut state).unwrap().text, text);
    assert_eq!(state.assembler.partial(0.0).text, text);
}

#[test]
fn test_cancelled_redecode_keeps_previous_live_and_committed_words() {
    let (engine, _tmp) = crate::test_support::rnnt_engine();
    let flag = Arc::new(AtomicBool::new(false));
    let mut guard = engine.pool.checkout_blocking().unwrap();
    install_abort(&mut guard, &flag, 1);
    let mut state = engine.create_state(false);
    state.abort = Some(flag);
    state.assembler.append(vec![word("already", 0.0, 0.1)]);
    state.assembler.commit_live();
    state
        .assembler
        .append(vec![word("readable", 0.1, 0.2), word("tail", 0.2, 0.3)]);
    assert!(matches!(
        engine.process_chunk(&[0.0; 16000], &mut state, &mut guard),
        Err(GigasttError::Cancelled)
    ));
    assert_eq!(state.assembler.partial(0.0).text, "already readable tail");
}

#[test]
fn test_cancelled_second_channel_keeps_first_channel_text() {
    let (engine, _tmp) = crate::test_support::rnnt_engine();
    let flag = Arc::new(AtomicBool::new(false));
    let partial = Arc::new(TranscriptSnapshot::default());
    let mut guard = engine.pool.checkout_blocking().unwrap();
    install_abort(&mut guard, &flag, 14);
    let channels = vec![vec![0.0; 16000]; 2];
    let request = TranscribeRequest::new(TranscribeSource::Channels(&channels))
        .with_abort(Some(flag))
        .with_partial(Some(partial.clone()));
    assert!(matches!(
        engine.transcribe_request(request, &mut guard),
        Err(GigasttError::Cancelled)
    ));
    let words = partial.get().unwrap().words;
    assert!(words.iter().any(|word| word.speaker == Some(0)));
    assert!(words.iter().any(|word| word.speaker == Some(1)));
}
