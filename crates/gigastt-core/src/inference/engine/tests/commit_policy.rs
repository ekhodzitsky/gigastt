use super::*;
use crate::inference::CommitPolicy;

#[test]
fn test_stream_snapshot_matches_emitted_commits_and_resets_after_final() {
    use crate::inference::TranscriptSnapshot;
    use std::sync::Arc;
    let (engine, _tmp) = crate::test_support::rnnt_engine();
    let engine = engine.with_stream_max_window_secs(2.4);
    let mut guard = engine.pool.checkout_blocking().unwrap();
    guard.joiner = Some(Box::new(crate::runtime::mock::MockSession::unconstrained(
        vec![crate::runtime::tensor::Tensor::new_checked(
            crate::runtime::tensor::Shape::new(vec![1, 1, 2]),
            crate::runtime::tensor::TensorData::F32(vec![10.0, 0.0]),
        )],
    )));
    let mut state = engine.create_state(false);
    state.endpoint_mode = EndpointMode::Manual;
    state.commit_policy = CommitPolicy::StablePrefix;
    let snapshot = Arc::new(TranscriptSnapshot::default());
    state.partial = Some(snapshot.clone());
    let mut saw_committed = false;
    for _ in 0..5 {
        for segment in engine
            .process_chunk(&[0.0; 16000], &mut state, &mut guard)
            .unwrap()
        {
            let saved = snapshot.get().unwrap();
            assert_eq!(saved.committed, segment.committed);
            assert_eq!(saved.tentative, segment.tentative);
            saw_committed |= !segment.committed.is_empty();
        }
    }
    assert!(saw_committed, "test must exercise a window-cap commit");
    engine.flush_state(&mut state).unwrap();
    assert!(
        snapshot.get().is_none(),
        "a new utterance must not replay the old snapshot"
    );
}

#[test]
fn test_stop_flush_keeps_committed_text() {
    let (engine, _tmp) = crate::test_support::rnnt_engine();
    let mut state = engine.create_state(false);
    state.assembler.append(vec![word("привет", 0.0, 0.4)]);
    state.assembler.commit_live();
    state.assembler.set_words(vec![word("мир", 0.4, 0.8)]);
    let seg = engine.flush_state(&mut state).unwrap();
    assert!(!seg.truncated, "Stop is an endpoint, not a cap");
    assert!(seg.is_final);
    assert_eq!(seg.text, "привет мир");
    assert_eq!(seg.committed, seg.text);
    assert!(seg.tentative.is_empty());
}

#[test]
fn test_session_cap_and_shutdown_keep_committed_text() {
    let (engine, _tmp) = crate::test_support::rnnt_engine();
    let mut state = engine.create_state(false);
    state.commit_policy = CommitPolicy::OnFinalize;
    state.assembler.append(vec![word("привет", 0.0, 0.4)]);
    state.assembler.commit_live();
    state.assembler.set_words(vec![word("мир", 0.4, 0.8)]);
    // What an in-flight cap sees: the published partial, whose committed
    // field is empty under on_finalize even though the words are there.
    let partial = engine.stream_partial(&state, 1.0);
    assert!(partial.committed.is_empty());
    let cut = partial.into_truncated_final();
    assert!(cut.truncated);
    assert!(cut.is_final);
    assert!(cut.speech_final);
    assert_eq!(cut.text, "привет мир");
    assert_eq!(cut.committed, cut.text);
    assert!(cut.tentative.is_empty());

    let flushed = engine.flush_truncated(&mut state);
    assert!(flushed.truncated);
    assert_eq!(flushed.text, "привет мир");
    assert_eq!(flushed.committed, flushed.text);
    assert!(flushed.tentative.is_empty());
}

#[test]
fn test_truncated_flush_of_empty_state_is_explicit() {
    let (engine, _tmp) = crate::test_support::rnnt_engine();
    let mut state = engine.create_state(false);
    let seg = engine.flush_truncated(&mut state);
    assert!(seg.truncated);
    assert!(seg.is_final);
    assert!(seg.text.is_empty());
    assert!(seg.committed.is_empty());
}

#[test]
fn test_commit_policy_parts_finalize_and_reset_per_utterance() {
    let (engine, _tmp) = crate::test_support::rnnt_engine();
    for policy in [
        CommitPolicy::Auto,
        CommitPolicy::OnFinalize,
        CommitPolicy::StablePrefix,
    ] {
        let mut state = engine.create_state(false);
        state.commit_policy = policy;
        state.assembler.append(vec![word("привет", 0.0, 0.5)]);
        state.assembler.commit_live();
        state.assembler.set_words(vec![word("мир", 0.5, 1.0)]);
        let partial = engine.stream_partial(&state, 1.0);
        assert_eq!(partial.text, "привет мир");
        assert_eq!(partial.text, partial.committed.clone() + &partial.tentative);
        assert_eq!(
            partial.committed,
            if policy == CommitPolicy::OnFinalize {
                ""
            } else {
                "привет"
            }
        );
        let final_ = engine.flush_state(&mut state).unwrap();
        assert!(final_.committed.starts_with(&partial.committed));
        assert_eq!(final_.committed, final_.text);
        assert!(final_.tentative.is_empty());
        state.assembler.append(vec![word("снова", 1.0, 1.5)]);
        let next = engine.stream_partial(&state, 2.0);
        assert!(next.committed.is_empty());
        assert_eq!(next.tentative, "снова");
    }
}

#[test]
fn test_auto_defers_commit_for_itn_and_preserves_legacy_final_text() {
    let (engine, _tmp) = crate::test_support::rnnt_engine();
    let engine = engine.with_itn(true);
    let mut state = engine.create_state(false);
    state.assembler.append(vec![word("двадцать", 0.0, 0.5)]);
    state.assembler.commit_live();
    state.assembler.set_words(vec![word("один", 0.5, 1.0)]);
    let partial = engine.stream_partial(&state, 1.0);
    assert!(partial.committed.is_empty());
    assert_eq!(partial.tentative, "двадцать один");
    // Per-session overrides govern AUTO, just like final post-processing.
    state.itn = Some(false);
    assert_eq!(engine.stream_partial(&state, 1.0).committed, "двадцать");
    state.itn = None;
    let final_ = engine.flush_state(&mut state).unwrap();
    assert_eq!(final_.text, "21");
    assert_eq!(final_.committed, "21");
    assert!(final_.tentative.is_empty());
}

#[test]
fn test_stable_prefix_final_only_processes_the_uncommitted_tail() {
    let (engine, _tmp) = crate::test_support::rnnt_engine();
    let engine = engine.with_itn(true);
    let mut state = engine.create_state(false);
    state.commit_policy = CommitPolicy::StablePrefix;
    state
        .assembler
        .append(vec![word("двадцать", 0.0, 0.5), word("один", 0.5, 1.0)]);
    state.assembler.commit_live();
    state.assembler.set_words(vec![word("два", 1.0, 1.5)]);
    let partial = engine.stream_partial(&state, 1.0);
    assert_eq!(partial.committed, "двадцать один");
    let final_ = engine.flush_state(&mut state).unwrap();
    assert_eq!(final_.committed, "двадцать один 2");
    assert!(final_.committed.starts_with(&partial.committed));
    assert_eq!(final_.text, final_.committed);
    assert!(final_.tentative.is_empty());
}

#[test]
fn test_cancelled_stream_snapshot_honors_commit_policy() {
    use crate::inference::TranscriptSnapshot;
    use std::sync::{Arc, atomic::AtomicBool};
    let (engine, _tmp) = crate::test_support::rnnt_engine();
    let mut guard = engine.pool.checkout_blocking().unwrap();
    for policy in [CommitPolicy::OnFinalize, CommitPolicy::StablePrefix] {
        let mut state = engine.create_state(false);
        state.commit_policy = policy;
        state.abort = Some(Arc::new(AtomicBool::new(true)));
        let snapshot = Arc::new(TranscriptSnapshot::default());
        state.partial = Some(snapshot.clone());
        state.assembler.append(vec![word("привет", 0.0, 0.5)]);
        state.assembler.commit_live();
        state.assembler.set_words(vec![word("мир", 0.5, 1.0)]);
        assert!(
            engine
                .process_chunk(&[0.0; 320], &mut state, &mut guard)
                .is_err()
        );
        for partial in [
            snapshot.get().unwrap(),
            engine.flush_state(&mut state).unwrap(),
        ] {
            assert!(!partial.is_final);
            assert_eq!(partial.text, partial.committed.clone() + &partial.tentative);
            assert_eq!(
                partial.committed,
                if policy == CommitPolicy::OnFinalize {
                    ""
                } else {
                    "привет"
                }
            );
        }
    }
}
