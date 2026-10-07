use super::*;
#[cfg(feature = "diarization")]
use crate::inference::diarization::SPEAKER_EMBEDDING_DIM;
#[cfg(feature = "diarization")]
use polyvoice::Embedder;
use std::path::Path;

#[test]
#[ignore = "requires model"]
fn test_create_state_initial_fields() {
    let engine = Engine::load_with_pool_size(&crate::model::default_model_dir(), 1)
        .expect("engine should load");
    let state = engine.create_state(false);
    assert!(state.audio_buffer.is_empty());
    assert!(state.assembler.is_empty());
    assert_eq!(state.window_start_samples, 0);
    assert_eq!(state.context_samples, 0);
    assert_eq!(state.pending_samples, 0);
    assert!(state.resampler.is_none());
    // No VAD attached on a default engine → no endpointer.
    assert!(state.vad_endpointer.is_none());
    // Decoder state seeded to blank.
    assert_eq!(state.decoder.consecutive_blanks, 0);
}

#[test]
#[ignore = "requires model"]
fn test_create_state_diarization_flag_ignored_without_feature() {
    // Without the `diarization` feature the flag is silently ignored and a
    // perfectly usable state still comes back.
    let engine = Engine::load_with_pool_size(&crate::model::default_model_dir(), 1)
        .expect("engine should load");
    let state = engine.create_state(true);
    assert!(state.audio_buffer.is_empty());
}

// Missing speaker models must return an error without panicking.
#[cfg(feature = "diarization")]
#[test]
fn test_load_speaker_encoder_missing_model_errors() {
    let missing = Path::new("/nonexistent/gigastt-test/wespeaker_resnet34.onnx");
    let result = diarization::load_speaker_encoder(missing, 1);
    assert!(
        result.is_err(),
        "a missing WeSpeaker model must surface as Err, not panic or Ok"
    );
}

#[cfg(feature = "diarization")]
#[test]
#[ignore = "requires the WeSpeaker diarization model"]
fn test_speaker_encoder_accepts_waveform_audio() {
    let model_path = Path::new(&crate::model::default_model_dir()).join("wespeaker_resnet34.onnx");
    let encoder =
        diarization::load_speaker_encoder(&model_path, 1).expect("speaker encoder should load");
    let samples: Vec<f32> = (0..24_000)
        .map(|i| {
            let phase = std::f32::consts::TAU * 220.0 * i as f32 / 16_000.0;
            0.1 * phase.sin()
        })
        .collect();

    let embedding = encoder
        .embed(&samples)
        .expect("waveform must be converted to rank-3 fbank features");

    assert_eq!(embedding.len(), SPEAKER_EMBEDDING_DIM);
    assert!(embedding.iter().all(|value| value.is_finite()));
}

#[test]
#[ignore = "requires model"]
fn test_process_chunk_empty_input_returns_no_segments() {
    let engine = Engine::load_with_pool_size(&crate::model::default_model_dir(), 1)
        .expect("engine should load");
    let mut guard = engine.pool.checkout_blocking().expect("checkout");
    let mut state = engine.create_state(false);
    let segs = engine
        .process_chunk(&[], &mut state, &mut guard)
        .expect("empty chunk must not error");
    assert!(segs.is_empty(), "empty input yields no segments");
    assert_eq!(state.audio_buffer.len(), 0);
}

#[test]
#[ignore = "requires model"]
fn test_process_chunk_sub_stride_buffers_without_decoding() {
    // A chunk smaller than the decode stride is buffered and triggers no
    // decode (the stride gate returns early).
    let engine = Engine::load_with_pool_size(&crate::model::default_model_dir(), 1)
        .expect("engine should load");
    let mut guard = engine.pool.checkout_blocking().expect("checkout");
    let mut state = engine.create_state(false);
    let small = vec![0.0f32; 1600]; // 0.1s ≪ 0.8s stride
    let segs = engine
        .process_chunk(&small, &mut state, &mut guard)
        .expect("sub-stride chunk must not error");
    assert!(segs.is_empty(), "sub-stride chunk yields no segments yet");
    assert_eq!(state.audio_buffer.len(), 1600, "samples are buffered");
    assert_eq!(state.pending_samples, 1600, "pending counter advances");
}

#[test]
#[ignore = "requires model"]
fn test_process_chunk_silence_over_stride_decodes_no_words() {
    // Enough silence to cross the decode stride: the encoder runs but
    // produces no words (silence), so the partial path returns no segments
    // and the pending counter resets.
    let engine = Engine::load_with_pool_size(&crate::model::default_model_dir(), 1)
        .expect("engine should load");
    let mut guard = engine.pool.checkout_blocking().expect("checkout");
    let mut state = engine.create_state(false);
    let chunk = vec![0.0f32; 16000]; // 1s of silence, > 0.8s stride
    let segs = engine
        .process_chunk(&chunk, &mut state, &mut guard)
        .expect("decode of silence must not error");
    // Silence → no words → empty assembler → no partial segment emitted.
    assert!(segs.is_empty(), "silence decodes to no words");
    assert_eq!(
        state.pending_samples, 0,
        "decode resets the pending counter"
    );
}

#[test]
#[ignore = "requires model"]
fn test_flush_state_empty_returns_none() {
    let engine = Engine::load_with_pool_size(&crate::model::default_model_dir(), 1)
        .expect("engine should load");
    let mut state = engine.create_state(false);
    assert!(
        engine.flush_state(&mut state).is_none(),
        "an empty assembler flushes to None"
    );
}

#[test]
#[ignore = "requires model"]
fn test_flush_state_nonempty_returns_final_segment() {
    let engine = Engine::load_with_pool_size(&crate::model::default_model_dir(), 1)
        .expect("engine should load");
    let mut state = engine.create_state(false);
    state.assembler.set_words(vec![word("hello", 0.0, 0.4)]);
    let seg = engine
        .flush_state(&mut state)
        .expect("non-empty assembler flushes to a Final segment");
    assert!(seg.is_final);
    assert_eq!(seg.text, "hello");
    assert!(
        engine.flush_state(&mut state).is_none(),
        "finalize resets the assembler"
    );
}

#[test]
#[ignore = "requires model"]
fn test_finish_stream_no_pending_flushes_assembler() {
    // No buffered audio and no pending samples: finish_stream skips the
    // forced decode and just flushes whatever the assembler holds.
    let engine = Engine::load_with_pool_size(&crate::model::default_model_dir(), 1)
        .expect("engine should load");
    let mut guard = engine.pool.checkout_blocking().expect("checkout");
    let mut state = engine.create_state(false);
    state.assembler.set_words(vec![word("trailing", 0.0, 0.4)]);
    let seg = engine
        .finish_stream(&mut state, &mut guard)
        .expect("finish_stream flushes the assembler");
    assert_eq!(seg.text, "trailing");
    assert!(seg.is_final);
}

#[test]
#[ignore = "requires model"]
fn test_finish_stream_empty_state_returns_none() {
    let engine = Engine::load_with_pool_size(&crate::model::default_model_dir(), 1)
        .expect("engine should load");
    let mut guard = engine.pool.checkout_blocking().expect("checkout");
    let mut state = engine.create_state(false);
    assert!(
        engine.finish_stream(&mut state, &mut guard).is_none(),
        "an idle stream finishes to None"
    );
}
