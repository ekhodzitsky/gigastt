use super::*;
use crate::runtime::RuntimeError;

struct Mock<F>(F);
impl<F> RuntimeSession for Mock<F>
where
    F: Fn(&[Tensor]) -> Result<Vec<Tensor>, RuntimeError> + Send + Sync + 'static,
{
    fn run(&self, inputs: &[Tensor]) -> Result<Vec<Tensor>, RuntimeError> {
        (self.0)(inputs)
    }
}
fn i64_value(input: &Tensor) -> usize {
    input.view().data().as_i64().unwrap()[0] as usize
}
fn floats(input: &Tensor) -> &[f32] {
    input.view().data().as_f32().unwrap()
}
fn add_caches(outputs: &mut Vec<Tensor>, inputs: &[Tensor]) {
    outputs.extend(inputs.iter().map(|t| {
        f32_tensor(
            t.shape().dims(),
            floats(t).iter().map(|x| x + 1.0).collect(),
        )
        .unwrap()
    }));
}
fn varint(mut n: u64, out: &mut Vec<u8>) {
    while n >= 128 {
        out.push((n & 127) as u8 | 128);
        n >>= 7;
    }
    out.push(n as u8);
}
fn field(n: u64, value: &[u8], out: &mut Vec<u8>) {
    varint(n * 8 + 2, out);
    varint(value.len() as u64, out);
    out.extend_from_slice(value);
}
fn integer(n: u64, value: u64, out: &mut Vec<u8>) {
    varint(n * 8, out);
    varint(value, out);
}
fn mock_tokenizer(style: TranscriptStyle) -> V4Tokenizer {
    // Small synthetic protobuf fixture with the fixed public geometry. No
    // actual trained weights or tokenizer assets required by unit tests.
    let mut proto = Vec::new();
    for id in 0..1024 {
        let piece = match id {
            0 => "<unk>".into(),
            1 => "▁alpha".into(),
            2 => "▁beta".into(),
            _ => format!("z{id}"),
        };
        let mut p = Vec::new();
        field(1, piece.as_bytes(), &mut p);
        integer(3, if id == 0 { 2 } else { 1 }, &mut p);
        field(1, &p, &mut proto);
    }
    let mut trainer = Vec::new();
    integer(3, 2, &mut trainer);
    integer(41, u64::MAX, &mut trainer);
    integer(42, u64::MAX, &mut trainer);
    field(2, &trainer, &mut proto);
    V4Tokenizer::from_bytes(style, &proto).unwrap()
}
#[derive(Default)]
struct Faults {
    stage: AtomicUsize,
    cancel: AtomicBool,
    mode: AtomicUsize,
    calls: [AtomicUsize; 4],
}
impl Faults {
    fn at(&self, stage: usize) -> Result<(), RuntimeError> {
        self.calls[stage - 1].fetch_add(1, Ordering::Relaxed);
        if self.stage.load(Ordering::Relaxed) == stage {
            if self.mode.load(Ordering::Relaxed) == 1 {
                self.cancel.store(true, Ordering::Relaxed)
            } else {
                return Err(RuntimeError::InferenceFailed(
                    "injected native failure".into(),
                ));
            }
        }
        Ok(())
    }
    fn set(&self, stage: usize, mode: usize) {
        self.stage.store(stage, Ordering::Relaxed);
        self.mode.store(mode, Ordering::Relaxed);
        self.cancel.store(false, Ordering::Relaxed);
    }
}
fn mock_head(style: TranscriptStyle, faults: Arc<Faults>) -> Head {
    let predict_faults = faults.clone();
    Head {
        tokenizer: mock_tokenizer(style),
        predictor: Box::new(Mock(move |inputs: &[Tensor]| {
            predict_faults.at(3)?;
            let label = i64_value(&inputs[0]) as f32;
            Ok(vec![
                f32_tensor(&[1, 1, 320], vec![label; 320]).unwrap(),
                f32_tensor(&[1, 1, 320], vec![1.; 320]).unwrap(),
                f32_tensor(&[1, 1, 320], vec![2.; 320]).unwrap(),
            ])
        })),
        joiner: Box::new(Mock(move |inputs: &[Tensor]| {
            faults.at(4)?;
            let label = floats(&inputs[1])[0] as usize;
            let next = match label {
                1024 => 1,
                1 => 2,
                _ => 1024,
            };
            let mut output = vec![-10.; 8 * 1025];
            for row in output.as_chunks_mut::<1025>().0 {
                row[next] = -0.1;
            }
            Ok(vec![f32_tensor(&[1, 8, 1, 1025], output).unwrap()])
        })),
    }
}
fn mock_engine(faults: Arc<Faults>) -> Arc<V4StreamingEngine> {
    let front_faults = faults.clone();
    let step_faults = faults.clone();
    Arc::new(V4StreamingEngine {
        mel: Box::new(|pcm| {
            let frames = complete_mel_frames(pcm.len());
            Ok((0..MELS)
                .flat_map(|m| (0..frames).map(move |f| pcm[f * HOP] + m as f32))
                .collect())
        }),
        front: Box::new(Mock(move |inputs: &[Tensor]| {
            front_faults.at(1)?;
            assert_eq!(inputs.len(), 2 + V4_STREAMING_FRONT_CACHE_SHAPES.len());
            let physical = inputs[0].shape().dims()[2] / 4;
            let valid = i64_value(&inputs[1]).div_ceil(4);
            let mut outputs = vec![
                f32_tensor(&[1, physical, CHANNELS], vec![3.; physical * CHANNELS]).unwrap(),
                scalar(valid).unwrap(),
            ];
            add_caches(&mut outputs, &inputs[2..]);
            Ok(outputs)
        })),
        step: Box::new(Mock(move |inputs: &[Tensor]| {
            step_faults.at(2)?;
            assert_eq!(inputs.len(), 5 + V4_STREAMING_LAYER_CACHE_SHAPES.len());
            let physical = inputs[0].shape().dims()[1];
            let valid = i64_value(&inputs[1]);
            let position = i64_value(&inputs[3]);
            let cache_valid = i64_value(&inputs[4]);
            let mut outputs = vec![
                f32_tensor(&[1, CHANNELS, physical], vec![5.; physical * CHANNELS]).unwrap(),
                scalar(valid).unwrap(),
                scalar(position + valid).unwrap(),
                scalar((cache_valid + valid).min(32)).unwrap(),
            ];
            add_caches(&mut outputs, &inputs[5..]);
            Ok(outputs)
        })),
        raw: mock_head(TranscriptStyle::Raw, faults.clone()),
        e2e: mock_head(TranscriptStyle::E2e, faults),
        request: Mutex::new(()),
        active: AtomicUsize::new(0),
        pack_identity: "test".into(),
    })
}

#[test]
fn test_v4_streaming_pcm_mel_partition_carry_and_short_final() {
    // Exact expected windows are independent of how the caller splits input.
    let source = (0..16037).map(|i| i as f32).collect::<Vec<_>>();
    for parts in [vec![319, 1, 160, 481, 2048, 13028], vec![8000, 8037]] {
        let mut wave = Vec::new();
        let mut pending = Vec::new();
        let mut seen = Vec::new();
        let mut start = 0;
        for n in parts {
            wave.extend_from_slice(&source[start..start + n]);
            start += n;
            let count = complete_mel_frames(wave.len());
            let values: Vec<f32> = (0..MELS)
                .flat_map(|m| {
                    let wave = &wave;
                    (0..count).map(move |f| wave[f * HOP] + m as f32)
                })
                .collect();
            wave.drain(..count * HOP);
            let ready = take_mel(&mut pending, values, 32, false);
            if ready.valid > 0 {
                seen.extend_from_slice(&ready.values[..ready.valid]);
            }
            assert!(wave.len() < 320);
            assert!(pending.len() / MELS < 32);
        }
        let ready = take_mel(&mut pending, vec![], 32, true);
        seen.extend_from_slice(&ready.values[..ready.valid]);
        assert_eq!(ready.valid, 3);
        assert_eq!(ready.physical, 32);
        assert!(ready.values[3..32].iter().all(|x| *x == 0.0));
        assert_eq!(seen, (0..99).map(|f| (f * HOP) as f32).collect::<Vec<_>>());
        assert_eq!(wave, source[99 * HOP..]);
        assert!(pending.is_empty());
    }
    assert_eq!(complete_mel_frames(319), 0);
}
#[test]
fn test_v4_streaming_publication_and_independent_context_reset() {
    let faults = Arc::new(Faults::default());
    let engine = mock_engine(faults);
    let mut a = engine
        .start_session(V4StreamContext::Ms160, V4StreamingStyles::Raw)
        .unwrap();
    let mut b = engine
        .start_session(V4StreamContext::Ms640, V4StreamingStyles::E2e)
        .unwrap();
    assert!(
        engine
            .start_session(V4StreamContext::Ms320, V4StreamingStyles::Both)
            .is_err()
    );
    assert_eq!(
        a.push_pcm(&[0.; 319], false, None).unwrap().encoded_frames,
        0
    );
    let out = a.push_pcm(&[0.; 2561], false, None).unwrap();
    assert_eq!(out.encoded_frames, 4);
    assert!(out.e2e.is_none());
    let delta = out.raw.unwrap();
    assert_eq!(delta.text, "alpha");
    assert_eq!(delta.tokens.len(), 2);
    assert_eq!(delta.tokens[0].frame, 0);
    assert_eq!(delta.words[0].word, "alpha");
    assert_eq!(delta.words[0].end, 0.04);
    assert_eq!(b.state.as_ref().unwrap().position, 0);
    assert_eq!(floats(&a.state.as_ref().unwrap().layers[0])[0], 1.);
    assert_eq!(floats(&b.state.as_ref().unwrap().layers[0])[0], 0.);
    let out = b.push_pcm(&[0.; 320], true, None).unwrap();
    assert!(out.raw.is_none());
    assert_eq!(out.encoded_frames, 1);
    let delta = out.e2e.unwrap();
    assert_eq!(delta.text, "alpha beta");
    assert_eq!(delta.words[1].end, 0.02);
    let out = a.flush(None).unwrap();
    assert_eq!(out.raw.unwrap().text, " beta");
    assert!(out.finished);
    let out = a.push_pcm(&[f32::NAN], false, None).unwrap();
    assert_eq!(out.samples, 2880);
    assert!(out.raw.unwrap().tokens.is_empty());
    a.reset(V4StreamContext::Ms320).unwrap();
    assert_eq!(a.state.as_ref().unwrap().samples, 0);
    assert_eq!(floats(&a.state.as_ref().unwrap().layers[0])[0], 0.);
    drop(b);
    assert_eq!(engine.active.load(Ordering::Acquire), 1);
    assert!(
        engine
            .start_session(V4StreamContext::Ms160, V4StreamingStyles::Both)
            .is_ok()
    );
}
#[test]
fn test_v4_streaming_preflight_rejections_leave_state_and_release_slot() {
    let faults = Arc::new(Faults::default());
    let engine = mock_engine(faults.clone());
    let mut s = engine
        .start_session(V4StreamContext::Ms160, V4StreamingStyles::Both)
        .unwrap();
    let cancel = AtomicBool::new(true);
    assert!(matches!(
        s.push_pcm(&[0.; 320], true, Some(&cancel)),
        Err(GigasttError::Cancelled)
    ));
    assert!(s.push_pcm(&[f32::NAN], false, None).is_err());
    assert!(s.push_pcm(&[0.; 16001], false, None).is_err());
    assert_eq!(s.state.as_ref().unwrap().samples, 0);
    s.state.as_mut().unwrap().samples = V4_STREAM_MAX_SESSION_SAMPLES;
    assert!(matches!(
        s.push_pcm(&[0.], false, None),
        Err(GigasttError::AudioTooLong { .. })
    ));
    assert_eq!(
        s.state.as_ref().unwrap().samples,
        V4_STREAM_MAX_SESSION_SAMPLES
    );
    assert!(faults.calls.iter().all(|n| n.load(Ordering::Relaxed) == 0));
    assert!(engine.request.try_lock().is_some());
}
#[test]
fn test_v4_streaming_runtime_errors_and_post_native_cancellation_require_reset() {
    for mode in [0, 1] {
        for stage in 1..=4 {
            let faults = Arc::new(Faults::default());
            let engine = mock_engine(faults.clone());
            let mut s = engine
                .start_session(V4StreamContext::Ms160, V4StreamingStyles::Both)
                .unwrap();
            let mut other = engine
                .start_session(V4StreamContext::Ms320, V4StreamingStyles::Raw)
                .unwrap();
            // On this preflight packet we have retained audio but no native call.
            s.push_pcm(&[0.; 319], false, None).unwrap();
            faults.set(stage, mode);
            let err = s
                .push_pcm(&[0.; 1], true, Some(&faults.cancel))
                .unwrap_err();
            if mode == 1 {
                assert!(matches!(err, GigasttError::Cancelled));
            } else {
                assert!(matches!(err, GigasttError::Inference { .. }));
            }
            assert!(s.state.is_none());
            assert!(s.flush(None).is_err());
            assert!(
                faults
                    .calls
                    .iter()
                    .skip(stage)
                    .all(|x| x.load(Ordering::Relaxed) == 0)
            );
            assert!(engine.request.try_lock().is_some());
            faults.set(0, 0);
            assert!(other.push_pcm(&[0.; 320], true, None).is_ok());
            s.reset(V4StreamContext::Ms160).unwrap();
            assert_eq!(s.state.as_ref().unwrap().samples, 0);
            assert!(s.push_pcm(&[0.; 320], true, None).is_ok());
            drop(s);
            drop(other);
            assert_eq!(engine.active.load(Ordering::Acquire), 0);
        }
    }
}
#[test]
fn test_v4_streaming_malformed_cache_and_position_outputs_fail_closed() {
    let shapes: &[&[usize]] = &[&[1, 2]];
    let data = || f32_tensor(&[1, 1, 1], vec![0.]).unwrap();
    let scalar_one = || scalar(1).unwrap();
    let good = || f32_tensor(&[1, 2], vec![0.; 2]).unwrap();
    assert!(
        acoustic_outputs(
            vec![data(), scalar_one(), scalar_one(), scalar_one(), good()],
            &[1, 1, 1],
            1,
            shapes,
            Some((1, 1))
        )
        .is_ok()
    );
    assert!(
        acoustic_outputs(
            vec![
                data(),
                scalar_one(),
                scalar(2).unwrap(),
                scalar_one(),
                good()
            ],
            &[1, 1, 1],
            1,
            shapes,
            Some((1, 1))
        )
        .is_err()
    );
    assert!(
        acoustic_outputs(
            vec![data(), scalar_one(), f32_tensor(&[2], vec![0.; 2]).unwrap()],
            &[1, 1, 1],
            1,
            shapes,
            None
        )
        .is_err()
    );
    assert!(
        acoustic_outputs(
            vec![
                data(),
                scalar_one(),
                f32_tensor(&[1, 2], vec![f32::NAN; 2]).unwrap()
            ],
            &[1, 1, 1],
            1,
            shapes,
            None
        )
        .is_err()
    );
}

#[test]
fn test_v4_streaming_error_after_prior_delta_preserves_only_returned_output() {
    let faults = Arc::new(Faults::default());
    let engine = mock_engine(faults.clone());
    let mut a = engine
        .start_session(V4StreamContext::Ms160, V4StreamingStyles::Both)
        .unwrap();
    let prior = a.push_pcm(&[0.; 2880], false, None).unwrap();
    assert_eq!(prior.raw.as_ref().unwrap().text, "alpha");
    faults.set(2, 0);
    assert!(matches!(
        a.push_pcm(&[0.; 2560], false, None),
        Err(GigasttError::Inference { .. })
    ));
    assert!(a.state.is_none());
    assert_eq!(prior.e2e.as_ref().unwrap().text, "alpha");
    assert!(engine.request.try_lock().is_some());
}

#[test]
fn test_v4_streaming_wait_cancel_leaves_session_untouched_and_other_owner_runs() {
    use std::{sync::Barrier, time::Duration};
    let faults = Arc::new(Faults::default());
    let engine = mock_engine(faults.clone());
    let mut a = engine
        .start_session(V4StreamContext::Ms160, V4StreamingStyles::Both)
        .unwrap();
    let mut b = engine
        .start_session(V4StreamContext::Ms320, V4StreamingStyles::Raw)
        .unwrap();
    let reserved = engine.request.lock();
    let barrier = Barrier::new(2);
    let cancel = AtomicBool::new(false);
    std::thread::scope(|scope| {
        let t = scope.spawn(|| {
            barrier.wait();
            a.push_pcm(&[0.; 320], true, Some(&cancel))
        });
        barrier.wait();
        // Keep the lease held throughout; cancellation, not acquisition,
        // must unblock the contender. No timing threshold is asserted.
        std::thread::sleep(Duration::from_millis(15));
        cancel.store(true, Ordering::Relaxed);
        assert!(matches!(t.join().unwrap(), Err(GigasttError::Cancelled)));
    });
    assert_eq!(a.state.as_ref().unwrap().samples, 0);
    assert!(faults.calls.iter().all(|n| n.load(Ordering::Relaxed) == 0));
    drop(reserved);
    assert!(b.push_pcm(&[0.; 320], true, None).is_ok());
    assert!(engine.request.try_lock().is_some());
}
