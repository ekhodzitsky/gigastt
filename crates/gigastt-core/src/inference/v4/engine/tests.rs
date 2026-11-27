use super::*;
use crate::runtime::{
    RuntimeError, RuntimeSession,
    tensor::{Shape, Tensor, TensorData},
};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

struct MockSession<F>(F);
impl<F> RuntimeSession for MockSession<F>
where
    F: Fn(&[Tensor]) -> Result<Vec<Tensor>, RuntimeError> + Send + Sync + 'static,
{
    fn run(&self, inputs: &[Tensor]) -> Result<Vec<Tensor>, RuntimeError> {
        (self.0)(inputs)
    }
}
fn f32s(shape: &[usize], data: Vec<f32>) -> Tensor {
    Tensor::new(Shape::new(shape.to_vec()), TensorData::F32(data)).unwrap()
}
fn i64s(shape: &[usize], data: Vec<i64>) -> Tensor {
    Tensor::new(Shape::new(shape.to_vec()), TensorData::I64(data)).unwrap()
}
fn predictor_outputs(n: f32) -> Vec<Tensor> {
    vec![
        f32s(&[1, 1, 320], vec![n; 320]),
        f32s(&[1, 1, 320], vec![n + 1.; 320]),
        f32s(&[1, 1, 320], vec![n + 2.; 320]),
    ]
}
fn logits(labels: [usize; 8]) -> Tensor {
    let mut p = vec![-10.0; 8 * 1025];
    for (row, &label) in labels.iter().enumerate() {
        p[row * 1025 + label] = -0.1;
    }
    f32s(&[1, 8, 1, 1025], p)
}

#[test]
fn test_v4_padded_physical_clamp_and_valid_only_scan() {
    let grid = V4FrameGrid::new(960, 641).unwrap();
    let data: Vec<f32> = (0..768).flat_map(|_| [0.0, 1.0]).collect();
    let encoded =
        decode::Encoded::from_outputs(vec![f32s(&[1, 768, 2], data), i64s(&[1], vec![1])], &grid)
            .unwrap();
    let predict = MockSession(|inputs: &[Tensor]| {
        assert_eq!(inputs[0].view().data().as_i64().unwrap(), &[1024]);
        Ok(predictor_outputs(0.))
    });
    let join = MockSession(|inputs: &[Tensor]| {
        assert_eq!(inputs[0].shape().dims(), &[1, 768, 8]);
        assert_eq!(
            &inputs[0].view().data().as_f32().unwrap()[..8],
            &[0., 1., 1., 1., 1., 1., 1., 1.]
        );
        Ok(vec![logits([1024, 7, 7, 7, 7, 7, 7, 7])])
    });
    let result = decode::greedy(&predict, &join, &encoded, None).unwrap();
    assert!(result.tokens.is_empty());
    assert_eq!(result.trace.predictor_inputs.len(), 1);
    assert_eq!(result.trace.joint_inputs.len(), 1);
}

#[test]
fn test_v4_encoder_rejects_wrong_physical_or_valid_shape() {
    let grid = V4FrameGrid::new(960, 641).unwrap();
    assert!(
        decode::Encoded::from_outputs(
            vec![f32s(&[1, 768, 1], vec![0.; 768]), i64s(&[1], vec![1])],
            &grid
        )
        .is_err()
    );
    assert!(
        decode::Encoded::from_outputs(
            vec![f32s(&[1, 768, 2], vec![0.; 1536]), i64s(&[1], vec![2])],
            &grid
        )
        .is_err()
    );
    assert!(
        decode::Encoded::from_outputs(
            vec![
                f32s(&[1, 768, 2], vec![f32::NAN; 1536]),
                i64s(&[1], vec![1])
            ],
            &grid
        )
        .is_err()
    );
}

#[test]
fn test_v4_first_argmax_tie_and_max_ten_labels() {
    let grid = V4FrameGrid::new(320, 320).unwrap();
    let encoded = decode::Encoded::from_outputs(
        vec![f32s(&[1, 768, 1], vec![0.; 768]), i64s(&[1], vec![1])],
        &grid,
    )
    .unwrap();
    let n = AtomicUsize::new(0);
    let predict = MockSession(move |inputs: &[Tensor]| {
        let k = n.fetch_add(1, Ordering::Relaxed);
        let expected = if k == 0 { 1024 } else { 2 };
        assert_eq!(inputs[0].view().data().as_i64().unwrap(), &[expected]);
        if k > 0 {
            assert_eq!(inputs[1].view().data().as_f32().unwrap()[0], k as f32);
        }
        Ok(predictor_outputs(k as f32))
    });
    let join = MockSession(|_: &[Tensor]| {
        let mut v = vec![-10.; 8 * 1025];
        v[2] = -0.1;
        v[3] = -0.1;
        Ok(vec![f32s(&[1, 8, 1, 1025], v)])
    });
    let r = decode::greedy(&predict, &join, &encoded, None).unwrap();
    assert_eq!(r.tokens.len(), 10);
    assert!(r.tokens.iter().all(|t| t.id == 2 && t.frame == 0));
    assert_eq!(r.trace.predictor_inputs.len(), 10);
    assert_eq!(r.trace.joint_inputs.len(), 10);
}

#[test]
fn test_v4_cancel_after_native_call_returns_no_trajectory() {
    use std::sync::Arc;
    let grid = V4FrameGrid::new(320, 320).unwrap();
    let encoded = decode::Encoded::from_outputs(
        vec![f32s(&[1, 768, 1], vec![0.; 768]), i64s(&[1], vec![1])],
        &grid,
    )
    .unwrap();
    let cancel = Arc::new(AtomicBool::new(false));
    let raised = cancel.clone();
    let predict = MockSession(move |_: &[Tensor]| {
        raised.store(true, Ordering::Relaxed);
        Ok(predictor_outputs(0.))
    });
    let join = MockSession(|_: &[Tensor]| -> Result<Vec<Tensor>, RuntimeError> {
        panic!("joiner must not run after abort")
    });
    assert!(matches!(
        decode::greedy(&predict, &join, &encoded, Some(&cancel)),
        Err(crate::error::GigasttError::Cancelled)
    ));
}

#[test]
fn test_v4_blank_window_reuses_prediction_and_commits_only_on_emission() {
    use std::sync::Arc;
    let grid = V4FrameGrid::new(8_000, 8_000).unwrap();
    let t = grid.encoder_frames();
    assert!(t > 9);
    let encoded = decode::Encoded::from_outputs(
        vec![
            f32s(&[1, 768, t], vec![0.; 768 * t]),
            i64s(&[1], vec![t as i64]),
        ],
        &grid,
    )
    .unwrap();
    // Repeat a new invocation with fresh mock state to prove no cross-request
    // predictor state is carried by the greedy function.
    for _ in 0..2 {
        let n = AtomicUsize::new(0);
        let predict = MockSession(move |inputs: &[Tensor]| {
            let k = n.fetch_add(1, Ordering::Relaxed);
            assert_eq!(
                inputs[0].view().data().as_i64().unwrap(),
                &[if k == 0 { 1024 } else { 17 }]
            );
            assert_eq!(
                inputs[1].view().data().as_f32().unwrap()[0],
                if k == 0 { 0. } else { 1. }
            );
            assert_eq!(
                inputs[2].view().data().as_f32().unwrap()[0],
                if k == 0 { 0. } else { 2. }
            );
            Ok(predictor_outputs(k as f32))
        });
        let n = Arc::new(AtomicUsize::new(0));
        let c = n.clone();
        let join = MockSession(move |inputs: &[Tensor]| {
            let k = c.fetch_add(1, Ordering::Relaxed);
            assert_eq!(
                inputs[1].view().data().as_f32().unwrap()[0],
                if k < 2 { 0. } else { 1. }
            );
            let mut labels = [1024; 8];
            if k == 1 {
                labels[1] = 17;
            }
            Ok(vec![logits(labels)])
        });
        let r = decode::greedy(&predict, &join, &encoded, None).unwrap();
        assert_eq!(
            r.tokens.iter().map(|t| (t.id, t.frame)).collect::<Vec<_>>(),
            vec![(17, 9)]
        );
        assert_eq!(r.trace.predictor_inputs, vec![[1024, 0], [17, 9]]);
        assert_eq!(r.trace.joint_inputs[0], [0, 1024, 0, 8]);
        assert_eq!(
            r.trace.joint_inputs[1],
            [8, 1024, 0, t.saturating_sub(8).min(8)]
        );
        assert_eq!(
            r.trace.joint_inputs[2],
            [9, 17, 1, t.saturating_sub(9).min(8)]
        );
    }
}

#[test]
fn test_v4_cancelled_slot_and_encoder_do_not_call_native() {
    let flag = AtomicBool::new(true);
    let slot = Mutex::new(());
    assert!(matches!(
        acquire_request_slot(&slot, Some(&flag)),
        Err(GigasttError::Cancelled)
    ));
    let grid = V4FrameGrid::new(320, 320).unwrap();
    let encoder = MockSession(|_: &[Tensor]| -> Result<Vec<Tensor>, RuntimeError> {
        panic!("cancelled encoder must not run")
    });
    assert!(matches!(
        decode::encode_features(&encoder, vec![0.; 64], &grid, Some(&flag)),
        Err(GigasttError::Cancelled)
    ));
    flag.store(false, Ordering::Relaxed);
    let held = slot.lock();
    std::thread::scope(|s| {
        s.spawn(|| {
            std::thread::sleep(Duration::from_millis(20));
            flag.store(true, Ordering::Relaxed);
        });
        assert!(matches!(
            acquire_request_slot(&slot, Some(&flag)),
            Err(GigasttError::Cancelled)
        ));
    });
    drop(held);
    assert!(slot.try_lock().is_some());
}

#[test]
fn test_v4_invalid_predictor_and_joiner_outputs_fail_closed() {
    let grid = V4FrameGrid::new(320, 320).unwrap();
    let encoded = decode::Encoded::from_outputs(
        vec![f32s(&[1, 768, 1], vec![0.; 768]), i64s(&[1], vec![1])],
        &grid,
    )
    .unwrap();
    let wrong = MockSession(|_: &[Tensor]| Ok(predictor_outputs(f32::NAN)));
    let never = MockSession(|_: &[Tensor]| -> Result<Vec<Tensor>, RuntimeError> {
        panic!("invalid predictor must stop before joiner")
    });
    assert!(decode::greedy(&wrong, &never, &encoded, None).is_err());
    let correct = MockSession(|_: &[Tensor]| Ok(predictor_outputs(0.)));
    let wrong = MockSession(|_: &[Tensor]| Ok(vec![f32s(&[1, 1, 1, 1025], vec![0.; 1025])]));
    assert!(decode::greedy(&correct, &wrong, &encoded, None).is_err());
}

#[test]
fn test_v4_streaming_blank_prediction_persists_across_packets_with_absolute_offset() {
    use std::sync::Arc;
    let grid = V4FrameGrid::new(960, 641).unwrap();
    let data: Vec<f32> = (0..768).flat_map(|_| [3.0, 99.0]).collect();
    let encoded =
        decode::Encoded::from_outputs(vec![f32s(&[1, 768, 2], data), i64s(&[1], vec![1])], &grid)
            .unwrap()
            .valid_prefix();
    let predicts = Arc::new(AtomicUsize::new(0));
    let pc = predicts.clone();
    let predict = MockSession(move |inputs: &[Tensor]| {
        let n = pc.fetch_add(1, Ordering::Relaxed);
        assert_eq!(
            inputs[0].view().data().as_i64().unwrap(),
            &[if n == 0 { 1024 } else { 7 }]
        );
        assert_eq!(
            inputs[1].view().data().as_f32().unwrap()[0],
            if n == 0 { 0. } else { 11. }
        );
        Ok(predictor_outputs(10. + n as f32))
    });
    let calls = AtomicUsize::new(0);
    let join = MockSession(move |inputs: &[Tensor]| {
        assert_eq!(&inputs[0].view().data().as_f32().unwrap()[..8], &[3.0; 8]);
        let n = calls.fetch_add(1, Ordering::Relaxed);
        let mut labels = [1024; 8];
        if n == 1 {
            labels[0] = 7;
        }
        Ok(vec![logits(labels)])
    });
    let mut state = decode::GreedyState::default();
    let first = decode::greedy_block(&predict, &join, &encoded, &mut state, None).unwrap();
    assert!(first.tokens.is_empty());
    assert_eq!(predicts.load(Ordering::Relaxed), 1);
    let second = decode::greedy_block(&predict, &join, &encoded, &mut state, None).unwrap();
    assert_eq!(second.tokens.len(), 1);
    assert_eq!((second.tokens[0].id, second.tokens[0].frame), (7, 1));
    assert_eq!(predicts.load(Ordering::Relaxed), 2);
    assert_eq!(second.trace.predictor_inputs, [[7, 0]]);
    let third = decode::greedy_block(&predict, &join, &encoded, &mut state, None).unwrap();
    assert!(third.tokens.is_empty());
    assert_eq!(predicts.load(Ordering::Relaxed), 2);
}
