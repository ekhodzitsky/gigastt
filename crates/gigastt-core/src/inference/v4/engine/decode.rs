//! Exact batch-one author windowed greedy schedule for verified v4 RNNT.

use anyhow::{Context, Result as AnyResult, ensure};
use std::sync::atomic::{AtomicBool, Ordering};

use super::{V4FrameGrid, V4Token, inference_error};
use crate::error::GigasttError;
use crate::runtime::{RuntimeSession, Shape, Tensor, TensorData};

const CHANNELS: usize = 768;
const HIDDEN: usize = 320;
const BLANK: usize = 1024;
const CLASSES: usize = BLANK + 1;
const WINDOW: usize = 8;
const MAX_SYMBOLS: usize = 10;

pub(super) fn check_cancel(cancel: Option<&AtomicBool>) -> Result<(), GigasttError> {
    if cancel.is_some_and(|flag| flag.load(Ordering::Relaxed)) {
        Err(GigasttError::Cancelled)
    } else {
        Ok(())
    }
}

pub(super) fn run_session(
    session: &dyn RuntimeSession,
    inputs: &[Tensor],
    cancel: Option<&AtomicBool>,
) -> Result<Vec<Tensor>, GigasttError> {
    check_cancel(cancel)?;
    let result = session.run(inputs);
    check_cancel(cancel)?;
    result.map_err(Into::into)
}

fn f32_tensor(shape: &[usize], data: Vec<f32>) -> Result<Tensor, GigasttError> {
    Ok(Tensor::new(
        Shape::new(shape.to_vec()),
        TensorData::F32(data),
    )?)
}
fn i64_tensor(shape: &[usize], data: Vec<i64>) -> Result<Tensor, GigasttError> {
    Ok(Tensor::new(
        Shape::new(shape.to_vec()),
        TensorData::I64(data),
    )?)
}
fn f32_output(tensor: Tensor, shape: &[usize], label: &str) -> AnyResult<Vec<f32>> {
    ensure!(tensor.shape().dims() == shape, "unexpected {label} shape");
    let TensorData::F32(data) = tensor.into_data() else {
        anyhow::bail!("unexpected {label} element type");
    };
    ensure!(data.iter().all(|v| v.is_finite()), "non-finite {label}");
    Ok(data)
}

pub(super) struct Encoded {
    pub(super) data: Vec<f32>,
    pub(super) physical: usize,
    pub(super) valid: usize,
}
impl Encoded {
    pub(super) fn from_outputs(outputs: Vec<Tensor>, grid: &V4FrameGrid) -> AnyResult<Self> {
        let [values, length]: [Tensor; 2] = outputs
            .try_into()
            .map_err(|_| anyhow::anyhow!("expected two v4 encoder outputs"))?;
        let physical = grid.encoder_frames();
        let data = f32_output(values, &[1, CHANNELS, physical], "v4 encoder")?;
        ensure!(
            length.shape().dims() == [1],
            "unexpected v4 encoded length shape"
        );
        let TensorData::I64(length) = length.into_data() else {
            anyhow::bail!("unexpected v4 encoded length type");
        };
        let valid = usize::try_from(length[0]).context("negative v4 encoded length")?;
        ensure!(
            valid == grid.valid_encoder_frames() && 0 < valid && valid <= physical,
            "v4 encoded valid length mismatch"
        );
        Ok(Self {
            data,
            physical,
            valid,
        })
    }
    fn window(&self, pos: usize) -> Result<Tensor, GigasttError> {
        // The upstream gather clamps to the physical final frame, not the
        // valid final frame. The
        // valid prefix is applied only when searching the joiner outputs.
        let mut values = Vec::with_capacity(CHANNELS * WINDOW);
        for channel in self.data.as_slice().chunks_exact(self.physical) {
            for offset in 0..WINDOW {
                values.push(channel[(pos + offset).min(self.physical - 1)]);
            }
        }
        f32_tensor(&[1, CHANNELS, WINDOW], values)
    }
}

pub(super) fn encode_features(
    session: &dyn RuntimeSession,
    values: Vec<f32>,
    grid: &V4FrameGrid,
    cancel: Option<&AtomicBool>,
) -> Result<Encoded, GigasttError> {
    let mel = grid.mel_frames();
    if values.len() != 64 * mel || values.iter().any(|x| !x.is_finite()) {
        return Err(inference_error(anyhow::anyhow!("invalid v4 mel tensor")));
    }
    let length =
        i64::try_from(grid.valid_mel_frames()).map_err(|err| inference_error(err.into()))?;
    let inputs = [
        f32_tensor(&[1, 64, mel], values)?,
        i64_tensor(&[1], vec![length])?,
    ];
    let outputs = run_session(session, &inputs, cancel)?;
    Encoded::from_outputs(outputs, grid).map_err(inference_error)
}

struct Prediction {
    values: Vec<f32>,
    h: Vec<f32>,
    c: Vec<f32>,
}

#[cfg(test)]
#[derive(Debug, Default, Clone)]
pub(super) struct Trace {
    pub predictor_inputs: Vec<[usize; 2]>,
    pub joint_inputs: Vec<[usize; 4]>,
    pub prediction: Vec<f32>,
    pub h: Vec<f32>,
    pub c: Vec<f32>,
    pub log_probs: Vec<f32>,
}

pub(super) struct Trajectory {
    pub tokens: Vec<V4Token>,
    #[cfg(test)]
    pub trace: Trace,
}

fn predictor(
    session: &dyn RuntimeSession,
    label: usize,
    h: &[f32],
    c: &[f32],
    cancel: Option<&AtomicBool>,
) -> Result<Prediction, GigasttError> {
    let label = i64::try_from(label).map_err(|err| inference_error(err.into()))?;
    let inputs = [
        i64_tensor(&[1, 1], vec![label])?,
        f32_tensor(&[1, 1, HIDDEN], h.to_vec())?,
        f32_tensor(&[1, 1, HIDDEN], c.to_vec())?,
    ];
    let outputs = run_session(session, &inputs, cancel)?;
    let [pred, h, c]: [Tensor; 3] = outputs
        .try_into()
        .map_err(|_| inference_error(anyhow::anyhow!("expected three v4 predictor outputs")))?;
    Ok(Prediction {
        values: f32_output(pred, &[1, 1, HIDDEN], "v4 prediction").map_err(inference_error)?,
        h: f32_output(h, &[1, 1, HIDDEN], "v4 predictor hidden state").map_err(inference_error)?,
        c: f32_output(c, &[1, 1, HIDDEN], "v4 predictor cell state").map_err(inference_error)?,
    })
}

fn argmax_first(row: &[f32]) -> usize {
    // Only validated, finite, nonempty log-softmax rows reach here. Strict
    // comparison retains the first index on an equal score (author argmax).
    let mut best = 0;
    for (index, &value) in row.iter().enumerate().skip(1) {
        if value > row[best] {
            best = index;
        }
    }
    best
}

pub(super) fn greedy(
    predict: &dyn RuntimeSession,
    join: &dyn RuntimeSession,
    encoded: &Encoded,
    cancel: Option<&AtomicBool>,
) -> Result<Trajectory, GigasttError> {
    let cap = encoded.valid * (MAX_SYMBOLS + 1);
    let mut pos = 0;
    let mut repeat = 0;
    let mut label = BLANK;
    // Fresh author prediction uses a zero embedding. The exact qualified
    // ONNX decoder's blank embedding row is zero; zero h/c reproduce it.
    let mut h = vec![0.0; HIDDEN];
    let mut c = vec![0.0; HIDDEN];
    let mut prediction = None;
    let mut tokens = Vec::new();
    let mut predictor_calls = 0;
    let mut joint_calls = 0;
    #[cfg(test)]
    let mut trace = Trace::default();
    while pos < encoded.valid {
        check_cancel(cancel)?;
        if prediction.is_none() {
            if predictor_calls >= cap {
                return Err(inference_error(anyhow::anyhow!(
                    "v4 predictor call cap exceeded"
                )));
            }
            let p = predictor(predict, label, &h, &c, cancel)?;
            predictor_calls += 1;
            #[cfg(test)]
            {
                trace.predictor_inputs.push([label, pos]);
                trace.prediction.extend_from_slice(&p.values);
                trace.h.extend_from_slice(&p.h);
                trace.c.extend_from_slice(&p.c);
            }
            prediction = Some(p);
        }
        if joint_calls >= cap {
            return Err(inference_error(anyhow::anyhow!(
                "v4 joiner call cap exceeded"
            )));
        }
        let p = prediction
            .as_ref()
            .ok_or_else(|| inference_error(anyhow::anyhow!("missing v4 cached prediction")))?;
        let inputs = [
            encoded.window(pos)?,
            f32_tensor(&[1, HIDDEN, 1], p.values.clone())?,
        ];
        let outputs = run_session(join, &inputs, cancel)?;
        joint_calls += 1;
        let [joint]: [Tensor; 1] = outputs
            .try_into()
            .map_err(|_| inference_error(anyhow::anyhow!("expected one v4 joiner output")))?;
        let logp = f32_output(joint, &[1, WINDOW, 1, CLASSES], "v4 log-softmax")
            .map_err(inference_error)?;
        let count = WINDOW.min(encoded.valid - pos);
        #[cfg(test)]
        {
            trace.joint_inputs.push([pos, label, repeat, count]);
            trace.log_probs.extend_from_slice(&logp);
        }
        let nonblank = logp
            .as_chunks::<CLASSES>()
            .0
            .iter()
            .take(count)
            .enumerate()
            .find_map(|(delta, row)| {
                let token = argmax_first(row);
                (token != BLANK).then_some((delta, token, row[token]))
            });
        let Some((delta, token, selected)) = nonblank else {
            pos += count;
            repeat = 0;
            continue;
        };
        pos += delta;
        repeat = if delta == 0 { repeat + 1 } else { 1 };
        let confidence = f64::from(selected).exp();
        if !(0.0..=1.0).contains(&confidence) {
            return Err(inference_error(anyhow::anyhow!(
                "invalid selected v4 log-softmax confidence"
            )));
        }
        tokens.push(V4Token {
            id: token,
            frame: pos,
            confidence,
        });
        label = token;
        let p = prediction
            .take()
            .ok_or_else(|| inference_error(anyhow::anyhow!("missing v4 committed prediction")))?;
        h = p.h;
        c = p.c;
        if repeat >= MAX_SYMBOLS {
            pos += 1;
            repeat = 0;
        }
    }
    check_cancel(cancel)?;
    Ok(Trajectory {
        tokens,
        #[cfg(test)]
        trace,
    })
}
