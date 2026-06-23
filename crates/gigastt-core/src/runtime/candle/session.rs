//! Candle-backed encoder / decoder / joiner [`RuntimeSession`] implementations.

use candle_core::{DType, Device, Tensor as CandleTensor};
use candle_nn::{Module, VarBuilder};

use crate::runtime::{
    error::RuntimeError,
    session::RuntimeSession,
    tensor::{Shape, Tensor, TensorData},
};

use super::conformer::ConformerEncoder;

/// LSTM hidden size / decoder output dim (mirrors `inference::PRED_HIDDEN`).
const PRED_HIDDEN: usize = 320;
/// Encoder output channel dim.
const ENC_DIM: usize = 768;
/// rnnt char vocab size (incl. blank).
const VOCAB: usize = 34;

fn backend_err(e: impl std::fmt::Display) -> RuntimeError {
    RuntimeError::InferenceFailed(e.to_string())
}

/// Wraps a loaded Candle Conformer encoder behind the [`RuntimeSession`] seam.
pub struct EncoderSession {
    enc: ConformerEncoder,
    device: Device,
}

impl EncoderSession {
    pub(crate) fn new(enc: ConformerEncoder, device: Device) -> Self {
        Self { enc, device }
    }
}

impl RuntimeSession for EncoderSession {
    /// Encoder contract (mirrors the ort ONNX encoder, which emits two outputs):
    /// `inputs[0] = audio_signal [1, 64, T] F32`, `inputs[1] = length [1]`
    /// (ignored; batch is always 1 here). Returns
    /// `[encoded [1, 768, T/4] F32, encoded_len [1] I64]` (channels-first), so
    /// the engine's decode loop can read `encoder_outputs[1]` for the output
    /// frame count exactly as it does for the ort backend.
    fn run(&self, inputs: &[Tensor]) -> Result<Vec<Tensor>, RuntimeError> {
        if inputs.is_empty() {
            return Err(RuntimeError::InvalidInputCount {
                expected: 1,
                got: inputs.len(),
            });
        }
        let mel = super::tensor::to_candle(&inputs[0], &self.device)?;
        let out = self
            .enc
            .forward(&mel)
            .map_err(|e| RuntimeError::InferenceFailed(e.to_string()))?;
        // Output time dimension (channels-first `[1, 768, T']`) is the encoded
        // frame count the RNN-T decode loop iterates over.
        let enc_len = out.dims().get(2).copied().ok_or_else(|| {
            RuntimeError::InferenceFailed(format!(
                "encoder output has unexpected rank {}",
                out.rank()
            ))
        })?;
        Ok(vec![
            super::tensor::from_candle(&out)?,
            Tensor::new(Shape::new(vec![1]), TensorData::I64(vec![enc_len as i64]))?,
        ])
    }
}

/// Candle-backed RNN-T prediction network (embedding + single-layer LSTM).
///
/// Mirrors the ONNX decoder graph: `Gather(embed, token) -> LSTM(1 layer)`.
/// The LSTM gate order is **iofc** (ONNX convention): the 1280-row weight
/// matrices are 4 blocks of 320 in input/output/forget/cell order. A manual
/// one-step cell is implemented (NOT candle's prebuilt LSTM, whose gate order
/// differs) so the iofc layout is honoured exactly.
pub struct DecoderSession {
    /// Embedding table `[VOCAB, PRED_HIDDEN]`.
    embed: CandleTensor,
    /// Input-hidden weights `[4*PRED_HIDDEN, PRED_HIDDEN]` (iofc), no transpose.
    w_ih: CandleTensor,
    /// Hidden-hidden weights `[4*PRED_HIDDEN, PRED_HIDDEN]` (iofc), no transpose.
    w_hh: CandleTensor,
    /// Input bias `[4*PRED_HIDDEN]` (iofc).
    b_ih: CandleTensor,
    /// Recurrent bias `[4*PRED_HIDDEN]` (iofc).
    b_hh: CandleTensor,
    device: Device,
}

impl DecoderSession {
    pub(crate) fn load(vb: VarBuilder, device: Device) -> Result<Self, RuntimeError> {
        let embed = vb
            .get((VOCAB, PRED_HIDDEN), "embed.weight")
            .map_err(backend_err)?;
        let w_ih = vb
            .get((4 * PRED_HIDDEN, PRED_HIDDEN), "lstm.w_ih")
            .map_err(backend_err)?;
        let w_hh = vb
            .get((4 * PRED_HIDDEN, PRED_HIDDEN), "lstm.w_hh")
            .map_err(backend_err)?;
        let b_ih = vb.get(4 * PRED_HIDDEN, "lstm.b_ih").map_err(backend_err)?;
        let b_hh = vb.get(4 * PRED_HIDDEN, "lstm.b_hh").map_err(backend_err)?;
        Ok(Self {
            embed,
            w_ih,
            w_hh,
            b_ih,
            b_hh,
            device,
        })
    }

    /// One LSTM timestep over column vectors (all `[PRED_HIDDEN]`).
    ///
    /// `g = W_ih·x + b_ih + W_hh·h_prev + b_hh`  (gate pre-activations, iofc)
    /// `i,o,f = sigmoid(g[..])`, `cg = tanh(g[..])`
    /// `c_new = f*c_prev + i*cg`, `h_new = o*tanh(c_new)`.
    fn lstm_step(
        &self,
        x: &CandleTensor,      // [PRED_HIDDEN]
        h_prev: &CandleTensor, // [PRED_HIDDEN]
        c_prev: &CandleTensor, // [PRED_HIDDEN]
    ) -> Result<(CandleTensor, CandleTensor), RuntimeError> {
        // matmul wants 2-D operands: weight [4H, H] · x [H, 1] -> [4H, 1].
        let x_col = x.reshape((PRED_HIDDEN, 1)).map_err(backend_err)?;
        let h_col = h_prev.reshape((PRED_HIDDEN, 1)).map_err(backend_err)?;

        let gates = self
            .w_ih
            .matmul(&x_col)
            .map_err(backend_err)?
            .add(&self.w_hh.matmul(&h_col).map_err(backend_err)?)
            .map_err(backend_err)?
            .reshape(4 * PRED_HIDDEN) // [4H]
            .map_err(backend_err)?
            .add(&self.b_ih)
            .map_err(backend_err)?
            .add(&self.b_hh)
            .map_err(backend_err)?;

        // iofc gate blocks.
        let i = gates.narrow(0, 0, PRED_HIDDEN).map_err(backend_err)?;
        let o = gates
            .narrow(0, PRED_HIDDEN, PRED_HIDDEN)
            .map_err(backend_err)?;
        let f = gates
            .narrow(0, 2 * PRED_HIDDEN, PRED_HIDDEN)
            .map_err(backend_err)?;
        let cg = gates
            .narrow(0, 3 * PRED_HIDDEN, PRED_HIDDEN)
            .map_err(backend_err)?;

        let i = candle_nn::ops::sigmoid(&i).map_err(backend_err)?;
        let o = candle_nn::ops::sigmoid(&o).map_err(backend_err)?;
        let f = candle_nn::ops::sigmoid(&f).map_err(backend_err)?;
        let cg = cg.tanh().map_err(backend_err)?;

        // c_new = f*c_prev + i*cg
        let c_new = f
            .mul(c_prev)
            .map_err(backend_err)?
            .add(&i.mul(&cg).map_err(backend_err)?)
            .map_err(backend_err)?;
        // h_new = o*tanh(c_new)
        let h_new = o
            .mul(&c_new.tanh().map_err(backend_err)?)
            .map_err(backend_err)?;

        Ok((h_new, c_new))
    }
}

impl RuntimeSession for DecoderSession {
    /// Decoder contract: `inputs = [prev_token [1,1] I64, h [1,1,320] F32,
    /// c [1,1,320] F32]`; returns `[dec, new_h, new_c]`, each flattened length
    /// 320 (F32). `dec == new_h` (the LSTM output equals the new hidden state).
    fn run(&self, inputs: &[Tensor]) -> Result<Vec<Tensor>, RuntimeError> {
        if inputs.len() != 3 {
            return Err(RuntimeError::InvalidInputCount {
                expected: 3,
                got: inputs.len(),
            });
        }

        let token = inputs[0]
            .view()
            .data()
            .as_i64()
            .and_then(|s| s.first().copied())
            .ok_or_else(|| {
                RuntimeError::InferenceFailed("decoder prev_token must be i64 [1,1]".to_string())
            })?;
        if !(0..VOCAB as i64).contains(&token) {
            return Err(RuntimeError::InferenceFailed(format!(
                "decoder prev_token {token} out of range [0,{VOCAB})"
            )));
        }

        let h_in = super::tensor::to_candle(&inputs[1], &self.device)?
            .reshape(PRED_HIDDEN)
            .map_err(backend_err)?;
        let c_in = super::tensor::to_candle(&inputs[2], &self.device)?
            .reshape(PRED_HIDDEN)
            .map_err(backend_err)?;

        // Embedding lookup: row `token` of `embed` -> x [PRED_HIDDEN].
        let x = self
            .embed
            .narrow(0, token as usize, 1)
            .map_err(backend_err)?
            .reshape(PRED_HIDDEN)
            .map_err(backend_err)?;

        let (h_new, c_new) = self.lstm_step(&x, &h_in, &c_in)?;

        // dec == h_new (ONNX decoder returns the LSTM output as `dec`).
        let dec = to_runtime_vec(&h_new)?;
        let new_h = to_runtime_vec(&h_new)?;
        let new_c = to_runtime_vec(&c_new)?;

        Ok(vec![
            runtime_tensor_3d(dec)?,
            runtime_tensor_3d(new_h)?,
            runtime_tensor_3d(new_c)?,
        ])
    }
}

/// Candle-backed RNN-T joint network.
///
/// `e = enc_proj·enc + enc.bias`, `d = dec_proj·dec + pred.bias`,
/// `j = relu(e + d)`, `out = out_proj·j + out.bias`,
/// `logits = log_softmax(out)`  (ONNX applies LogSoftmax — matched here).
pub struct JoinerSession {
    enc_proj: candle_nn::Linear, // 768 -> 320
    dec_proj: candle_nn::Linear, // 320 -> 320
    out: candle_nn::Linear,      // 320 -> 34
    device: Device,
}

impl JoinerSession {
    pub(crate) fn load(vb: VarBuilder, device: Device) -> Result<Self, RuntimeError> {
        let enc_proj =
            candle_nn::linear(ENC_DIM, PRED_HIDDEN, vb.pp("enc_proj")).map_err(backend_err)?;
        let dec_proj =
            candle_nn::linear(PRED_HIDDEN, PRED_HIDDEN, vb.pp("dec_proj")).map_err(backend_err)?;
        let out = candle_nn::linear(PRED_HIDDEN, VOCAB, vb.pp("out")).map_err(backend_err)?;
        Ok(Self {
            enc_proj,
            dec_proj,
            out,
            device,
        })
    }
}

impl RuntimeSession for JoinerSession {
    /// Joiner contract: `inputs = [enc_frame [1,768,1] F32, dec_data [1,320,1]
    /// F32]`; output `[logits]` flattened length 34 (F32).
    fn run(&self, inputs: &[Tensor]) -> Result<Vec<Tensor>, RuntimeError> {
        if inputs.len() != 2 {
            return Err(RuntimeError::InvalidInputCount {
                expected: 2,
                got: inputs.len(),
            });
        }

        // Flatten [1,768,1] -> [1,768] and [1,320,1] -> [1,320] (row vectors, so
        // candle's Linear can matmul against the [out,in] weight).
        let enc = super::tensor::to_candle(&inputs[0], &self.device)?
            .reshape((1, ENC_DIM))
            .map_err(backend_err)?;
        let dec = super::tensor::to_candle(&inputs[1], &self.device)?
            .reshape((1, PRED_HIDDEN))
            .map_err(backend_err)?;

        let e = self.enc_proj.forward(&enc).map_err(backend_err)?; // [1,320]
        let d = self.dec_proj.forward(&dec).map_err(backend_err)?; // [1,320]
        let j = e
            .add(&d)
            .map_err(backend_err)?
            .relu()
            .map_err(backend_err)?; // [1,320]
        let logits = self.out.forward(&j).map_err(backend_err)?; // [1,34]
        let log_probs = candle_nn::ops::log_softmax(&logits, 1).map_err(backend_err)?; // [1,34]

        let data = to_runtime_vec(&log_probs)?;
        Ok(vec![Tensor::new(
            Shape::new(vec![1, 1, 1, VOCAB]),
            TensorData::F32(data),
        )?])
    }
}

/// Flatten a 1-D candle tensor to an owned `Vec<f32>`.
fn to_runtime_vec(t: &CandleTensor) -> Result<Vec<f32>, RuntimeError> {
    t.to_dtype(DType::F32)
        .map_err(backend_err)?
        .flatten_all()
        .map_err(backend_err)?
        .to_vec1::<f32>()
        .map_err(backend_err)
}

/// Wrap a `[PRED_HIDDEN]` vector as a runtime tensor with the ONNX decoder
/// output shape `[1,1,PRED_HIDDEN]` (decode.rs reads it flattened as 320 f32).
fn runtime_tensor_3d(data: Vec<f32>) -> Result<Tensor, RuntimeError> {
    Tensor::new(Shape::new(vec![1, 1, PRED_HIDDEN]), TensorData::F32(data))
}
