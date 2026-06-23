// Vendored from askidmobile/RustASR (crates/model-gigaam, commit 33060b8d),
// dual-licensed MIT OR Apache-2.0. Copyright (c) the RustASR authors.
// Adapted for gigastt: import paths changed; compiled against upstream candle 0.9;
// CTC head not used here (only the v3 Conformer encoder).
#![allow(dead_code)] // wired into a RuntimeSession in a later task

//! Conformer-энкодер для GigaAM.
//!
//! Реализация архитектуры Conformer (Gulati et al., 2020)
//! с Rotary Position Embeddings (RoPE), Macaron-style FFN,
//! depthwise separable convolution, и стриженной субдискретизацией.
//!
//! Совместимость весов: ключи тензоров совпадают с PyTorch-реализацией
//! GigaAM (salute-developers/GigaAM), что позволяет загружать
//! сконвертированные safetensors напрямую через VarBuilder.

use candle_core::{Device, Result, Tensor};
use candle_nn::{Conv1d, Conv1dConfig, LayerNorm, Linear, Module, VarBuilder};

use super::config::EncoderConfig;

// -----------------------------------------------------------------------
// Rotary Position Embedding (RoPE)
// -----------------------------------------------------------------------

/// Создать таблицу cos/sin для RoPE.
///
/// Возвращает два тензора (cos, sin) формы (max_len, 1, 1, dim).
fn create_rope_table(dim: usize, max_len: usize, device: &Device) -> Result<(Tensor, Tensor)> {
    // GigaAM v3 uses a RoPE base of 5000 (not the common 10000); verified by
    // matching the encoder's baked cos/sin table to 5.96e-6 over the first 100
    // positions. Using 10000 silently corrupts attention from the first layer.
    let base = 5_000f32;
    // inv_freq = 1 / (base^(2i/dim)) для i = 0, 2, 4, ..., dim-2
    let half_dim = dim / 2;
    let inv_freq: Vec<f32> = (0..half_dim)
        .map(|i| 1.0 / base.powf(2.0 * i as f32 / dim as f32))
        .collect();

    let inv_freq_t = Tensor::from_vec(inv_freq, half_dim, device)?;
    let positions: Vec<f32> = (0..max_len).map(|i| i as f32).collect();
    let positions_t = Tensor::from_vec(positions, max_len, device)?;

    // freqs = outer(positions, inv_freq) → (max_len, half_dim)
    let freqs = positions_t
        .unsqueeze(1)?
        .matmul(&inv_freq_t.unsqueeze(0)?)?;

    // emb = cat(freqs, freqs, dim=-1) → (max_len, dim)
    let emb = Tensor::cat(&[&freqs, &freqs], 1)?;

    let cos = emb.cos()?;
    let sin = emb.sin()?;

    // Формы: (max_len, 1, 1, dim) для broadcasting с (seq, batch, heads, d_k)
    let cos = cos.unsqueeze(1)?.unsqueeze(1)?;
    let sin = sin.unsqueeze(1)?.unsqueeze(1)?;

    Ok((cos, sin))
}

/// Применить RoPE к Q и K.
///
/// q, k: (seq_len, batch, n_heads, d_k)
/// cos, sin: (seq_len, 1, 1, d_k) — обрезанные до seq_len
fn apply_rotary_pos_emb(
    q: &Tensor,
    k: &Tensor,
    cos: &Tensor,
    sin: &Tensor,
) -> Result<(Tensor, Tensor)> {
    let q_rot = rotate_half(q)?;
    let k_rot = rotate_half(k)?;

    let q_embed = q.broadcast_mul(cos)?.add(&q_rot.broadcast_mul(sin)?)?;
    let k_embed = k.broadcast_mul(cos)?.add(&k_rot.broadcast_mul(sin)?)?;

    Ok((q_embed, k_embed))
}

/// Разделить последнее измерение пополам и повернуть: (-x2, x1).
fn rotate_half(x: &Tensor) -> Result<Tensor> {
    let d = x.dim(candle_core::D::Minus1)?;
    let half = d / 2;
    let x1 = x.narrow(candle_core::D::Minus1, 0, half)?;
    let x2 = x.narrow(candle_core::D::Minus1, half, half)?;
    let neg_x2 = x2.neg()?;
    Tensor::cat(&[&neg_x2, &x1], candle_core::D::Minus1)
}

// -----------------------------------------------------------------------
// Strided Subsampling (conv1d)
// -----------------------------------------------------------------------

/// Страйденная субдискретизация через свёрточные слои.
///
/// Уменьшает длину последовательности в `subsampling_factor` раз.
/// Для factor=4 используется 2 слоя Conv1d с stride=2.
pub struct StridingSubsampling {
    /// Свёрточные слои (без ReLU — он применяется в forward).
    convs: Vec<Conv1d>,
    /// Фактор субдискретизации.
    factor: usize,
}

impl StridingSubsampling {
    pub fn load(
        feat_in: usize,
        d_model: usize,
        kernel_size: usize,
        factor: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let n_layers = (factor as f64).log2() as usize;
        let padding = (kernel_size - 1) / 2;
        let cfg = Conv1dConfig {
            padding,
            stride: 2,
            dilation: 1,
            groups: 1,
            ..Default::default()
        };

        let mut convs = Vec::with_capacity(n_layers);
        let mut in_ch = feat_in;

        for i in 0..n_layers {
            // Ключи: conv.0, conv.2 (рядом с ReLU на позициях 1, 3)
            let layer_idx = i * 2;
            let conv = candle_nn::conv1d(
                in_ch,
                d_model,
                kernel_size,
                cfg,
                vb.pp(format!("conv.{layer_idx}")),
            )?;
            convs.push(conv);
            in_ch = d_model;
        }

        Ok(Self { convs, factor })
    }

    /// Прямой проход: (batch, feat_in, seq_len) → (batch, seq_len/factor, d_model).
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // Вход: (batch, seq_len, feat_in) — транспонируем в (batch, feat_in, seq_len)
        let mut h = x.transpose(1, 2)?;

        for conv in &self.convs {
            h = conv.forward(&h)?;
            h = h.relu()?;
        }

        // Выход: (batch, d_model, seq_len/factor) → (batch, seq_len/factor, d_model)
        h.transpose(1, 2)
    }

    /// Вычислить длину после субдискретизации.
    pub fn output_length(&self, input_length: usize) -> usize {
        let mut length = input_length;
        let n_layers = (self.factor as f64).log2() as usize;
        for _ in 0..n_layers {
            length = length.div_ceil(2);
        }
        length
    }
}

// -----------------------------------------------------------------------
// Multi-Head Attention с Rotary Position Embeddings
// -----------------------------------------------------------------------

/// Multi-Head Self-Attention с RoPE (Rotary Position Embeddings).
pub struct RotaryMHSA {
    linear_q: Linear,
    linear_k: Linear,
    linear_v: Linear,
    linear_out: Linear,
    n_heads: usize,
    d_k: usize,
}

impl RotaryMHSA {
    pub fn load(d_model: usize, n_heads: usize, vb: VarBuilder) -> Result<Self> {
        let d_k = d_model / n_heads;
        let linear_q = candle_nn::linear(d_model, d_model, vb.pp("linear_q"))?;
        let linear_k = candle_nn::linear(d_model, d_model, vb.pp("linear_k"))?;
        let linear_v = candle_nn::linear(d_model, d_model, vb.pp("linear_v"))?;
        let linear_out = candle_nn::linear(d_model, d_model, vb.pp("linear_out"))?;

        Ok(Self {
            linear_q,
            linear_k,
            linear_v,
            linear_out,
            n_heads,
            d_k,
        })
    }

    /// Прямой проход MHSA с RoPE.
    ///
    /// # Аргументы
    /// * `x` — (batch, seq, d_model) — входной тензор (query=key=value=x)
    /// * `cos_emb`, `sin_emb` — RoPE таблицы, обрезанные до seq_len
    ///   формы (seq_len, 1, 1, d_k)
    /// * `att_mask` — маска внимания (batch, seq, seq) или None
    pub fn forward(
        &self,
        x: &Tensor,
        cos_emb: &Tensor,
        sin_emb: &Tensor,
        att_mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let (b, t, _d) = x.dims3()?;

        // 1. Применить RoPE к сырому входу (до проекции).
        //    GigaAM применяет RoPE ДО линейных проекций Q/K.
        //    x: (batch, seq, d_model) → reshape → (seq, batch, heads, d_k)
        let x_rope = x
            .transpose(0, 1)? // (seq, batch, d_model)
            .reshape((t, b, self.n_heads, self.d_k))?; // (seq, batch, heads, d_k)

        let (q_rope, k_rope) = apply_rotary_pos_emb(&x_rope, &x_rope, cos_emb, sin_emb)?;

        // Обратно: (seq, batch, heads, d_k) → (batch, seq, d_model)
        let q_in = q_rope
            .reshape((t, b, self.n_heads * self.d_k))?
            .transpose(0, 1)?; // (batch, seq, d_model)
        let k_in = k_rope
            .reshape((t, b, self.n_heads * self.d_k))?
            .transpose(0, 1)?; // (batch, seq, d_model)
        let v_in = x_rope
            .reshape((t, b, self.n_heads * self.d_k))?
            .transpose(0, 1)?; // (batch, seq, d_model)

        // 2. Проекция через линейные слои.
        let q = self
            .linear_q
            .forward(&q_in)? // (batch, seq, d_model)
            .reshape((b, t, self.n_heads, self.d_k))?
            .transpose(1, 2)? // (batch, heads, seq, d_k)
            .contiguous()?;
        let k = self
            .linear_k
            .forward(&k_in)?
            .reshape((b, t, self.n_heads, self.d_k))?
            .transpose(1, 2)?
            .contiguous()?;
        let v = self
            .linear_v
            .forward(&v_in)?
            .reshape((b, t, self.n_heads, self.d_k))?
            .transpose(1, 2)?
            .contiguous()?;

        // 3. Scaled dot-product attention.
        let scale = (self.d_k as f64).sqrt();
        let mut scores = q.matmul(&k.transpose(2, 3)?)?;
        scores = (scores / scale)?;

        // Применить маску (если есть).
        if let Some(mask) = att_mask {
            // mask: (batch, seq, seq) → (batch, 1, seq, seq)
            let mask = mask.unsqueeze(1)?;
            // Маскированные позиции заполняем -10000.
            let fill_val =
                Tensor::new(-10_000f32, scores.device())?.broadcast_as(scores.shape())?;
            scores = mask.where_cond(&fill_val, &scores)?;
        }

        let attn = candle_nn::ops::softmax_last_dim(&scores)?;

        if let Some(mask) = att_mask {
            let mask = mask.unsqueeze(1)?;
            let zeros = Tensor::zeros_like(&attn)?;
            let attn = mask.where_cond(&zeros, &attn)?;

            // 4. Weighted sum и выходная проекция.
            let context = attn.matmul(&v)?; // (batch, heads, seq, d_k)
            let context = context
                .transpose(1, 2)? // (batch, seq, heads, d_k)
                .reshape((b, t, self.n_heads * self.d_k))?;
            return self.linear_out.forward(&context);
        }

        // 4. Weighted sum и выходная проекция (без маски).
        let context = attn.matmul(&v)?;
        let context = context
            .transpose(1, 2)?
            .reshape((b, t, self.n_heads * self.d_k))?;
        self.linear_out.forward(&context)
    }
}

// -----------------------------------------------------------------------
// Conformer Feed-Forward Module
// -----------------------------------------------------------------------

/// Conformer Feed-Forward: Linear → SiLU → Linear.
pub struct ConformerFeedForward {
    linear1: Linear,
    linear2: Linear,
}

impl ConformerFeedForward {
    pub fn load(d_model: usize, d_ff: usize, vb: VarBuilder) -> Result<Self> {
        let linear1 = candle_nn::linear(d_model, d_ff, vb.pp("linear1"))?;
        let linear2 = candle_nn::linear(d_ff, d_model, vb.pp("linear2"))?;
        Ok(Self { linear1, linear2 })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let h = self.linear1.forward(x)?;
        let h = candle_nn::Activation::Silu.forward(&h)?;
        self.linear2.forward(&h)
    }
}

// -----------------------------------------------------------------------
// Conformer Convolution Module
// -----------------------------------------------------------------------

/// Conformer Convolution:
/// Pointwise Conv1d → GLU → Depthwise Conv1d → LayerNorm → SiLU → Pointwise Conv1d
pub struct ConformerConvolution {
    pointwise_conv1: Conv1d,
    depthwise_conv: Conv1d,
    norm: LayerNorm,
    pointwise_conv2: Conv1d,
    d_model: usize,
}

impl ConformerConvolution {
    pub fn load(d_model: usize, kernel_size: usize, vb: VarBuilder) -> Result<Self> {
        let padding = (kernel_size - 1) / 2;

        // Pointwise conv1: (d_model → 2*d_model, kernel=1) для GLU
        let pw1_cfg = Conv1dConfig {
            padding: 0,
            stride: 1,
            dilation: 1,
            groups: 1,
            ..Default::default()
        };
        let pointwise_conv1 =
            candle_nn::conv1d(d_model, d_model * 2, 1, pw1_cfg, vb.pp("pointwise_conv1"))?;

        // Depthwise conv: (d_model → d_model, kernel=kernel_size, groups=d_model)
        let dw_cfg = Conv1dConfig {
            padding,
            stride: 1,
            dilation: 1,
            groups: d_model,
            ..Default::default()
        };
        let depthwise_conv = candle_nn::conv1d(
            d_model,
            d_model,
            kernel_size,
            dw_cfg,
            vb.pp("depthwise_conv"),
        )?;

        // LayerNorm (ключ "batch_norm" для совместимости с PyTorch)
        let norm = candle_nn::layer_norm(d_model, 1e-5, vb.pp("batch_norm"))?;

        // Pointwise conv2: (d_model → d_model, kernel=1)
        let pw2_cfg = Conv1dConfig {
            padding: 0,
            stride: 1,
            dilation: 1,
            groups: 1,
            ..Default::default()
        };
        let pointwise_conv2 =
            candle_nn::conv1d(d_model, d_model, 1, pw2_cfg, vb.pp("pointwise_conv2"))?;

        Ok(Self {
            pointwise_conv1,
            depthwise_conv,
            norm,
            pointwise_conv2,
            d_model,
        })
    }

    /// Прямой проход: x (batch, seq, d_model) → (batch, seq, d_model).
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // x: (batch, seq, d_model)

        // Транспонируем для Conv1d: (batch, d_model, seq)
        let h = x.transpose(1, 2)?;

        // Pointwise conv1: (batch, 2*d_model, seq)
        let h = self.pointwise_conv1.forward(&h)?;

        // GLU: разделить по каналам, sigmoid(вторая половина) * первая половина
        let h1 = h.narrow(1, 0, self.d_model)?;
        let h2 = h.narrow(1, self.d_model, self.d_model)?;
        let h = (h1 * candle_nn::ops::sigmoid(&h2)?)?;

        // Depthwise conv: (batch, d_model, seq)
        let h = self.depthwise_conv.forward(&h)?;

        // LayerNorm: нужно транспонировать в (batch, seq, d_model)
        let h = h.transpose(1, 2)?; // (batch, seq, d_model)
        let h = self.norm.forward(&h)?;
        let h = h.transpose(1, 2)?; // обратно (batch, d_model, seq)

        // SiLU
        let h = candle_nn::Activation::Silu.forward(&h)?;

        // Pointwise conv2
        let h = self.pointwise_conv2.forward(&h)?;

        // Обратно в (batch, seq, d_model)
        h.transpose(1, 2)
    }
}

// -----------------------------------------------------------------------
// Conformer Layer (Macaron-style)
// -----------------------------------------------------------------------

/// Один слой Conformer (Macaron-style):
/// FFN1 → Self-Attention → Convolution → FFN2 → LayerNorm
pub struct ConformerLayer {
    norm_feed_forward1: LayerNorm,
    feed_forward1: ConformerFeedForward,
    norm_self_att: LayerNorm,
    self_attn: RotaryMHSA,
    norm_conv: LayerNorm,
    conv: ConformerConvolution,
    norm_feed_forward2: LayerNorm,
    feed_forward2: ConformerFeedForward,
    norm_out: LayerNorm,
}

impl ConformerLayer {
    pub fn load(
        d_model: usize,
        d_ff: usize,
        n_heads: usize,
        conv_kernel_size: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let norm_feed_forward1 = candle_nn::layer_norm(d_model, 1e-5, vb.pp("norm_feed_forward1"))?;
        let feed_forward1 = ConformerFeedForward::load(d_model, d_ff, vb.pp("feed_forward1"))?;
        let norm_self_att = candle_nn::layer_norm(d_model, 1e-5, vb.pp("norm_self_att"))?;
        let self_attn = RotaryMHSA::load(d_model, n_heads, vb.pp("self_attn"))?;
        let norm_conv = candle_nn::layer_norm(d_model, 1e-5, vb.pp("norm_conv"))?;
        let conv = ConformerConvolution::load(d_model, conv_kernel_size, vb.pp("conv"))?;
        let norm_feed_forward2 = candle_nn::layer_norm(d_model, 1e-5, vb.pp("norm_feed_forward2"))?;
        let feed_forward2 = ConformerFeedForward::load(d_model, d_ff, vb.pp("feed_forward2"))?;
        let norm_out = candle_nn::layer_norm(d_model, 1e-5, vb.pp("norm_out"))?;

        Ok(Self {
            norm_feed_forward1,
            feed_forward1,
            norm_self_att,
            self_attn,
            norm_conv,
            conv,
            norm_feed_forward2,
            feed_forward2,
            norm_out,
        })
    }

    /// Прямой проход одного слоя Conformer.
    ///
    /// x: (batch, seq, d_model)
    /// cos_emb, sin_emb: RoPE таблицы (seq, 1, 1, d_k)
    /// att_mask: (batch, seq, seq) или None
    pub fn forward(
        &self,
        x: &Tensor,
        cos_emb: &Tensor,
        sin_emb: &Tensor,
        att_mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        const FC_FACTOR: f64 = 0.5;

        // 1. FFN1 (с фактором 0.5)
        let h = self.norm_feed_forward1.forward(x)?;
        let h = self.feed_forward1.forward(&h)?;
        let residual = (x + (h * FC_FACTOR)?)?;

        // 2. Self-Attention
        let h = self.norm_self_att.forward(&residual)?;
        let h = self.self_attn.forward(&h, cos_emb, sin_emb, att_mask)?;
        let residual = (residual + h)?;

        // 3. Convolution
        let h = self.norm_conv.forward(&residual)?;
        let h = self.conv.forward(&h)?;
        let residual = (residual + h)?;

        // 4. FFN2 (с фактором 0.5)
        let h = self.norm_feed_forward2.forward(&residual)?;
        let h = self.feed_forward2.forward(&h)?;
        let residual = (residual + (h * FC_FACTOR)?)?;

        // 5. Финальная нормализация
        self.norm_out.forward(&residual)
    }
}

// -----------------------------------------------------------------------
// Conformer Encoder
// -----------------------------------------------------------------------

/// Полный Conformer-энкодер GigaAM:
/// Subsampling → Positional Encoding → N × ConformerLayer
pub struct ConformerEncoder {
    pre_encode: StridingSubsampling,
    layers: Vec<ConformerLayer>,
    /// Предвычисленная таблица cos для RoPE.
    rope_cos: Tensor,
    /// Предвычисленная таблица sin для RoPE.
    rope_sin: Tensor,
    /// Размерность одной головы (для пересоздания RoPE).
    d_k: usize,
    /// Количество входных mel-бинов.
    #[allow(dead_code)]
    feat_in: usize,
}

impl ConformerEncoder {
    pub fn load(config: &EncoderConfig, vb: VarBuilder) -> Result<Self> {
        let d_k = config.d_model / config.n_heads;
        let d_ff = config.d_model * config.ff_expansion_factor;

        // Субдискретизация
        let pre_encode = StridingSubsampling::load(
            config.feat_in,
            config.d_model,
            config.subs_kernel_size,
            config.subsampling_factor,
            vb.pp("pre_encode"),
        )?;

        // Создать таблицу RoPE
        let (rope_cos, rope_sin) = create_rope_table(d_k, config.pos_emb_max_len, vb.device())?;
        // Привести к тому же dtype, что и модель
        let rope_cos = rope_cos.to_dtype(vb.dtype())?;
        let rope_sin = rope_sin.to_dtype(vb.dtype())?;

        // Слои Conformer
        let mut layers = Vec::with_capacity(config.n_layers);
        for i in 0..config.n_layers {
            let layer = ConformerLayer::load(
                config.d_model,
                d_ff,
                config.n_heads,
                config.conv_kernel_size,
                vb.pp(format!("layers.{i}")),
            )?;
            layers.push(layer);
        }

        Ok(Self {
            pre_encode,
            layers,
            rope_cos,
            rope_sin,
            d_k,
            feat_in: config.feat_in,
        })
    }

    /// Прямой проход энкодера.
    ///
    /// # Аргументы
    /// * `features` — mel-спектрограмма (batch, feat_in, seq_len)
    ///
    /// # Возвращает
    /// Тензор (batch, d_model, encoded_len) — закодированные фичи.
    pub fn forward(&self, features: &Tensor) -> Result<Tensor> {
        // 1. Субдискретизация: (batch, feat_in, seq) → (batch, seq/4, d_model)
        // Входной features в формате (batch, feat_in, seq_len)
        // Субдискретизация ожидает (batch, seq, feat_in)
        let x = features.transpose(1, 2)?;
        let x = self.pre_encode.forward(&x)?;

        let (_b, t, _d) = x.dims3()?;

        // 2. RoPE — обрезать cos/sin до текущей длины.
        //    Если последовательность длиннее предвычисленной таблицы,
        //    пересоздаём таблицу на лету.
        let (cos_emb, sin_emb) = if t <= self.rope_cos.dim(0)? {
            (
                self.rope_cos.narrow(0, 0, t)?,
                self.rope_sin.narrow(0, 0, t)?,
            )
        } else {
            tracing::warn!(
                "GigaAM: RoPE таблица расширена с {} до {} позиций",
                self.rope_cos.dim(0)?,
                t,
            );
            let (cos, sin) = create_rope_table(self.d_k, t, x.device())?;
            (cos.to_dtype(x.dtype())?, sin.to_dtype(x.dtype())?)
        };

        // 3. Прогоняем через все слои Conformer.
        // Маску не используем (batch_size=1 при инференсе).
        //
        // Metal workaround: каждые SYNC_EVERY слоёв вставляем device.synchronize()
        // для сброса Metal command buffer pool. Это предотвращает накопление
        // слишком большого количества буферов в in-flight состоянии, что может
        // вызвать краш AGXMetalG16X::fillBuffer на M4 / macOS 26.x.
        const SYNC_EVERY: usize = 4;
        let is_metal = x.device().is_metal();

        let mut h = x;
        for (i, layer) in self.layers.iter().enumerate() {
            h = layer.forward(&h, &cos_emb, &sin_emb, None)?;

            if is_metal && (i + 1) % SYNC_EVERY == 0 {
                h.device().synchronize().map_err(|e| {
                    candle_core::Error::Msg(format!("Metal sync at layer {}: {e}", i + 1))
                })?;
            }
        }

        // 4. Выход: (batch, seq/4, d_model) → (batch, d_model, seq/4)
        h.transpose(1, 2)
    }
}
