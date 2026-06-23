// Vendored from askidmobile/RustASR (crates/model-gigaam, commit 33060b8d),
// dual-licensed MIT OR Apache-2.0. Copyright (c) the RustASR authors.
// Adapted for gigastt: import paths changed; compiled against upstream candle 0.9;
// CTC head not used here (only the v3 Conformer encoder).
#![allow(dead_code)] // wired into a RuntimeSession in a later task

//! Конфигурация GigaAM v3 E2E CTC.

use serde::{Deserialize, Serialize};

/// Конфигурация модели GigaAM.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GigaAmConfig {
    /// Название модели (например, "gigaam-v3-e2e-ctc").
    pub model_name: String,

    /// Тип декодера: "ctc" или "rnnt".
    pub model_class: String,

    /// Частота дискретизации аудио (16000).
    pub sample_rate: usize,

    /// Конфигурация препроцессора (mel-спектрограмма).
    pub preprocessor: PreprocessorConfig,

    /// Конфигурация Conformer-энкодера.
    pub encoder: EncoderConfig,

    /// Конфигурация CTC-головы.
    pub head: HeadConfig,
}

/// Конфигурация mel-спектрограммы.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PreprocessorConfig {
    /// Частота дискретизации (16000).
    pub sample_rate: usize,

    /// Количество mel-бинов (64).
    pub features: usize,

    /// Длина окна (320).
    pub win_length: usize,

    /// Шаг между фреймами (160).
    pub hop_length: usize,

    /// Размер FFT (320).
    pub n_fft: usize,

    /// Шкала mel-фильтров ("htk").
    pub mel_scale: String,

    /// Центрирование STFT (false для GigaAM).
    #[serde(default)]
    pub center: bool,
}

/// Конфигурация Conformer-энкодера.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EncoderConfig {
    /// Количество входных mel-бинов (64).
    pub feat_in: usize,

    /// Количество слоёв Conformer (16).
    pub n_layers: usize,

    /// Размерность модели (768).
    pub d_model: usize,

    /// Тип субдискретизации: "conv1d" или "conv2d".
    pub subsampling: String,

    /// Размер ядра свёртки субдискретизации (5).
    pub subs_kernel_size: usize,

    /// Фактор субдискретизации (4).
    pub subsampling_factor: usize,

    /// Фактор расширения feed-forward (4).
    pub ff_expansion_factor: usize,

    /// Тип позиционного кодирования: "rotary" или "rel_pos".
    pub self_attention_model: String,

    /// Максимальная длина позиционного кодирования (5000).
    pub pos_emb_max_len: usize,

    /// Количество голов внимания (16).
    pub n_heads: usize,

    /// Размер ядра свёртки в Conformer-блоке (5).
    pub conv_kernel_size: usize,

    /// Тип нормализации свёртки: "layer_norm" или "batch_norm".
    pub conv_norm_type: String,
}

/// Конфигурация CTC-головы.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HeadConfig {
    /// Входная размерность (d_model = 768).
    pub feat_in: usize,

    /// Количество классов (257 для v3_e2e_ctc: 256 токенов + blank).
    pub num_classes: usize,
}

impl GigaAmConfig {
    /// Конфигурация по умолчанию для GigaAM v3 E2E CTC.
    pub fn v3_e2e_ctc() -> Self {
        Self {
            model_name: "gigaam-v3-e2e-ctc".to_string(),
            model_class: "ctc".to_string(),
            sample_rate: 16000,
            preprocessor: PreprocessorConfig {
                sample_rate: 16000,
                features: 64,
                win_length: 320,
                hop_length: 160,
                n_fft: 320,
                mel_scale: "htk".to_string(),
                center: false,
            },
            encoder: EncoderConfig {
                feat_in: 64,
                n_layers: 16,
                d_model: 768,
                subsampling: "conv1d".to_string(),
                subs_kernel_size: 5,
                subsampling_factor: 4,
                ff_expansion_factor: 4,
                self_attention_model: "rotary".to_string(),
                pos_emb_max_len: 5000,
                n_heads: 16,
                conv_kernel_size: 5,
                conv_norm_type: "layer_norm".to_string(),
            },
            head: HeadConfig {
                feat_in: 768,
                num_classes: 257,
            },
        }
    }

    /// d_k — размерность на одну голову внимания.
    pub fn d_k(&self) -> usize {
        self.encoder.d_model / self.encoder.n_heads
    }

    /// Размерность feed-forward слоя.
    pub fn d_ff(&self) -> usize {
        self.encoder.d_model * self.encoder.ff_expansion_factor
    }
}

impl EncoderConfig {
    /// GigaAM v3 encoder config (shared across CTC/rnnt heads).
    pub fn v3_rnnt() -> Self {
        Self {
            feat_in: 64,
            n_layers: 16,
            d_model: 768,
            subsampling: "conv1d".to_string(),
            subs_kernel_size: 5,
            subsampling_factor: 4,
            ff_expansion_factor: 4,
            self_attention_model: "rotary".to_string(),
            pos_emb_max_len: 5000,
            n_heads: 16,
            conv_kernel_size: 5,
            conv_norm_type: "layer_norm".to_string(),
        }
    }
}
