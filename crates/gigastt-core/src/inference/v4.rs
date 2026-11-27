//! SentencePiece, timestamp and optional MKL frontend contracts for the qualified v4 ABI.
//!
//! These helpers consume a verified pack, not a precision flag or a legacy
//! variant. They do not create inference sessions or change the legacy frontend.

use anyhow::{Context, Result, ensure};
use sentencepiece_rs::{ModelType, PieceType, SentencePieceProcessor};

use crate::model::{QualifiedV4Pack, TranscriptStyle};

#[cfg(all(
    feature = "v4-mkl",
    target_os = "linux",
    target_arch = "x86_64",
    target_env = "gnu"
))]
mod frontend;
#[cfg(all(
    feature = "v4-mkl",
    target_os = "linux",
    target_arch = "x86_64",
    target_env = "gnu"
))]
pub use frontend::{V4Features, V4Frontend};

const WINDOW_SAMPLES: usize = 320;
const HOP_SAMPLES: usize = 160;
const SUBSAMPLING: usize = 4;
const SAMPLE_RATE: f64 = 16_000.0;
const PIECES: usize = 1024;

/// Physical and valid geometry of a batch-one, center=false v4 input.
///
/// Encoder lengths are expected from the pinned author encoder's subsampling;
/// the future inference runtime must verify the returned encoder length.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct V4FrameGrid {
    physical_samples: usize,
    valid_samples: usize,
}

impl V4FrameGrid {
    /// Construct a grid from actual 16 kHz samples and a valid prefix length.
    /// The author STFT rejects buffers shorter than one full window. A zero
    /// valid mel sequence is not an executable utterance in this profile.
    pub fn new(physical_samples: usize, valid_samples: usize) -> Result<Self> {
        ensure!(
            valid_samples >= WINDOW_SAMPLES && valid_samples <= physical_samples,
            "v4 requires at least one complete valid 320-sample window"
        );
        Ok(Self {
            physical_samples,
            valid_samples,
        })
    }

    fn frame_count(samples: usize) -> usize {
        (samples - WINDOW_SAMPLES) / HOP_SAMPLES + 1
    }

    /// Actual sample count, including any explicit padding supplied by the caller.
    pub fn physical_samples(&self) -> usize {
        self.physical_samples
    }

    /// Number of valid 16 kHz samples, excluding explicit padding.
    pub fn valid_samples(&self) -> usize {
        self.valid_samples
    }

    /// Physical time dimension of the log-mel tensor.
    pub fn mel_frames(&self) -> usize {
        Self::frame_count(self.physical_samples)
    }

    /// Valid mel length passed to the encoder.
    pub fn valid_mel_frames(&self) -> usize {
        Self::frame_count(self.valid_samples)
    }

    /// Expected physical encoder time dimension after causal subsampling.
    pub fn encoder_frames(&self) -> usize {
        self.mel_frames().div_ceil(SUBSAMPLING)
    }

    /// Expected valid encoder length after causal subsampling.
    pub fn valid_encoder_frames(&self) -> usize {
        self.valid_mel_frames().div_ceil(SUBSAMPLING)
    }

    /// Start sample of a physical center=false mel frame.
    pub fn mel_frame_start_sample(&self, frame: usize) -> Option<usize> {
        (frame < self.mel_frames()).then(|| frame * HOP_SAMPLES)
    }

    /// Author full-context timestamp step, using the valid utterance duration
    /// divided by the valid encoder length, not the nominal fixed 40 ms grid.
    pub fn frame_shift_seconds(&self) -> f64 {
        self.valid_samples as f64 / SAMPLE_RATE / self.valid_encoder_frames() as f64
    }

    /// Inclusive frame start and exclusive frame end relative to the utterance.
    pub fn frame_bounds(&self, frame: usize) -> Option<(f64, f64)> {
        if frame >= self.valid_encoder_frames() {
            return None;
        }
        let shift = self.frame_shift_seconds();
        Some((frame as f64 * shift, (frame + 1) as f64 * shift))
    }
}

/// Author-style word timing derived from literal SentencePiece strings.
/// No word confidence is fabricated by the tokenizer.
#[derive(Debug, Clone, PartialEq)]
pub struct V4WordTiming {
    /// Literal grouped pieces (which may include `<unk>` even if full decode
    /// uses the model-specific unknown surface).
    pub text: String,
    /// First constituent token frame, in seconds.
    pub start: f64,
    /// End of the last constituent token frame, in seconds.
    pub end: f64,
}

/// Tokenizer of exactly one verified v4 transcript style.
///
/// The appended RNNT blank belongs to the decoder, not the 1024-piece model.
/// The SentencePiece implementation is qualified for these exact two models;
/// this is not a generic arbitrary-model tokenizer API.
pub struct V4Tokenizer {
    processor: SentencePieceProcessor,
    style: TranscriptStyle,
}

impl V4Tokenizer {
    /// Load the selected tokenizer from the verified pack. The parser consumes
    /// the same bytes that are rehashed at this boundary, not a second open.
    pub fn from_pack(pack: &QualifiedV4Pack, style: TranscriptStyle) -> Result<Self> {
        Self::from_bytes(style, &pack.read_tokenizer_bytes(style)?)
    }

    fn from_bytes(style: TranscriptStyle, bytes: &[u8]) -> Result<Self> {
        let processor = SentencePieceProcessor::from_serialized_model(bytes)
            .context("invalid v4 SentencePiece model")?;
        let model = processor.model();
        ensure!(
            model.model_type() == ModelType::Bpe && model.vocab_size() == PIECES,
            "unexpected v4 SentencePiece model type or size"
        );
        ensure!(
            processor.unk_id() == 0
                && processor.bos_id().is_none()
                && processor.eos_id().is_none()
                && processor.pad_id().is_none(),
            "unexpected v4 special token IDs"
        );
        ensure!(
            model.pieces().iter().enumerate().all(|(id, piece)| {
                piece.kind
                    == if id == 0 {
                        PieceType::Unknown
                    } else {
                        PieceType::Normal
                    }
            }),
            "unexpected control or byte pieces in v4 SentencePiece vocabulary"
        );
        Ok(Self { processor, style })
    }

    /// Selected independent transcript style.
    pub fn style(&self) -> TranscriptStyle {
        self.style
    }

    /// Number of actual SentencePiece entries, excluding RNNT blank.
    pub fn piece_count(&self) -> usize {
        PIECES
    }

    /// Appended RNNT blank label, outside the SentencePiece vocabulary.
    pub fn blank_id(&self) -> usize {
        PIECES
    }

    /// Unknown SentencePiece ID (not blank).
    pub fn unknown_id(&self) -> usize {
        self.processor.unk_id()
    }

    /// Apply the exact embedded identity normalization and BPE score merging.
    pub fn encode(&self, text: &str) -> Result<Vec<usize>> {
        self.processor
            .encode_to_ids(text)
            .context("v4 SentencePiece encode failed")
    }

    /// Decode a sequence of nonblank token IDs with the selected model's
    /// unknown-token and whitespace semantics. Blank/out-of-range is an error.
    pub fn decode(&self, ids: &[usize]) -> Result<String> {
        ensure!(
            ids.iter().all(|&id| id < PIECES),
            "v4 token ID outside SentencePiece vocabulary"
        );
        self.processor
            .decode_ids(ids)
            .context("v4 SentencePiece decode failed")
    }

    /// Exact literal SentencePiece name; this is not singleton decoded text.
    pub fn id_to_piece(&self, id: usize) -> Result<&str> {
        ensure!(id < PIECES, "v4 token ID outside SentencePiece vocabulary");
        self.processor
            .id_to_piece(id)
            .context("v4 SentencePiece lookup failed")
    }

    /// Lookup a literal piece, returning the official unknown ID for missing pieces.
    pub fn piece_to_id(&self, piece: &str) -> usize {
        self.processor.piece_to_id(piece)
    }

    /// Group token IDs and their monotone absolute encoder-frame indices on
    /// the author full-context utterance grid. This helper is deliberately
    /// distinct from `decode`: author word grouping uses literal pieces.
    pub fn words(
        &self,
        ids: &[usize],
        frames: &[usize],
        grid: &V4FrameGrid,
    ) -> Result<Vec<V4WordTiming>> {
        let pieces = ids
            .iter()
            .map(|&id| self.id_to_piece(id))
            .collect::<Result<Vec<_>>>()?;
        group_words(pieces.into_iter(), frames, grid)
    }
}

fn group_words<'a>(
    mut pieces: impl Iterator<Item = &'a str>,
    frames: &[usize],
    grid: &V4FrameGrid,
) -> Result<Vec<V4WordTiming>> {
    ensure!(
        frames.windows(2).all(|pair| pair[0] <= pair[1]),
        "nonmonotone v4 token frames"
    );
    let mut words = Vec::new();
    let mut current = String::new();
    let mut span: Option<(f64, f64)> = None;
    let commit =
        |words: &mut Vec<V4WordTiming>, current: &mut String, span: &mut Option<(f64, f64)>| {
            let text = current.trim();
            if let Some((start, end)) = span.take()
                && !text.is_empty()
            {
                words.push(V4WordTiming {
                    text: text.to_owned(),
                    start,
                    end,
                });
            }
            current.clear();
        };
    for &frame in frames {
        let piece = pieces.next().context("v4 token/frame count mismatch")?;
        let (start, end) = grid
            .frame_bounds(frame)
            .context("v4 token frame outside valid encoder length")?;
        let piece = if let Some(remainder) = piece.strip_prefix('▁') {
            commit(&mut words, &mut current, &mut span);
            remainder
        } else if piece == " " {
            commit(&mut words, &mut current, &mut span);
            continue;
        } else {
            piece
        };
        current.push_str(piece);
        if let Some((_, previous_end)) = &mut span {
            *previous_end = end;
        } else {
            span = Some((start, end));
        }
    }
    ensure!(pieces.next().is_none(), "v4 token/frame count mismatch");
    commit(&mut words, &mut current, &mut span);
    Ok(words)
}

#[cfg(all(test, not(miri)))]
mod tests;
