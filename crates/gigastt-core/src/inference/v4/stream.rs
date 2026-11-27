//! Session-owned, native cached v4 RNNT streaming. No legacy window cursor.

use anyhow::{Context, Result as AnyResult, ensure};
use parking_lot::Mutex;
use std::path::Path;
use std::sync::{
    Arc,
    atomic::{AtomicBool, AtomicUsize, Ordering},
};

use super::{
    V4Frontend, V4Token, V4Tokenizer,
    engine::{
        acquire_request_slot,
        decode::{self, Encoded, GreedyState, f32_output, f32_tensor, i64_tensor, run_session},
        inference_error, model_error,
    },
    group_word_spans_at,
};
use crate::{
    error::GigasttError,
    model::{
        QualifiedV4StreamingPack, StreamingArtifactRole as Role, TranscriptStyle,
        V4_STREAMING_FRONT_CACHE_SHAPES, V4_STREAMING_LAYER_CACHE_SHAPES,
        V4_STREAMING_LEFT_CONTEXT_FRAMES, V4StreamContext,
    },
    runtime::{RuntimeSession, Tensor, TensorData, ort::v4::load_verified_streaming_cpu},
    transcript::WordInfo,
};

const MELS: usize = 64;
const CHANNELS: usize = 768;
const WINDOW: usize = 320;
const HOP: usize = 160;
const SAMPLE_RATE: usize = 16_000;
const FRAME_SECONDS: f64 = 0.04;

/// Native streaming accepts at most one second of canonical PCM per call.
pub const V4_STREAM_MAX_PACKET_SAMPLES: usize = SAMPLE_RATE;
/// Maximum total canonical samples in a session, including its final packet.
pub const V4_STREAM_MAX_SESSION_SAMPLES: usize = 3600 * SAMPLE_RATE;
/// Maximum concurrently owned stream states per loaded engine. Inference
/// calls share one execution slot; the states themselves never share caches.
pub const V4_STREAM_MAX_SESSIONS: usize = 2;

/// Select either independent RNNT head, or both sharing the acoustic steps.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum V4StreamingStyles {
    /// Unformatted raw head.
    Raw,
    /// Formatted end-to-end head.
    E2e,
    /// Both distinct heads, sharing the acoustic computation.
    Both,
}
impl V4StreamingStyles {
    fn raw(self) -> bool {
        matches!(self, Self::Raw | Self::Both)
    }
    fn e2e(self) -> bool {
        matches!(self, Self::E2e | Self::Both)
    }
}
impl From<TranscriptStyle> for V4StreamingStyles {
    fn from(style: TranscriptStyle) -> Self {
        match style {
            TranscriptStyle::Raw => Self::Raw,
            TranscriptStyle::E2e => Self::E2e,
        }
    }
}

/// New output of a selected head. Newly emitted raw tokens may precede text
/// and word publication (which waits for a completed lexical boundary).
#[derive(Debug, Clone)]
pub struct V4StreamingDelta {
    /// Newly committed SentencePiece-decoded text, without any extra normalization.
    pub text: String,
    /// New literal word groups on the absolute 40ms grid, with derived confidence.
    pub words: Vec<WordInfo>,
    /// Newly emitted nonblank IDs and absolute frame offsets, even if their
    /// lexical word is not yet publishable. These tokens are never replayed.
    pub tokens: Vec<V4Token>,
    #[cfg(test)]
    trace: decode::Trace,
}
impl V4StreamingDelta {
    fn empty() -> Self {
        Self {
            text: String::new(),
            words: Vec::new(),
            tokens: Vec::new(),
            #[cfg(test)]
            trace: decode::Trace::default(),
        }
    }
}

/// One atomic native streaming packet result. An error returns no head delta.
#[derive(Debug)]
pub struct V4StreamingPacket {
    /// Raw output, absent unless explicitly selected.
    pub raw: Option<V4StreamingDelta>,
    /// Formatted output, absent unless explicitly selected.
    pub e2e: Option<V4StreamingDelta>,
    /// Total number of accepted canonical samples in this stream.
    pub samples: usize,
    /// Absolute number of valid encoder frames already decoded.
    pub encoded_frames: usize,
    /// True after explicit final packet/flush (including finished no-ops).
    pub finished: bool,
    #[cfg(test)]
    trace: PacketTrace,
}

struct Head {
    tokenizer: V4Tokenizer,
    predictor: Box<dyn RuntimeSession>,
    joiner: Box<dyn RuntimeSession>,
}
impl Head {
    fn load(pack: &QualifiedV4StreamingPack, style: TranscriptStyle) -> AnyResult<Self> {
        let (predictor, joiner) = match style {
            TranscriptStyle::Raw => (Role::RawDecoder, Role::RawJoint),
            TranscriptStyle::E2e => (Role::E2eDecoder, Role::E2eJoint),
        };
        Ok(Self {
            tokenizer: V4Tokenizer::from_bytes(style, &pack.read_tokenizer_bytes(style)?)?,
            predictor: load_verified_streaming_cpu(pack.read_graph_bytes(predictor)?)?,
            joiner: load_verified_streaming_cpu(pack.read_graph_bytes(joiner)?)?,
        })
    }
}

#[derive(Default)]
struct HeadState {
    greedy: GreedyState,
    tokens: Vec<V4Token>,
    published: String,
    published_words: usize,
}
impl HeadState {
    fn advance(
        &mut self,
        head: &Head,
        encoded: Option<&Encoded>,
        samples: usize,
        final_packet: bool,
        cancel: Option<&AtomicBool>,
    ) -> Result<V4StreamingDelta, GigasttError> {
        let mut delta = V4StreamingDelta::empty();
        if let Some(encoded) = encoded {
            let trajectory = decode::greedy_block(
                head.predictor.as_ref(),
                head.joiner.as_ref(),
                encoded,
                &mut self.greedy,
                cancel,
            )?;
            self.tokens.extend_from_slice(&trajectory.tokens);
            delta.tokens = trajectory.tokens;
            #[cfg(test)]
            {
                delta.trace = trajectory.trace;
            }
        }
        // The publishable prefix depends only on tokens, not packet count.
        // Avoid re-decoding an unchanged history on blank-only packets.
        if delta.tokens.is_empty() && !final_packet {
            return Ok(delta);
        }
        let mut count = self.tokens.len();
        if !final_packet {
            count = 0;
            for (index, token) in self.tokens.iter().enumerate().rev() {
                let piece = head
                    .tokenizer
                    .id_to_piece(token.id)
                    .map_err(inference_error)?;
                if piece == " " || piece.starts_with('▁') {
                    count = if piece == " " { index + 1 } else { index };
                    break;
                }
            }
        }
        let tokens = &self.tokens[..count];
        let ids: Vec<_> = tokens.iter().map(|token| token.id).collect();
        let frames: Vec<_> = tokens.iter().map(|token| token.frame).collect();
        let text = head.tokenizer.decode(&ids).map_err(inference_error)?;
        // The upstream helper computes a character-count suffix; a rewrite
        // of already published text is not a representable append-only delta.
        delta.text = text
            .strip_prefix(&self.published)
            .context("v4 streaming publication rewrites an earlier prefix")
            .map_err(inference_error)?
            .to_owned();
        let pieces = ids
            .iter()
            .map(|&id| head.tokenizer.id_to_piece(id))
            .collect::<AnyResult<Vec<_>>>()
            .map_err(inference_error)?;
        let groups = group_word_spans_at(pieces.into_iter(), &frames, |frame| {
            (frame < self.greedy.frame_offset).then_some((
                frame as f64 * FRAME_SECONDS,
                (frame + 1) as f64 * FRAME_SECONDS,
            ))
        })
        .map_err(inference_error)?;
        if groups.len() < self.published_words {
            return Err(inference_error(anyhow::anyhow!(
                "v4 streaming publication removes completed words"
            )));
        }
        for (word, span) in &groups[self.published_words..] {
            let confidence = (tokens[span.clone()]
                .iter()
                .map(|t| t.confidence)
                .sum::<f64>()
                / span.len() as f64) as f32;
            let end = if final_packet {
                word.end.min(samples as f64 / SAMPLE_RATE as f64)
            } else {
                word.end
            };
            delta.words.push(WordInfo::new(
                word.text.clone(),
                word.start,
                end,
                confidence,
                None,
            ));
        }
        self.published = text;
        self.published_words = groups.len();
        Ok(delta)
    }
}

// Test doubles exercise the same state and slot ownership without loading MKL
// or models. The concrete production closure is a verified V4Frontend.
type MelComputer = dyn Fn(&[f32]) -> AnyResult<Vec<f32>> + Send + Sync;

/// Loaded verified cached-streaming FP32 CPU graph group. This is distinct
/// from the full-context V4Engine and the legacy streaming engine.
pub struct V4StreamingEngine {
    mel: Box<MelComputer>,
    front: Box<dyn RuntimeSession>,
    step: Box<dyn RuntimeSession>,
    raw: Head,
    e2e: Head,
    request: Mutex<()>,
    active: AtomicUsize,
    pack_identity: String,
}
impl V4StreamingEngine {
    /// Load a side-loaded verified streaming pack and explicit local MKL runtime.
    pub fn load(pack_dir: &Path, mkl_runtime_dir: &Path) -> Result<Self, GigasttError> {
        let pack = QualifiedV4StreamingPack::load(pack_dir)
            .map_err(|e| model_error(pack_dir.display().to_string(), e))?;
        Self::from_pack(&pack, mkl_runtime_dir)
    }
    /// Load verified graph/tokenizer bytes from a validated streaming pack.
    pub fn from_pack(
        pack: &QualifiedV4StreamingPack,
        mkl_runtime_dir: &Path,
    ) -> Result<Self, GigasttError> {
        let load = || -> AnyResult<Self> {
            let frontend = V4Frontend::from_streaming_pack(pack, mkl_runtime_dir)?;
            Ok(Self {
                mel: Box::new(move |pcm| Ok(frontend.compute(pcm)?.values)),
                front: load_verified_streaming_cpu(pack.read_graph_bytes(Role::Frontend)?)?,
                step: load_verified_streaming_cpu(pack.read_graph_bytes(Role::Step)?)?,
                e2e: Head::load(pack, TranscriptStyle::E2e)?,
                raw: Head::load(pack, TranscriptStyle::Raw)?,
                request: Mutex::new(()),
                active: AtomicUsize::new(0),
                pack_identity: pack.identity().to_owned(),
            })
        };
        load().map_err(|e| {
            model_error(
                format!("qualified v4 streaming pack {}", pack.identity()),
                e,
            )
        })
    }
    /// Content identity of the pinned graph group and metadata.
    pub fn pack_identity(&self) -> &str {
        &self.pack_identity
    }
    /// Reserve ownership of one independent state. No graph call is made.
    /// Drop the session to release its reservation (including failed/finished states).
    pub fn start_session(
        self: &Arc<Self>,
        context: V4StreamContext,
        styles: V4StreamingStyles,
    ) -> Result<V4StreamingSession, GigasttError> {
        self.active
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |count| {
                (count < V4_STREAM_MAX_SESSIONS).then_some(count + 1)
            })
            .map_err(|_| GigasttError::InvalidInput {
                message: "v4 streaming session limit reached".into(),
            })?;
        let mut session = V4StreamingSession {
            engine: self.clone(),
            context,
            styles,
            state: None,
        };
        // On failure Drop rolls back the reserved owner slot.
        session.reset(context)?;
        Ok(session)
    }
}

/// One session owns both the acoustic caches and selected head histories.
/// A preflight/checkout error leaves the state untouched. An error after
/// entering packet processing discards the live state and requires reset;
/// no partial packet result can escape. Native calls cannot be preempted.
pub struct V4StreamingSession {
    engine: Arc<V4StreamingEngine>,
    context: V4StreamContext,
    styles: V4StreamingStyles,
    state: Option<State>,
}
impl Drop for V4StreamingSession {
    fn drop(&mut self) {
        // Release state memory before admitting a replacement session.
        self.state = None;
        self.engine.active.fetch_sub(1, Ordering::AcqRel);
    }
}
impl V4StreamingSession {
    /// The explicit currently configured context.
    pub fn context(&self) -> V4StreamContext {
        self.context
    }
    /// The fixed head selection for this session.
    pub fn styles(&self) -> V4StreamingStyles {
        self.styles
    }
    /// Discard all PCM/mel, cache and head history; context is explicit.
    pub fn reset(&mut self, context: V4StreamContext) -> Result<(), GigasttError> {
        self.state = None;
        self.context = context;
        self.state = Some(State::new(self.styles)?);
        Ok(())
    }
    /// Submit one canonical finite 16 kHz mono f32 packet. No hidden padding,
    /// resampling or splitting occurs. EOF pads only pending complete mel
    /// frames to the context block, not PCM. Finished sessions ignore all
    /// later input until explicit reset (including repeated flush).
    pub fn push_pcm(
        &mut self,
        pcm: &[f32],
        final_packet: bool,
        cancel: Option<&AtomicBool>,
    ) -> Result<V4StreamingPacket, GigasttError> {
        let state = self
            .state
            .as_mut()
            .ok_or_else(|| GigasttError::InvalidInput {
                message: "v4 streaming session failed; reset required".into(),
            })?;
        if state.finished {
            return Ok(state.empty_packet());
        }
        decode::check_cancel(cancel)?;
        if pcm.len() > V4_STREAM_MAX_PACKET_SAMPLES || pcm.iter().any(|x| !x.is_finite()) {
            return Err(GigasttError::InvalidAudio {
                reason: "v4 streaming packet must be finite mono 16 kHz PCM, at most 16000 samples"
                    .into(),
            });
        }
        let samples = state
            .samples
            .checked_add(pcm.len())
            .filter(|&n| n <= V4_STREAM_MAX_SESSION_SAMPLES)
            .ok_or_else(|| GigasttError::AudioTooLong {
                observed_secs: (state.samples as f64 + pcm.len() as f64) / SAMPLE_RATE as f64,
                limit_secs: V4_STREAM_MAX_SESSION_SAMPLES as f64 / SAMPLE_RATE as f64,
            })?;
        let slot = acquire_request_slot(&self.engine.request, cancel)?;
        let result = state.process(
            &self.engine,
            self.context,
            pcm,
            samples,
            final_packet,
            cancel,
        );
        if result.is_err() {
            self.state = None;
        }
        drop(slot);
        result
    }
    /// Explicit EOF without inventing audio samples.
    pub fn flush(
        &mut self,
        cancel: Option<&AtomicBool>,
    ) -> Result<V4StreamingPacket, GigasttError> {
        self.push_pcm(&[], true, cancel)
    }
}

fn zeros(shapes: &[&[usize]]) -> Result<Vec<Tensor>, GigasttError> {
    shapes
        .iter()
        .map(|shape| f32_tensor(shape, vec![0.0; shape.iter().product()]))
        .collect()
}

#[derive(Default, Debug)]
#[cfg(test)]
struct PacketTrace {
    feature_input: Vec<f32>,
    new_mel: Vec<f32>,
    ready_mel: Vec<f32>,
    padded_mel: Vec<f32>,
    pre_encoded: Vec<f32>,
    encoded_physical: Vec<f32>,
    encoded_valid: Vec<f32>,
    valid_mel: usize,
    physical_mel: usize,
    valid_encoded: usize,
    physical_encoded: usize,
}

struct State {
    samples: usize,
    finished: bool,
    wave: Vec<f32>,
    pending: Vec<f32>,
    position: usize,
    cache_valid: usize,
    front: Vec<Tensor>,
    layers: Vec<Tensor>,
    raw: Option<HeadState>,
    e2e: Option<HeadState>,
}
impl State {
    fn new(styles: V4StreamingStyles) -> Result<Self, GigasttError> {
        Ok(Self {
            samples: 0,
            finished: false,
            wave: Vec::new(),
            pending: Vec::new(),
            position: 0,
            cache_valid: 0,
            front: zeros(V4_STREAMING_FRONT_CACHE_SHAPES)?,
            layers: zeros(V4_STREAMING_LAYER_CACHE_SHAPES)?,
            raw: styles.raw().then(HeadState::default),
            e2e: styles.e2e().then(HeadState::default),
        })
    }
    fn empty_packet(&self) -> V4StreamingPacket {
        V4StreamingPacket {
            raw: self.raw.as_ref().map(|_| V4StreamingDelta::empty()),
            e2e: self.e2e.as_ref().map(|_| V4StreamingDelta::empty()),
            samples: self.samples,
            encoded_frames: self.position,
            finished: self.finished,
            #[cfg(test)]
            trace: PacketTrace::default(),
        }
    }
    fn process(
        &mut self,
        engine: &V4StreamingEngine,
        context: V4StreamContext,
        pcm: &[f32],
        samples: usize,
        final_packet: bool,
        cancel: Option<&AtomicBool>,
    ) -> Result<V4StreamingPacket, GigasttError> {
        #[cfg(test)]
        let mut trace = PacketTrace::default();
        decode::check_cancel(cancel)?;
        self.wave.extend_from_slice(pcm);
        let frames = complete_mel_frames(self.wave.len());
        let new_mel = if frames == 0 {
            Vec::new()
        } else {
            let count = (frames - 1) * HOP + WINDOW;
            #[cfg(test)]
            {
                trace.feature_input = self.wave[..count].to_vec();
            }
            let mel = (engine.mel)(&self.wave[..count]);
            decode::check_cancel(cancel)?;
            let mel = mel.map_err(inference_error)?;
            if mel.len() != MELS * frames || mel.iter().any(|v| !v.is_finite()) {
                return Err(inference_error(anyhow::anyhow!(
                    "invalid v4 streaming mel tensor"
                )));
            }
            self.wave.drain(..frames * HOP);
            mel
        };
        #[cfg(test)]
        {
            trace.new_mel = new_mel.clone();
        }
        let ready = take_mel(
            &mut self.pending,
            new_mel,
            context.mel_block(),
            final_packet,
        );
        let encoded = if ready.valid == 0 {
            None
        } else {
            #[cfg(test)]
            {
                trace.valid_mel = ready.valid;
                trace.physical_mel = ready.physical;
                trace.ready_mel = ready
                    .values
                    .chunks_exact(ready.physical)
                    .flat_map(|row| row[..ready.valid].iter().copied())
                    .collect();
                trace.padded_mel = ready.values.clone();
            }
            let encoded = self.encode(
                engine,
                context,
                ready,
                cancel,
                #[cfg(test)]
                &mut trace,
            )?;
            Some(encoded)
        };
        // Match the official saved head order. Neither head is returned if
        // the other selected head fails or cancellation arrives.
        let e2e = self
            .e2e
            .as_mut()
            .map(|state| {
                state.advance(&engine.e2e, encoded.as_ref(), samples, final_packet, cancel)
            })
            .transpose()?;
        let raw = self
            .raw
            .as_mut()
            .map(|state| {
                state.advance(&engine.raw, encoded.as_ref(), samples, final_packet, cancel)
            })
            .transpose()?;
        decode::check_cancel(cancel)?;
        self.samples = samples;
        self.finished = final_packet;
        Ok(V4StreamingPacket {
            raw,
            e2e,
            samples,
            encoded_frames: self.position,
            finished: final_packet,
            #[cfg(test)]
            trace,
        })
    }
    fn encode(
        &mut self,
        engine: &V4StreamingEngine,
        context: V4StreamContext,
        ready: ReadyMel,
        cancel: Option<&AtomicBool>,
        #[cfg(test)] trace: &mut PacketTrace,
    ) -> Result<Encoded, GigasttError> {
        let valid = ready.valid.div_ceil(4);
        let physical = ready.physical / 4;
        let mut front_inputs = vec![
            f32_tensor(&[1, MELS, ready.physical], ready.values)?,
            scalar(ready.valid)?,
        ];
        front_inputs.append(&mut self.front);
        let outputs = run_session(engine.front.as_ref(), &front_inputs, cancel)?;
        let (pre_encoded, next_front) = acoustic_outputs(
            outputs,
            &[1, physical, CHANNELS],
            valid,
            V4_STREAMING_FRONT_CACHE_SHAPES,
            None,
        )
        .map_err(inference_error)?;
        drop(front_inputs);
        self.front = next_front;
        #[cfg(test)]
        {
            trace.pre_encoded = pre_encoded.clone();
        }
        let next_position = self.position + valid;
        let next_valid = (self.cache_valid + valid).min(V4_STREAMING_LEFT_CONTEXT_FRAMES);
        let mut step_inputs = vec![
            f32_tensor(&[1, physical, CHANNELS], pre_encoded)?,
            scalar(valid)?,
            scalar(context.context_frames())?,
            scalar(self.position)?,
            scalar(self.cache_valid)?,
        ];
        step_inputs.append(&mut self.layers);
        let outputs = run_session(engine.step.as_ref(), &step_inputs, cancel)?;
        let (data, next_layers) = acoustic_outputs(
            outputs,
            &[1, CHANNELS, physical],
            valid,
            V4_STREAMING_LAYER_CACHE_SHAPES,
            Some((next_position, next_valid)),
        )
        .map_err(inference_error)?;
        self.layers = next_layers;
        self.position = next_position;
        self.cache_valid = next_valid;
        #[cfg(test)]
        {
            trace.encoded_physical = data.clone();
            trace.valid_encoded = valid;
            trace.physical_encoded = physical;
        }
        let encoded = Encoded {
            data,
            physical,
            valid,
        }
        .valid_prefix();
        #[cfg(test)]
        {
            trace.encoded_valid = encoded.data.clone();
        }
        Ok(encoded)
    }
}

fn scalar(value: usize) -> Result<Tensor, GigasttError> {
    i64_tensor(
        &[1],
        vec![i64::try_from(value).map_err(|e| inference_error(e.into()))?],
    )
}
fn expect_scalar(tensor: Tensor, expected: usize) -> AnyResult<()> {
    ensure!(
        tensor.shape().dims() == [1],
        "invalid v4 streaming scalar shape"
    );
    let TensorData::I64(values) = tensor.into_data() else {
        anyhow::bail!("invalid v4 streaming scalar type");
    };
    ensure!(
        values == [i64::try_from(expected)?],
        "v4 streaming scalar mismatch"
    );
    Ok(())
}
fn acoustic_outputs(
    outputs: Vec<Tensor>,
    shape: &[usize],
    valid: usize,
    cache_shapes: &[&[usize]],
    step: Option<(usize, usize)>,
) -> AnyResult<(Vec<f32>, Vec<Tensor>)> {
    ensure!(
        outputs.len() == 2 + cache_shapes.len() + if step.is_some() { 2 } else { 0 },
        "v4 streaming output count mismatch"
    );
    let mut outputs = outputs.into_iter();
    let data = f32_output(
        outputs.next().context("missing v4 acoustic output")?,
        shape,
        "v4 streaming acoustic output",
    )?;
    expect_scalar(outputs.next().context("missing v4 encoded length")?, valid)?;
    if let Some((position, cache_valid)) = step {
        expect_scalar(
            outputs.next().context("missing v4 next_position")?,
            position,
        )?;
        expect_scalar(
            outputs.next().context("missing v4 next_cache_valid")?,
            cache_valid,
        )?;
    }
    let mut caches = Vec::with_capacity(cache_shapes.len());
    for (tensor, shape) in outputs.zip(cache_shapes) {
        ensure!(
            tensor.shape().dims() == *shape
                && tensor
                    .view()
                    .data()
                    .as_f32()
                    .is_some_and(|vals| vals.iter().all(|x| x.is_finite())),
            "invalid v4 streaming cache output"
        );
        caches.push(tensor);
    }
    Ok((data, caches))
}
fn complete_mel_frames(samples: usize) -> usize {
    if samples < WINDOW {
        0
    } else {
        (samples - WINDOW) / HOP + 1
    }
}
struct ReadyMel {
    values: Vec<f32>,
    valid: usize,
    physical: usize,
}
fn take_mel(
    pending: &mut Vec<f32>,
    new_mel: Vec<f32>,
    block: usize,
    final_packet: bool,
) -> ReadyMel {
    let prior = pending.len() / MELS;
    let new = new_mel.len() / MELS;
    let total = prior + new;
    let valid = if final_packet {
        total
    } else {
        total / block * block
    };
    let physical = valid.div_ceil(block) * block;
    let mut ready = Vec::with_capacity(MELS * physical);
    let mut carry = Vec::with_capacity(MELS * (total - valid));
    for m in 0..MELS {
        let combined = pending[m * prior..(m + 1) * prior]
            .iter()
            .chain(&new_mel[m * new..(m + 1) * new])
            .copied();
        ready.extend(combined.clone().take(valid));
        ready.resize(ready.len() + physical - valid, 0.0);
        carry.extend(combined.skip(valid));
    }
    *pending = carry;
    ReadyMel {
        values: ready,
        valid,
        physical,
    }
}

#[cfg(all(test, not(miri)))]
mod oracle;
#[cfg(all(test, not(miri)))]
mod tests;
