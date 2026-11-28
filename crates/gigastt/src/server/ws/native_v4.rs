//! Explicit native cached v4 WebSocket implementation, separate from v3.
use std::{
    path::Path,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    time::Duration,
};

use axum::{
    Json,
    extract::{
        Query,
        ws::{CloseFrame, Message, WebSocket, WebSocketUpgrade},
    },
    http::{HeaderMap, StatusCode, Uri},
    response::{IntoResponse, Response},
};
use futures_util::{Sink, SinkExt, StreamExt};
use gigastt_core::{
    error::GigasttError,
    inference::v4::{
        V4_STREAM_MAX_PACKET_SAMPLES, V4_STREAM_MAX_SESSIONS, V4StreamingEngine, V4StreamingPacket,
        V4StreamingSession,
    },
    model::{TranscriptStyle, V4_STREAMING_CONTEXTS_MS, V4StreamContext},
    protocol::{ServerMessage, native_v4 as wire},
    transcript::WordInfo,
};
use serde::Deserialize;
use serde_json::json;
use tokio::sync::{OwnedSemaphorePermit, Semaphore};
use tokio_util::{sync::CancellationToken, task::TaskTracker};
use tracing::Instrument;

use super::{incoming::Incoming, native_protocol_requested, session_deadline};
use crate::server::{RuntimeLimits, config::pool_retry_after_ms, json_text};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Selection {
    context: V4StreamContext,
    style: TranscriptStyle,
}
#[derive(Deserialize, Default)]
#[serde(deny_unknown_fields)]
struct Selectors {
    future_context_ms: Option<u16>,
    transcript_style: Option<String>,
    model_family: Option<String>,
    model_head: Option<String>,
    model_precision: Option<String>,
}
fn parse_selection(uri: &Uri, default: Selection) -> Result<Selection, Problem> {
    let Query(q) = Query::<Selectors>::try_from_uri(uri)
        .map_err(|_| Problem::new("invalid_query", "Invalid native v4 WebSocket query"))?;
    if q.model_family.as_deref().is_some_and(|x| x != "gigaam_v4")
        || q.model_head.as_deref().is_some_and(|x| x != "rnnt")
        || q.model_precision.as_deref().is_some_and(|x| x != "fp32")
    {
        return Err(Problem::new(
            "unsupported_model",
            "Native v4 streaming requires the GigaAM v4 RNNT FP32 profile",
        ));
    }
    let context = q
        .future_context_ms
        .map(V4StreamContext::from_millis)
        .transpose()
        .map_err(|_| {
            Problem::new(
                "invalid_future_context",
                "Native v4 future context must be 160, 320, or 640 ms",
            )
        })?
        .unwrap_or(default.context);
    let style = q
        .transcript_style
        .map(|x| x.parse())
        .transpose()
        .map_err(|_| {
            Problem::new(
                "invalid_transcript_style",
                "Native v4 transcript style must be raw or e2e",
            )
        })?
        .unwrap_or(default.style);
    Ok(Selection { context, style })
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
enum Control {
    Stop,
    Configure {
        sample_rate: Option<u32>,
        protocol_version: Option<String>,
        diarization: Option<bool>,
        punctuation: Option<bool>,
        itn: Option<bool>,
        endpoint_mode: Option<String>,
        future_context_ms: Option<u16>,
        transcript_style: Option<String>,
    },
}
#[derive(Debug, PartialEq, Eq)]
enum Command {
    Configure,
    Stop,
}
fn command(text: &str, selection: Selection, started: bool) -> Result<Command, Problem> {
    let decoded = serde_json::from_str::<Control>(text).map_err(|_| {
        Problem::new(
            "invalid_configure",
            "Unsupported native v4 WebSocket command or option",
        )
    })?;
    let Control::Configure {
        sample_rate,
        protocol_version,
        diarization,
        punctuation,
        itn,
        endpoint_mode,
        future_context_ms,
        transcript_style,
    } = decoded
    else {
        return Ok(Command::Stop);
    };
    if started {
        return Err(Problem::new(
            "configure_too_late",
            "Configure must precede the first audio frame",
        ));
    }
    if sample_rate.is_some_and(|rate| rate != 16000) {
        return Err(Problem::new(
            "invalid_sample_rate",
            "Native v4 streaming requires 16 kHz mono PCM16",
        ));
    }
    if protocol_version
        .as_deref()
        .is_some_and(|v| !matches!(v, "1.0" | "1.1" | "1.2" | "1.3" | wire::PROTOCOL_VERSION))
    {
        return Err(Problem::new(
            "unsupported_protocol_version",
            "Native v4 streaming accepts protocol 1.0 through 1.4",
        ));
    }
    if diarization == Some(true)
        || punctuation == Some(true)
        || itn == Some(true)
        || endpoint_mode.as_deref().is_some_and(|m| m != "manual")
    {
        return Err(Problem::new(
            "unsupported_option",
            "Native v4 streaming supports only manual Stop and no postprocessing",
        ));
    }
    if future_context_ms.is_some_and(|v| v != selection.context.future_context_ms())
        || transcript_style
            .as_deref()
            .is_some_and(|v| v != selection.style.as_str())
    {
        return Err(Problem::new(
            "configuration_locked",
            "Select native v4 style and context in the WebSocket URL; reconnect to change them",
        ));
    }
    Ok(Command::Configure)
}

#[derive(Debug, Clone, Copy)]
struct Problem {
    code: &'static str,
    message: &'static str,
}
impl Problem {
    const fn new(code: &'static str, message: &'static str) -> Self {
        Self { code, message }
    }
    async fn send(
        self,
        sink: &mut super::frames::WsSink,
        retry_after_ms: Option<u32>,
    ) -> anyhow::Result<()> {
        send(
            sink,
            Message::Text(
                json_text(&ServerMessage::Error {
                    code: self.code.into(),
                    message: self.message.into(),
                    retry_after_ms,
                })
                .into(),
            ),
        )
        .await
    }
}
fn pcm16(bytes: &[u8], limit: usize) -> Result<Vec<f32>, Problem> {
    if bytes.len() > limit.min(V4_STREAM_MAX_PACKET_SAMPLES * 2) {
        return Err(Problem::new(
            "payload_too_large",
            "Native v4 audio frame exceeds the negotiated byte limit",
        ));
    }
    let (samples, rest) = bytes.as_chunks::<2>();
    if samples.is_empty() || !rest.is_empty() {
        return Err(Problem::new(
            "invalid_audio",
            "Native v4 audio frames must contain complete PCM16 samples",
        ));
    }
    Ok(samples
        .iter()
        .map(|&b| f32::from(i16::from_le_bytes(b)) * (1.0 / 32768.0))
        .collect())
}

/// Explicit side-loaded native streaming model for a v4 file server.
pub struct NativeV4Streaming {
    engine: Arc<V4StreamingEngine>,
    default_context: V4StreamContext,
}
impl NativeV4Streaming {
    /// Verify and load the exact cached streaming pack. No download or fallback.
    pub fn load(
        pack: &Path,
        runtime: &Path,
        default_context: V4StreamContext,
    ) -> Result<Self, GigasttError> {
        Ok(Self {
            engine: Arc::new(V4StreamingEngine::load(pack, runtime)?),
            default_context,
        })
    }
}

pub(crate) struct NativeState {
    engine: Arc<V4StreamingEngine>,
    default: Selection,
    slots: Arc<Semaphore>,
    limits: Arc<RuntimeLimits>,
    shutdown: CancellationToken,
    tracker: TaskTracker,
}
impl NativeState {
    pub(crate) fn new(
        stream: NativeV4Streaming,
        style: TranscriptStyle,
        limits: Arc<RuntimeLimits>,
        shutdown: CancellationToken,
        tracker: TaskTracker,
    ) -> Self {
        Self {
            engine: stream.engine,
            default: Selection {
                context: stream.default_context,
                style,
            },
            slots: Arc::new(Semaphore::new(V4_STREAM_MAX_SESSIONS)),
            limits,
            shutdown,
            tracker,
        }
    }
    fn max_bytes(&self) -> usize {
        self.limits
            .ws_frame_max_bytes
            .min(V4_STREAM_MAX_PACKET_SAMPLES * 2)
    }
    pub(crate) fn capability(&self) -> wire::ReadyCapability {
        self.capability_for(self.default)
    }
    fn capability_for(&self, selection: Selection) -> wire::ReadyCapability {
        wire::ReadyCapability {
            model_family: "gigaam_v4",
            model_head: "rnnt",
            model_precision: "fp32",
            transcript_style: selection.style.as_str(),
            future_context_ms: selection.context.future_context_ms(),
            supported_future_context_ms: V4_STREAMING_CONTEXTS_MS,
            max_packet_bytes: self.max_bytes(),
            endpoint_mode: "manual",
        }
    }
    pub(crate) fn upgrade(
        self: Arc<Self>,
        ws: WebSocketUpgrade,
        uri: Uri,
        headers: &HeaderMap,
    ) -> Response {
        if self.shutdown.is_cancelled() {
            return (
                StatusCode::SERVICE_UNAVAILABLE,
                Json(json!({"error":"Server shutting down","code":"shutting_down"})),
            )
                .into_response();
        }
        if !native_protocol_requested(headers) {
            return (StatusCode::BAD_REQUEST,Json(json!({"error":"Select the gigastt-native-v4 WebSocket subprotocol explicitly","code":"native_protocol_required"}))).into_response();
        }
        let selection = match parse_selection(&uri, self.default) {
            Ok(selection) => selection,
            Err(e) => {
                return (
                    StatusCode::BAD_REQUEST,
                    Json(json!({"error":e.message,"code":e.code})),
                )
                    .into_response();
            }
        };
        let request_id = headers
            .get("x-request-id")
            .and_then(|v| v.to_str().ok())
            .unwrap_or("unknown")
            .to_owned();
        let cap = self.max_bytes();
        ws.protocols([wire::SUBPROTOCOL])
            .max_message_size(cap)
            .max_frame_size(cap)
            .on_upgrade(move |socket| {
                let span = tracing::info_span!("native_v4_ws",request_id=%request_id);
                async move {
                    let tracker = self.tracker.clone();
                    tracker.track_future(self.handle(socket, selection)).await
                }
                .instrument(span)
            })
    }
    async fn handle(self: Arc<Self>, socket: WebSocket, selection: Selection) {
        if let Err(error) = self.run(socket, selection).await {
            tracing::debug!(%error,"Native v4 WebSocket closed");
        }
    }
    async fn run(&self, socket: WebSocket, selection: Selection) -> anyhow::Result<()> {
        let (mut sink, source) = socket.split();
        let abort = Arc::new(AtomicBool::new(false));
        let _abort = crate::server::file_transcribe::AbortOnDrop(abort.clone());
        let mut incoming = Incoming::new(source, abort.clone());
        let permits = self.slots.clone();
        let permit = tokio::select! {biased;
            _=self.shutdown.cancelled()=>{close(&mut sink,1001,"server shutdown").await;return Ok(())},
            _=incoming.closed.cancelled()=>return Ok(()),
            result=tokio::time::timeout(Duration::from_secs(self.limits.pool_checkout_timeout_secs),permits.acquire_owned())=>match result {
                Ok(Ok(permit))=>permit,
                Ok(Err(_))=>return Ok(()),
                Err(_)=>{Problem::new("timeout","Server busy, try again later").send(&mut sink,Some(pool_retry_after_ms(&self.limits))).await?;close(&mut sink,1013,"server busy").await;return Ok(())}
            }
        };
        let session = self
            .engine
            .start_session(selection.context, selection.style.into())?;
        let mut leased = Leased {
            session,
            _permit: permit,
        };
        let ready = json!({"type":"ready","model":format!("gigaam-v4-rnnt-{}-fp32-streaming",selection.style.as_str()),"sample_rate":16000,"version":wire::PROTOCOL_VERSION,"min_protocol_version":"1.0","supported_rates":[16000],"max_session_secs":self.limits.max_session_secs,"idle_timeout_secs":self.limits.idle_timeout_secs,"native_v4":self.capability_for(selection)});
        send(&mut sink, Message::Text(json_text(&ready).into())).await?;
        let deadline = session_deadline(self.limits.max_session_secs);
        let idle = Duration::from_secs(self.limits.idle_timeout_secs);
        let mut started = false;
        let mut snapshot = Snapshot::default();
        let mut idle_deadline = next_idle_deadline(idle);
        let mut ping = tokio::time::interval(Duration::from_secs(super::WS_PING_INTERVAL_SECS));
        ping.tick().await;
        let mut unanswered = 0;
        loop {
            let frame = tokio::select! {biased;
                _=self.shutdown.cancelled()=>{close(&mut sink,1001,"server shutdown").await;return Ok(())},
                _=incoming.closed.cancelled()=>return Ok(()),
                _=tokio::time::sleep_until(deadline)=>{Problem::new("max_session_duration_exceeded","Maximum session duration exceeded").send(&mut sink,None).await?;close(&mut sink,1008,"session limit").await;return Ok(())},
                _=ping.tick()=>{if super::keepalive_should_close(unanswered){close(&mut sink,1008,"ping timeout").await;return Ok(())} unanswered+=1;send(&mut sink,Message::Ping(Vec::new().into())).await?;continue;},
                _=tokio::time::sleep_until(idle_deadline)=>{Problem::new("idle_timeout","Idle timeout").send(&mut sink,None).await?;close(&mut sink,1008,"idle timeout").await;return Ok(())},
                r=incoming.rx.recv()=>match r { Some(Ok(frame))=>frame,_=>return Ok(()) }
            };
            unanswered = 0;
            idle_deadline = next_idle_deadline(idle);
            let (pcm, final_packet) = match frame {
                Message::Binary(data) => match pcm16(&data, self.max_bytes()) {
                    Ok(pcm) => {
                        started = true;
                        (pcm, false)
                    }
                    Err(e) => {
                        e.send(&mut sink, None).await?;
                        close(&mut sink, 1008, "invalid audio").await;
                        return Ok(());
                    }
                },
                Message::Text(text) => match command(text.as_str(), selection, started) {
                    Ok(Command::Configure) => continue,
                    Ok(Command::Stop) => (Vec::new(), true),
                    Err(e) => {
                        e.send(&mut sink, None).await?;
                        close(&mut sink, 1008, "invalid configure").await;
                        return Ok(());
                    }
                },
                Message::Close(_) => return Ok(()),
                Message::Ping(_) | Message::Pong(_) => continue,
            };
            let worker_abort = abort.clone();
            let worker = self.tracker.spawn_blocking(move || {
                let result = if final_packet {
                    leased.session.flush(Some(&worker_abort))
                } else {
                    leased.session.push_pcm(&pcm, false, Some(&worker_abort))
                };
                (leased, result)
            });
            match await_packet(
                worker,
                &abort,
                &self.shutdown,
                &incoming.closed,
                deadline,
                self.limits.inference_timeout_secs,
            )
            .await
            {
                Ok((returned, Ok(packet))) => {
                    leased = returned;
                    if self.shutdown.is_cancelled() {
                        close(&mut sink, 1001, "server shutdown").await;
                        return Ok(());
                    }
                    if abort.load(Ordering::Relaxed) {
                        return Ok(());
                    }
                    snapshot
                        .publish(&mut sink, packet, selection.style, final_packet)
                        .await?;
                }
                Ok((_, Err(e))) => {
                    let code = e.code();
                    let mapped = match code {
                        "audio_too_long" => Problem::new(
                            "max_session_duration_exceeded",
                            "Native v4 audio sample limit exceeded",
                        ),
                        "cancelled" => {
                            Problem::new("cancelled", "Native v4 transcription cancelled")
                        }
                        _ => Problem::new("inference_error", "Native v4 inference failed"),
                    };
                    mapped.send(&mut sink, None).await?;
                    close(&mut sink, 1011, "stream failed").await;
                    return Ok(());
                }
                Err(WaitFailure::Disconnected) => return Ok(()),
                Err(WaitFailure::Shutdown) => {
                    close(&mut sink, 1001, "server shutdown").await;
                    return Ok(());
                }
                Err(reason) => {
                    let problem = match reason {
                        WaitFailure::Timeout => {
                            Problem::new("inference_timeout", "Native v4 inference timed out")
                        }
                        WaitFailure::Deadline => Problem::new(
                            "max_session_duration_exceeded",
                            "Maximum session duration exceeded",
                        ),
                        _ => Problem::new("inference_panic", "Native v4 inference worker failed"),
                    };
                    problem.send(&mut sink, None).await?;
                    close(&mut sink, 1011, "stream ended").await;
                    return Ok(());
                }
            }
            if final_packet {
                close(&mut sink, 1000, "finished").await;
                return Ok(());
            }
        }
    }
}

fn next_idle_deadline(idle: Duration) -> tokio::time::Instant {
    let now = tokio::time::Instant::now();
    now.checked_add(idle).unwrap_or(now + super::FAR_FUTURE)
}

struct Leased {
    session: V4StreamingSession,
    _permit: OwnedSemaphorePermit,
}
#[derive(Debug, PartialEq, Eq)]
enum WaitFailure {
    Disconnected,
    Shutdown,
    Timeout,
    Deadline,
    Worker,
}
async fn await_packet<T>(
    mut work: tokio::task::JoinHandle<T>,
    abort: &AtomicBool,
    shutdown: &CancellationToken,
    closed: &CancellationToken,
    deadline: tokio::time::Instant,
    timeout_secs: u64,
) -> Result<T, WaitFailure> {
    let timeout = async {
        if timeout_secs == 0 {
            std::future::pending::<()>().await;
        } else {
            tokio::time::sleep(Duration::from_secs(timeout_secs)).await;
        }
    };
    let e = tokio::select! {biased;
        _=shutdown.cancelled()=>WaitFailure::Shutdown,
        _=closed.cancelled()=>WaitFailure::Disconnected,
        _=tokio::time::sleep_until(deadline)=>WaitFailure::Deadline,
        value=&mut work=>return value.map_err(|_|WaitFailure::Worker),
        _=timeout=>WaitFailure::Timeout,
    };
    // Dropping the join handle detaches rather than stops the task. The
    // tracked blocking worker still owns the state and permit until exit.
    abort.store(true, Ordering::Relaxed);
    Err(e)
}

#[derive(Default)]
struct Snapshot {
    text: String,
    words: Vec<WordInfo>,
    next: usize,
}
impl Snapshot {
    async fn publish(
        &mut self,
        sink: &mut super::frames::WsSink,
        packet: V4StreamingPacket,
        style: TranscriptStyle,
        is_final: bool,
    ) -> anyhow::Result<()> {
        let delta = self.apply(packet, style, is_final)?;
        if let Some(delta) = delta {
            let message = wire::Transcript::new(&self.text, &self.words, &delta, is_final);
            send(sink, Message::Text(json_text(&message).into())).await?;
        }
        Ok(())
    }
    fn apply(
        &mut self,
        packet: V4StreamingPacket,
        style: TranscriptStyle,
        is_final: bool,
    ) -> anyhow::Result<Option<wire::Delta>> {
        anyhow::ensure!(packet.finished == is_final, "native v4 EOF mismatch");
        let delta = match (style, packet.raw, packet.e2e) {
            (TranscriptStyle::Raw, Some(d), None) | (TranscriptStyle::E2e, None, Some(d)) => d,
            _ => anyhow::bail!("native v4 style mismatch"),
        };
        self.next += 1;
        self.text.push_str(&delta.text);
        self.words.extend_from_slice(&delta.words);
        if !is_final && delta.text.is_empty() && delta.words.is_empty() && delta.tokens.is_empty() {
            return Ok(None);
        }
        Ok(Some(wire::Delta {
            packet_one_based: self.next,
            samples: packet.samples,
            encoded_frames: packet.encoded_frames,
            text: delta.text,
            words: delta.words,
            tokens: delta
                .tokens
                .into_iter()
                .map(|t| wire::Token {
                    id: t.id,
                    frame: t.frame,
                    confidence: t.confidence,
                })
                .collect(),
        }))
    }
}
// A peer that stops reading must not retain a streaming state indefinitely.
const WRITE_TIMEOUT: Duration = Duration::from_secs(5);
async fn send_within<S>(sink: &mut S, message: Message, max: Duration) -> anyhow::Result<()>
where
    S: Sink<Message, Error = axum::Error> + Unpin,
{
    tokio::time::timeout(max, sink.send(message)).await??;
    Ok(())
}
async fn send(sink: &mut super::frames::WsSink, message: Message) -> anyhow::Result<()> {
    send_within(sink, message, WRITE_TIMEOUT).await
}
async fn close(sink: &mut super::frames::WsSink, code: u16, reason: &'static str) {
    let _ = send(
        sink,
        Message::Close(Some(CloseFrame {
            code,
            reason: reason.into(),
        })),
    )
    .await;
}

#[cfg(test)]
mod tests;
