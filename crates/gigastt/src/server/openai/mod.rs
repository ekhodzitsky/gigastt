//! OpenAI Audio Transcriptions compatibility layer.
//!
//! Implements the subset of
//! [`POST /v1/audio/transcriptions`](https://platform.openai.com/docs/api-reference/audio/createTranscription)
//! that local agents (llama-swap, Hermes, OpenAI SDKs with a custom `base_url`)
//! actually exercise:
//!
//! | Form field | Behaviour |
//! |---|---|
//! | `file` | required audio bytes |
//! | `model` | accepted, ignored (single loaded head) |
//! | `response_format` | `json` (default) · `text` · `srt` · `vtt` · `verbose_json` |
//! | `language` | accepted; echoed in `verbose_json` (default `ru`) |
//! | `timestamp_granularities[]` | `word` / `segment` for `verbose_json` |
//! | `stream` | `true`/`false` — SSE of `transcript.text.delta` + `done` + `[DONE]` |
//! | `prompt`, `temperature` | accepted, ignored |
//!
//! Inference reuses the native pipeline; this module only shapes the request
//! and response envelopes. Streaming runs the real chunked encoder path and
//! maps progressive text to OpenAI transcript events (append-only deltas).

use axum::body::Bytes;
use axum::extract::Multipart;
use axum::http::{StatusCode, header};
use axum::response::{IntoResponse, Json, Response};
use gigastt_core::export::{RenderOpts, to_srt, to_transcript_segments, to_vtt};
use gigastt_core::inference::{TranscribeResult, WordInfo};
use serde::Serialize;

/// Wire value of OpenAI `response_format`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum OpenAIResponseFormat {
    /// `{"text":"..."}` — OpenAI default for whisper-1.
    #[default]
    Json,
    /// Raw transcript body (`text/plain`).
    Text,
    /// SubRip captions.
    Srt,
    /// WebVTT captions.
    Vtt,
    /// Whisper-style verbose JSON with optional segments/words.
    VerboseJson,
}

impl OpenAIResponseFormat {
    /// Parse an OpenAI `response_format` token. Unknown values are errors so
    /// clients get a typed 400 instead of a silent fallback.
    pub fn parse(raw: &str) -> Result<Self, String> {
        match raw.trim().to_ascii_lowercase().as_str() {
            "" | "json" => Ok(Self::Json),
            "text" => Ok(Self::Text),
            "srt" => Ok(Self::Srt),
            "vtt" => Ok(Self::Vtt),
            "verbose_json" => Ok(Self::VerboseJson),
            other => Err(format!(
                "Unsupported response_format '{other}'. Supported: json, text, srt, vtt, verbose_json"
            )),
        }
    }

    /// Wire token (for error messages).
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Json => "json",
            Self::Text => "text",
            Self::Srt => "srt",
            Self::Vtt => "vtt",
            Self::VerboseJson => "verbose_json",
        }
    }
}

impl std::fmt::Display for OpenAIResponseFormat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Parsed OpenAI multipart options (everything except the audio file).
#[derive(Debug, Clone, Default)]
pub struct OpenAITranscriptionOptions {
    /// OpenAI `model` string — accepted for client compatibility, never used
    /// to select a head (single-engine server).
    pub model: Option<String>,
    /// OpenAI `language` (ISO-639-1 or longer names). Echoed in verbose JSON;
    /// does not reconfigure the loaded head.
    pub language: Option<String>,
    /// `response_format` (default `json`).
    pub response_format: OpenAIResponseFormat,
    /// Whether `verbose_json` should include word-level timestamps.
    pub include_words: bool,
    /// Whether `verbose_json` should include segment-level timestamps.
    pub include_segments: bool,
    /// When true, respond with an SSE stream of OpenAI transcript events
    /// instead of a single buffered body.
    pub stream: bool,
}

/// Fully parsed multipart request.
#[derive(Debug)]
pub struct OpenAITranscriptionRequest {
    pub file: Bytes,
    pub options: OpenAITranscriptionOptions,
}

/// Default JSON body: `{"text":"..."}`.
#[derive(Debug, Serialize, PartialEq, Eq)]
pub struct OpenAIJsonResponse {
    pub text: String,
}

/// One word in OpenAI `verbose_json.words[]`.
#[derive(Debug, Serialize)]
pub struct OpenAIWord {
    pub word: String,
    pub start: f64,
    pub end: f64,
}

/// One segment in OpenAI `verbose_json.segments[]` (Whisper-compatible fields
/// clients commonly read: `id`, `start`, `end`, `text`. Extra Whisper-only
/// fields are filled with stable zeros so typed clients do not break).
#[derive(Debug, Serialize)]
pub struct OpenAISegment {
    pub id: u32,
    pub seek: u32,
    pub start: f64,
    pub end: f64,
    pub text: String,
    pub tokens: Vec<u32>,
    pub temperature: f64,
    pub avg_logprob: f64,
    pub compression_ratio: f64,
    pub no_speech_prob: f64,
}

/// OpenAI `verbose_json` envelope.
#[derive(Debug, Serialize)]
pub struct OpenAIVerboseResponse {
    pub task: &'static str,
    pub language: String,
    pub duration: f64,
    pub text: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub segments: Option<Vec<OpenAISegment>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub words: Option<Vec<OpenAIWord>>,
}

/// Map an internal error into the gigastt REST error envelope.
fn openai_error(status: StatusCode, msg: &str, code: &str) -> Response {
    (
        status,
        Json(serde_json::json!({"error": msg, "code": code})),
    )
        .into_response()
}

/// Normalize a client-supplied language tag for echo in `verbose_json`.
///
/// Accepts ISO-639-1 (`ru`, `en`) and a few common full names. Unknown tokens
/// are lowercased and passed through unchanged so multi-lingual heads keep
/// whatever the client sent.
pub fn normalize_language(raw: &str) -> String {
    let t = raw.trim();
    if t.is_empty() {
        return "ru".into();
    }
    match t.to_ascii_lowercase().as_str() {
        "ru" | "rus" | "russian" | "ru-ru" => "ru".into(),
        "en" | "eng" | "english" | "en-us" | "en-gb" => "en".into(),
        "kk" | "kaz" | "kazakh" => "kk".into(),
        "ky" | "kir" | "kyrgyz" => "ky".into(),
        "uz" | "uzb" | "uzbek" => "uz".into(),
        other => other.to_string(),
    }
}

/// Apply one form field into `options`. Returns `Err(message)` for invalid
/// `response_format` / granularities. Pure: unit-testable without Multipart.
pub fn apply_openai_form_field(
    options: &mut OpenAITranscriptionOptions,
    name: &str,
    value: &[u8],
) -> Result<(), String> {
    let text = || String::from_utf8_lossy(value).into_owned();
    match name {
        "model" => {
            let s = text();
            if !s.is_empty() {
                options.model = Some(s);
            }
        }
        "language" => {
            let s = text();
            if !s.is_empty() {
                options.language = Some(s);
            }
        }
        "response_format" => {
            options.response_format = OpenAIResponseFormat::parse(&text())?;
        }
        // OpenAI SDKs send either `timestamp_granularities[]` or bare
        // `timestamp_granularities` as repeated fields.
        "timestamp_granularities[]" | "timestamp_granularities" => {
            match text().trim().to_ascii_lowercase().as_str() {
                "word" => options.include_words = true,
                "segment" => options.include_segments = true,
                "" => {}
                other => {
                    return Err(format!(
                        "Unsupported timestamp_granularity '{other}'. Supported: word, segment"
                    ));
                }
            }
        }
        "stream" => {
            options.stream = parse_bool_form(&text()).ok_or_else(|| {
                format!(
                    "Invalid stream value '{}'. Use true or false",
                    text().trim()
                )
            })?;
        }
        // Accepted for SDK compatibility; no server-side effect.
        "prompt" | "temperature" => {}
        _ => {}
    }
    Ok(())
}

/// Parse OpenAI-style form booleans (`true`/`false`/`1`/`0`).
fn parse_bool_form(raw: &str) -> Option<bool> {
    match raw.trim().to_ascii_lowercase().as_str() {
        "true" | "1" | "yes" | "on" => Some(true),
        "false" | "0" | "no" | "off" => Some(false),
        "" => Some(false),
        _ => None,
    }
}

/// After all fields are applied, resolve default granularities for
/// `verbose_json`: if the client requested neither, include segments only
/// (OpenAI historical default). Streaming is incompatible with caption
/// `response_format`s (SSE is always text-delta events).
pub fn finalize_openai_options(options: &mut OpenAITranscriptionOptions) -> Result<(), String> {
    if options.response_format == OpenAIResponseFormat::VerboseJson
        && !options.include_words
        && !options.include_segments
    {
        options.include_segments = true;
    }
    if options.stream {
        match options.response_format {
            OpenAIResponseFormat::Json | OpenAIResponseFormat::Text => {}
            other => {
                return Err(format!(
                    "stream=true is not supported with response_format='{other}'. Use json or text (or omit response_format)"
                ));
            }
        }
    }
    Ok(())
}

mod sse;
pub use sse::{OpenAIStreamAssembler, sse_delta_payload, sse_done_payload, sse_event_data};
// Event payload types stay on the public openai path for existing callers.
#[allow(unused_imports)]
pub use sse::{OpenAITranscriptDelta, OpenAITranscriptDone};

/// Parse an OpenAI-style multipart body into a typed request.
pub async fn parse_openai_multipart(
    multipart: Multipart,
) -> Result<OpenAITranscriptionRequest, Box<Response>> {
    parse_openai_multipart_mode(multipart, false).await
}
#[cfg(all(
    feature = "v4-mkl",
    target_os = "linux",
    target_arch = "x86_64",
    target_env = "gnu"
))]
pub(crate) async fn parse_v4_openai_multipart(
    multipart: Multipart,
) -> Result<OpenAITranscriptionRequest, Box<Response>> {
    parse_openai_multipart_mode(multipart, true).await
}
async fn parse_openai_multipart_mode(
    mut multipart: Multipart,
    strict_v4: bool,
) -> Result<OpenAITranscriptionRequest, Box<Response>> {
    let mut file: Option<Bytes> = None;
    let mut options = OpenAITranscriptionOptions::default();
    let mut seen = std::collections::BTreeSet::new();

    while let Some(field) = multipart.next_field().await.map_err(|e| {
        tracing::error!("OpenAI multipart parse failed: {e:#}");
        Box::new(openai_error(
            StatusCode::BAD_REQUEST,
            "Invalid multipart body",
            "invalid_multipart",
        ))
    })? {
        let name = field.name().unwrap_or("").to_string();
        if strict_v4 && !seen.insert(name.clone()) {
            return Err(Box::new(openai_error(
                StatusCode::BAD_REQUEST,
                "Duplicate v4 multipart field",
                "invalid_multipart",
            )));
        }
        let data = field.bytes().await.map_err(|e| {
            tracing::error!("OpenAI multipart field read failed: {e:#}");
            Box::new(openai_error(
                StatusCode::BAD_REQUEST,
                "Invalid multipart body",
                "invalid_multipart",
            ))
        })?;
        if name == "file" {
            file = Some(data);
            continue;
        }
        if !strict_v4
            && (super::router::is_v4_selector(&name)
                || (name == "model"
                    && std::str::from_utf8(&data).is_ok_and(super::router::is_v4_model_id)))
        {
            return Err(Box::new(openai_error(
                StatusCode::CONFLICT,
                "Explicit v4 model selection requires a v4 file server",
                "variant_not_loaded",
            )));
        }
        if strict_v4
            && !matches!(
                name.as_str(),
                "model"
                    | "response_format"
                    | "stream"
                    | "timestamp_granularities"
                    | "timestamp_granularities[]"
            )
        {
            return Err(Box::new(openai_error(
                StatusCode::BAD_REQUEST,
                "Unsupported v4 OpenAI option; language hints, prompts and sampling options are unavailable",
                "unsupported_v4_option",
            )));
        }
        if let Err(msg) = apply_openai_form_field(&mut options, &name, &data) {
            let code = if msg.contains("response_format") {
                "invalid_response_format"
            } else if msg.contains("timestamp_granularity") {
                "invalid_timestamp_granularity"
            } else if msg.contains("stream") {
                "invalid_stream"
            } else {
                "invalid_multipart"
            };
            return Err(Box::new(openai_error(StatusCode::BAD_REQUEST, &msg, code)));
        }
    }

    if let Err(msg) = finalize_openai_options(&mut options) {
        return Err(Box::new(openai_error(
            StatusCode::BAD_REQUEST,
            &msg,
            "invalid_stream_options",
        )));
    }

    let file = file.ok_or_else(|| {
        Box::new(openai_error(
            StatusCode::BAD_REQUEST,
            "Missing required form field: file",
            "missing_file",
        ))
    })?;
    if file.is_empty() {
        return Err(Box::new(openai_error(
            StatusCode::BAD_REQUEST,
            "Empty request body",
            "empty_body",
        )));
    }

    Ok(OpenAITranscriptionRequest { file, options })
}

fn openai_words(words: &[WordInfo]) -> Vec<OpenAIWord> {
    words
        .iter()
        .map(|w| OpenAIWord {
            word: w.word.clone(),
            start: w.start,
            end: w.end,
        })
        .collect()
}

fn openai_segments(words: &[WordInfo]) -> Vec<OpenAISegment> {
    to_transcript_segments(words)
        .into_iter()
        .enumerate()
        .map(|(i, seg)| {
            // Mean per-word confidence → fake avg_logprob in [-1, 0] so
            // clients that read the field get a plausible number. Purely
            // cosmetic; not a real log-probability.
            let avg_conf = if seg.words.is_empty() {
                0.0
            } else {
                seg.words.iter().map(|w| w.confidence as f64).sum::<f64>() / seg.words.len() as f64
            };
            OpenAISegment {
                id: i as u32,
                seek: (seg.start * 100.0).round() as u32,
                start: seg.start,
                end: seg.end,
                // OpenAI/Whisper often prefixes segment text with a space.
                text: format!(" {}", seg.text.trim()),
                tokens: Vec::new(),
                temperature: 0.0,
                avg_logprob: (avg_conf - 1.0).clamp(-1.0, 0.0),
                compression_ratio: 1.0,
                no_speech_prob: 0.0,
            }
        })
        .collect()
}

/// Build the OpenAI `verbose_json` value (pure, unit-testable).
pub fn build_verbose_response(
    result: &TranscribeResult,
    options: &OpenAITranscriptionOptions,
) -> OpenAIVerboseResponse {
    let language = options
        .language
        .as_deref()
        .map(normalize_language)
        .unwrap_or_else(|| "ru".into());
    OpenAIVerboseResponse {
        task: "transcribe",
        language,
        duration: result.duration_s,
        text: result.text.clone(),
        segments: options
            .include_segments
            .then(|| openai_segments(&result.words)),
        words: options.include_words.then(|| openai_words(&result.words)),
    }
}

/// Render a transcription result into an OpenAI-shaped HTTP response.
pub fn render_openai_response(
    result: &TranscribeResult,
    options: &OpenAITranscriptionOptions,
) -> Response {
    let opts = RenderOpts::default();
    match options.response_format {
        OpenAIResponseFormat::Json => Json(OpenAIJsonResponse {
            text: result.text.clone(),
        })
        .into_response(),
        OpenAIResponseFormat::Text => (
            StatusCode::OK,
            [(header::CONTENT_TYPE, "text/plain; charset=utf-8")],
            result.text.clone(),
        )
            .into_response(),
        OpenAIResponseFormat::Srt => (
            StatusCode::OK,
            [(header::CONTENT_TYPE, "application/x-subrip; charset=utf-8")],
            to_srt(
                &result.words,
                opts.max_chars_per_line,
                opts.max_words_per_line,
            ),
        )
            .into_response(),
        OpenAIResponseFormat::Vtt => (
            StatusCode::OK,
            [(header::CONTENT_TYPE, "text/vtt; charset=utf-8")],
            to_vtt(
                &result.words,
                opts.max_chars_per_line,
                opts.max_words_per_line,
            ),
        )
            .into_response(),
        OpenAIResponseFormat::VerboseJson => {
            Json(build_verbose_response(result, options)).into_response()
        }
    }
}

#[cfg(test)]
mod tests;
