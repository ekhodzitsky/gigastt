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
use axum::response::sse::Event;
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

/// OpenAI streaming event: incremental text.
#[derive(Debug, Serialize)]
pub struct OpenAITranscriptDelta {
    #[serde(rename = "type")]
    pub event_type: &'static str,
    pub delta: String,
}

/// OpenAI streaming event: full transcript at end of stream.
#[derive(Debug, Serialize)]
pub struct OpenAITranscriptDone {
    #[serde(rename = "type")]
    pub event_type: &'static str,
    pub text: String,
}

/// SSE `data:` payload for a text delta.
pub fn sse_delta_payload(delta: &str) -> String {
    serde_json::to_string(&OpenAITranscriptDelta {
        event_type: "transcript.text.delta",
        delta: delta.to_string(),
    })
    .unwrap_or_else(|_| r#"{"type":"transcript.text.delta","delta":""}"#.into())
}

/// SSE `data:` payload for the terminal done event.
pub fn sse_done_payload(text: &str) -> String {
    serde_json::to_string(&OpenAITranscriptDone {
        event_type: "transcript.text.done",
        text: text.to_string(),
    })
    .unwrap_or_else(|_| r#"{"type":"transcript.text.done","text":""}"#.into())
}

/// Build an axum SSE event for a delta / done / `[DONE]` marker.
pub fn sse_event_data(data: impl Into<String>) -> Event {
    Event::default().data(data.into())
}

/// Tracks append-only OpenAI text for progressive streaming.
///
/// Gigastt partials rewrite the *current* utterance; finals close it.
/// Deltas are emitted only when the cumulative view is a prefix extension of
/// what was already sent (OpenAI deltas are append-only — never retract).
#[derive(Debug, Default)]
pub struct OpenAIStreamAssembler {
    /// Text from completed (final) utterances.
    committed: String,
    /// Full cumulative text already sent as deltas.
    last_emitted: String,
}

impl OpenAIStreamAssembler {
    pub fn new() -> Self {
        Self::default()
    }

    /// Full transcript for the terminal `transcript.text.done` event.
    pub fn text(&self) -> &str {
        // Prefer committed after finals; fall back to live-emitted partials.
        if self.committed.len() >= self.last_emitted.len() {
            &self.committed
        } else {
            &self.last_emitted
        }
    }

    /// Ingest one native segment; return an optional delta to send.
    pub fn push_segment(&mut self, text: &str, is_final: bool) -> Option<String> {
        let live = text.trim();
        let candidate = match (self.committed.is_empty(), live.is_empty()) {
            (_, true) => self.committed.clone(),
            (true, false) => live.to_string(),
            (false, false) => format!("{} {live}", self.committed),
        };

        let mut delta = None;
        if candidate.starts_with(&self.last_emitted) {
            let d = candidate[self.last_emitted.len()..].to_string();
            if !d.is_empty() {
                self.last_emitted.clone_from(&candidate);
                delta = Some(d);
            }
        }
        // else: partial rewrote earlier tokens — skip (cannot unsend)

        if is_final {
            if !live.is_empty() {
                self.committed = if self.committed.is_empty() {
                    live.to_string()
                } else {
                    format!("{} {live}", self.committed)
                };
            }
            // Align emission cursor with committed when it is a pure extension.
            if self.committed.starts_with(&self.last_emitted) {
                let d = self.committed[self.last_emitted.len()..].to_string();
                self.last_emitted.clone_from(&self.committed);
                if delta.is_none() && !d.is_empty() {
                    delta = Some(d);
                }
            } else {
                // Final disagrees with what we streamed — snap cursor for `done`
                // accuracy without inventing a retracting delta.
                self.last_emitted.clone_from(&self.committed);
            }
        }
        delta
    }
}

/// Parse an OpenAI-style multipart body into a typed request.
pub async fn parse_openai_multipart(
    mut multipart: Multipart,
) -> Result<OpenAITranscriptionRequest, Response> {
    let mut file: Option<Bytes> = None;
    let mut options = OpenAITranscriptionOptions::default();

    while let Some(field) = multipart.next_field().await.map_err(|e| {
        openai_error(
            StatusCode::BAD_REQUEST,
            &format!("Invalid multipart body: {e}"),
            "invalid_multipart",
        )
    })? {
        let name = field.name().unwrap_or("").to_string();
        let data = field.bytes().await.map_err(|e| {
            openai_error(
                StatusCode::BAD_REQUEST,
                &format!("Failed to read multipart field: {e}"),
                "invalid_multipart",
            )
        })?;
        if name == "file" {
            file = Some(data);
            continue;
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
            return Err(openai_error(StatusCode::BAD_REQUEST, &msg, code));
        }
    }

    if let Err(msg) = finalize_openai_options(&mut options) {
        return Err(openai_error(
            StatusCode::BAD_REQUEST,
            &msg,
            "invalid_stream_options",
        ));
    }

    let file = file.ok_or_else(|| {
        openai_error(
            StatusCode::BAD_REQUEST,
            "Missing required form field: file",
            "missing_file",
        )
    })?;
    if file.is_empty() {
        return Err(openai_error(
            StatusCode::BAD_REQUEST,
            "Empty request body",
            "empty_body",
        ));
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
mod tests {
    use super::*;
    use gigastt_core::inference::WordInfo;

    fn sample_result() -> TranscribeResult {
        TranscribeResult {
            text: "привет мир".into(),
            words: vec![
                WordInfo::new("привет", 0.0, 0.5, 0.98, None),
                WordInfo::new("мир", 1.6, 2.0, 0.97, None),
            ],
            duration_s: 2.0,
            confidence: Some(0.975),
        }
    }

    #[test]
    fn test_response_format_parse() {
        assert_eq!(
            OpenAIResponseFormat::parse("json").unwrap(),
            OpenAIResponseFormat::Json
        );
        assert_eq!(
            OpenAIResponseFormat::parse("TEXT").unwrap(),
            OpenAIResponseFormat::Text
        );
        assert_eq!(
            OpenAIResponseFormat::parse("verbose_json").unwrap(),
            OpenAIResponseFormat::VerboseJson
        );
        assert!(OpenAIResponseFormat::parse("docx").is_err());
    }

    #[test]
    fn test_normalize_language() {
        assert_eq!(normalize_language("Russian"), "ru");
        assert_eq!(normalize_language("en-US"), "en");
        assert_eq!(normalize_language(""), "ru");
        assert_eq!(normalize_language("kk"), "kk");
        assert_eq!(normalize_language("pt-BR"), "pt-br");
    }

    #[test]
    fn test_apply_form_fields_and_finalize() {
        let mut opts = OpenAITranscriptionOptions::default();
        apply_openai_form_field(&mut opts, "model", b"whisper-1").unwrap();
        apply_openai_form_field(&mut opts, "language", b"Russian").unwrap();
        apply_openai_form_field(&mut opts, "response_format", b"verbose_json").unwrap();
        apply_openai_form_field(&mut opts, "timestamp_granularities[]", b"word").unwrap();
        apply_openai_form_field(&mut opts, "prompt", b"ignored").unwrap();
        apply_openai_form_field(&mut opts, "temperature", b"0").unwrap();
        finalize_openai_options(&mut opts).unwrap();

        assert_eq!(opts.model.as_deref(), Some("whisper-1"));
        assert_eq!(opts.language.as_deref(), Some("Russian"));
        assert_eq!(opts.response_format, OpenAIResponseFormat::VerboseJson);
        assert!(opts.include_words);
        // word-only request must not force segments on.
        assert!(!opts.include_segments);
    }

    #[test]
    fn test_finalize_defaults_segments_for_verbose() {
        let mut opts = OpenAITranscriptionOptions {
            response_format: OpenAIResponseFormat::VerboseJson,
            ..Default::default()
        };
        finalize_openai_options(&mut opts).unwrap();
        assert!(opts.include_segments);
        assert!(!opts.include_words);
    }

    #[test]
    fn test_stream_flag_and_incompatible_format() {
        let mut opts = OpenAITranscriptionOptions::default();
        apply_openai_form_field(&mut opts, "stream", b"true").unwrap();
        assert!(opts.stream);
        finalize_openai_options(&mut opts).unwrap();

        let mut opts = OpenAITranscriptionOptions {
            stream: true,
            response_format: OpenAIResponseFormat::Srt,
            ..Default::default()
        };
        let err = finalize_openai_options(&mut opts).unwrap_err();
        assert!(err.contains("stream=true"));
    }

    #[test]
    fn test_stream_assembler_grows_and_finalizes() {
        let mut a = OpenAIStreamAssembler::new();
        assert_eq!(a.push_segment("привет", false).as_deref(), Some("привет"));
        assert_eq!(a.push_segment("привет мир", false).as_deref(), Some(" мир"));
        assert_eq!(a.push_segment("привет мир", true).as_deref(), None);
        assert_eq!(a.text(), "привет мир");
        // Second utterance
        assert_eq!(
            a.push_segment("как дела", true).as_deref(),
            Some(" как дела")
        );
        assert_eq!(a.text(), "привет мир как дела");
    }

    #[test]
    fn test_sse_payloads() {
        let d = sse_delta_payload(" hi");
        let v: serde_json::Value = serde_json::from_str(&d).unwrap();
        assert_eq!(v["type"], "transcript.text.delta");
        assert_eq!(v["delta"], " hi");
        let done = sse_done_payload("hello");
        let v: serde_json::Value = serde_json::from_str(&done).unwrap();
        assert_eq!(v["type"], "transcript.text.done");
        assert_eq!(v["text"], "hello");
    }

    #[test]
    fn test_invalid_response_format_field() {
        let mut opts = OpenAITranscriptionOptions::default();
        let err = apply_openai_form_field(&mut opts, "response_format", b"pdf").unwrap_err();
        assert!(err.contains("response_format"));
    }

    #[test]
    fn test_json_response_is_text_only() {
        let resp = OpenAIJsonResponse {
            text: "привет".into(),
        };
        let v = serde_json::to_value(&resp).unwrap();
        assert_eq!(v["text"], "привет");
        assert_eq!(v.as_object().unwrap().len(), 1);
    }

    #[test]
    fn test_verbose_default_has_segments_no_words() {
        let result = sample_result();
        let mut opts = OpenAITranscriptionOptions {
            response_format: OpenAIResponseFormat::VerboseJson,
            language: Some("ru".into()),
            ..Default::default()
        };
        finalize_openai_options(&mut opts).unwrap();
        let v = serde_json::to_value(build_verbose_response(&result, &opts)).unwrap();
        assert_eq!(v["task"], "transcribe");
        assert_eq!(v["language"], "ru");
        assert_eq!(v["duration"], 2.0);
        assert_eq!(v["text"], "привет мир");
        assert!(v["segments"].is_array());
        // Two words with a long pause → two segments.
        assert_eq!(v["segments"].as_array().unwrap().len(), 2);
        assert!(v.get("words").is_none());
        // Whisper-shaped segment fields present.
        assert_eq!(v["segments"][0]["id"], 0);
        assert!(
            v["segments"][0]["text"]
                .as_str()
                .unwrap()
                .contains("привет")
        );
    }

    #[test]
    fn test_verbose_word_granularity() {
        let result = sample_result();
        let mut opts = OpenAITranscriptionOptions {
            response_format: OpenAIResponseFormat::VerboseJson,
            include_words: true,
            include_segments: false,
            language: Some("English".into()),
            ..Default::default()
        };
        finalize_openai_options(&mut opts).unwrap();
        let v = serde_json::to_value(build_verbose_response(&result, &opts)).unwrap();
        assert_eq!(v["language"], "en");
        assert!(v.get("segments").is_none());
        assert_eq!(v["words"].as_array().unwrap().len(), 2);
        assert_eq!(v["words"][0]["word"], "привет");
        assert_eq!(v["words"][0]["start"], 0.0);
        assert_eq!(v["words"][0]["end"], 0.5);
    }

    #[test]
    fn test_verbose_both_granularities() {
        let result = sample_result();
        let mut opts = OpenAITranscriptionOptions {
            response_format: OpenAIResponseFormat::VerboseJson,
            include_words: true,
            include_segments: true,
            ..Default::default()
        };
        finalize_openai_options(&mut opts).unwrap();
        let v = serde_json::to_value(build_verbose_response(&result, &opts)).unwrap();
        assert!(v["segments"].is_array());
        assert!(v["words"].is_array());
    }

    #[test]
    fn test_render_json_and_text_content_types() {
        let result = sample_result();
        let json_opts = OpenAITranscriptionOptions::default();
        let resp = render_openai_response(&result, &json_opts);
        assert_eq!(resp.status(), StatusCode::OK);

        let text_opts = OpenAITranscriptionOptions {
            response_format: OpenAIResponseFormat::Text,
            ..Default::default()
        };
        let resp = render_openai_response(&result, &text_opts);
        assert_eq!(resp.status(), StatusCode::OK);
        let ct = resp
            .headers()
            .get(header::CONTENT_TYPE)
            .unwrap()
            .to_str()
            .unwrap();
        assert!(ct.starts_with("text/plain"), "got {ct}");
    }

    #[test]
    fn test_render_srt_and_vtt() {
        let result = sample_result();
        for fmt in [OpenAIResponseFormat::Srt, OpenAIResponseFormat::Vtt] {
            let opts = OpenAITranscriptionOptions {
                response_format: fmt,
                ..Default::default()
            };
            let resp = render_openai_response(&result, &opts);
            assert_eq!(resp.status(), StatusCode::OK);
        }
    }

    /// Multipart helper shared by integration-style unit tests.
    fn multipart_body(boundary: &str, fields: &[(&str, Option<&str>, &[u8])]) -> Vec<u8> {
        let mut body = Vec::new();
        for (name, filename, value) in fields {
            body.extend_from_slice(format!("--{boundary}\r\n").as_bytes());
            match filename {
                Some(fname) => body.extend_from_slice(
                    format!(
                        "Content-Disposition: form-data; name=\"{name}\"; filename=\"{fname}\"\r\n\
                         Content-Type: application/octet-stream\r\n\r\n"
                    )
                    .as_bytes(),
                ),
                None => body.extend_from_slice(
                    format!("Content-Disposition: form-data; name=\"{name}\"\r\n\r\n").as_bytes(),
                ),
            }
            body.extend_from_slice(value);
            body.extend_from_slice(b"\r\n");
        }
        body.extend_from_slice(format!("--{boundary}--\r\n").as_bytes());
        body
    }

    #[tokio::test]
    async fn test_parse_multipart_full_form() {
        use axum::Router;
        use axum::routing::post;

        let app = Router::new().route(
            "/t",
            post(|multipart: Multipart| async move {
                match parse_openai_multipart(multipart).await {
                    Ok(req) => {
                        let v = serde_json::json!({
                            "file_len": req.file.len(),
                            "model": req.options.model,
                            "language": req.options.language,
                            "format": match req.options.response_format {
                                OpenAIResponseFormat::VerboseJson => "verbose_json",
                                OpenAIResponseFormat::Json => "json",
                                OpenAIResponseFormat::Text => "text",
                                OpenAIResponseFormat::Srt => "srt",
                                OpenAIResponseFormat::Vtt => "vtt",
                            },
                            "words": req.options.include_words,
                            "segments": req.options.include_segments,
                        });
                        (StatusCode::OK, Json(v)).into_response()
                    }
                    Err(resp) => resp,
                }
            }),
        );

        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = listener.local_addr().unwrap().port();
        tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });

        let boundary = "----gigasttTestBoundary";
        let file_bytes = b"RIFF....fake-wav-payload";
        let body = multipart_body(
            boundary,
            &[
                ("model", None, b"whisper-1"),
                ("language", None, b"ru"),
                ("response_format", None, b"verbose_json"),
                ("timestamp_granularities[]", None, b"word"),
                ("timestamp_granularities[]", None, b"segment"),
                ("file", Some("clip.wav"), file_bytes),
            ],
        );

        let resp = reqwest::Client::new()
            .post(format!("http://127.0.0.1:{port}/t"))
            .header(
                "content-type",
                format!("multipart/form-data; boundary={boundary}"),
            )
            .body(body)
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 200);
        let v: serde_json::Value = serde_json::from_str(&resp.text().await.unwrap()).unwrap();
        assert_eq!(v["file_len"], file_bytes.len());
        assert_eq!(v["model"], "whisper-1");
        assert_eq!(v["language"], "ru");
        assert_eq!(v["format"], "verbose_json");
        assert_eq!(v["words"], true);
        assert_eq!(v["segments"], true);
    }

    #[tokio::test]
    async fn test_parse_multipart_missing_file() {
        use axum::Router;
        use axum::routing::post;

        let app = Router::new().route(
            "/t",
            post(|multipart: Multipart| async move {
                match parse_openai_multipart(multipart).await {
                    Ok(_) => StatusCode::OK.into_response(),
                    Err(resp) => resp,
                }
            }),
        );

        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = listener.local_addr().unwrap().port();
        tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });

        let boundary = "----gigasttTestBoundary";
        let body = multipart_body(boundary, &[("model", None, b"whisper-1")]);
        let resp = reqwest::Client::new()
            .post(format!("http://127.0.0.1:{port}/t"))
            .header(
                "content-type",
                format!("multipart/form-data; boundary={boundary}"),
            )
            .body(body)
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 400);
        let v: serde_json::Value = serde_json::from_str(&resp.text().await.unwrap()).unwrap();
        assert_eq!(v["code"], "missing_file");
    }

    #[tokio::test]
    async fn test_parse_multipart_invalid_format() {
        use axum::Router;
        use axum::routing::post;

        let app = Router::new().route(
            "/t",
            post(|multipart: Multipart| async move {
                match parse_openai_multipart(multipart).await {
                    Ok(_) => StatusCode::OK.into_response(),
                    Err(resp) => resp,
                }
            }),
        );

        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = listener.local_addr().unwrap().port();
        tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });

        let boundary = "----gigasttTestBoundary";
        let body = multipart_body(
            boundary,
            &[
                ("response_format", None, b"pdf"),
                ("file", Some("a.wav"), b"x"),
            ],
        );
        let resp = reqwest::Client::new()
            .post(format!("http://127.0.0.1:{port}/t"))
            .header(
                "content-type",
                format!("multipart/form-data; boundary={boundary}"),
            )
            .body(body)
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 400);
        let v: serde_json::Value = serde_json::from_str(&resp.text().await.unwrap()).unwrap();
        assert_eq!(v["code"], "invalid_response_format");
    }

    #[tokio::test]
    async fn test_parse_multipart_empty_file() {
        use axum::Router;
        use axum::routing::post;

        let app = Router::new().route(
            "/t",
            post(|multipart: Multipart| async move {
                match parse_openai_multipart(multipart).await {
                    Ok(_) => StatusCode::OK.into_response(),
                    Err(resp) => resp,
                }
            }),
        );

        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = listener.local_addr().unwrap().port();
        tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });

        let boundary = "----gigasttTestBoundary";
        let body = multipart_body(boundary, &[("file", Some("empty.wav"), b"")]);
        let resp = reqwest::Client::new()
            .post(format!("http://127.0.0.1:{port}/t"))
            .header(
                "content-type",
                format!("multipart/form-data; boundary={boundary}"),
            )
            .body(body)
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 400);
        let v: serde_json::Value = serde_json::from_str(&resp.text().await.unwrap()).unwrap();
        assert_eq!(v["code"], "empty_body");
    }
}
