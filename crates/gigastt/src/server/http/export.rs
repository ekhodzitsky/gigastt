//! Export query params and response rendering for `/v1/transcribe` and jobs.

use axum::http::StatusCode;
use axum::http::header;
use axum::response::Response;
use serde::Deserialize;
use std::str::FromStr;

use gigastt_core::export::{ExportFormat, RenderOpts};

use super::error::{ApiError, api_error};

/// Query parameters for `/v1/transcribe` export formatting.
#[derive(Debug, Default, Clone, Deserialize)]
pub struct ExportParams {
    /// Export format: `json` (default), `txt`, `srt`, `vtt`, `md`.
    pub format: Option<String>,
    /// When set, the response carries `Content-Disposition: attachment` with this
    /// filename (or `transcript.<ext>` if the value is empty).
    pub download: Option<String>,
    /// Maximum characters per subtitle/caption line. `0` = unlimited.
    #[serde(default)]
    pub max_chars_per_line: Option<usize>,
    /// Maximum words per subtitle/caption line. `0` = unlimited.
    #[serde(default)]
    pub max_words_per_line: Option<usize>,
    /// Include per-word timestamps in Markdown output.
    #[serde(default)]
    pub word_timestamps: Option<bool>,
    /// Return cue-sized segments. In the default (JSON) response this adds a
    /// `segments` array; combined with `format=md` it switches Markdown to
    /// `### [mm:ss]` segment headers. Ignored for `txt`/`srt`/`vtt` (those are
    /// already flat / cue-based).
    #[serde(default)]
    pub segments: Option<bool>,
    /// Per-request override for the punctuation / casing restoration pass.
    /// `Some(true)` forces it on (409 `punctuation_not_available` if no
    /// punctuation model is loaded), `Some(false)` skips it, absent = the
    /// server's boot default. Applies to `POST /v1/transcribe` only.
    #[serde(default)]
    pub punctuation: Option<bool>,
    /// Per-request override for inverse text normalization (number-words →
    /// digits). `Some(true)`/`Some(false)` force the state, absent = boot
    /// default. Pure code (no model), so always accepted. `POST /v1/transcribe`
    /// only.
    #[serde(default)]
    pub itn: Option<bool>,
    /// Per-request override for VAD gating. `Some(true)` decodes only detected
    /// speech (409 `vad_not_loaded` if no VAD is loaded), `Some(false)` decodes
    /// the whole buffer, absent = boot default. `POST /v1/transcribe` only.
    #[serde(default)]
    pub vad: Option<bool>,
    /// Per-request hotword phrases, comma-separated
    /// (`?hotwords=сбер,тинькофф`). Absent = keep the engine boot biaser;
    /// present with an empty value (or only empty segments) forces biasing
    /// off for this request; non-empty replaces the biaser for this request
    /// only. Capped at 64 phrases / 64 chars each (400 on overflow).
    /// `POST /v1/transcribe` only.
    #[serde(default)]
    pub hotwords: Option<String>,
    /// Additive logit boost for per-request hotwords. Only applied when
    /// `hotwords` is also present; absent defaults to 5.0.
    /// `POST /v1/transcribe` only.
    #[serde(default)]
    pub hotwords_boost: Option<f32>,
    /// Forward-compatibility guard for a future multi-model server: names the
    /// recognition head the request expects. A single-variant engine can't
    /// switch, so any value other than the loaded variant returns 409
    /// `variant_not_loaded`; matching (or absent) proceeds. `POST /v1/transcribe`
    /// only.
    #[serde(default)]
    pub variant: Option<String>,
    /// Channel handling for file transcription. `split` transcribes left/right
    /// channels as separate speakers. Defaults to mono mix.
    #[serde(default)]
    pub channels: Option<String>,
    /// Request speaker diarization. Mutually exclusive with `channels=split`.
    #[serde(default)]
    pub diarization: Option<bool>,
    /// Raw headerless telephony codec of the upload body: `pcmu` (alias
    /// `ulaw`), `pcma` (alias `alaw`), or `g722`. When set, the body is
    /// decoded as a raw byte stream of that codec instead of sniffing a
    /// container — for RTP dumps and Asterisk Monitor raw captures. Requires
    /// `sample_rate`.
    #[serde(default)]
    pub codec: Option<String>,
    /// Sample rate (Hz) of a raw `codec` upload — mandatory when `codec` is
    /// set (typical telephony: 8000). G.722 decodes to its native 16 kHz
    /// regardless; both 8000 (the SDP clock-rate convention) and 16000 are
    /// accepted for it.
    #[serde(default)]
    pub sample_rate: Option<u32>,
}

/// Split a comma-separated `?hotwords=` value into trimmed non-empty phrases.
/// Empty input (or only empty segments) yields an empty list, which the engine
/// treats as force-off for this request.
pub(crate) fn parse_hotwords_query(raw: &str) -> Vec<String> {
    raw.split(',')
        .map(str::trim)
        .filter(|p| !p.is_empty())
        .map(str::to_string)
        .collect()
}

/// Build [`TranscribeOverrides`] from REST query params. Absent knob params
/// leave the corresponding override field as `None` (engine boot default).
pub(crate) fn overrides_from_export_params(
    params: &ExportParams,
) -> gigastt_core::inference::TranscribeOverrides {
    gigastt_core::inference::TranscribeOverrides {
        punctuation: params.punctuation,
        itn: params.itn,
        vad: params.vad,
    }
}

/// Build optional [`HotwordOverride`] from REST query params. `None` when the
/// `hotwords` key is absent (engine boot biaser). `Some` when the key is
/// present — even empty — so `?hotwords=` can force biasing off.
pub(crate) fn hotwords_from_export_params(
    params: &ExportParams,
) -> Option<gigastt_core::inference::HotwordOverride> {
    params.hotwords.as_ref().map(|raw| {
        gigastt_core::inference::HotwordOverride::new(
            parse_hotwords_query(raw),
            // Boost only applies when hotwords is present; ignore alone.
            params.hotwords_boost,
        )
    })
}

/// Render a transcription result into the requested export format.
///
/// Returns `None` when the caller explicitly requested the default JSON
/// response, so the handler can keep serving the existing `TranscribeResponse`
/// contract unchanged.
#[allow(clippy::result_large_err)]
pub(super) fn render_export_response(
    result: &gigastt_core::inference::TranscribeResult,
    params: &ExportParams,
) -> Result<Option<Response>, ApiError> {
    let format_str = params.format.as_deref().unwrap_or("json");
    if format_str.eq_ignore_ascii_case("json") {
        return Ok(None);
    }

    let format = ExportFormat::from_str(format_str)
        .map_err(|e| api_error(StatusCode::BAD_REQUEST, &format!("{e}"), "invalid_format"))?;

    let opts = RenderOpts {
        max_chars_per_line: params.max_chars_per_line.unwrap_or(80),
        max_words_per_line: params.max_words_per_line.unwrap_or(14),
        include_word_timestamps: params.word_timestamps.unwrap_or(false),
    };

    // Precedence for `format` × `segments`: only Markdown composes with
    // `segments=true`, switching to `### [mm:ss]` section headers over the same
    // cue boundaries as SRT/VTT. `txt`/`srt`/`vtt` ignore `segments` (flat /
    // already cue-based); plain `format=md` keeps the frontmatter + `# Transcript`
    // blob unchanged.
    let body = if format == ExportFormat::Md && params.segments.unwrap_or(false) {
        gigastt_core::export::to_md_segments(
            result,
            opts.max_chars_per_line,
            opts.max_words_per_line,
        )
    } else {
        format.render(result, &opts)
    };
    let mut builder = Response::builder()
        .status(StatusCode::OK)
        .header(header::CONTENT_TYPE, format.content_type());

    if let Some(filename) = &params.download {
        let filename = if filename.is_empty() {
            format!("transcript.{}", format.extension())
        } else {
            filename.clone()
        };
        // The filename is user-controlled (query param), so emit it as an
        // RFC 6266 value: quotes/semicolons can no longer inject extra header
        // parameters (filename spoofing) and non-ASCII names survive via
        // `filename*`. The helper output is pure ASCII, which makes the header
        // conversion infallible; keep the defensive fallback anyway.
        let disposition = header::HeaderValue::from_str(&content_disposition_attachment(&filename))
            .unwrap_or_else(|_| {
                header::HeaderValue::from_str(&format!(
                    "attachment; filename=\"transcript.{}\"",
                    format.extension()
                ))
                .expect("static content-disposition is always a valid header value")
            });
        builder = builder.header(header::CONTENT_DISPOSITION, disposition);
    }

    let response = builder.body(axum::body::Body::from(body)).map_err(|e| {
        api_error(
            StatusCode::INTERNAL_SERVER_ERROR,
            &format!("failed to build response: {e}"),
            "internal_error",
        )
    })?;
    Ok(Some(response))
}

/// Build a `Content-Disposition: attachment` value for a user-controlled
/// filename per RFC 6266. The legacy quoted `filename=` parameter carries an
/// ASCII fallback (`"` / `\` / control / non-ASCII characters replaced by
/// `_`), and the full name travels percent-encoded in `filename*=UTF-8''…`
/// (RFC 5987 attr-char set emitted verbatim, everything else as `%XX` UTF-8
/// bytes). Without the fallback sanitization a name like `x"; filename*=…`
/// would splice extra parameters into the header — `HeaderValue` only rejects
/// control characters, not quotes or semicolons — letting an attacker spoof
/// the download filename seen by the client. The output is always printable
/// ASCII, so `HeaderValue::from_str` accepts it unconditionally.
pub(super) fn content_disposition_attachment(filename: &str) -> String {
    let mut fallback = String::with_capacity(filename.len());
    for c in filename.chars() {
        match c {
            '"' | '\\' => fallback.push('_'),
            c if c.is_ascii() && !c.is_ascii_control() => fallback.push(c),
            _ => fallback.push('_'),
        }
    }
    let mut encoded = String::with_capacity(filename.len());
    for &b in filename.as_bytes() {
        if b.is_ascii_alphanumeric() || b"!#$&+-.^_`|~".contains(&b) {
            encoded.push(char::from(b));
        } else {
            encoded.push_str(&format!("%{b:02X}"));
        }
    }
    format!("attachment; filename=\"{fallback}\"; filename*=UTF-8''{encoded}")
}
