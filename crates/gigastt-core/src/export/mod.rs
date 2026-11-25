//! Output formatters for transcription results.
//!
//! Supports plain text, JSON, SRT, WebVTT, and Markdown export from the
//! [`TranscribeResult`] structure returned by the inference engine.

use crate::error::GigasttError;
use crate::transcript::{TranscribeResult, WordInfo};
use serde::Serialize;
use std::str::FromStr;

/// Supported export formats for transcription results.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ExportFormat {
    /// JSON with word-level metadata (default).
    #[default]
    Json,
    /// Plain text transcript only.
    Txt,
    /// SubRip subtitles.
    Srt,
    /// WebVTT subtitles.
    Vtt,
    /// Markdown with YAML frontmatter and optional speaker/timing sections.
    Md,
}

impl FromStr for ExportFormat {
    type Err = GigasttError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_ascii_lowercase().as_str() {
            "json" => Ok(Self::Json),
            "txt" | "text" => Ok(Self::Txt),
            "srt" => Ok(Self::Srt),
            "vtt" => Ok(Self::Vtt),
            "md" | "markdown" => Ok(Self::Md),
            _ => Err(GigasttError::InvalidInput {
                message: format!("unsupported export format: {s}"),
            }),
        }
    }
}

impl std::fmt::Display for ExportFormat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Json => write!(f, "json"),
            Self::Txt => write!(f, "txt"),
            Self::Srt => write!(f, "srt"),
            Self::Vtt => write!(f, "vtt"),
            Self::Md => write!(f, "md"),
        }
    }
}

impl ExportFormat {
    /// MIME type to serve for this format over HTTP.
    pub fn content_type(&self) -> &'static str {
        match self {
            Self::Json => "application/json; charset=utf-8",
            Self::Txt => "text/plain; charset=utf-8",
            Self::Srt => "application/x-subrip; charset=utf-8",
            Self::Vtt => "text/vtt; charset=utf-8",
            Self::Md => "text/markdown; charset=utf-8",
        }
    }

    /// Default file extension (without leading dot).
    pub fn extension(&self) -> &'static str {
        match self {
            Self::Json => "json",
            Self::Txt => "txt",
            Self::Srt => "srt",
            Self::Vtt => "vtt",
            Self::Md => "md",
        }
    }

    /// Render a [`TranscribeResult`] into this format.
    pub fn render(&self, result: &TranscribeResult, opts: &RenderOpts) -> String {
        match self {
            Self::Json => to_json(result),
            Self::Txt => to_txt(result),
            Self::Srt => to_srt(
                &result.words,
                opts.max_chars_per_line,
                opts.max_words_per_line,
            ),
            Self::Vtt => to_vtt(
                &result.words,
                opts.max_chars_per_line,
                opts.max_words_per_line,
            ),
            Self::Md => to_md(result, opts.include_word_timestamps),
        }
    }
}

/// Options controlling subtitle line breaking and Markdown detail level.
#[derive(Clone, Copy, Debug)]
pub struct RenderOpts {
    /// Maximum characters per subtitle/caption line. `0` means unlimited.
    pub max_chars_per_line: usize,
    /// Maximum words per subtitle/caption line. `0` means unlimited.
    pub max_words_per_line: usize,
    /// Include per-word timestamps in Markdown output.
    pub include_word_timestamps: bool,
}

impl Default for RenderOpts {
    fn default() -> Self {
        Self {
            max_chars_per_line: 80,
            max_words_per_line: 14,
            include_word_timestamps: false,
        }
    }
}

/// Serialize the full result as JSON, mirroring the current REST contract.
///
/// The REST API exposes `duration` rather than the internal `duration_s` field
/// name, so this formatter maps the field explicitly.
pub fn to_json(result: &TranscribeResult) -> String {
    serde_json::json!({
        "text": result.text,
        "words": result.words,
        "duration": result.duration_s,
    })
    .to_string()
}

/// Plain text transcript only.
pub fn to_txt(result: &TranscribeResult) -> String {
    result.text.clone()
}

/// SubRip (SRT) subtitles from word-level timings.
///
/// Words are grouped into lines respecting `max_chars_per_line` and
/// `max_words_per_line`. Speaker labels are rendered as `[SPEAKER_N] text`.
pub fn to_srt(words: &[WordInfo], max_chars_per_line: usize, max_words_per_line: usize) -> String {
    let cues = build_cues(words, max_chars_per_line, max_words_per_line);
    let mut out = String::new();
    for (i, cue) in cues.iter().enumerate() {
        if i > 0 {
            out.push('\n');
        }
        out.push_str(&(i + 1).to_string());
        out.push('\n');
        out.push_str(&format_srt_time(cue.start));
        out.push_str(" --> ");
        out.push_str(&format_srt_time(cue.end));
        out.push('\n');
        out.push_str(&cue.text);
        out.push('\n');
    }
    out
}

/// WebVTT subtitles from word-level timings.
pub fn to_vtt(words: &[WordInfo], max_chars_per_line: usize, max_words_per_line: usize) -> String {
    let cues = build_cues(words, max_chars_per_line, max_words_per_line);
    let mut out = String::from("WEBVTT\n\n");
    for cue in &cues {
        out.push_str(&format_vtt_time(cue.start));
        out.push_str(" --> ");
        out.push_str(&format_vtt_time(cue.end));
        out.push('\n');
        out.push_str(&cue.text);
        out.push('\n');
        out.push('\n');
    }
    out
}

/// Markdown export with YAML frontmatter and an optional word-level appendix.
pub fn to_md(result: &TranscribeResult, include_word_timestamps: bool) -> String {
    let speaker_count = result
        .words
        .iter()
        .filter_map(|w| w.speaker)
        .max()
        .map(|m| m + 1)
        .unwrap_or(0);

    let mut out = String::new();
    out.push_str("---\n");
    out.push_str(&format!("duration: {}\n", result.duration_s));
    out.push_str("language: ru\n");
    out.push_str(&format!("speakers: {speaker_count}\n"));
    out.push_str("---\n\n");

    out.push_str("# Transcript\n\n");
    out.push_str(&result.text);
    out.push_str("\n\n");

    if include_word_timestamps && !result.words.is_empty() {
        out.push_str("# Word timings\n\n");
        out.push_str("| Word | Start | End | Confidence | Speaker |\n");
        out.push_str("|------|-------|-----|------------|---------|\n");
        for w in &result.words {
            let speaker = w
                .speaker
                .map(|s| format!("SPEAKER_{s}"))
                .unwrap_or_else(|| "-".to_string());
            out.push_str(&format!(
                "| {} | {:.3}s | {:.3}s | {:.3} | {speaker} |\n",
                w.word.replace('|', "\\|"),
                w.start,
                w.end,
                w.confidence
            ));
        }
    }

    out
}

/// Internal cue used for SRT/VTT line grouping.
///
/// Carries the words that fall within the cue's span so higher-level exports
/// (segment JSON, segment-grouped Markdown) can reuse the exact same grouping
/// boundaries as SRT/VTT instead of re-deriving them.
#[derive(Clone, Debug)]
struct Cue {
    start: f64,
    end: f64,
    text: String,
    words: Vec<WordInfo>,
}

/// A grouped transcript segment: a span of words with an aggregate start/end,
/// text, and optional speaker label. Used both for the natural-boundary
/// segments returned by `?segments=true` and for the cue-based segments behind
/// `format=md&segments=true`, SRT, and VTT.
#[derive(Clone, Debug, Serialize)]
pub struct Segment {
    /// Segment start time in seconds (start of its first word).
    pub start: f64,
    /// Segment end time in seconds (end of its last word).
    pub end: f64,
    /// Rendered segment text (speaker label prefix included only for cue-based
    /// caption exports; natural segments keep the label in `speaker`).
    pub text: String,
    /// The words that fall within this segment's span.
    pub words: Vec<WordInfo>,
    /// Speaker label when the segment came from diarization or channel split.
    /// Omitted from JSON for plain mono transcription to keep responses small.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub speaker: Option<u32>,
}

/// Pause gap that triggers a new natural segment (seconds). Chosen to split
/// typical conversational pauses without fragmenting normal word spacing.
const SEGMENT_PAUSE_THRESHOLD_S: f64 = 0.9;

/// Maximum duration of a natural segment (seconds). Utterances longer than
/// this are split even when no pause, punctuation, or speaker change occurred.
const MAX_SEGMENT_DURATION_S: f64 = 30.0;

/// Sentence-ending punctuation that forces a segment boundary after the word
/// that carries it.
const SEGMENT_SENTENCE_END_PUNCTUATION: &[char] = &['.', '!', '?'];

/// Return the common speaker label for a group of words, if all words share
/// the same non-`None` speaker. Used to populate `Segment::speaker` only when
/// diarization or channel split produced labels.
fn segment_speaker(words: &[WordInfo]) -> Option<u32> {
    let first = words.first()?.speaker?;
    if words.iter().all(|w| w.speaker == Some(first)) {
        Some(first)
    } else {
        None
    }
}

/// Group words into caption cues with speaker-aware line breaking.
fn build_cues(words: &[WordInfo], max_chars: usize, max_words: usize) -> Vec<Cue> {
    if words.is_empty() {
        return Vec::new();
    }

    let mut cues = Vec::new();
    let mut current = Cue {
        start: words[0].start,
        end: words[0].end,
        text: String::new(),
        words: Vec::new(),
    };
    let mut current_speaker: Option<u32> = None;
    let mut word_count = 0;

    let flush = |cue: &mut Cue, cues: &mut Vec<Cue>| {
        if !cue.text.is_empty() {
            // Trim trailing space left by append_word.
            cue.text = cue.text.trim_end().to_string();
            cues.push(cue.clone());
            cue.text.clear();
            cue.words.clear();
        }
    };

    for word in words {
        let speaker_changed = word.speaker != current_speaker;
        if speaker_changed {
            flush(&mut current, &mut cues);
            current.start = word.start;
            current_speaker = word.speaker;
            word_count = 0;
            if let Some(speaker) = word.speaker {
                current.text.push_str(&format!("[SPEAKER_{speaker}] "));
            }
        }

        let would_chars = if current.text.is_empty() {
            word.word.len()
        } else {
            current.text.len() + 1 + word.word.len()
        };
        let would_words = word_count + 1;

        let break_line = !current.text.is_empty()
            && ((max_chars > 0 && would_chars > max_chars)
                || (max_words > 0 && would_words > max_words));

        if break_line {
            flush(&mut current, &mut cues);
            current.start = word.start;
            current.end = word.end;
            word_count = 0;
            if let Some(speaker) = word.speaker {
                current.text.push_str(&format!("[SPEAKER_{speaker}] "));
            }
        }

        if !current.text.is_empty() && !current.text.ends_with(' ') {
            current.text.push(' ');
        }
        current.text.push_str(&word.word);
        current.end = word.end;
        current.words.push(word.clone());
        word_count += 1;
    }

    flush(&mut current, &mut cues);
    cues
}

/// Group a word list into cue-sized segments, reusing the SRT/VTT cue
/// boundaries so every export format agrees on segment spans.
///
/// Each returned [`Segment`] carries the words that fall within its span, so a
/// consumer can render segment-level UI (e.g. `### [mm:ss]` sections) without
/// re-deriving offsets from the flat per-word list.
pub fn to_segments(words: &[WordInfo], max_chars: usize, max_words: usize) -> Vec<Segment> {
    build_cues(words, max_chars, max_words)
        .into_iter()
        .map(|cue| Segment {
            start: cue.start,
            end: cue.end,
            text: cue.text,
            words: cue.words.clone(),
            speaker: segment_speaker(&cue.words),
        })
        .collect()
}

/// Group a word list into natural transcript segments using pause, sentence-end
/// punctuation, speaker change, and a maximum duration boundary.
///
/// This is the segmenter behind `POST /v1/transcribe?segments=true`. It is kept
/// separate from the SRT/VTT cue builder so subtitle line-breaking can remain
/// driven by `max_chars_per_line` / `max_words_per_line` while the JSON segment
/// array uses conversation-level boundaries.
pub fn to_transcript_segments(words: &[WordInfo]) -> Vec<Segment> {
    build_segments(words)
}

fn build_segments(words: &[WordInfo]) -> Vec<Segment> {
    if words.is_empty() {
        return Vec::new();
    }

    let mut segments = Vec::new();
    let mut current = Segment {
        start: words[0].start,
        end: words[0].end,
        text: words[0].word.clone(),
        words: vec![words[0].clone()],
        speaker: words[0].speaker,
    };

    for i in 1..words.len() {
        let word = &words[i];
        let prev = &words[i - 1];

        let pause = word.start - prev.end;
        let speaker_changed = word.speaker != prev.speaker;
        let prev_ended_sentence = prev
            .word
            .trim_end()
            .ends_with(SEGMENT_SENTENCE_END_PUNCTUATION);
        let would_exceed_duration = word.end - current.start > MAX_SEGMENT_DURATION_S;

        if pause > SEGMENT_PAUSE_THRESHOLD_S
            || speaker_changed
            || prev_ended_sentence
            || would_exceed_duration
        {
            current.speaker = segment_speaker(&current.words);
            segments.push(current);
            current = Segment {
                start: word.start,
                end: word.end,
                text: word.word.clone(),
                words: vec![word.clone()],
                speaker: word.speaker,
            };
        } else {
            current.text.push(' ');
            current.text.push_str(&word.word);
            current.end = word.end;
            current.words.push(word.clone());
        }
    }

    current.speaker = segment_speaker(&current.words);
    segments.push(current);
    segments
}

/// Segment-grouped Markdown: `### [mm:ss]` (or `[hh:mm:ss]` past one hour)
/// section headers per cue-sized segment, followed by that segment's text.
///
/// Shares its boundaries with SRT/VTT and `?segments=true` (all via
/// `build_cues`). Motivated by downstream consumers that otherwise fabricate
/// `### mm:ss` offsets because only flat per-word timings were exposed.
pub fn to_md_segments(result: &TranscribeResult, max_chars: usize, max_words: usize) -> String {
    let segments = to_segments(&result.words, max_chars, max_words);

    let speaker_count = result
        .words
        .iter()
        .filter_map(|w| w.speaker)
        .max()
        .map(|m| m + 1)
        .unwrap_or(0);

    let mut out = String::new();
    out.push_str("---\n");
    out.push_str(&format!("duration: {}\n", result.duration_s));
    out.push_str("language: ru\n");
    out.push_str(&format!("speakers: {speaker_count}\n"));
    out.push_str("---\n\n");

    for segment in &segments {
        out.push_str(&format!(
            "### [{}]\n\n",
            format_timestamp_hms(segment.start)
        ));
        out.push_str(&segment.text);
        out.push_str("\n\n");
    }

    out
}

/// Format a timestamp as `mm:ss`, widening to `hh:mm:ss` once it reaches one
/// hour. Used for the `### [mm:ss]` segment-Markdown headers.
fn format_timestamp_hms(seconds: f64) -> String {
    let total_s = seconds.max(0.0).round() as u64;
    let s = total_s % 60;
    let total_m = total_s / 60;
    let m = total_m % 60;
    let h = total_m / 60;
    if h > 0 {
        format!("{h:02}:{m:02}:{s:02}")
    } else {
        format!("{m:02}:{s:02}")
    }
}

fn format_srt_time(seconds: f64) -> String {
    let total_ms = (seconds.max(0.0) * 1000.0).round() as u64;
    let ms = total_ms % 1000;
    let total_s = total_ms / 1000;
    let s = total_s % 60;
    let total_m = total_s / 60;
    let m = total_m % 60;
    let h = total_m / 60;
    format!("{h:02}:{m:02}:{s:02},{ms:03}")
}

fn format_vtt_time(seconds: f64) -> String {
    let total_ms = (seconds.max(0.0) * 1000.0).round() as u64;
    let ms = total_ms % 1000;
    let total_s = total_ms / 1000;
    let s = total_s % 60;
    let total_m = total_s / 60;
    let m = total_m % 60;
    let h = total_m / 60;
    format!("{h:02}:{m:02}:{s:02}.{ms:03}")
}

#[cfg(test)]
mod tests;
