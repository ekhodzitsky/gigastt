//! Output formatters for transcription results.
//!
//! Supports plain text, JSON, SRT, WebVTT, and Markdown export from the
//! [`TranscribeResult`] structure returned by the inference engine.

use crate::error::GigasttError;
use crate::inference::{TranscribeResult, WordInfo};
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
#[derive(Clone, Debug)]
struct Cue {
    start: f64,
    end: f64,
    text: String,
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
    };
    let mut current_speaker: Option<u32> = None;
    let mut word_count = 0;

    let flush = |cue: &mut Cue, cues: &mut Vec<Cue>| {
        if !cue.text.is_empty() {
            // Trim trailing space left by append_word.
            cue.text = cue.text.trim_end().to_string();
            cues.push(cue.clone());
            cue.text.clear();
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
        word_count += 1;
    }

    flush(&mut current, &mut cues);
    cues
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
mod tests {
    use super::*;

    fn sample_words() -> Vec<WordInfo> {
        vec![
            WordInfo {
                word: "привет".to_string(),
                start: 0.0,
                end: 0.5,
                confidence: 0.98,
                speaker: Some(0),
            },
            WordInfo {
                word: "как".to_string(),
                start: 0.6,
                end: 0.9,
                confidence: 0.95,
                speaker: Some(0),
            },
            WordInfo {
                word: "дела".to_string(),
                start: 1.0,
                end: 1.4,
                confidence: 0.97,
                speaker: Some(1),
            },
        ]
    }

    fn sample_result() -> TranscribeResult {
        TranscribeResult {
            text: "привет как дела".to_string(),
            words: sample_words(),
            duration_s: 1.4,
        }
    }

    #[test]
    fn test_to_txt() {
        let result = sample_result();
        assert_eq!(to_txt(&result), "привет как дела");
    }

    #[test]
    fn test_to_json() {
        let result = sample_result();
        let json = to_json(&result);
        assert!(json.contains("привет как дела"));
        assert!(json.contains("\"duration\":1.4"));
    }

    #[test]
    fn test_to_srt() {
        let words = sample_words();
        let srt = to_srt(&words, 80, 14);
        assert!(srt.contains("00:00:00,000 -->"));
        assert!(srt.contains("[SPEAKER_0] привет как"));
        assert!(srt.contains("[SPEAKER_1] дела"));
        assert!(srt.starts_with("1\n"));
    }

    #[test]
    fn test_to_vtt() {
        let words = sample_words();
        let vtt = to_vtt(&words, 80, 14);
        assert!(vtt.starts_with("WEBVTT\n\n"));
        assert!(vtt.contains("00:00:00.000 -->"));
        assert!(vtt.contains("[SPEAKER_1] дела"));
    }

    #[test]
    fn test_to_md() {
        let result = sample_result();
        let md = to_md(&result, true);
        assert!(md.contains("duration: 1.4"));
        assert!(md.contains("speakers: 2"));
        assert!(md.contains("привет как дела"));
        assert!(md.contains("| Word | Start | End |"));
    }

    #[test]
    fn test_format_srt_time() {
        assert_eq!(format_srt_time(0.0), "00:00:00,000");
        assert_eq!(format_srt_time(61.123), "00:01:01,123");
        assert_eq!(format_srt_time(3661.5), "01:01:01,500");
    }

    #[test]
    fn test_format_vtt_time() {
        assert_eq!(format_vtt_time(0.0), "00:00:00.000");
        assert_eq!(format_vtt_time(61.123), "00:01:01.123");
    }

    #[test]
    fn test_export_format_from_str() {
        assert_eq!(ExportFormat::from_str("srt").unwrap(), ExportFormat::Srt);
        assert_eq!(ExportFormat::from_str("SRT").unwrap(), ExportFormat::Srt);
        assert_eq!(
            ExportFormat::from_str("markdown").unwrap(),
            ExportFormat::Md
        );
        assert!(ExportFormat::from_str("docx").is_err());
    }

    #[test]
    fn test_empty_words() {
        let words: Vec<WordInfo> = Vec::new();
        assert!(to_srt(&words, 80, 14).is_empty());
        assert!(to_vtt(&words, 80, 14) == "WEBVTT\n\n");
    }

    #[test]
    fn test_export_format_display_all_variants() {
        assert_eq!(ExportFormat::Json.to_string(), "json");
        assert_eq!(ExportFormat::Txt.to_string(), "txt");
        assert_eq!(ExportFormat::Srt.to_string(), "srt");
        assert_eq!(ExportFormat::Vtt.to_string(), "vtt");
        assert_eq!(ExportFormat::Md.to_string(), "md");
    }

    #[test]
    fn test_export_format_content_type_all_variants() {
        assert_eq!(
            ExportFormat::Json.content_type(),
            "application/json; charset=utf-8"
        );
        assert_eq!(
            ExportFormat::Txt.content_type(),
            "text/plain; charset=utf-8"
        );
        assert_eq!(
            ExportFormat::Srt.content_type(),
            "application/x-subrip; charset=utf-8"
        );
        assert_eq!(ExportFormat::Vtt.content_type(), "text/vtt; charset=utf-8");
        assert_eq!(
            ExportFormat::Md.content_type(),
            "text/markdown; charset=utf-8"
        );
    }

    #[test]
    fn test_export_format_extension_all_variants() {
        assert_eq!(ExportFormat::Json.extension(), "json");
        assert_eq!(ExportFormat::Txt.extension(), "txt");
        assert_eq!(ExportFormat::Srt.extension(), "srt");
        assert_eq!(ExportFormat::Vtt.extension(), "vtt");
        assert_eq!(ExportFormat::Md.extension(), "md");
    }

    #[test]
    fn test_render_dispatches_each_format() {
        let result = sample_result();
        let opts = RenderOpts::default();

        let json = ExportFormat::Json.render(&result, &opts);
        assert_eq!(json, to_json(&result));

        let txt = ExportFormat::Txt.render(&result, &opts);
        assert_eq!(txt, "привет как дела");

        let srt = ExportFormat::Srt.render(&result, &opts);
        assert!(srt.starts_with("1\n"));

        let vtt = ExportFormat::Vtt.render(&result, &opts);
        assert!(vtt.starts_with("WEBVTT\n\n"));

        let md = ExportFormat::Md.render(&result, &opts);
        assert!(md.starts_with("---\n"));
        // Default opts disable word timestamps, so no table is emitted.
        assert!(!md.contains("| Word | Start | End |"));
    }

    #[test]
    fn test_render_md_with_word_timestamps_opt_in() {
        let result = sample_result();
        let opts = RenderOpts {
            include_word_timestamps: true,
            ..RenderOpts::default()
        };
        let md = ExportFormat::Md.render(&result, &opts);
        assert!(md.contains("# Word timings"));
        assert!(md.contains("| Word | Start | End |"));
    }

    #[test]
    fn test_render_opts_default_values() {
        let opts = RenderOpts::default();
        assert_eq!(opts.max_chars_per_line, 80);
        assert_eq!(opts.max_words_per_line, 14);
        assert!(!opts.include_word_timestamps);
    }

    #[test]
    fn test_from_str_all_aliases() {
        assert_eq!(ExportFormat::from_str("json").unwrap(), ExportFormat::Json);
        assert_eq!(ExportFormat::from_str("txt").unwrap(), ExportFormat::Txt);
        assert_eq!(ExportFormat::from_str("text").unwrap(), ExportFormat::Txt);
        assert_eq!(ExportFormat::from_str("vtt").unwrap(), ExportFormat::Vtt);
        assert_eq!(ExportFormat::from_str("md").unwrap(), ExportFormat::Md);
    }

    #[test]
    fn test_to_md_no_speakers_zero_count() {
        let result = TranscribeResult {
            text: "no speaker words".to_string(),
            words: vec![WordInfo {
                word: "no".to_string(),
                start: 0.0,
                end: 0.3,
                confidence: 0.9,
                speaker: None,
            }],
            duration_s: 0.3,
        };
        let md = to_md(&result, true);
        assert!(md.contains("speakers: 0"));
        // Speaker column renders "-" when no speaker is assigned.
        assert!(md.contains("| - |"));
    }

    #[test]
    fn test_to_md_word_timestamps_skipped_when_empty() {
        let result = TranscribeResult {
            text: String::new(),
            words: Vec::new(),
            duration_s: 0.0,
        };
        let md = to_md(&result, true);
        // Empty word list means the appendix table is omitted entirely.
        assert!(!md.contains("# Word timings"));
        assert!(md.contains("speakers: 0"));
    }

    #[test]
    fn test_to_md_escapes_pipe_in_word() {
        let result = TranscribeResult {
            text: "a|b".to_string(),
            words: vec![WordInfo {
                word: "a|b".to_string(),
                start: 0.0,
                end: 0.5,
                confidence: 0.91,
                speaker: Some(2),
            }],
            duration_s: 0.5,
        };
        let md = to_md(&result, true);
        // Pipe in the word must be escaped to avoid breaking the table column.
        assert!(md.contains("a\\|b"));
        assert!(md.contains("SPEAKER_2"));
        assert!(md.contains("speakers: 3"));
    }

    #[test]
    fn test_srt_speaker_change_breaks_cue_with_label() {
        // Two speakers force a cue break; each cue carries its speaker label.
        let words = sample_words();
        let cues = build_cues(&words, 80, 14);
        assert_eq!(cues.len(), 2);
        assert!(cues[0].text.starts_with("[SPEAKER_0]"));
        assert!(cues[1].text.starts_with("[SPEAKER_1]"));
        assert!(cues[1].text.contains("дела"));
    }

    #[test]
    fn test_line_breaking() {
        let words: Vec<WordInfo> = (0..20)
            .map(|i| WordInfo {
                word: format!("word{i}"),
                start: i as f64,
                end: i as f64 + 0.4,
                confidence: 0.9,
                speaker: None,
            })
            .collect();
        let srt = to_srt(&words, 40, 5);
        let cue_count = srt.trim().split("\n\n").count();
        // 20 words / 5 per line = 4 cues, but exact count depends on chars.
        assert!(cue_count >= 2);
    }
}
