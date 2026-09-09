use clap::{CommandFactory, FromArgMatches, Parser, Subcommand};
use gigastt::boot::{ItnMode, PunctuationMode, parse_itn_mode, parse_punctuation_mode};
use gigastt_core::model;
use gigastt_core::model::{ModelVariant, ProgressMode};
use tracing_subscriber::EnvFilter;

mod packaging;
mod serve;
mod transcribe_cmd;
use packaging::{run_cache_gc, run_download, run_quantize};
use serve::{ServeArgs, run_serve};
use transcribe_cmd::{
    BatchOutputArgs, OfflineEngineArgs, run_transcribe, run_transcribe_batch, run_watch,
};

#[derive(Parser)]
#[command(
    name = "gigastt",
    version,
    about = "Local STT server powered by GigaAM v3",
    after_long_help = "Engine and post-processing options (--model-variant, --punctuation, --itn, --vad, ...) are defined on the subcommands, not at the top level.\nSee `gigastt serve --help` or `gigastt transcribe --help` for the full list."
)]
struct Cli {
    /// Log level [default: info]
    #[arg(long, global = true, default_value = "info")]
    log_level: String,

    /// Air-gapped mode: refuse every network fetch (model download, punctuation /
    /// diarization / VAD auto-fetch) with an instruction naming the missing file
    /// instead of a connect timeout. Equivalent to GIGASTT_OFFLINE=1.
    #[arg(long, global = true)]
    offline: bool,

    #[command(subcommand)]
    command: Commands,
}

#[allow(clippy::large_enum_variant)]
#[derive(Subcommand)]
enum Commands {
    /// Start WebSocket STT server (auto-downloads model if missing)
    Serve(ServeArgs),

    /// Download model without starting server
    Download {
        /// Model directory
        #[arg(long, default_value_t = model::default_model_dir())]
        model_dir: String,

        /// Recognition head to download: `rnnt` (default — lower WER, bare
        /// lowercase), `e2e_rnnt` (punctuation / casing / ITN), or `ml_ctc` /
        /// `ml_ctc_large` (GigaAM Multilingual charwise CTC, 220M / 600M —
        /// ru/en/kk/ky/uz, pre-quantized INT8 fetched directly).
        #[arg(
            long,
            env = "GIGASTT_MODEL_VARIANT",
            default_value = "rnnt",
            value_parser = parse_model_variant
        )]
        model_variant: ModelVariant,

        /// Skip downloading the speaker diarization model
        #[cfg(feature = "diarization")]
        #[arg(long, default_value_t = false)]
        skip_diarization: bool,

        /// Progress reporting format: `human` (default — interactive `\r`
        /// progress on stderr) or `json` (NDJSON events on stdout, one object
        /// per line, for sidecar integrators; human progress and tracing logs
        /// stay off stdout in this mode).
        #[arg(
            long,
            env = "GIGASTT_DOWNLOAD_PROGRESS",
            default_value = "human",
            value_parser = parse_progress_mode
        )]
        progress: ProgressMode,

        /// Also fetch the per-bucket palettized ANE (Core ML) encoder packages
        /// into `~/.gigastt/models/ane/` for the macOS Neural Engine backend.
        /// Requires a published ANE release.
        #[cfg(feature = "ane")]
        #[arg(long, default_value_t = false)]
        ane: bool,
    },

    /// Packaging: quantize a local **FP32** encoder ONNX to INT8.
    /// Runtime inference never uses FP32 — prefer `gigastt download` for INT8.
    Quantize {
        /// Model directory holding the FP32 encoder source
        #[arg(long, default_value_t = model::default_model_dir())]
        model_dir: String,

        /// Force re-quantization even if INT8 model exists
        #[arg(long)]
        force: bool,
    },

    /// Prune stale ONNX Runtime optimized graphs and stale CoreML compiled-model
    /// caches (and optionally hardlink exact duplicate files) under the model
    /// directory. Reclaims disk on multi-head / FP32-polluted installs and after
    /// an ONNX Runtime upgrade, without changing accuracy.
    CacheGc {
        /// Model directory
        #[arg(long, default_value_t = model::default_model_dir())]
        model_dir: String,

        /// Report reclaimable files without deleting or hardlinking
        #[arg(long, default_value_t = false)]
        dry_run: bool,

        /// Also hardlink content-identical files under the model dir
        /// (SHA-256 groups). Off by default — optimized_cache prune always runs.
        #[arg(long, default_value_t = false)]
        dedupe: bool,

        /// Directory of the ORT optimized-graph cache to prune. Defaults to
        /// `<model-dir>/optimized_cache`; set this when the cache was
        /// relocated via `serve --optimized-cache-dir`. Env:
        /// GIGASTT_OPTIMIZED_CACHE_DIR.
        #[arg(long, env = "GIGASTT_OPTIMIZED_CACHE_DIR", value_name = "PATH")]
        optimized_cache_dir: Option<String>,
    },

    /// Transcribe an audio file (offline)
    Transcribe {
        /// Path to audio file (WAV, M4A, MP3, OGG, Opus, WebM, FLAC; raw telephony via --codec)
        file: String,

        /// Model directory
        #[arg(long, default_value_t = model::default_model_dir())]
        model_dir: String,

        /// Recognition head to use. Omit to auto-detect from the model
        /// directory (existing install used as-is; only downloads if empty).
        /// `rnnt` (lower WER, bare lowercase), `e2e_rnnt` (punctuation /
        /// casing / ITN), or `ml_ctc` / `ml_ctc_large` (GigaAM Multilingual
        /// charwise CTC, 220M / 600M — ru/en/kk/ky/uz, bare lowercase).
        /// Env: GIGASTT_MODEL_VARIANT.
        #[arg(
            long,
            env = "GIGASTT_MODEL_VARIANT",
            value_parser = parse_model_variant
        )]
        model_variant: Option<ModelVariant>,

        /// Punctuation + capitalization restoration: `on`, `off`, or `auto`.
        /// `auto` (default) enables it for `rnnt`, disables it for `e2e_rnnt`.
        /// Env: GIGASTT_PUNCTUATION.
        #[arg(
            long,
            env = "GIGASTT_PUNCTUATION",
            default_value = "auto",
            value_parser = parse_punctuation_mode
        )]
        punctuation: PunctuationMode,

        /// Directory holding the optional punctuation model. Defaults to
        /// `~/.gigastt/models/punct/`. Auto-downloaded from
        /// `ekhodzitsky/rupunct-small-onnx` when enabled and absent.
        /// Env: GIGASTT_PUNCT_MODEL_DIR.
        #[arg(
            long,
            env = "GIGASTT_PUNCT_MODEL_DIR",
            default_value_t = model::default_punct_model_dir()
        )]
        punct_model_dir: String,

        /// Inverse text normalization (Russian number-words → digits):
        /// `on`, `off`, or `auto`. `auto` (default) enables it for `rnnt`,
        /// disables it for `e2e_rnnt`. Runs before punctuation. Env: GIGASTT_ITN.
        #[arg(
            long,
            env = "GIGASTT_ITN",
            default_value = "auto",
            value_parser = parse_itn_mode
        )]
        itn: ItnMode,

        /// Contextual hotword biasing: path to a file of phrases to boost during
        /// recognition (one phrase per line, optional `\t<weight>` suffix; blank
        /// lines and `#` comments ignored). Off when unset. Env:
        /// GIGASTT_HOTWORDS_FILE.
        #[arg(long, env = "GIGASTT_HOTWORDS_FILE")]
        hotwords_file: Option<String>,

        /// Also bias the built-in Russian brand/acronym lexicon. Combined with
        /// any `--hotwords-file` phrases. Env: GIGASTT_HOTWORDS_DEFAULT.
        #[arg(long, env = "GIGASTT_HOTWORDS_DEFAULT", default_value_t = false)]
        hotwords_default: bool,

        /// Additive logit boost applied to hotword continuation tokens during
        /// greedy decode [default: 5.0]. Higher = stronger bias. No effect
        /// unless hotwords are configured. Env: GIGASTT_HOTWORDS_BOOST.
        #[arg(long, env = "GIGASTT_HOTWORDS_BOOST")]
        hotwords_boost: Option<f32>,

        /// Voice activity detection: skip silence before decoding. Off by
        /// default; downloads the Silero VAD model (MIT) on first use. Env:
        /// GIGASTT_VAD.
        #[arg(long, env = "GIGASTT_VAD", default_value_t = false)]
        vad: bool,

        /// VAD speech-probability threshold in [0,1] [default: 0.5]. Higher =
        /// stricter. No effect unless `--vad`. Env: GIGASTT_VAD_THRESHOLD.
        #[arg(long, env = "GIGASTT_VAD_THRESHOLD")]
        vad_threshold: Option<f32>,

        /// Minimum trailing silence (ms) to close a speech region [default: 500].
        /// No effect unless `--vad`. Env: GIGASTT_VAD_MIN_SILENCE_MS.
        #[arg(long, env = "GIGASTT_VAD_MIN_SILENCE_MS")]
        vad_min_silence_ms: Option<u32>,

        /// Directory holding the Silero VAD model (`silero_vad.onnx`). Defaults
        /// to `~/.gigastt/models/vad/`. Auto-downloaded when `--vad` is set and
        /// the model is absent. Env: GIGASTT_VAD_MODEL_DIR.
        #[arg(long, env = "GIGASTT_VAD_MODEL_DIR", default_value_t = model::default_vad_model_dir())]
        vad_model_dir: String,

        /// Intra-op thread count for the encoder session on the CPU build. The
        /// encoder dominates the single-utterance cost, so more threads speed up
        /// long single-file jobs on weak CPUs. When unset, defaults to the logical
        /// CPU count (offline transcription runs a single triplet). Do not set
        /// `1` on multi-core hosts unless debugging — it is ~3× slower than auto.
        /// An explicit value (flag or env, including `1`) is still honoured as-is
        /// for debug passthrough. No effect on CoreML / CUDA builds.
        #[arg(long, env = "GIGASTT_ENCODER_INTRA_THREADS")]
        encoder_intra_threads: Option<usize>,

        /// Max pooled triplets this file decode may hold to run overlapping
        /// 24 s windows in parallel. Default `1` is serial (one triplet, all
        /// cores). `2` loads two triplets and splits encoder threads across
        /// them. Env: GIGASTT_FILE_WINDOW_CONCURRENCY.
        #[arg(long, env = "GIGASTT_FILE_WINDOW_CONCURRENCY", default_value_t = 1)]
        file_window_concurrency: usize,

        /// Export format: json, txt, srt, vtt, md [default: txt]
        #[arg(short, long, env = "GIGASTT_FORMAT", default_value = "txt")]
        format: String,

        /// Output file. When omitted, prints to stdout.
        #[arg(short, long, env = "GIGASTT_OUTPUT")]
        output: Option<String>,

        /// Maximum characters per subtitle/caption line (SRT/VTT) [default: 80]
        #[arg(long, env = "GIGASTT_MAX_CHARS_PER_LINE")]
        max_chars_per_line: Option<usize>,

        /// Maximum words per subtitle/caption line (SRT/VTT) [default: 14]
        #[arg(long, env = "GIGASTT_MAX_WORDS_PER_LINE")]
        max_words_per_line: Option<usize>,

        /// Include per-word timestamps in Markdown output
        #[arg(long, env = "GIGASTT_WORD_TIMESTAMPS", default_value_t = false)]
        word_timestamps: bool,

        /// Transcribe left/right channels as separate speakers (channel 0 = speaker_0,
        /// channel 1 = speaker_1). Falls back to mono for mono files, dual-mono stereo
        /// files, and files with more than two channels. Env: GIGASTT_STEREO_SPEAKERS.
        #[arg(long, env = "GIGASTT_STEREO_SPEAKERS", default_value_t = false)]
        stereo_speakers: bool,

        /// Raw headerless telephony codec of the input file: `pcmu` (alias
        /// `ulaw`), `pcma` (alias `alaw`), or `g722`. When set, the file is
        /// decoded as a raw byte stream (RTP dump, Asterisk Monitor raw)
        /// instead of sniffing a container. Requires `--sample-rate`.
        /// Env: GIGASTT_CODEC.
        #[arg(long, env = "GIGASTT_CODEC", requires = "sample_rate")]
        codec: Option<String>,

        /// Sample rate (Hz) of a raw `--codec` input (typical telephony: 8000).
        /// G.722 decodes to its native 16 kHz; both 8000 (the SDP clock-rate
        /// convention) and 16000 are accepted for it. Env: GIGASTT_SAMPLE_RATE.
        #[arg(long, env = "GIGASTT_SAMPLE_RATE")]
        sample_rate: Option<u32>,
    },

    /// Transcribe every audio file in a directory (offline, one-shot)
    TranscribeBatch {
        /// Directory scanned recursively for audio files (WAV, MP3, M4A, OGG, FLAC, WebM)
        input_dir: String,

        /// Directory the `<stem>.<ext>` transcripts are written into
        output_dir: String,

        #[command(flatten)]
        engine: OfflineEngineArgs,

        #[command(flatten)]
        output: BatchOutputArgs,
    },

    /// Watch a directory and transcribe new/changed audio files as they appear
    Watch {
        /// Directory polled for audio files (WAV, MP3, M4A, OGG, FLAC, WebM). Files
        /// already present at startup are registered but not transcribed.
        input_dir: String,

        /// Directory the `<stem>.<ext>` transcripts are written into
        output_dir: String,

        #[command(flatten)]
        engine: OfflineEngineArgs,

        #[command(flatten)]
        output: BatchOutputArgs,

        /// Poll interval in milliseconds [default: 1000]. Polling with a
        /// stability check keeps the watcher dependency-free and handles
        /// files still being copied into the directory.
        /// Env: GIGASTT_WATCH_POLL_INTERVAL_MS.
        #[arg(long, env = "GIGASTT_WATCH_POLL_INTERVAL_MS", default_value_t = 1000)]
        poll_interval_ms: u64,

        /// Consecutive polls with an identical size+mtime required before a
        /// file is scheduled [default: 2]. Env: GIGASTT_WATCH_SETTLE_POLLS.
        #[arg(long, env = "GIGASTT_WATCH_SETTLE_POLLS", default_value_t = 2)]
        settle_polls: u32,
    },
}

/// clap value parser for `--model-variant`. Accepts `rnnt` / `e2e_rnnt` /
/// `ml_ctc` / `ml_ctc_large` (case-insensitive); see [`ModelVariant::from_str`].
fn parse_model_variant(s: &str) -> Result<ModelVariant, String> {
    s.parse()
}

/// Parse the `download --progress` value (`human` | `json`).
fn parse_progress_mode(s: &str) -> Result<ProgressMode, String> {
    s.parse()
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let matches = Cli::command().get_matches();
    let cli = Cli::from_arg_matches(&matches).unwrap_or_else(|e| e.exit());

    if cli.offline {
        // Translate the flag into the env var the download guard in
        // gigastt-core reads, so both spellings behave identically.
        // Safety: this is the first statement after argument parsing — nothing
        // has read or written the process environment concurrently yet (the
        // only env readers live further down this same call path).
        unsafe { std::env::set_var("GIGASTT_OFFLINE", "1") };
    }
    // NDJSON download progress owns stdout: in `--progress=json` mode the
    // tracing writer moves to stderr so stdout carries nothing but event
    // lines (the default writer is stdout).
    let json_progress = matches!(
        &cli.command,
        Commands::Download { progress, .. } if *progress == ProgressMode::Json
    );

    let directive = format!("gigastt={}", cli.log_level);
    let filter = EnvFilter::from_default_env().add_directive(directive.parse()?);
    if json_progress {
        tracing_subscriber::fmt()
            .with_env_filter(filter)
            .with_writer(std::io::stderr)
            .init();
    } else {
        tracing_subscriber::fmt().with_env_filter(filter).init();
    }

    match cli.command {
        Commands::Serve(args) => {
            run_serve(&matches, args).await?;
        }
        Commands::Download {
            model_dir,
            model_variant,
            #[cfg(feature = "diarization")]
            skip_diarization,
            progress,
            #[cfg(feature = "ane")]
            ane,
        } => {
            run_download(
                model_dir,
                model_variant,
                #[cfg(feature = "diarization")]
                skip_diarization,
                progress,
                #[cfg(feature = "ane")]
                ane,
            )
            .await?;
        }
        Commands::Quantize { model_dir, force } => {
            run_quantize(model_dir, force)?;
        }
        Commands::CacheGc {
            model_dir,
            dry_run,
            dedupe,
            optimized_cache_dir,
        } => {
            run_cache_gc(model_dir, dry_run, dedupe, optimized_cache_dir)?;
        }
        Commands::Transcribe {
            file,
            model_dir,
            model_variant,
            punctuation,
            punct_model_dir,
            itn,
            hotwords_file,
            hotwords_default,
            hotwords_boost,
            vad,
            vad_threshold,
            vad_min_silence_ms,
            vad_model_dir,
            encoder_intra_threads,
            file_window_concurrency,
            format,
            output,
            max_chars_per_line,
            max_words_per_line,
            word_timestamps,
            stereo_speakers,
            codec,
            sample_rate,
        } => {
            run_transcribe(
                file,
                model_dir,
                model_variant,
                punctuation,
                punct_model_dir,
                itn,
                hotwords_file,
                hotwords_default,
                hotwords_boost,
                vad,
                vad_threshold,
                vad_min_silence_ms,
                vad_model_dir,
                encoder_intra_threads,
                file_window_concurrency,
                format,
                output,
                max_chars_per_line,
                max_words_per_line,
                word_timestamps,
                stereo_speakers,
                codec,
                sample_rate,
            )
            .await?;
        }
        Commands::TranscribeBatch {
            input_dir,
            output_dir,
            engine: eng,
            output: out,
        } => {
            run_transcribe_batch(input_dir, output_dir, eng, out).await?;
        }
        Commands::Watch {
            input_dir,
            output_dir,
            engine: eng,
            output: out,
            poll_interval_ms,
            settle_polls,
        } => {
            run_watch(
                input_dir,
                output_dir,
                eng,
                out,
                poll_interval_ms,
                settle_polls,
            )
            .await?;
        }
    }

    Ok(())
}

#[cfg(test)]
mod cli_tests;
