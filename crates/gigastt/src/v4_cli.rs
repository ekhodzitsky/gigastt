//! Explicit family selection, isolated from the legacy model variant recipe.
use anyhow::{Result, ensure};
use clap::{ArgMatches, Args, parser::ValueSource};
use gigastt_core::model::TranscriptStyle;
use std::path::PathBuf;

#[derive(Args, Default)]
pub(crate) struct FamilyArgs {
    /// Explicit model family: gigaam_v3 (legacy default) or gigaam_v4.
    #[arg(long, global=true, env="GIGASTT_MODEL_FAMILY", value_parser=["gigaam_v3","gigaam_v4"])]
    model_family: Option<String>,
    /// Explicit v4 recognition head (rnnt only).
    #[arg(long, global=true, env="GIGASTT_MODEL_HEAD", value_parser=["rnnt"])]
    model_head: Option<String>,
    /// Explicit v4 transcript style (raw or e2e).
    #[arg(long, global=true, env="GIGASTT_TRANSCRIPT_STYLE", value_parser=["raw","e2e"])]
    transcript_style: Option<String>,
    /// Explicit v4 model precision (fp32 only).
    #[arg(long, global=true, env="GIGASTT_MODEL_PRECISION", value_parser=["fp32"])]
    model_precision: Option<String>,
    /// Operator-supplied verified v4 MKL runtime directory.
    #[arg(
        long,
        global = true,
        env = "GIGASTT_V4_MKL_RUNTIME_DIR",
        value_name = "PATH"
    )]
    v4_mkl_runtime_dir: Option<PathBuf>,
}

pub(crate) struct V4Selection {
    pub style: TranscriptStyle,
    pub mkl_dir: PathBuf,
}

impl FamilyArgs {
    pub(crate) fn resolve(&self) -> Result<Option<V4Selection>> {
        if self.model_family.as_deref() != Some("gigaam_v4") {
            ensure!(
                self.model_head.is_none()
                    && self.transcript_style.is_none()
                    && self.model_precision.is_none()
                    && self.v4_mkl_runtime_dir.is_none(),
                "v4 model head/style/precision/runtime options require --model-family gigaam_v4"
            );
            return Ok(None);
        }
        ensure!(
            self.model_head.as_deref() == Some("rnnt")
                && self.model_precision.as_deref() == Some("fp32"),
            "v4 requires explicit --model-head rnnt and --model-precision fp32"
        );
        let style = match self.transcript_style.as_deref() {
            Some("raw") => TranscriptStyle::Raw,
            Some("e2e") => TranscriptStyle::E2e,
            _ => anyhow::bail!("v4 requires explicit --transcript-style raw or e2e"),
        };
        let mkl_dir = self
            .v4_mkl_runtime_dir
            .clone()
            .ok_or_else(|| anyhow::anyhow!("v4 requires --v4-mkl-runtime-dir PATH"))?;
        Ok(Some(V4Selection { style, mkl_dir }))
    }
}

fn explicit(args: &ArgMatches, id: &str) -> bool {
    matches!(
        args.value_source(id),
        Some(ValueSource::CommandLine | ValueSource::EnvVariable)
    )
}
pub(crate) fn validate_command(matches: &ArgMatches) -> Result<()> {
    let (command, args) = matches
        .subcommand()
        .ok_or_else(|| anyhow::anyhow!("missing subcommand"))?;
    ensure!(
        matches!(command, "serve" | "transcribe"),
        "v4 currently supports only serve and transcribe; no download, quantize, cache-gc, batch or watch"
    );
    let unsupported = [
        "model_variant",
        "punct_model_dir",
        "vad_model_dir",
        "hotwords_file",
        "hotwords_boost",
        "vad_threshold",
        "vad_min_silence_ms",
    ];
    for id in unsupported {
        ensure!(
            !explicit(args, id),
            "v4 does not support --{}",
            id.replace('_', "-")
        );
    }
    use gigastt::boot::{ItnMode, PunctuationMode};
    ensure!(
        !explicit(args, "punctuation")
            || args.get_one::<PunctuationMode>("punctuation") == Some(&PunctuationMode::Off),
        "v4 does not support punctuation restoration; use --punctuation off or omit it"
    );
    ensure!(
        !explicit(args, "itn") || args.get_one::<ItnMode>("itn") == Some(&ItnMode::Off),
        "v4 does not support extra ITN; use --itn off or omit it"
    );
    ensure!(
        !args.get_flag("hotwords_default") && !args.get_flag("vad"),
        "v4 does not support hotwords or VAD"
    );
    ensure!(
        args.get_one::<usize>("encoder_intra_threads")
            .is_none_or(|&n| n == 1)
            && args.get_one::<usize>("file_window_concurrency") == Some(&1),
        "v4 requires one CPU thread and one file request slot"
    );
    if command == "serve" {
        use gigastt_core::ExecutionProviderChoice;
        ensure!(
            !explicit(args, "execution_provider")
                || args.get_one::<ExecutionProviderChoice>("execution_provider")
                    == Some(&ExecutionProviderChoice::Cpu),
            "v4 requires explicit CPU execution (or omit provider)"
        );
        ensure!(
            !explicit(args, "pool_size") || args.get_one::<usize>("pool_size") == Some(&1),
            "v4 requires one request slot"
        );
        ensure!(
            args.get_one::<usize>("pool_min_size") == Some(&1)
                && args.get_one::<usize>("batch_pool_size") == Some(&0),
            "v4 requires pool minimum one and no batch pool"
        );
        ensure!(
            args.get_one::<u32>("jobs_retry").is_none_or(|&n| n == 0),
            "v4 job retries are unsupported; omit or set --jobs-retry 0"
        );
        ensure!(
            !args.get_flag("jobs_file_zero_context") && !args.get_flag("metrics"),
            "v4 does not support the legacy jobs zero-context or metrics option"
        );
        for id in [
            "stream_ctc_model_dir",
            "optimized_cache_dir",
            "endpoint_mode",
            "endpoint_silence_ms",
            "stream_max_window_secs",
            "stream_stable_prefix",
            "metrics_listen",
            "config",
        ] {
            ensure!(
                !explicit(args, id),
                "v4 file server does not support --{}",
                id.replace('_', "-")
            );
        }
        ensure!(
            args.get_one::<u64>("max_audio_secs")
                .is_none_or(|&n| n <= 30),
            "v4 supports at most 30 seconds per full-context file"
        );
        if explicit(args, "v4_streaming_model_dir") {
            gigastt_core::model::V4StreamContext::from_millis(
                args.get_one::<u16>("v4_streaming_future_context_ms")
                    .copied()
                    .unwrap_or(320),
            )?;
            ensure!(
                args.get_one::<usize>("ws_frame_max_bytes")
                    .is_none_or(|n| *n >= 2),
                "native v4 WebSocket frame limit must admit at least one PCM16 sample"
            );
        } else {
            for id in [
                "idle_timeout_secs",
                "ws_frame_max_bytes",
                "max_session_secs",
            ] {
                ensure!(
                    !explicit(args, id),
                    "v4 file server does not support --{} without a native streaming pack",
                    id.replace('_', "-")
                );
            }
        }
        // The edge profile implies legacy VAD/pool changes.
        ensure!(
            !explicit(args, "profile")
                || args.get_one::<super::serve::ServeProfile>("profile")
                    == Some(&super::serve::ServeProfile::Default),
            "v4 does not support the legacy edge profile"
        );
    } else {
        ensure!(
            !args.get_flag("stereo_speakers")
                && !explicit(args, "codec")
                && !explicit(args, "sample_rate"),
            "v4 does not support speaker splitting or raw headerless telephony"
        );
    }
    ensure!(
        cfg!(all(
            feature = "v4-mkl",
            target_os = "linux",
            target_arch = "x86_64",
            target_env = "gnu"
        )),
        "v4 requires a GNU Linux x86-64 build with --features v4-mkl"
    );
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub(crate) async fn transcribe(
    selection: V4Selection,
    file: String,
    model_dir: String,
    format: String,
    output: Option<String>,
    max_chars_per_line: Option<usize>,
    max_words_per_line: Option<usize>,
    word_timestamps: bool,
) -> Result<()> {
    #[cfg(all(
        feature = "v4-mkl",
        target_os = "linux",
        target_arch = "x86_64",
        target_env = "gnu"
    ))]
    {
        use gigastt_core::{
            export::{ExportFormat, RenderOpts},
            inference::v4::V4Engine,
        };
        use std::path::Path;
        use std::str::FromStr;
        let format = ExportFormat::from_str(&format).map_err(|e| anyhow::anyhow!("{e}"))?;
        let result = tokio::task::spawn_blocking(move || -> Result<_> {
            let engine = V4Engine::load(Path::new(&model_dir), &selection.mkl_dir)?;
            Ok(engine.transcribe_file(Path::new(&file), selection.style, None)?)
        })
        .await??;
        let result = gigastt::server::v4_result(result);
        let rendered = format.render(
            &result,
            &RenderOpts {
                max_chars_per_line: max_chars_per_line.unwrap_or(80),
                max_words_per_line: max_words_per_line.unwrap_or(14),
                include_word_timestamps: word_timestamps,
            },
        );
        if let Some(path) = output {
            std::fs::write(path, rendered)?;
        } else {
            println!("{rendered}");
        }
        Ok(())
    }
    #[cfg(not(all(
        feature = "v4-mkl",
        target_os = "linux",
        target_arch = "x86_64",
        target_env = "gnu"
    )))]
    {
        let _ = (
            selection.style,
            selection.mkl_dir,
            file,
            model_dir,
            format,
            output,
            max_chars_per_line,
            max_words_per_line,
            word_timestamps,
        );
        anyhow::bail!("v4 requires a GNU Linux x86-64 build with --features v4-mkl")
    }
}

pub(crate) async fn serve(
    selection: V4Selection,
    model_dir: String,
    streaming_dir: Option<PathBuf>,
    future_context_ms: u16,
    config: gigastt::server::ServerConfig,
) -> Result<()> {
    #[cfg(all(
        feature = "v4-mkl",
        target_os = "linux",
        target_arch = "x86_64",
        target_env = "gnu"
    ))]
    {
        use gigastt_core::inference::v4::V4Engine;
        use std::path::Path;
        let style = selection.style;
        let (engine, streaming) = tokio::task::spawn_blocking(move || -> Result<_> {
            let context = gigastt_core::model::V4StreamContext::from_millis(future_context_ms)?;
            let engine = V4Engine::load(Path::new(&model_dir), &selection.mkl_dir)?;
            let streaming = streaming_dir
                .as_deref()
                .map(|dir| {
                    gigastt::server::NativeV4Streaming::load(dir, &selection.mkl_dir, context)
                })
                .transpose()?;
            Ok((engine, streaming))
        })
        .await??;
        gigastt::server::run_v4_files_with_streaming(engine, style, config, streaming, None).await
    }
    #[cfg(not(all(
        feature = "v4-mkl",
        target_os = "linux",
        target_arch = "x86_64",
        target_env = "gnu"
    )))]
    {
        let _ = (
            selection.style,
            selection.mkl_dir,
            model_dir,
            streaming_dir,
            future_context_ms,
            config,
        );
        anyhow::bail!("v4 requires a GNU Linux x86-64 build with --features v4-mkl")
    }
}
#[cfg(test)]
mod tests {
    use clap::Parser;

    #[test]
    fn test_v4_requires_explicit_full_profile_and_rejects_conflicts() {
        let cli = crate::Cli::try_parse_from([
            "gigastt",
            "transcribe",
            "audio.wav",
            "--model-family",
            "gigaam_v4",
        ])
        .unwrap();
        assert!(cli.family.resolve().is_err());
        let cli = crate::Cli::try_parse_from([
            "gigastt",
            "transcribe",
            "audio.wav",
            "--model-precision",
            "fp32",
        ])
        .unwrap();
        assert!(cli.family.resolve().is_err());
        let cli = crate::Cli::try_parse_from(["gigastt", "transcribe", "audio.wav"]).unwrap();
        assert!(cli.family.resolve().unwrap().is_none());
    }
    #[test]
    fn test_v4_command_rejects_conflicting_legacy_options() {
        use clap::CommandFactory;
        for extra in [
            vec!["--vad"],
            vec!["--punctuation", "auto"],
            vec!["--hotwords-file", "words.txt"],
            vec!["--model-variant", "rnnt"],
            vec!["--encoder-intra-threads", "2"],
            vec!["--stereo-speakers"],
        ] {
            let mut argv = vec![
                "gigastt",
                "transcribe",
                "audio.wav",
                "--model-family",
                "gigaam_v4",
                "--model-head",
                "rnnt",
                "--model-precision",
                "fp32",
                "--transcript-style",
                "raw",
                "--v4-mkl-runtime-dir",
                "/tmp/mkl",
            ];
            argv.extend(extra);
            let m = crate::Cli::command().try_get_matches_from(argv).unwrap();
            let err = super::validate_command(&m).unwrap_err().to_string();
            assert!(!err.contains("GNU Linux"), "{err}");
        }
        let m = crate::Cli::command()
            .try_get_matches_from([
                "gigastt",
                "serve",
                "--model-family",
                "gigaam_v4",
                "--model-head",
                "rnnt",
                "--model-precision",
                "fp32",
                "--transcript-style",
                "raw",
                "--v4-mkl-runtime-dir",
                "/tmp/mkl",
                "--pool-size",
                "2",
            ])
            .unwrap();
        assert!(
            super::validate_command(&m)
                .unwrap_err()
                .to_string()
                .contains("one request slot")
        );
        let m = crate::Cli::command()
            .try_get_matches_from([
                "gigastt",
                "transcribe",
                "audio.wav",
                "--model-family",
                "gigaam_v4",
                "--model-head",
                "rnnt",
                "--model-precision",
                "fp32",
                "--transcript-style",
                "raw",
                "--v4-mkl-runtime-dir",
                "/tmp/mkl",
            ])
            .unwrap();
        let result = super::validate_command(&m);
        assert_eq!(
            result.is_ok(),
            cfg!(all(
                feature = "v4-mkl",
                target_os = "linux",
                target_arch = "x86_64",
                target_env = "gnu"
            ))
        );
    }
}
