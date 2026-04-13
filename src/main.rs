use clap::{Parser, Subcommand};
use gigastt::{inference, model, server};
use tracing_subscriber::EnvFilter;

#[derive(Parser)]
#[command(
    name = "gigastt",
    version,
    about = "Local STT server powered by GigaAM v3"
)]
struct Cli {
    /// Log level [default: info]
    #[arg(long, global = true, default_value = "info")]
    log_level: String,

    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Start WebSocket STT server (auto-downloads model if missing)
    Serve {
        /// Port to listen on
        #[arg(short, long, default_value_t = 9876)]
        port: u16,

        /// Bind address (use 0.0.0.0 for Docker)
        #[arg(long, default_value = "127.0.0.1")]
        host: String,

        /// Model directory
        #[arg(long, default_value_t = model::default_model_dir())]
        model_dir: String,

        /// Number of concurrent inference sessions
        #[arg(long, default_value_t = 4)]
        pool_size: usize,
    },

    /// Download model without starting server
    Download {
        /// Model directory
        #[arg(long, default_value_t = model::default_model_dir())]
        model_dir: String,

        /// Also download speaker diarization model
        #[cfg(feature = "diarization")]
        #[arg(long, default_value_t = false)]
        diarization: bool,
    },

    /// Quantize encoder model to INT8 (replaces scripts/quantize.py)
    #[cfg(feature = "quantize")]
    Quantize {
        /// Model directory
        #[arg(long, default_value_t = model::default_model_dir())]
        model_dir: String,

        /// Force re-quantization even if INT8 model exists
        #[arg(long)]
        force: bool,
    },

    /// Transcribe an audio file (offline)
    Transcribe {
        /// Path to WAV file (PCM16 mono)
        file: String,

        /// Model directory
        #[arg(long, default_value_t = model::default_model_dir())]
        model_dir: String,
    },
}

fn log_rss() {
    #[cfg(target_os = "linux")]
    {
        if let Ok(status) = std::fs::read_to_string("/proc/self/status")
            && let Some(line) = status.lines().find(|l| l.starts_with("VmRSS:"))
        {
            tracing::info!("{}", line.trim());
        }
    }
    // On macOS/other platforms, use `ps` as a simple cross-platform fallback
    #[cfg(not(target_os = "linux"))]
    {
        if let Ok(output) = std::process::Command::new("ps")
            .args(["-o", "rss=", "-p", &std::process::id().to_string()])
            .output()
            && let Ok(rss) = String::from_utf8_lossy(&output.stdout)
                .trim()
                .parse::<u64>()
        {
            tracing::info!(rss_mb = rss / 1024, "memory_after_load");
        }
    }
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    let directive = format!("gigastt={}", cli.log_level);
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env().add_directive(directive.parse()?))
        .init();

    match cli.command {
        Commands::Serve {
            port,
            host,
            model_dir,
            pool_size,
        } => {
            model::ensure_model(&model_dir).await?;
            #[cfg(feature = "quantize")]
            {
                let int8_path =
                    std::path::Path::new(&model_dir).join("v3_e2e_rnnt_encoder_int8.onnx");
                if !int8_path.exists() {
                    tracing::info!("Auto-quantizing encoder to INT8 (4x smaller, same quality)...");
                    let input = std::path::Path::new(&model_dir).join("v3_e2e_rnnt_encoder.onnx");
                    gigastt::quantize::quantize_model(&input, &int8_path)?;
                    tracing::info!("INT8 encoder saved to {}", int8_path.display());
                }
            }
            let engine = inference::Engine::load_with_pool_size(&model_dir, pool_size)?;
            log_rss();
            server::run(engine, port, &host).await?;
        }
        Commands::Download {
            model_dir,
            #[cfg(feature = "diarization")]
            diarization,
        } => {
            model::ensure_model(&model_dir).await?;
            #[cfg(feature = "diarization")]
            {
                if diarization {
                    model::ensure_speaker_model(&model_dir).await?;
                }
            }
            // Auto-quantize encoder to INT8 if not already done
            #[cfg(feature = "quantize")]
            {
                let int8_path =
                    std::path::Path::new(&model_dir).join("v3_e2e_rnnt_encoder_int8.onnx");
                if !int8_path.exists() {
                    tracing::info!("Auto-quantizing encoder to INT8 (4x smaller, same quality)...");
                    let input = std::path::Path::new(&model_dir).join("v3_e2e_rnnt_encoder.onnx");
                    gigastt::quantize::quantize_model(&input, &int8_path)?;
                    tracing::info!("INT8 encoder saved to {}", int8_path.display());
                }
            }
            #[cfg(not(feature = "quantize"))]
            {
                let int8_path =
                    std::path::Path::new(&model_dir).join("v3_e2e_rnnt_encoder_int8.onnx");
                if !int8_path.exists() {
                    tracing::info!(
                        "Tip: install with --features quantize for 4x smaller model: cargo install gigastt --features quantize && gigastt quantize"
                    );
                }
            }
            tracing::info!("Model ready at {model_dir}");
        }
        #[cfg(feature = "quantize")]
        Commands::Quantize { model_dir, force } => {
            model::ensure_model(&model_dir).await?;
            let input = std::path::Path::new(&model_dir).join("v3_e2e_rnnt_encoder.onnx");
            let output = std::path::Path::new(&model_dir).join("v3_e2e_rnnt_encoder_int8.onnx");
            if output.exists() && !force {
                tracing::info!("INT8 model already exists: {}", output.display());
                tracing::info!("Use --force to re-quantize.");
                return Ok(());
            }
            gigastt::quantize::quantize_model(&input, &output)?;
            tracing::info!("Quantized model saved to {}", output.display());
        }
        Commands::Transcribe { file, model_dir } => {
            model::ensure_model(&model_dir).await?;
            let engine = inference::Engine::load_with_pool_size(&model_dir, 1)?;
            log_rss();
            let mut triplet = engine.pool.checkout().await;
            let result = engine.transcribe_file(&file, &mut triplet);
            engine.pool.checkin(triplet).await;
            println!("{}", result?.text);
        }
    }

    Ok(())
}
