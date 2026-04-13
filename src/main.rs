use clap::{Parser, Subcommand};
use gigastt::{inference, model, server};
use tracing_subscriber::EnvFilter;

#[derive(Parser)]
#[command(name = "gigastt", version, about = "Local STT server powered by GigaAM v3")]
struct Cli {
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

    /// Transcribe an audio file (offline)
    Transcribe {
        /// Path to WAV file (PCM16 mono)
        file: String,

        /// Model directory
        #[arg(long, default_value_t = model::default_model_dir())]
        model_dir: String,
    },
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env().add_directive("gigastt=info".parse()?))
        .init();

    let cli = Cli::parse();

    match cli.command {
        Commands::Serve { port, host, model_dir } => {
            model::ensure_model(&model_dir).await?;
            let engine = inference::Engine::load(&model_dir)?;
            server::run(engine, port, &host).await?;
        }
        Commands::Download {
            model_dir,
            #[cfg(feature = "diarization")]
            diarization,
        } => {
            model::ensure_model(&model_dir).await?;
            #[cfg(feature = "diarization")]
            if diarization {
                model::ensure_speaker_model(&model_dir).await?;
            }
            tracing::info!("Model ready at {model_dir}");
        }
        Commands::Transcribe { file, model_dir } => {
            model::ensure_model(&model_dir).await?;
            let engine = inference::Engine::load(&model_dir)?;
            let mut triplet = engine.pool.checkout().await;
            let result = engine.transcribe_file(&file, &mut triplet);
            engine.pool.checkin(triplet).await;
            println!("{}", result?.text);
        }
    }

    Ok(())
}
