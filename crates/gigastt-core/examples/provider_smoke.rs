//! Temporary paired-revision diagnostic; never substitutes CPU for CoreML.
use anyhow::{Context, Result, ensure};
use gigastt_core::{ExecutionProviderChoice, inference::Engine, model::ModelVariant};

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let models = args.next().context("missing model directory")?;
    let fixture = args.next().context("missing audio fixture")?;
    let provider = match args.next().as_deref() {
        Some("cpu") => ExecutionProviderChoice::Cpu,
        Some("coreml") => ExecutionProviderChoice::Coreml,
        _ => anyhow::bail!("provider must be cpu or coreml"),
    };
    let engine = Engine::load_with_execution_provider(
        &models,
        Some(ModelVariant::Rnnt),
        1,
        1,
        0,
        2,
        None,
        provider,
    )?;
    let mut slot = engine.pool.checkout_blocking()?;
    let result = engine.transcribe_file(&fixture, &mut slot)?;
    println!("{}", result.text);
    ensure!(result.text.contains("тенге"), "unexpected reference text");
    Ok(())
}
