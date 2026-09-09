//! `download` / `quantize` / `cache-gc` command bodies.

use gigastt_core::model;
use gigastt_core::model::{ModelVariant, ProgressMode};

/// Packaging: rebuild INT8 encoder from a local FP32 ONNX (no download).
pub(crate) fn run_quantize(model_dir: String, force: bool) -> anyhow::Result<()> {
    // Order matters for lean INT8 installs: if INT8 is already present
    // and `--force` is off, no-op without looking for FP32 (CI/e2e and
    // operators who only have the prequantized bundle).
    let dir = std::path::Path::new(&model_dir);
    let resolved = model::ModelVariant::detect_in_dir(dir).unwrap_or_default();
    let input = dir.join(resolved.encoder_file());
    let output = dir.join(resolved.encoder_int8_file());
    if output.exists() && !force {
        tracing::info!("INT8 model already exists: {}", output.display());
        tracing::info!("Use --force to re-quantize.");
        return Ok(());
    }
    if !input.is_file() {
        anyhow::bail!(
            "FP32 encoder not found at {} — `quantize` needs the FP32 ONNX as packaging source. \
             For runtime, use `gigastt download` (lean INT8 only).",
            input.display()
        );
    }
    gigastt_core::quantize::quantize_model(&input, &output)?;
    tracing::info!("Quantized model saved to {}", output.display());
    Ok(())
}

/// Prune optimized/CoreML caches and optionally content-hash-dedupe the model dir.
pub(crate) fn run_cache_gc(
    model_dir: String,
    dry_run: bool,
    dedupe: bool,
    optimized_cache_dir: Option<String>,
) -> anyhow::Result<()> {
    let dir = std::path::Path::new(&model_dir);
    let prune = match optimized_cache_dir {
        Some(cache_dir) => {
            model::prune_optimized_cache_dir(std::path::Path::new(&cache_dir), dir, dry_run)?
        }
        None => model::prune_optimized_cache(dir, dry_run)?,
    };
    let action = if dry_run { "would free" } else { "freed" };
    println!(
        "optimized_cache: kept {} graph(s), removed {} ({} {:.1} MiB)",
        prune.kept.len(),
        prune.removed.len(),
        action,
        prune.freed_bytes as f64 / (1024.0 * 1024.0),
    );
    for p in &prune.removed {
        println!("  - {}", p.display());
    }
    let coreml = model::prune_coreml_cache(dir, dry_run)?;
    println!(
        "coreml_cache: kept {}, removed {} stale ({} {:.1} MiB)",
        if coreml.kept.is_some() {
            "current"
        } else {
            "none"
        },
        coreml.removed.len(),
        action,
        coreml.freed_bytes as f64 / (1024.0 * 1024.0),
    );
    for p in &coreml.removed {
        println!("  - {}", p.display());
    }
    if dedupe {
        let d = model::dedupe_model_dir(dir, dry_run)?;
        println!(
            "dedupe: {} group(s), {} hardlink(s), {} {:.1} MiB",
            d.groups,
            d.hardlinked,
            action,
            d.freed_bytes as f64 / (1024.0 * 1024.0),
        );
    }
    Ok(())
}

/// Download the lean INT8 model (and optional side assets). On failure or
/// SIGINT exits the process with a progress-kind exit code (never returns `Err`
/// for those paths).
pub(crate) async fn run_download(
    model_dir: String,
    model_variant: ModelVariant,
    #[cfg(feature = "diarization")] skip_diarization: bool,
    progress: ProgressMode,
    #[cfg(feature = "ane")] ane: bool,
) -> anyhow::Result<()> {
    model::set_progress_mode(progress);
    // `download` is an explicit action: the requested variant maps to
    // the default (Rnnt) so a bare `gigastt download` fetches something
    // useful. **INT8 only** — lean prequantized bundle (or CTC INT8 from HF).
    //
    // The flow runs on its own task: large-file SHA-256 verify is
    // synchronous, and polled inline it would starve the select's signal
    // branch — Ctrl-C must interrupt immediately (sidecar cancel path).
    let dl_model_dir = model_dir.clone();
    let mut download = tokio::spawn(async move {
        let model_dir = dl_model_dir;
        if model_variant.is_ctc() {
            // CTC heads: HF pre-quantized INT8 encoder (+ vocab) directly.
            model::ensure_model_variant(Some(model_variant), &model_dir).await?;
        } else {
            // RNN-T: lean INT8 bundle from the pinned Release.
            model::ensure_prequantized_model_variant(Some(model_variant), &model_dir).await?;
        }
        #[cfg(feature = "diarization")]
        {
            if !skip_diarization {
                model::ensure_speaker_model(&model_dir).await?;
            }
        }
        #[cfg(feature = "ane")]
        if ane {
            let ane_dir = model::default_ane_model_dir();
            model::ensure_ane_packages(&ane_dir).await?;
            tracing::info!("ANE encoder packages ready at {ane_dir}");
        }
        tracing::info!("Model ready at {model_dir}");
        anyhow::Ok(())
    });
    // Resolves only on a *delivered* SIGINT. A failed handler
    // registration is logged and parks forever — it must not fabricate
    // an interrupt and abort a healthy download with exit 130.
    let interrupted = async {
        match tokio::signal::ctrl_c().await {
            Ok(()) => (),
            Err(e) => {
                tracing::warn!("Failed to listen for Ctrl-C: {e}");
                std::future::pending::<()>().await
            }
        }
    };
    tokio::select! {
        joined = &mut download => {
            // Flatten the JoinHandle: a panic inside the download task
            // is reported through the same error contract.
            let result = joined
                .map_err(|e| anyhow::anyhow!("download task failed: {e}"))
                .and_then(|r| r);
            match result {
                Ok(()) => {
                    model::emit_progress_event(&model::ProgressEvent::Done {
                        model_dir: model_dir.clone(),
                    });
                }
                Err(e) => {
                    let kind = model::classify_download_error(&e);
                    model::emit_progress_event(&model::ProgressEvent::Error {
                        kind,
                        message: format!("{e:#}"),
                    });
                    // Same rendering anyhow's `Termination` would print,
                    // then the documented per-kind exit code (all != 0).
                    eprintln!("Error: {e:?}");
                    std::process::exit(kind.exit_code());
                }
            }
        }
        _ = interrupted => {
            model::emit_progress_event(&model::ProgressEvent::Error {
                kind: model::ProgressErrorKind::Interrupted,
                message: "interrupted by SIGINT".to_string(),
            });
            eprintln!("Interrupted by Ctrl-C");
            std::process::exit(model::ProgressErrorKind::Interrupted.exit_code());
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_run_quantize_noops_when_int8_already_present() {
        let tmp = tempfile::tempdir().expect("tempdir");
        std::fs::write(tmp.path().join("v3_rnnt_encoder_int8.onnx"), b"int8").unwrap();
        run_quantize(tmp.path().display().to_string(), false).expect("existing INT8 is a no-op");
        assert!(!tmp.path().join("v3_rnnt_encoder.onnx").exists());
    }

    #[test]
    fn test_run_quantize_errors_when_fp32_missing() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let err = run_quantize(tmp.path().display().to_string(), true)
            .expect_err("force still needs FP32");
        let msg = format!("{err:#}");
        assert!(
            msg.contains("FP32 encoder not found"),
            "unexpected error: {msg}"
        );
    }

    #[test]
    fn test_run_cache_gc_empty_dir_is_ok() {
        let tmp = tempfile::tempdir().expect("tempdir");
        run_cache_gc(tmp.path().display().to_string(), true, true, None).expect("empty dry-run");
        run_cache_gc(tmp.path().display().to_string(), false, false, None).expect("empty prune");
    }

    #[test]
    fn test_run_cache_gc_prunes_explicit_optimized_cache_dir() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let model_dir = tmp.path().join("models");
        // rnnt INT8 stub so the prune has a keep-name.
        std::fs::create_dir_all(&model_dir).unwrap();
        std::fs::write(model_dir.join("v3_rnnt_encoder_int8.onnx"), b"int8").unwrap();
        let cache_dir = tmp.path().join("elsewhere");
        std::fs::create_dir_all(&cache_dir).unwrap();
        let keep = cache_dir.join("v3_rnnt_encoder_int8_optimized.ort");
        let zombie = cache_dir.join("v3_e2e_rnnt_encoder_int8_optimized.ort");
        std::fs::write(&keep, b"keep").unwrap();
        std::fs::write(&zombie, b"zombie").unwrap();

        run_cache_gc(
            model_dir.display().to_string(),
            false,
            false,
            Some(cache_dir.display().to_string()),
        )
        .expect("prune explicit cache dir");
        assert!(keep.exists());
        assert!(!zombie.exists());
    }
}
