use super::*;

#[test]
fn test_cli_download_parsing() {
    let cli = Cli::parse_from(["gigastt", "download", "--model-dir", "/tmp/models"]);
    match cli.command {
        Commands::Download {
            model_dir,
            model_variant,
            ..
        } => {
            assert_eq!(model_dir, "/tmp/models");
            assert_eq!(model_variant, ModelVariant::Rnnt);
        }
        _ => panic!("expected Download"),
    }
}

#[test]
fn test_cli_download_model_variant_override() {
    let cli = Cli::parse_from(["gigastt", "download", "--model-variant", "e2e_rnnt"]);
    match cli.command {
        Commands::Download { model_variant, .. } => {
            assert_eq!(model_variant, ModelVariant::E2eRnnt);
        }
        _ => panic!("expected Download"),
    }
}

#[test]
fn test_cli_cache_gc_parsing() {
    let cli = Cli::try_parse_from([
        "gigastt",
        "cache-gc",
        "--model-dir",
        "/tmp/models",
        "--dry-run",
        "--dedupe",
        "--optimized-cache-dir",
        "/var/cache/gigastt",
    ])
    .expect("parse cache-gc");
    match cli.command {
        Commands::CacheGc {
            model_dir,
            dry_run,
            dedupe,
            optimized_cache_dir,
        } => {
            assert_eq!(model_dir, "/tmp/models");
            assert!(dry_run);
            assert!(dedupe);
            assert_eq!(optimized_cache_dir.as_deref(), Some("/var/cache/gigastt"));
        }
        _ => panic!("expected CacheGc"),
    }
}

#[test]
fn test_cli_cache_gc_optimized_cache_dir_defaults_to_none() {
    let _guard = ENV_LOCK.lock().unwrap();
    let _restore = EnvRestore(
        "GIGASTT_OPTIMIZED_CACHE_DIR",
        std::env::var("GIGASTT_OPTIMIZED_CACHE_DIR").ok(),
    );
    unsafe {
        std::env::remove_var("GIGASTT_OPTIMIZED_CACHE_DIR");
    }
    let cli = Cli::try_parse_from(["gigastt", "cache-gc"]).expect("parse cache-gc");
    match cli.command {
        Commands::CacheGc {
            optimized_cache_dir,
            ..
        } => {
            assert_eq!(optimized_cache_dir, None);
        }
        _ => panic!("expected CacheGc"),
    }
}

#[test]
fn test_cli_cache_gc_optimized_cache_dir_env_var() {
    let _guard = ENV_LOCK.lock().unwrap();
    let _restore = EnvRestore(
        "GIGASTT_OPTIMIZED_CACHE_DIR",
        std::env::var("GIGASTT_OPTIMIZED_CACHE_DIR").ok(),
    );
    unsafe {
        std::env::set_var("GIGASTT_OPTIMIZED_CACHE_DIR", "/var/cache/gigastt");
    }
    let cli = Cli::try_parse_from(["gigastt", "cache-gc"]).expect("parse cache-gc");
    match cli.command {
        Commands::CacheGc {
            optimized_cache_dir,
            ..
        } => {
            assert_eq!(optimized_cache_dir.as_deref(), Some("/var/cache/gigastt"));
        }
        _ => panic!("expected CacheGc"),
    }
}

#[test]
fn test_cli_quantize_parsing() {
    let cli = Cli::parse_from(["gigastt", "quantize", "--force"]);
    match cli.command {
        Commands::Quantize { force, .. } => {
            assert!(force);
        }
        _ => panic!("expected Quantize"),
    }
}

#[test]
fn test_cli_download_parses() {
    let cli = Cli::try_parse_from(["gigastt", "download"]).expect("parse");
    match cli.command {
        Commands::Download { model_variant, .. } => {
            assert_eq!(model_variant, ModelVariant::Rnnt);
        }
        _ => panic!("expected Download"),
    }
}

#[test]
fn test_cli_download_rejects_fp32_flag() {
    let err = match Cli::try_parse_from(["gigastt", "download", "--fp32"]) {
        Ok(_) => panic!("fp32 flag must be removed from download"),
        Err(e) => e,
    };
    let msg = err.to_string();
    assert!(
        msg.contains("unexpected") || msg.contains("fp32") || msg.contains("unknown"),
        "expected clap to reject --fp32, got: {msg}"
    );
}

#[test]
fn test_cli_download_progress_defaults_to_human() {
    let _guard = ENV_LOCK.lock().unwrap();
    let _restore = EnvRestore(
        "GIGASTT_DOWNLOAD_PROGRESS",
        std::env::var("GIGASTT_DOWNLOAD_PROGRESS").ok(),
    );
    unsafe {
        std::env::remove_var("GIGASTT_DOWNLOAD_PROGRESS");
    }
    let cli = Cli::parse_from(["gigastt", "download"]);
    match cli.command {
        Commands::Download { progress, .. } => assert_eq!(progress, ProgressMode::Human),
        _ => panic!("expected Download"),
    }
}

#[test]
fn test_cli_download_progress_json_flag() {
    let _guard = ENV_LOCK.lock().unwrap();
    let _restore = EnvRestore(
        "GIGASTT_DOWNLOAD_PROGRESS",
        std::env::var("GIGASTT_DOWNLOAD_PROGRESS").ok(),
    );
    unsafe {
        std::env::remove_var("GIGASTT_DOWNLOAD_PROGRESS");
    }
    let cli = Cli::parse_from(["gigastt", "download", "--progress", "json"]);
    match cli.command {
        Commands::Download { progress, .. } => assert_eq!(progress, ProgressMode::Json),
        _ => panic!("expected Download"),
    }
}

#[test]
fn test_cli_download_progress_env_var() {
    let _guard = ENV_LOCK.lock().unwrap();
    let _restore = EnvRestore(
        "GIGASTT_DOWNLOAD_PROGRESS",
        std::env::var("GIGASTT_DOWNLOAD_PROGRESS").ok(),
    );
    unsafe {
        std::env::set_var("GIGASTT_DOWNLOAD_PROGRESS", "json");
    }
    let cli = Cli::parse_from(["gigastt", "download"]);
    match cli.command {
        Commands::Download { progress, .. } => assert_eq!(progress, ProgressMode::Json),
        _ => panic!("expected Download"),
    }
}

#[cfg(feature = "ane")]
#[test]
fn test_cli_download_ane_flag() {
    let cli = Cli::parse_from(["gigastt", "download", "--ane"]);
    match cli.command {
        Commands::Download { ane, .. } => assert!(ane),
        _ => panic!("expected Download"),
    }
    // Absent by default.
    let cli = Cli::parse_from(["gigastt", "download"]);
    match cli.command {
        Commands::Download { ane, .. } => assert!(!ane),
        _ => panic!("expected Download"),
    }
}
