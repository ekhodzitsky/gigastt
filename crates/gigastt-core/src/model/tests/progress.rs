use super::*;

#[test]
fn test_download_progress_basic() {
    let sink = ProgressSink::human();
    let mut progress = DownloadProgress::new(1_000_000);
    // Should not panic on normal update.
    progress.update(500_000, &sink, "model.onnx");
    assert_eq!(progress.current, 500_000);
    assert_eq!(progress.last_percent, 50);
    progress.finish(&sink, "model.onnx");
}

#[test]
fn test_download_progress_zero_total() {
    let sink = ProgressSink::human();
    let mut progress = DownloadProgress::new(0);
    // Must not divide by zero.
    progress.update(100, &sink, "model.onnx");
    assert_eq!(progress.last_percent, 0);
    progress.finish(&sink, "model.onnx");
}

// ── progress events (NDJSON / human sink) ───────────────────────────────

#[test]
fn test_progress_mode_from_str() {
    use std::str::FromStr;
    assert_eq!(
        ProgressMode::from_str("human").unwrap(),
        ProgressMode::Human
    );
    assert_eq!(ProgressMode::from_str("json").unwrap(), ProgressMode::Json);
    assert_eq!(
        ProgressMode::from_str(" JSON ").unwrap(),
        ProgressMode::Json
    );
    assert!(ProgressMode::from_str("xml").is_err());
    assert_eq!(ProgressMode::default(), ProgressMode::Human);
    assert_eq!(ProgressMode::Json.as_str(), "json");
}

/// The NDJSON wire shape is the integrator contract: one line per event,
/// `phase` as the discriminator, exactly the fields sidecars match on.
#[test]
fn test_progress_event_ndjson_schema() {
    let cases: Vec<(ProgressEvent, &str)> = vec![
        (
            ProgressEvent::Download {
                file: "v3_rnnt_encoder.onnx".to_string(),
                bytes_done: 50,
                bytes_total: 100,
            },
            "{\"phase\":\"download\",\"file\":\"v3_rnnt_encoder.onnx\",\"bytes_done\":50,\"bytes_total\":100}",
        ),
        (
            ProgressEvent::Quantize {
                file: "v3_rnnt_encoder.onnx".to_string(),
            },
            "{\"phase\":\"quantize\",\"file\":\"v3_rnnt_encoder.onnx\"}",
        ),
        (
            ProgressEvent::Verify {
                file: "v3_vocab.txt".to_string(),
            },
            "{\"phase\":\"verify\",\"file\":\"v3_vocab.txt\"}",
        ),
        (
            ProgressEvent::Done {
                model_dir: "/home/u/.gigastt/models".to_string(),
            },
            "{\"phase\":\"done\",\"model_dir\":\"/home/u/.gigastt/models\"}",
        ),
        (
            ProgressEvent::Error {
                kind: ProgressErrorKind::Network,
                message: "connection refused".to_string(),
            },
            "{\"phase\":\"error\",\"kind\":\"network\",\"message\":\"connection refused\"}",
        ),
        (
            ProgressEvent::Error {
                kind: ProgressErrorKind::Interrupted,
                message: "SIGINT".to_string(),
            },
            "{\"phase\":\"error\",\"kind\":\"interrupted\",\"message\":\"SIGINT\"}",
        ),
    ];
    for (event, want) in cases {
        let line = event.to_ndjson();
        assert_eq!(line, want);
        // Every line must round-trip as a JSON object with a `phase` tag.
        let parsed: serde_json::Value =
            serde_json::from_str(&line).expect("NDJSON line must parse");
        assert!(parsed.get("phase").is_some(), "phase tag missing: {line}");
    }
}

#[test]
fn test_progress_error_kind_exit_codes_keep_nonzero_contract() {
    assert_eq!(ProgressErrorKind::Other.exit_code(), 1);
    assert_eq!(ProgressErrorKind::Network.exit_code(), 69);
    assert_eq!(ProgressErrorKind::Disk.exit_code(), 74);
    assert_eq!(ProgressErrorKind::Checksum.exit_code(), 65);
    assert_eq!(ProgressErrorKind::Interrupted.exit_code(), 130);
    // 2 stays reserved for clap usage errors: a misconfigured invocation
    // must never look like a transient (retryable) download failure.
    for kind in [
        ProgressErrorKind::Other,
        ProgressErrorKind::Network,
        ProgressErrorKind::Disk,
        ProgressErrorKind::Checksum,
        ProgressErrorKind::Interrupted,
    ] {
        assert_ne!(kind.exit_code(), 2, "{kind:?} must not collide with clap");
    }
    for kind in [
        ProgressErrorKind::Network,
        ProgressErrorKind::Disk,
        ProgressErrorKind::Checksum,
        ProgressErrorKind::Interrupted,
        ProgressErrorKind::Other,
    ] {
        assert_ne!(kind.exit_code(), 0, "{kind:?} must keep != 0");
    }
}

/// Human mode must stay byte-for-byte identical to the legacy `\r`
/// reporter: same format strings, same trailing-space padding.
#[test]
fn test_download_progress_human_render_matches_legacy() {
    let mut progress = DownloadProgress::new(10 * 1_048_576);
    // current=0, percent 0 == last_percent 0 -> no redraw.
    assert_eq!(progress.human_tick(), None);
    progress.current = 5 * 1_048_576;
    assert_eq!(
        progress.human_tick().as_deref(),
        Some("\rDownloading... 50% (5.0MB / 10.0MB)")
    );
    // Same percentage -> throttled (no redraw).
    assert_eq!(progress.human_tick(), None);
    progress.current = 10 * 1_048_576;
    assert_eq!(
        progress.human_tick().as_deref(),
        Some("\rDownloading... 100% (10.0MB / 10.0MB)")
    );
    assert_eq!(
        progress.human_finish(),
        "\rDownload complete (10.0MB)                    "
    );
}

/// Json mode: first chunk emits immediately, rapid chunks within the 200 ms
/// window are throttled, and 100% always emits exactly once.
#[test]
fn test_download_progress_json_first_throttled_then_final() {
    let (sink, log) = ProgressSink::capturing();
    let mut progress = DownloadProgress::new(1_000);
    // First chunk emits. Later chunks stay inside the throttle window even
    // when the interpreter is slower than 200 ms per call: refresh the stamp
    // instead of assuming the loop itself outruns the window. The last chunk
    // is 100% and must still emit exactly once.
    progress.update(10, &sink, "model.onnx");
    for _ in 0..98 {
        progress.last_json_emit = Some(std::time::Instant::now());
        progress.update(10, &sink, "model.onnx");
    }
    progress.last_json_emit = Some(std::time::Instant::now());
    progress.update(10, &sink, "model.onnx");
    let events = log.lock().unwrap();
    assert_eq!(
        events.as_slice(),
        [
            ProgressEvent::Download {
                file: "model.onnx".to_string(),
                bytes_done: 10,
                bytes_total: 1_000,
            },
            ProgressEvent::Download {
                file: "model.onnx".to_string(),
                bytes_done: 1_000,
                bytes_total: 1_000,
            },
        ]
    );
    // finish() must not duplicate the already-emitted 100% event.
    drop(events);
    progress.finish(&sink, "model.onnx");
    assert_eq!(log.lock().unwrap().len(), 2);
}

/// Json mode: once the throttle window has elapsed, mid-file progress
/// emits again (integrators get a steady cadence on long downloads).
#[test]
fn test_download_progress_json_emits_after_throttle_window() {
    let (sink, log) = ProgressSink::capturing();
    let mut progress = DownloadProgress::new(1_000);
    progress.update(10, &sink, "model.onnx");
    // Backdate the last emission past the throttle window.
    progress.last_json_emit = Some(
        std::time::Instant::now() - JSON_PROGRESS_THROTTLE - std::time::Duration::from_millis(50),
    );
    progress.update(10, &sink, "model.onnx");
    let events = log.lock().unwrap();
    assert_eq!(events.len(), 2, "elapsed window must re-emit: {events:?}");
    assert_eq!(
        events[1],
        ProgressEvent::Download {
            file: "model.onnx".to_string(),
            bytes_done: 20,
            bytes_total: 1_000,
        }
    );
}

/// Json mode with an unknown (chunked) total: throttled events carry
/// `bytes_total: 0`, and `finish` closes the file with one final event.
#[test]
fn test_download_progress_json_zero_total_emits_final_on_finish() {
    let (sink, log) = ProgressSink::capturing();
    let mut progress = DownloadProgress::new(0);
    progress.update(512, &sink, "model.onnx");
    progress.finish(&sink, "model.onnx");
    let events = log.lock().unwrap();
    assert_eq!(
        events.as_slice(),
        [
            ProgressEvent::Download {
                file: "model.onnx".to_string(),
                bytes_done: 512,
                bytes_total: 0,
            },
            ProgressEvent::Download {
                file: "model.onnx".to_string(),
                bytes_done: 512,
                bytes_total: 0,
            },
        ]
    );
}

/// Human mode never routes events to the sink (the `\r` line is the whole
/// output), so an integrator never sees a stray JSON line.
#[test]
fn test_download_progress_human_sink_emits_no_events() {
    let (sink, log) = ProgressSink::capturing();
    let sink = ProgressSink {
        mode: ProgressMode::Human,
        ..sink
    };
    let mut progress = DownloadProgress::new(1_000);
    progress.update(1_000, &sink, "model.onnx");
    progress.finish(&sink, "model.onnx");
    assert!(log.lock().unwrap().is_empty());
}

#[test]
fn test_classify_download_error_checksum_message() {
    let err = anyhow::anyhow!("SHA-256 mismatch for encoder.onnx: expected aa, got bb");
    assert_eq!(classify_download_error(&err), ProgressErrorKind::Checksum);
}

#[test]
fn test_classify_download_error_http_status_is_network() {
    let err = anyhow::anyhow!("Download failed for model.onnx: HTTP 404");
    assert_eq!(classify_download_error(&err), ProgressErrorKind::Network);
}

#[test]
fn test_classify_download_error_io_root_cause_is_disk() {
    let io = std::io::Error::new(std::io::ErrorKind::PermissionDenied, "denied");
    let err = anyhow::Error::from(io).context("Failed to create partial model file");
    assert_eq!(classify_download_error(&err), ProgressErrorKind::Disk);
}

#[test]
fn test_classify_download_error_other() {
    let err = anyhow::anyhow!("Failed to acquire download lock (another process is downloading)");
    assert_eq!(classify_download_error(&err), ProgressErrorKind::Other);
}

/// A real reqwest connection failure classifies as network via the typed
/// root cause, not message matching.
#[tokio::test]
#[cfg_attr(miri, ignore = "tokio runtime is unsupported under Miri")]
async fn test_classify_download_error_reqwest_is_network() {
    let err = reqwest::Client::new()
        .get("http://127.0.0.1:9/unreachable")
        .send()
        .await
        .expect_err("port 9 must refuse");
    let err = anyhow::Error::from(err).context("HTTP request failed");
    assert_eq!(classify_download_error(&err), ProgressErrorKind::Network);
}
