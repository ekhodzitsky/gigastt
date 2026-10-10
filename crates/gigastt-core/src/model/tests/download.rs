use super::*;

#[test]
fn test_home_dir_returns_some() {
    // On any CI or developer machine HOME / USERPROFILE should be set.
    assert!(
        home_dir().is_some(),
        "home_dir() must return Some on this platform"
    );
}

#[test]
fn test_hash_file_sha256_and_verify() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let path = tmp.path().join("blob.bin");
    std::fs::write(&path, b"hello").unwrap();
    let digest = hash_file_sha256(&path).expect("hash");
    assert_eq!(digest.len(), 64);
    verify_pinned_checksum(&path, &digest).expect("matching digest");
    let wrong = "00".repeat(32);
    let err = verify_pinned_checksum(&path, &wrong).expect_err("wrong digest");
    let msg = format!("{err:#}");
    assert!(
        msg.contains("SHA-256 mismatch") || msg.contains("model load error"),
        "unexpected: {msg}"
    );
}

#[cfg(feature = "net")]
#[test]
fn test_reject_if_over_download_cap() {
    assert!(reject_if_over_download_cap(0, MAX_DOWNLOAD_BYTES).is_ok());
    assert!(reject_if_over_download_cap(0, MAX_DOWNLOAD_BYTES + 1).is_err());
    assert!(reject_if_over_download_cap(MAX_DOWNLOAD_BYTES, 1).is_err());
    assert!(reject_if_over_download_cap(MAX_DOWNLOAD_BYTES - 10, 10).is_ok());
}

#[test]
fn test_default_model_dir_contains_gigastt() {
    let dir = default_model_dir();
    assert!(
        dir.contains(".gigastt"),
        "default_model_dir() should contain \".gigastt\", got: {dir}"
    );
}

#[test]
fn test_partial_path_appends_suffix() {
    let p = partial_path(Path::new("/tmp/gigastt/encoder.onnx"));
    assert_eq!(
        p,
        std::path::PathBuf::from("/tmp/gigastt/encoder.onnx.partial"),
    );
}

/// On the success path, `.partial` disappears and the final path
/// appears in a single atomic step.
#[test]
fn test_download_writes_partial_then_renames() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let final_path = tmp.path().join("encoder.onnx");
    let payload = b"fake encoder weights";
    let expected = sha256_hex(payload);

    let partial = stage_partial(&final_path, payload);
    assert!(partial.exists(), "precondition: partial is present");
    assert!(!final_path.exists(), "precondition: final is absent");

    finalize_download(&partial, &final_path, Some(&expected), "encoder.onnx")
        .expect("finalize should succeed");

    assert!(
        !partial.exists(),
        "partial must be gone after atomic rename"
    );
    assert!(
        final_path.exists(),
        "final path must exist after atomic rename"
    );
    assert_eq!(std::fs::read(&final_path).unwrap(), payload);
}

/// If the process dies between the network write and the
/// SHA verification / rename, `is_model_present` must NOT see the
/// file under its final name. We simulate the crash by staging a
/// `.partial` and never calling `finalize_download`.
#[test]
fn test_download_crash_before_rename_leaves_no_final_file() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let final_path = tmp.path().join("encoder.onnx");
    let partial = stage_partial(&final_path, b"half-written junk");

    assert!(partial.exists(), "partial must exist to simulate crash");
    assert!(
        !final_path.exists(),
        "crash before rename must never leave the final artefact visible"
    );

    // No variant's files exist in this tempdir, so is_model_present must
    // refuse to short-circuit the download path.
    assert!(
        !is_model_present(ModelVariant::Rnnt, tmp.path()),
        "is_model_present must not accept a staged partial"
    );
    assert!(
        !is_model_present(ModelVariant::E2eRnnt, tmp.path()),
        "is_model_present must not accept a staged partial"
    );
}

/// SHA mismatch removes the partial and leaves the final path
/// empty, so a retry starts from a clean slate.
#[test]
fn test_download_rejects_sha256_mismatch() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let final_path = tmp.path().join("decoder.onnx");
    let payload = b"real bytes";
    // Intentionally wrong expected hash (hash of different bytes).
    let wrong_expected = sha256_hex(b"different bytes");

    let partial = stage_partial(&final_path, payload);

    let err = finalize_download(&partial, &final_path, Some(&wrong_expected), "decoder.onnx")
        .expect_err("mismatch must error");
    let msg = format!("{err}");
    assert!(msg.contains("SHA-256 mismatch"), "unexpected error: {msg}");

    assert!(!partial.exists(), "partial must be removed on SHA mismatch");
    assert!(
        !final_path.exists(),
        "final must never appear on SHA mismatch"
    );
}

/// Success path with no checksum available still renames
/// atomically (partial gone, final present, bytes preserved).
#[test]
fn test_download_atomic_on_success_without_checksum() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let final_path = tmp.path().join("vocab.txt");
    let payload = b"token0\ntoken1\n";

    let partial = stage_partial(&final_path, payload);

    finalize_download(&partial, &final_path, None, "vocab.txt")
        .expect("no-checksum finalize should succeed");

    assert!(!partial.exists(), "partial must be gone after rename");
    assert!(final_path.exists(), "final path must exist");
    assert_eq!(std::fs::read(&final_path).unwrap(), payload);
}

/// sha256_file matches the in-memory hash of the same bytes.
#[test]
fn test_sha256_file_matches_in_memory_hash() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let p = tmp.path().join("blob");
    let payload = b"gigastt-model-bytes";
    std::fs::write(&p, payload).unwrap();

    let got = sha256_file(&p).expect("sha256_file");
    let want = sha256_hex(payload);
    assert_eq!(got, want);
}

/// `SPEAKER_MODEL_SHA256` is a 64-char lowercase hex digest
/// matching the SHA-256 of the upstream `onnx/model.onnx` blob
/// (no accidental truncation / placeholder at compile time).
#[cfg(feature = "diarization")]
#[test]
fn test_speaker_model_sha256_shape() {
    assert_eq!(
        SPEAKER_MODEL_SHA256.len(),
        64,
        "SPEAKER_MODEL_SHA256 must be a 64-char hex digest"
    );
    assert!(
        SPEAKER_MODEL_SHA256
            .chars()
            .all(|c| c.is_ascii_digit() || ('a'..='f').contains(&c)),
        "SPEAKER_MODEL_SHA256 must be lowercase hex; got: {SPEAKER_MODEL_SHA256}"
    );
}

/// Mismatching bytes against `SPEAKER_MODEL_SHA256` must delete
/// the partial and refuse to promote it — exercises the full
/// speaker-model finalize contract without touching the network.
#[cfg(feature = "diarization")]
#[test]
fn test_speaker_model_rejects_sha256_mismatch() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let final_path = tmp.path().join(SPEAKER_MODEL_FILE);
    // Definitely not the real speaker-model bytes.
    let partial = stage_partial(&final_path, b"not the real wespeaker weights");

    let err = finalize_download(
        &partial,
        &final_path,
        Some(SPEAKER_MODEL_SHA256),
        SPEAKER_MODEL_FILE,
    )
    .expect_err("speaker mismatch must error");
    assert!(
        format!("{err}").contains("SHA-256 mismatch"),
        "unexpected error: {err}"
    );

    assert!(
        !partial.exists(),
        "partial speaker model must be removed on mismatch"
    );
    assert!(
        !final_path.exists(),
        "final speaker model must never appear on mismatch"
    );
}

/// When the partial bytes DO hash to `SPEAKER_MODEL_SHA256`, the
/// finalize path promotes them atomically. Network-free: we forge a
/// "matching" partial by precomputing the hash of an arbitrary payload
/// and passing it as the expected value.
#[cfg(feature = "diarization")]
#[test]
fn test_speaker_model_partial_promoted_on_match() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let final_path = tmp.path().join(SPEAKER_MODEL_FILE);
    let payload = b"wespeaker-surrogate";
    let expected = sha256_hex(payload);

    let partial = stage_partial(&final_path, payload);

    finalize_download(&partial, &final_path, Some(&expected), SPEAKER_MODEL_FILE)
        .expect("matching partial must promote");

    assert!(!partial.exists());
    assert!(final_path.exists());
    assert_eq!(std::fs::read(&final_path).unwrap(), payload);
}

#[test]
fn test_partial_path_unique_contains_pid_and_timestamp() {
    let p = partial_path_unique(Path::new("/tmp/final.onnx"));
    let s = p.to_string_lossy();
    assert!(s.contains(".partial."));
    assert!(s.contains(&std::process::id().to_string()));
}

#[cfg(unix)]
#[test]
fn test_acquire_download_lock_creates_lock_file() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let lock = acquire_download_lock(tmp.path()).expect("acquire lock");
    assert!(tmp.path().join(".download.lock").exists());
    drop(lock);
}

#[tokio::test]
#[cfg_attr(miri, ignore = "tokio runtime is unsupported under Miri")]
async fn test_stream_to_partial_then_finalize_success() {
    let server = wiremock::MockServer::start().await;
    let payload = b"fake model bytes";
    wiremock::Mock::given(wiremock::matchers::method("GET"))
        .and(wiremock::matchers::path("/model.onnx"))
        .respond_with(
            wiremock::ResponseTemplate::new(200)
                .set_body_bytes(payload.as_slice())
                .insert_header("content-length", payload.len().to_string()),
        )
        .mount(&server)
        .await;

    let tmp = tempfile::tempdir().expect("tempdir");
    let final_path = tmp.path().join("model.onnx");
    let url = format!("{}/model.onnx", server.uri());

    stream_to_partial_then_finalize(&url, &final_path, None, "model.onnx")
        .await
        .expect("download should succeed");

    assert!(final_path.exists());
    assert_eq!(std::fs::read(&final_path).unwrap(), payload);
}

#[tokio::test]
#[cfg_attr(miri, ignore = "tokio runtime is unsupported under Miri")]
async fn test_stream_to_partial_then_finalize_http_error() {
    let server = wiremock::MockServer::start().await;
    wiremock::Mock::given(wiremock::matchers::method("GET"))
        .and(wiremock::matchers::path("/missing.onnx"))
        .respond_with(wiremock::ResponseTemplate::new(404))
        .mount(&server)
        .await;

    let tmp = tempfile::tempdir().expect("tempdir");
    let final_path = tmp.path().join("missing.onnx");
    let url = format!("{}/missing.onnx", server.uri());

    let err = stream_to_partial_then_finalize(&url, &final_path, None, "missing.onnx")
        .await
        .expect_err("404 should fail");
    let msg = format!("{err}");
    assert!(msg.contains("404"), "error should mention 404: {msg}");
}

#[tokio::test]
#[cfg_attr(miri, ignore = "tokio runtime is unsupported under Miri")]
async fn test_stream_to_partial_then_finalize_checksum_mismatch() {
    let server = wiremock::MockServer::start().await;
    let payload = b"wrong bytes";
    wiremock::Mock::given(wiremock::matchers::method("GET"))
        .and(wiremock::matchers::path("/model.onnx"))
        .respond_with(wiremock::ResponseTemplate::new(200).set_body_bytes(payload.as_slice()))
        .mount(&server)
        .await;

    let tmp = tempfile::tempdir().expect("tempdir");
    let final_path = tmp.path().join("model.onnx");
    let url = format!("{}/model.onnx", server.uri());
    let wrong_hash = sha256_hex(b"different bytes");

    let err = stream_to_partial_then_finalize(&url, &final_path, Some(&wrong_hash), "model.onnx")
        .await
        .expect_err("checksum mismatch should fail");
    let msg = format!("{err}");
    assert!(
        msg.contains("SHA-256 mismatch"),
        "error should mention mismatch: {msg}"
    );
}

/// The RAII guard removes the staged partial on drop (early return) but
/// leaves it alone once disarmed (successful rename).
#[test]
fn test_partial_file_guard_drop_removes_staged_file() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let partial = tmp.path().join("model.onnx.partial");
    std::fs::write(&partial, b"half-written junk").expect("write partial");

    drop(PartialFileGuard::new(partial.clone()));
    assert!(!partial.exists(), "guard drop must remove the partial");

    std::fs::write(&partial, b"half-written junk").expect("write partial");
    PartialFileGuard::new(partial.clone()).disarm();
    assert!(partial.exists(), "disarmed guard must keep the partial");
}

/// A connection cut mid-stream leaves no orphan `.partial` behind: the
/// guard cleans up the staged bytes so a retry starts clean.
#[tokio::test]
#[cfg_attr(miri, ignore = "tokio runtime is unsupported under Miri")]
async fn test_stream_to_partial_then_finalize_stream_error_removes_partial() {
    use tokio::io::AsyncReadExt as _;

    // Hand-rolled stub: hyper (and therefore wiremock) refuses to send a
    // body shorter than the declared content-length, so speak raw HTTP/1.1
    // instead — send the headers, a fraction of the promised body, then
    // close the socket to cut the stream.
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind stub server");
    let addr = listener.local_addr().expect("local addr");
    let server = tokio::spawn(async move {
        let (mut socket, _) = listener.accept().await.expect("accept");
        // Drain the tiny GET request so the response does not race it.
        let mut buf = [0u8; 1024];
        let _ = socket.read(&mut buf).await;
        let head = "HTTP/1.1 200 OK\r\ncontent-length: 4096\r\nconnection: close\r\n\r\n";
        socket.write_all(head.as_bytes()).await.expect("write head");
        socket
            .write_all(b"half of the model")
            .await
            .expect("write body");
        // The socket closes on drop, truncating the promised body.
    });

    let tmp = tempfile::tempdir().expect("tempdir");
    let final_path = tmp.path().join("model.onnx");
    let url = format!("http://{addr}/model.onnx");

    let err = stream_to_partial_then_finalize(&url, &final_path, None, "model.onnx")
        .await
        .expect_err("truncated body should fail");
    let msg = format!("{err:#}");
    assert!(
        msg.contains("Download stream error"),
        "error should come from the stream read: {msg}"
    );
    server.await.expect("stub server task");

    assert!(!final_path.exists(), "final must not appear on failure");
    let orphans: Vec<_> = std::fs::read_dir(tmp.path())
        .expect("read dir")
        .filter_map(|e| e.ok())
        .filter(|e| e.file_name().to_string_lossy().contains(".partial"))
        .collect();
    assert!(orphans.is_empty(), "no orphan .partial files: {orphans:?}");
}

/// End-to-end NDJSON contract on a local HTTP stub: the download of one
/// file emits a 100% `download` event followed by a `verify` event, and
/// every line round-trips through a JSON parser (true NDJSON).
#[tokio::test]
#[cfg_attr(miri, ignore = "tokio runtime is unsupported under Miri")]
async fn test_stream_to_partial_then_finalize_json_event_sequence() {
    let server = wiremock::MockServer::start().await;
    let payload = b"fake model bytes";
    wiremock::Mock::given(wiremock::matchers::method("GET"))
        .and(wiremock::matchers::path("/model.onnx"))
        .respond_with(
            wiremock::ResponseTemplate::new(200)
                .set_body_bytes(payload.as_slice())
                .insert_header("content-length", payload.len().to_string()),
        )
        .mount(&server)
        .await;

    let tmp = tempfile::tempdir().expect("tempdir");
    let final_path = tmp.path().join("model.onnx");
    let url = format!("{}/model.onnx", server.uri());
    let expected = sha256_hex(payload);
    let (sink, log) = ProgressSink::capturing();

    stream_to_partial_then_finalize_with_sink(
        &url,
        &final_path,
        Some(&expected),
        "model.onnx",
        &sink,
    )
    .await
    .expect("download should succeed");

    let events = log.lock().unwrap();
    assert_eq!(
        events.as_slice(),
        [
            ProgressEvent::Download {
                file: "model.onnx".to_string(),
                bytes_done: payload.len() as u64,
                bytes_total: payload.len() as u64,
            },
            ProgressEvent::Verify {
                file: "model.onnx".to_string(),
            },
        ]
    );
    // The integrator view: serialize each event to a line and parse it
    // back — the stream must be well-formed NDJSON with a phase tag.
    for event in events.iter() {
        let parsed: serde_json::Value =
            serde_json::from_str(&event.to_ndjson()).expect("event must be NDJSON");
        assert!(parsed.get("phase").is_some());
    }
}

/// No checksum pinned → no `verify` event (verification did not happen).
#[tokio::test]
#[cfg_attr(miri, ignore = "tokio runtime is unsupported under Miri")]
async fn test_stream_to_partial_then_finalize_json_no_verify_without_checksum() {
    let server = wiremock::MockServer::start().await;
    let payload = b"fake model bytes";
    wiremock::Mock::given(wiremock::matchers::method("GET"))
        .and(wiremock::matchers::path("/model.onnx"))
        .respond_with(
            wiremock::ResponseTemplate::new(200)
                .set_body_bytes(payload.as_slice())
                .insert_header("content-length", payload.len().to_string()),
        )
        .mount(&server)
        .await;

    let tmp = tempfile::tempdir().expect("tempdir");
    let final_path = tmp.path().join("model.onnx");
    let url = format!("{}/model.onnx", server.uri());
    let (sink, log) = ProgressSink::capturing();

    stream_to_partial_then_finalize_with_sink(&url, &final_path, None, "model.onnx", &sink)
        .await
        .expect("download should succeed");

    let events = log.lock().unwrap();
    assert!(
        events
            .iter()
            .all(|e| !matches!(e, ProgressEvent::Verify { .. })),
        "no checksum -> no verify event: {events:?}"
    );
    assert!(
        events
            .iter()
            .any(|e| matches!(e, ProgressEvent::Download { .. })),
        "download events expected: {events:?}"
    );
}

#[test]
fn test_punct_files_checksums_are_pinned() {
    // Three files, each with a 64-char lowercase hex digest — no truncation
    // or placeholder slipping into a release.
    assert_eq!(PUNCT_FILES.len(), 3);
    for (file, sum) in PUNCT_FILES {
        assert_eq!(
            sum.len(),
            64,
            "{file} punct checksum must be 64 hex chars, got: {sum}"
        );
        assert!(
            sum.chars()
                .all(|c| c.is_ascii_digit() || ('a'..='f').contains(&c)),
            "{file} punct checksum must be lowercase hex, got: {sum}"
        );
    }
}

/// `ensure_punct_model` short-circuits (no network, no `.partial`) when all
/// three files are already present.
#[tokio::test]
#[cfg_attr(miri, ignore = "tokio runtime is unsupported under Miri")]
async fn test_ensure_punct_model_present_no_download() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let dir = tmp.path();
    for (file, _) in PUNCT_FILES {
        std::fs::write(dir.join(file), b"stub").unwrap();
    }

    ensure_punct_model(dir.to_str().unwrap())
        .await
        .expect("present model must short-circuit");

    let partials: Vec<_> = std::fs::read_dir(dir)
        .unwrap()
        .filter_map(|e| e.ok())
        .filter(|e| e.file_name().to_string_lossy().contains(".partial"))
        .collect();
    assert!(partials.is_empty(), "no .partial files: {partials:?}");
    for (file, _) in PUNCT_FILES {
        assert_eq!(std::fs::read(dir.join(file)).unwrap(), b"stub");
    }
}

/// Verify that `ensure_model(None, dir)` with a complete E2eRnnt **INT8**
/// install does NOT create any `.partial` files (no download triggered).
#[tokio::test]
#[cfg_attr(miri, ignore = "tokio runtime is unsupported under Miri")]
async fn test_ensure_model_none_respects_existing_e2e_install() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let dir = tmp.path();

    // Stage a full E2eRnnt prequantized INT8 set (stub bytes, no real ONNX).
    for f in ModelVariant::E2eRnnt.prequantized_files() {
        std::fs::write(dir.join(f), b"stub").unwrap();
    }

    // ensure_model_variant with None must return E2eRnnt without downloading.
    let variant = ensure_model_variant(None, dir.to_str().unwrap())
        .await
        .expect("ensure_model_variant should succeed");

    assert_eq!(
        variant,
        ModelVariant::E2eRnnt,
        "must use the installed E2eRnnt"
    );

    // No .partial files must have been created.
    let partials: Vec<_> = std::fs::read_dir(dir)
        .unwrap()
        .filter_map(|e| e.ok())
        .filter(|e| e.file_name().to_string_lossy().contains(".partial"))
        .collect();
    assert!(
        partials.is_empty(),
        "no .partial files must exist: {partials:?}"
    );

    // Files must be untouched (still stub bytes).
    for f in ModelVariant::E2eRnnt.prequantized_files() {
        assert_eq!(
            std::fs::read(dir.join(f)).unwrap(),
            b"stub",
            "{f} must be unchanged"
        );
    }
}

/// FP32-only E2eRnnt is not usable: ensure falls through to default Rnnt
/// download path would run if the network were hit — we only assert the
/// presence filter rejects FP32-only (no Use short-circuit).
#[test]
fn test_fp32_only_e2e_is_not_usable() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let dir = tmp.path();
    for f in ModelVariant::E2eRnnt.download_files() {
        std::fs::write(dir.join(f), b"stub").unwrap();
    }
    assert!(
        is_model_present(ModelVariant::E2eRnnt, dir),
        "FP32 download set is complete"
    );
    assert!(
        !is_usable_present(ModelVariant::E2eRnnt, dir),
        "FP32-only must not count as a usable install"
    );
    let existing = ModelVariant::detect_in_dir(dir).filter(|&v| is_usable_present(v, dir));
    assert_eq!(existing, None, "detect+usable filter must ignore FP32-only");
}

/// The legacy public `ensure_model(dir)` wrapper delegates to
/// `ensure_model_variant(None, dir)`: with a complete INT8 install already
/// on disk it must succeed without touching the network (no `.partial` files).
#[tokio::test]
#[cfg_attr(miri, ignore = "tokio runtime is unsupported under Miri")]
async fn test_ensure_model_wrapper_uses_existing_install() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let dir = tmp.path();

    // Stage a complete default (Rnnt) prequantized INT8 set.
    for f in ModelVariant::Rnnt.prequantized_files() {
        std::fs::write(dir.join(f), b"stub").unwrap();
    }

    ensure_model(dir.to_str().unwrap())
        .await
        .expect("ensure_model must succeed against an existing install");

    let partials: Vec<_> = std::fs::read_dir(dir)
        .unwrap()
        .filter_map(|e| e.ok())
        .filter(|e| e.file_name().to_string_lossy().contains(".partial"))
        .collect();
    assert!(
        partials.is_empty(),
        "ensure_model must not download when the set is present: {partials:?}"
    );
}

/// `ensure_model_variant(Some(Rnnt), dir)` against a matching INT8 install
/// is the `VariantAction::Use` branch: returns Rnnt with no download.
#[tokio::test]
#[cfg_attr(miri, ignore = "tokio runtime is unsupported under Miri")]
async fn test_ensure_model_variant_explicit_match_uses_existing() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let dir = tmp.path();

    for f in ModelVariant::Rnnt.prequantized_files() {
        std::fs::write(dir.join(f), b"stub").unwrap();
    }

    let variant = ensure_model_variant(Some(ModelVariant::Rnnt), dir.to_str().unwrap())
        .await
        .expect("explicit matching variant must short-circuit");
    assert_eq!(variant, ModelVariant::Rnnt);

    let has_partial = std::fs::read_dir(dir)
        .unwrap()
        .filter_map(|e| e.ok())
        .any(|e| e.file_name().to_string_lossy().contains(".partial"));
    assert!(!has_partial, "no download for an explicit matching variant");
}

/// `ensure_vad_model` short-circuits (no network, no `.partial`) when the
/// Silero ONNX file is already present in the VAD directory.
#[tokio::test]
#[cfg_attr(miri, ignore = "tokio runtime is unsupported under Miri")]
async fn test_ensure_vad_model_present_no_download() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let dir = tmp.path();
    std::fs::write(dir.join(crate::vad::VAD_MODEL_FILE), b"stub vad").unwrap();

    ensure_vad_model(dir.to_str().unwrap())
        .await
        .expect("present VAD model must short-circuit");

    let partials: Vec<_> = std::fs::read_dir(dir)
        .unwrap()
        .filter_map(|e| e.ok())
        .filter(|e| e.file_name().to_string_lossy().contains(".partial"))
        .collect();
    assert!(partials.is_empty(), "no .partial files: {partials:?}");
    assert_eq!(
        std::fs::read(dir.join(crate::vad::VAD_MODEL_FILE)).unwrap(),
        b"stub vad",
        "existing VAD model must be left untouched"
    );
}

/// `default_punct_model_dir` and `default_vad_model_dir` are siblings of the
/// main model dir under `.gigastt/models`, with the expected leaf names.
#[test]
fn test_default_punct_and_vad_dirs_are_model_siblings() {
    let punct = default_punct_model_dir();
    let vad = default_vad_model_dir();
    assert!(
        punct.contains(".gigastt") && punct.ends_with("punct"),
        "punct dir should be under .gigastt and end with 'punct', got: {punct}"
    );
    assert!(
        vad.contains(".gigastt") && vad.ends_with("vad"),
        "vad dir should be under .gigastt and end with 'vad', got: {vad}"
    );
}

/// `VAD_MODEL_SHA256` is a 64-char lowercase hex digest (no truncation or
/// placeholder slipping into a release).
#[test]
fn test_vad_model_sha256_shape() {
    assert_eq!(
        VAD_MODEL_SHA256.len(),
        64,
        "VAD_MODEL_SHA256 must be a 64-char hex digest"
    );
    assert!(
        VAD_MODEL_SHA256
            .chars()
            .all(|c| c.is_ascii_digit() || ('a'..='f').contains(&c)),
        "VAD_MODEL_SHA256 must be lowercase hex; got: {VAD_MODEL_SHA256}"
    );
}

/// `ensure_model_variant` tolerates a deep, freshly-created model directory:
/// a complete Rnnt **INT8** set pre-staged under a nested path is detected
/// and used as-is (early `Use(...)` return) without any network access.
#[tokio::test]
#[cfg_attr(miri, ignore = "tokio runtime is unsupported under Miri")]
async fn test_ensure_model_variant_uses_complete_set_in_nested_path() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let nested = tmp.path().join("a").join("b").join("models");
    std::fs::create_dir_all(&nested).unwrap();
    for f in ModelVariant::Rnnt.prequantized_files() {
        std::fs::write(nested.join(f), b"stub").unwrap();
    }

    let variant = ensure_model_variant(None, nested.to_str().unwrap())
        .await
        .expect("nested complete install must be used as-is");
    assert_eq!(variant, ModelVariant::Rnnt);
}

// ── pre-quantized bundle ────────────────────────────────────────────────

#[test]
fn test_prequantized_files_mapping() {
    assert_eq!(
        ModelVariant::Rnnt.prequantized_files(),
        [
            "v3_rnnt_encoder_int8.onnx",
            "v3_rnnt_decoder.onnx",
            "v3_rnnt_joint.onnx",
            "v3_vocab.txt",
        ]
    );
    assert_eq!(
        ModelVariant::E2eRnnt.prequantized_files(),
        [
            "v3_e2e_rnnt_encoder_int8.onnx",
            "v3_e2e_rnnt_decoder.onnx",
            "v3_e2e_rnnt_joint.onnx",
            "v3_e2e_rnnt_vocab.txt",
        ]
    );
    // CTC is encoder-only: pre-quantized set is just the INT8 encoder + vocab.
    assert_eq!(
        ModelVariant::MlCtc.prequantized_files(),
        ["multilingual_ctc.int8.onnx", "multilingual_vocab.txt"]
    );
    assert_eq!(
        ModelVariant::MlCtcLarge.prequantized_files(),
        ["multilingual_large_ctc.int8.onnx", "multilingual_vocab.txt"]
    );
}

#[test]
fn test_encoder_int8_checksums_are_pinned() {
    for variant in [
        ModelVariant::Rnnt,
        ModelVariant::E2eRnnt,
        ModelVariant::MlCtc,
        ModelVariant::MlCtcLarge,
    ] {
        let sum = variant.encoder_int8_checksum();
        assert_eq!(
            sum.len(),
            64,
            "{variant:?} INT8 checksum must be 64 hex chars, got: {sum}"
        );
        assert!(
            sum.chars()
                .all(|c| c.is_ascii_digit() || ('a'..='f').contains(&c)),
            "{variant:?} INT8 checksum must be lowercase hex, got: {sum}"
        );
    }
}

#[test]
fn test_prequantized_checksum_int8_encoder_and_reuses_fp32_for_rest() {
    let v = ModelVariant::Rnnt;
    // Encoder → the INT8-specific checksum.
    assert_eq!(
        v.prequantized_checksum(v.encoder_int8_file()),
        Some(v.encoder_int8_checksum())
    );
    // Decoder/joiner/vocab → the same pins as the FP32 download set.
    for f in [v.decoder_file(), v.joint_file(), v.vocab_file()] {
        assert_eq!(v.prequantized_checksum(f), v.checksum(f));
        assert!(v.prequantized_checksum(f).is_some(), "{f} must be pinned");
    }
}

#[test]
fn test_is_prequantized_present() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let dir = tmp.path();
    assert!(!is_prequantized_present(ModelVariant::Rnnt, dir));
    for f in ModelVariant::Rnnt.prequantized_files() {
        std::fs::write(dir.join(f), b"x").unwrap();
    }
    assert!(is_prequantized_present(ModelVariant::Rnnt, dir));
    // The FP32 set is absent (no FP32 encoder), yet the prequantized set is
    // complete — the two presence checks are independent.
    assert!(!is_model_present(ModelVariant::Rnnt, dir));
}

/// Regression: an INT8-only (prequantized) tree must count as a usable
/// install for ensure/bootstrap — same filter as `ensure_model_variant`.
/// Without `is_usable_present`, serve would re-download the ~844 MB FP32 set.
#[test]
fn test_ensure_filter_accepts_prequantized_only_dir() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let dir = tmp.path();
    for f in ModelVariant::Rnnt.prequantized_files() {
        std::fs::write(dir.join(f), b"stub").unwrap();
    }
    assert_eq!(
        ModelVariant::detect_in_dir(dir),
        Some(ModelVariant::Rnnt),
        "encoder_int8 alone is enough for detect_in_dir"
    );
    assert!(
        is_prequantized_present(ModelVariant::Rnnt, dir),
        "prequantized set is complete"
    );
    assert!(
        !is_model_present(ModelVariant::Rnnt, dir),
        "FP32 download set must still be absent"
    );
    assert!(
        is_usable_present(ModelVariant::Rnnt, dir),
        "INT8-only set is usable without FP32"
    );
    // Same filter as ensure_model_variant:
    let existing = ModelVariant::detect_in_dir(dir).filter(|&v| is_usable_present(v, dir));
    assert_eq!(
        existing,
        Some(ModelVariant::Rnnt),
        "ensure treats prequantized-only as present"
    );
    assert_eq!(
        resolve_variant(None, existing),
        VariantAction::Use(ModelVariant::Rnnt),
        "usable existing → no download"
    );
}

/// `ensure_model_variant` short-circuits (no network, no `.partial`) when
/// only the pre-quantized INT8 set is on disk.
#[tokio::test]
#[cfg_attr(miri, ignore = "tokio runtime is unsupported under Miri")]
async fn test_ensure_model_variant_accepts_prequantized_only() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let dir = tmp.path();
    for f in ModelVariant::Rnnt.prequantized_files() {
        std::fs::write(dir.join(f), b"stub").unwrap();
    }

    let variant = ensure_model_variant(None, dir.to_str().unwrap())
        .await
        .expect("prequantized-only dir must short-circuit without download");
    assert_eq!(variant, ModelVariant::Rnnt);

    let has_partial = std::fs::read_dir(dir)
        .unwrap()
        .filter_map(|e| e.ok())
        .any(|e| e.file_name().to_string_lossy().contains(".partial"));
    assert!(
        !has_partial,
        "no download when the prequantized set is present"
    );
    // Still no FP32 encoder on disk.
    assert!(!dir.join(ModelVariant::Rnnt.encoder_file()).exists());
}

/// `ensure_prequantized_model_variant` short-circuits (no network, no
/// `.partial`) when the pre-quantized set is already present.
#[tokio::test]
#[cfg_attr(miri, ignore = "tokio runtime is unsupported under Miri")]
async fn test_ensure_prequantized_present_no_download() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let dir = tmp.path();
    for f in ModelVariant::Rnnt.prequantized_files() {
        std::fs::write(dir.join(f), b"stub").unwrap();
    }

    let variant = ensure_prequantized_model_variant(None, dir.to_str().unwrap())
        .await
        .expect("present prequantized set must short-circuit");
    assert_eq!(variant, ModelVariant::Rnnt);

    let has_partial = std::fs::read_dir(dir)
        .unwrap()
        .filter_map(|e| e.ok())
        .any(|e| e.file_name().to_string_lossy().contains(".partial"));
    assert!(
        !has_partial,
        "no download when the prequantized set is present"
    );
}

/// A custom pack is authoritative even when standard model files also exist.
#[tokio::test]
#[cfg_attr(miri, ignore = "tokio runtime is unsupported under Miri")]
async fn test_ensure_model_variant_custom_manifest_controls_bootstrap() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path();
    for file in ModelVariant::Rnnt.prequantized_files() {
        std::fs::write(dir.join(file), b"standard").unwrap();
    }
    std::fs::write(
        dir.join("manifest.toml"),
        "architecture = \"ml_ctc\"\n[files]\nencoder = \"custom.int8.onnx\"\nencoder_int8 = \"custom.int8.onnx\"\nvocab = \"custom_vocab.txt\"\n",
    ).unwrap();
    std::fs::write(dir.join("custom.int8.onnx"), b"custom").unwrap();
    std::fs::write(dir.join("custom_vocab.txt"), b"vocab").unwrap();
    assert_eq!(
        ensure_model_variant(None, dir.to_str().unwrap())
            .await
            .unwrap(),
        ModelVariant::MlCtc
    );
    // Incomplete custom packs must not silently select or download another model.
    std::fs::remove_file(dir.join("custom_vocab.txt")).unwrap();
    assert!(
        ensure_model_variant(None, dir.to_str().unwrap())
            .await
            .is_err()
    );
}

#[tokio::test]
#[cfg_attr(miri, ignore = "tokio runtime is unsupported under Miri")]
async fn test_ensure_model_variant_standalone_pack_validation() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path();
    let manifest = "architecture = \"ml_ctc\"\n[files]\nencoder = \"custom.int8.onnx\"\nencoder_int8 = \"custom.int8.onnx\"\nvocab = \"custom_vocab.txt\"\n";
    std::fs::write(dir.join("manifest.toml"), manifest).unwrap();
    std::fs::write(dir.join("custom.int8.onnx"), b"custom").unwrap();
    std::fs::write(dir.join("custom_vocab.txt"), b"vocab").unwrap();
    assert_eq!(
        ensure_model_variant(Some(ModelVariant::MlCtc), dir.to_str().unwrap())
            .await
            .unwrap(),
        ModelVariant::MlCtc
    );
    assert!(
        ensure_model_variant(Some(ModelVariant::Rnnt), dir.to_str().unwrap())
            .await
            .unwrap_err()
            .to_string()
            .contains("conflicts")
    );
    std::fs::remove_file(dir.join("custom.int8.onnx")).unwrap();
    assert!(
        ensure_model_variant(None, dir.to_str().unwrap())
            .await
            .unwrap_err()
            .to_string()
            .contains("encoder_int8")
    );
    std::fs::create_dir(dir.join("custom.int8.onnx")).unwrap();
    assert!(
        ensure_model_variant(None, dir.to_str().unwrap())
            .await
            .unwrap_err()
            .to_string()
            .contains("file missing")
    );
    std::fs::write(dir.join("manifest.toml"), "invalid manifest").unwrap();
    assert!(
        ensure_model_variant(None, dir.to_str().unwrap())
            .await
            .is_err()
    );
    assert_eq!(
        std::fs::read_dir(dir).unwrap().count(),
        3,
        "validation must not create download files"
    );
}
