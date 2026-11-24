use super::*;
use gigastt_core::inference::WordInfo;
use std::sync::atomic::{AtomicUsize, Ordering};

fn sample_result() -> TranscribeResult {
    TranscribeResult {
        text: "привет мир".to_string(),
        words: vec![
            WordInfo::new("привет", 0.0, 0.5, 0.98, None),
            WordInfo::new("мир", 0.6, 1.0, 0.97, None),
        ],
        confidence: Some(0.975),
        duration_s: 1.0,
    }
}

fn ok_transcribe() -> TranscribeFn {
    Arc::new(|_| Ok(sample_result()))
}

fn test_opts(input: &Path, output: &Path) -> BatchOptions {
    BatchOptions {
        input_dir: input.to_path_buf(),
        output_dir: output.to_path_buf(),
        formats: vec![ExportFormat::Txt, ExportFormat::Json],
        render_opts: RenderOpts::default(),
        move_to: None,
        delete_source: false,
        concurrency: 2,
        retries: 0,
    }
}

/// Write a minimal PCM16 WAV file (silence) — enough for the walker; the
/// stubbed transcribe closure never decodes it.
fn write_wav(path: &Path) {
    let samples = [0_i16; 160];
    let data_len = (samples.len() * 2) as u32;
    let mut buf = Vec::with_capacity(44 + data_len as usize);
    buf.extend_from_slice(b"RIFF");
    buf.extend_from_slice(&(36 + data_len).to_le_bytes());
    buf.extend_from_slice(b"WAVEfmt ");
    buf.extend_from_slice(&16_u32.to_le_bytes());
    buf.extend_from_slice(&1_u16.to_le_bytes()); // PCM
    buf.extend_from_slice(&1_u16.to_le_bytes()); // mono
    buf.extend_from_slice(&16000_u32.to_le_bytes());
    buf.extend_from_slice(&32000_u32.to_le_bytes());
    buf.extend_from_slice(&2_u16.to_le_bytes());
    buf.extend_from_slice(&16_u16.to_le_bytes());
    buf.extend_from_slice(b"data");
    buf.extend_from_slice(&data_len.to_le_bytes());
    for s in samples {
        buf.extend_from_slice(&s.to_le_bytes());
    }
    std::fs::write(path, buf).unwrap();
}

#[test]
fn test_is_audio_file_accepts_supported_case_insensitive() {
    assert!(is_audio_file(Path::new("a.wav")));
    assert!(is_audio_file(Path::new("a.MP3")));
    assert!(is_audio_file(Path::new("a.m4a")));
    assert!(is_audio_file(Path::new("a.OGG")));
    assert!(is_audio_file(Path::new("a.flac")));
    // A folder of browser recordings is a plausible batch input, and every
    // one of them carries this extension.
    assert!(is_audio_file(Path::new("a.webm")));
}

#[test]
fn test_is_audio_file_rejects_other_extensions() {
    assert!(!is_audio_file(Path::new("a.txt")));
    assert!(!is_audio_file(Path::new("a.json")));
    assert!(!is_audio_file(Path::new("a")));
    assert!(!is_audio_file(Path::new(".wav")));
}

#[test]
fn test_collect_audio_files_recurses_and_sorts() {
    let tmp = tempfile::tempdir().unwrap();
    let root = tmp.path();
    write_wav(&root.join("b.wav"));
    write_wav(&root.join("a.mp3"));
    std::fs::write(root.join("notes.txt"), b"x").unwrap();
    std::fs::create_dir(root.join("sub")).unwrap();
    write_wav(&root.join("sub").join("c.flac"));

    let files = collect_audio_files(root).unwrap();
    let names: Vec<_> = files
        .iter()
        .map(|p| p.file_name().unwrap().to_str().unwrap().to_string())
        .collect();
    assert_eq!(names, vec!["a.mp3", "b.wav", "c.flac"]);
}

#[test]
fn test_collect_audio_files_missing_dir_errors() {
    let result = collect_audio_files(Path::new("/nonexistent/dir"));
    assert!(result.is_err());
}

#[test]
fn test_output_path_for_uses_stem_and_extension() {
    let out = output_path_for(Path::new("/in/sub/rec.wav"), Path::new("/out"), "json");
    assert_eq!(out, PathBuf::from("/out/rec.json"));
}

#[test]
fn test_output_path_for_dotfile_keeps_full_name() {
    // `.hidden` has no extension; the whole name is the stem.
    let out = output_path_for(Path::new("/in/.hidden"), Path::new("/out"), "txt");
    assert_eq!(out, PathBuf::from("/out/.hidden.txt"));
}

#[cfg(unix)]
#[test]
fn test_output_path_for_non_utf8_stem_falls_back() {
    use std::os::unix::ffi::OsStrExt;
    let path = PathBuf::from(std::ffi::OsStr::from_bytes(b"/in/\xff\xfe.wav"));
    let out = output_path_for(&path, Path::new("/out"), "txt");
    assert_eq!(out, PathBuf::from("/out/transcript.txt"));
}

#[test]
fn test_parse_formats_csv_dedup_and_order() {
    let formats = parse_formats("txt, json ,txt,md").unwrap();
    assert_eq!(
        formats,
        vec![ExportFormat::Txt, ExportFormat::Json, ExportFormat::Md]
    );
}

#[test]
fn test_parse_formats_rejects_unknown_and_empty() {
    assert!(parse_formats("txt,docx").is_err());
    assert!(parse_formats(" , ").is_err());
}

#[tokio::test]
async fn test_run_batch_writes_all_formats() {
    let tmp = tempfile::tempdir().unwrap();
    let input = tmp.path().join("in");
    let output = tmp.path().join("out");
    std::fs::create_dir(&input).unwrap();
    write_wav(&input.join("one.wav"));
    write_wav(&input.join("two.mp3"));

    let summary = run_batch(
        &test_opts(&input, &output),
        ok_transcribe(),
        tokio_util::sync::CancellationToken::new(),
    )
    .await
    .unwrap();

    assert_eq!(summary.processed, 2);
    assert_eq!(summary.failed, 0);
    assert!(!summary.interrupted);
    for stem in ["one", "two"] {
        let txt = std::fs::read_to_string(output.join(format!("{stem}.txt"))).unwrap();
        assert_eq!(txt, "привет мир");
        let json = std::fs::read_to_string(output.join(format!("{stem}.json"))).unwrap();
        assert!(json.contains("привет мир"));
    }
}

#[tokio::test]
async fn test_run_batch_continues_after_failure() {
    let tmp = tempfile::tempdir().unwrap();
    let input = tmp.path().join("in");
    let output = tmp.path().join("out");
    std::fs::create_dir(&input).unwrap();
    write_wav(&input.join("good.wav"));
    write_wav(&input.join("bad.wav"));

    let transcribe: TranscribeFn = Arc::new(|path| {
        if path.file_stem().unwrap() == "bad" {
            anyhow::bail!("decode exploded");
        }
        Ok(sample_result())
    });
    let summary = run_batch(
        &test_opts(&input, &output),
        transcribe,
        tokio_util::sync::CancellationToken::new(),
    )
    .await
    .unwrap();

    assert_eq!(summary.processed, 1);
    assert_eq!(summary.failed, 1);
    assert!(output.join("good.txt").exists());
    assert!(!output.join("bad.txt").exists());
    // A failed source is left in place for inspection.
    assert!(input.join("bad.wav").exists());
}

#[tokio::test]
async fn test_run_batch_move_to_moves_source_after_success() {
    let tmp = tempfile::tempdir().unwrap();
    let input = tmp.path().join("in");
    let output = tmp.path().join("out");
    let done = input.join("done");
    std::fs::create_dir(&input).unwrap();
    write_wav(&input.join("a.wav"));
    // A backlog file already inside done/ must not be reprocessed.
    std::fs::create_dir(&done).unwrap();
    write_wav(&done.join("old.wav"));

    let mut opts = test_opts(&input, &output);
    opts.move_to = Some(done.clone());
    let summary = run_batch(
        &opts,
        ok_transcribe(),
        tokio_util::sync::CancellationToken::new(),
    )
    .await
    .unwrap();

    assert_eq!(summary.processed, 1);
    assert!(!input.join("a.wav").exists());
    assert!(done.join("a.wav").exists());
    assert!(output.join("a.txt").exists());
    assert!(!output.join("old.txt").exists());
}

#[tokio::test]
async fn test_run_batch_delete_source_removes_file() {
    let tmp = tempfile::tempdir().unwrap();
    let input = tmp.path().join("in");
    let output = tmp.path().join("out");
    std::fs::create_dir(&input).unwrap();
    write_wav(&input.join("a.wav"));

    let mut opts = test_opts(&input, &output);
    opts.delete_source = true;
    let summary = run_batch(
        &opts,
        ok_transcribe(),
        tokio_util::sync::CancellationToken::new(),
    )
    .await
    .unwrap();

    assert_eq!(summary.processed, 1);
    assert!(!input.join("a.wav").exists());
    assert!(output.join("a.txt").exists());
}

#[tokio::test]
async fn test_run_batch_retries_transient_failure() {
    let tmp = tempfile::tempdir().unwrap();
    let input = tmp.path().join("in");
    let output = tmp.path().join("out");
    std::fs::create_dir(&input).unwrap();
    write_wav(&input.join("a.wav"));

    let calls = Arc::new(AtomicUsize::new(0));
    let calls2 = calls.clone();
    let transcribe: TranscribeFn = Arc::new(move |_| {
        if calls2.fetch_add(1, Ordering::SeqCst) == 0 {
            anyhow::bail!("transient");
        }
        Ok(sample_result())
    });
    let mut opts = test_opts(&input, &output);
    opts.retries = 2;
    let summary = run_batch(
        &opts,
        transcribe,
        tokio_util::sync::CancellationToken::new(),
    )
    .await
    .unwrap();

    assert_eq!(summary.processed, 1);
    assert_eq!(summary.failed, 0);
    assert_eq!(calls.load(Ordering::SeqCst), 2);
}

#[tokio::test]
async fn test_run_batch_cancelled_before_start_processes_nothing() {
    let tmp = tempfile::tempdir().unwrap();
    let input = tmp.path().join("in");
    let output = tmp.path().join("out");
    std::fs::create_dir(&input).unwrap();
    write_wav(&input.join("a.wav"));
    write_wav(&input.join("b.wav"));

    let token = tokio_util::sync::CancellationToken::new();
    token.cancel();
    let summary = run_batch(&test_opts(&input, &output), ok_transcribe(), token)
        .await
        .unwrap();

    assert_eq!(summary.processed, 0);
    assert_eq!(summary.skipped, 2);
    assert!(summary.interrupted);
}

#[tokio::test]
async fn test_run_batch_empty_dir_is_ok() {
    let tmp = tempfile::tempdir().unwrap();
    let input = tmp.path().join("in");
    std::fs::create_dir(&input).unwrap();
    let summary = run_batch(
        &test_opts(&input, &tmp.path().join("out")),
        ok_transcribe(),
        tokio_util::sync::CancellationToken::new(),
    )
    .await
    .unwrap();
    assert_eq!(summary, BatchSummary::default());
}

#[tokio::test]
async fn test_watch_processes_new_file_and_shuts_down() {
    let tmp = tempfile::tempdir().unwrap();
    let input = tmp.path().join("in");
    let output = tmp.path().join("out");
    std::fs::create_dir(&input).unwrap();
    // Pre-existing backlog file: watch must leave it alone.
    write_wav(&input.join("old.wav"));

    let token = tokio_util::sync::CancellationToken::new();
    let opts = WatchOptions {
        batch: BatchOptions {
            concurrency: 1,
            ..test_opts(&input, &output)
        },
        poll_interval: Duration::from_millis(10),
        settle_polls: 2,
    };
    // Drive the scenario from a separate task: drop a new file, wait for
    // its outputs, then shut the watch down.
    let driver = tokio::spawn({
        let input = input.clone();
        let output = output.clone();
        let token = token.clone();
        async move {
            tokio::time::sleep(Duration::from_millis(50)).await;
            write_wav(&input.join("new.wav"));
            let deadline = std::time::Instant::now() + Duration::from_secs(10);
            while !output.join("new.txt").exists() {
                assert!(std::time::Instant::now() < deadline, "watch timed out");
                tokio::time::sleep(Duration::from_millis(20)).await;
            }
            token.cancel();
        }
    });
    let summary = run_watch(&opts, ok_transcribe(), token).await.unwrap();
    driver.await.unwrap();

    assert_eq!(summary.processed, 1);
    assert_eq!(summary.failed, 0);
    assert!(!output.join("old.txt").exists());
}

#[tokio::test]
async fn test_watch_skips_move_to_directory() {
    let tmp = tempfile::tempdir().unwrap();
    let input = tmp.path().join("in");
    let output = tmp.path().join("out");
    let done = input.join("done");
    std::fs::create_dir(&input).unwrap();
    std::fs::create_dir(&done).unwrap();

    let token = tokio_util::sync::CancellationToken::new();
    let opts = WatchOptions {
        batch: BatchOptions {
            move_to: Some(done.clone()),
            concurrency: 1,
            ..test_opts(&input, &output)
        },
        poll_interval: Duration::from_millis(10),
        settle_polls: 1,
    };
    let driver = tokio::spawn({
        let input = input.clone();
        let done = done.clone();
        let token = token.clone();
        async move {
            tokio::time::sleep(Duration::from_millis(30)).await;
            write_wav(&input.join("a.wav"));
            let deadline = std::time::Instant::now() + Duration::from_secs(10);
            while !done.join("a.wav").exists() {
                assert!(std::time::Instant::now() < deadline, "watch timed out");
                tokio::time::sleep(Duration::from_millis(20)).await;
            }
            // Let a few extra polls run: the moved file must not be
            // re-processed.
            tokio::time::sleep(Duration::from_millis(100)).await;
            token.cancel();
        }
    });
    let summary = run_watch(&opts, ok_transcribe(), token).await.unwrap();
    driver.await.unwrap();

    assert_eq!(summary.processed, 1);
    assert!(output.join("a.txt").exists());
}

mod collisions;
