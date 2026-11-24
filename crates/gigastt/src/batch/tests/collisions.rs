use super::*;
use std::sync::Barrier;
use std::sync::atomic::{AtomicUsize, Ordering};
use tokio_util::sync::CancellationToken;

#[tokio::test]
async fn test_batch_refuses_duplicate_destinations_before_transcribing() {
    for names in [["a/call.wav", "b/call.mp3"], ["a/call.wav", "b/call.wav"]] {
        let tmp = tempfile::tempdir().unwrap();
        let input = tmp.path().join("in");
        let output = tmp.path().join("out");
        for (name, audio) in names.into_iter().zip([b"first", b"other"]) {
            let path = input.join(name);
            std::fs::create_dir_all(path.parent().unwrap()).unwrap();
            std::fs::write(path, audio).unwrap();
        }

        let calls = Arc::new(AtomicUsize::new(0));
        let count = Arc::clone(&calls);
        let transcribe: TranscribeFn = Arc::new(move |_| {
            count.fetch_add(1, Ordering::Relaxed);
            Ok(sample_result())
        });
        let mut opts = test_opts(&input, &output);
        opts.delete_source = true;

        let result = run_batch(&opts, transcribe, CancellationToken::new()).await;
        assert!(result.is_err());
        assert_eq!(calls.load(Ordering::Relaxed), 0);
        assert_eq!(std::fs::read(input.join(names[0])).unwrap(), b"first");
        assert_eq!(std::fs::read(input.join(names[1])).unwrap(), b"other");
        assert_eq!(std::fs::read_dir(&output).unwrap().count(), 0);
    }
}

#[tokio::test]
async fn test_batch_preserves_existing_archive_and_both_sources() {
    let tmp = tempfile::tempdir().unwrap();
    let input = tmp.path().join("in");
    let output = tmp.path().join("out");
    let archive = tmp.path().join("archive");
    std::fs::create_dir(&input).unwrap();
    std::fs::create_dir(&archive).unwrap();
    std::fs::write(input.join("call.wav"), b"source").unwrap();
    std::fs::write(archive.join("call.wav"), b"older archive").unwrap();
    let mut opts = test_opts(&input, &output);
    opts.move_to = Some(archive.clone());

    let result = run_batch(&opts, ok_transcribe(), CancellationToken::new()).await;
    assert!(result.is_err());
    assert_eq!(
        std::fs::read(archive.join("call.wav")).unwrap(),
        b"older archive"
    );
    assert_eq!(std::fs::read(input.join("call.wav")).unwrap(), b"source");
    assert_eq!(std::fs::read_dir(&output).unwrap().count(), 0);
}

#[tokio::test]
async fn test_concurrent_batches_cannot_clobber_outputs() {
    let tmp = tempfile::tempdir().unwrap();
    let a = tmp.path().join("a");
    let b = tmp.path().join("b");
    let output = tmp.path().join("out");
    std::fs::create_dir(&a).unwrap();
    std::fs::create_dir(&b).unwrap();
    std::fs::write(a.join("call.wav"), b"first").unwrap();
    std::fs::write(b.join("call.wav"), b"other").unwrap();

    let barrier = Arc::new(Barrier::new(2));
    let transcribe: TranscribeFn = Arc::new(move |source| {
        let mut result = sample_result();
        result.text = std::fs::read_to_string(source)?;
        barrier.wait();
        Ok(result)
    });
    let a_opts = BatchOptions {
        delete_source: true,
        ..test_opts(&a, &output)
    };
    let b_opts = BatchOptions {
        delete_source: true,
        ..test_opts(&b, &output)
    };
    let (a_result, b_result) = tokio::join!(
        run_batch(&a_opts, transcribe.clone(), CancellationToken::new()),
        run_batch(&b_opts, transcribe, CancellationToken::new())
    );
    let a_result = a_result.unwrap();
    let b_result = b_result.unwrap();

    assert_eq!(a_result.processed + b_result.processed, 1);
    assert_eq!(a_result.failed + b_result.failed, 1);
    let (winner, loser, losing_audio) = if a_result.processed == 1 {
        (&a, &b, b"other")
    } else {
        (&b, &a, b"first")
    };
    assert!(!winner.join("call.wav").exists());
    assert_eq!(std::fs::read(loser.join("call.wav")).unwrap(), losing_audio);
    let text = std::fs::read_to_string(output.join("call.txt")).unwrap();
    let json = std::fs::read_to_string(output.join("call.json")).unwrap();
    let word = if a_result.processed == 1 {
        "first"
    } else {
        "other"
    };
    assert_eq!(text.trim(), word);
    assert_eq!(
        serde_json::from_str::<serde_json::Value>(&json).unwrap()["text"],
        word
    );
}

#[tokio::test]
async fn test_multiformat_failure_preserves_source() {
    let tmp = tempfile::tempdir().unwrap();
    let input = tmp.path().join("in");
    let output = tmp.path().join("out");
    std::fs::create_dir(&input).unwrap();
    let source = input.join("call.wav");
    std::fs::write(&source, b"audio").unwrap();

    let occupied = output.join("call.json");
    let unexpected_writer = occupied.clone();
    let transcribe: TranscribeFn = Arc::new(move |_| {
        std::fs::write(&unexpected_writer, b"racing report")?;
        Ok(sample_result())
    });
    let opts = BatchOptions {
        delete_source: true,
        ..test_opts(&input, &output)
    };
    let result = run_batch(&opts, transcribe, CancellationToken::new())
        .await
        .unwrap();

    assert_eq!(result.failed, 1);
    assert!(!output.join("call.txt").exists());
    assert_eq!(std::fs::read(occupied).unwrap(), b"racing report");
    assert_eq!(std::fs::read(source).unwrap(), b"audio");
}

#[tokio::test]
async fn test_batch_preserves_source_changed_by_producer_during_transcription() {
    let tmp = tempfile::tempdir().unwrap();
    let input = tmp.path().join("in");
    let output = tmp.path().join("out");
    std::fs::create_dir(&input).unwrap();
    let source = input.join("call.wav");
    std::fs::write(&source, b"audio").unwrap();
    let transcribe: TranscribeFn = Arc::new(move |path| {
        std::fs::write(path, b"longer audio from producer")?;
        Ok(sample_result())
    });
    let opts = BatchOptions {
        delete_source: true,
        ..test_opts(&input, &output)
    };

    let result = run_batch(&opts, transcribe, CancellationToken::new())
        .await
        .unwrap();

    assert_eq!(result.failed, 1);
    assert!(!output.join("call.txt").exists());
    assert_eq!(
        std::fs::read(source).unwrap(),
        b"longer audio from producer"
    );
}

#[tokio::test]
async fn test_watch_preserves_colliding_input_and_processes_unrelated_file() {
    let tmp = tempfile::tempdir().unwrap();
    let input = tmp.path().join("in");
    let output = tmp.path().join("out");
    let archive = tmp.path().join("archive");
    std::fs::create_dir(&input).unwrap();
    std::fs::create_dir(&output).unwrap();
    std::fs::create_dir(&archive).unwrap();
    std::fs::write(output.join("call.txt"), b"previous transcript").unwrap();
    std::fs::write(archive.join("call.wav"), b"previous archive").unwrap();

    let token = CancellationToken::new();
    let opts = WatchOptions {
        batch: BatchOptions {
            move_to: Some(archive.clone()),
            ..test_opts(&input, &output)
        },
        poll_interval: Duration::from_millis(5),
        settle_polls: 1,
    };
    let driver = tokio::spawn({
        let input = input.clone();
        let archive = archive.clone();
        let token = token.clone();
        async move {
            // The watch task registers its backlog before the first poll.
            tokio::time::sleep(Duration::from_millis(25)).await;
            std::fs::write(input.join("call.wav"), b"new audio").unwrap();
            std::fs::write(input.join("safe.wav"), b"safe audio").unwrap();
            tokio::time::timeout(Duration::from_secs(10), async {
                while !archive.join("safe.wav").exists() {
                    tokio::time::sleep(Duration::from_millis(5)).await;
                }
            })
            .await
            .unwrap();
            token.cancel();
        }
    });
    let summary = run_watch(&opts, ok_transcribe(), token).await.unwrap();
    driver.await.unwrap();
    assert_eq!(summary.failed, 1);
    assert_eq!(summary.processed, 1);
    assert_eq!(std::fs::read(input.join("call.wav")).unwrap(), b"new audio");
    assert_eq!(
        std::fs::read(archive.join("call.wav")).unwrap(),
        b"previous archive"
    );
    assert_eq!(
        std::fs::read(output.join("call.txt")).unwrap(),
        b"previous transcript"
    );
    assert_eq!(
        std::fs::read(archive.join("safe.wav")).unwrap(),
        b"safe audio"
    );
}

#[cfg(unix)]
#[tokio::test]
async fn test_batch_does_not_replace_dangling_destination_symlink() {
    let tmp = tempfile::tempdir().unwrap();
    let input = tmp.path().join("in");
    let output = tmp.path().join("out");
    std::fs::create_dir(&input).unwrap();
    std::fs::create_dir(&output).unwrap();
    std::fs::write(input.join("call.wav"), b"audio").unwrap();
    let absent = tmp.path().join("absent");
    std::os::unix::fs::symlink(&absent, output.join("call.txt")).unwrap();

    let result = run_batch(
        &test_opts(&input, &output),
        ok_transcribe(),
        CancellationToken::new(),
    )
    .await;

    assert!(result.is_err());
    assert_eq!(std::fs::read_link(output.join("call.txt")).unwrap(), absent);
    assert!(!absent.exists());
    assert_eq!(std::fs::read(input.join("call.wav")).unwrap(), b"audio");
}

#[cfg(target_os = "linux")]
#[test]
fn test_cross_filesystem_archive_keeps_bytes_on_conflict_and_then_moves() {
    use std::os::unix::fs::{MetadataExt, PermissionsExt};

    let input = tempfile::tempdir().unwrap();
    let Ok(archive) = tempfile::tempdir_in("/dev/shm") else {
        eprintln!("cross-filesystem fixture unavailable");
        return;
    };
    if input.path().metadata().unwrap().dev() == archive.path().metadata().unwrap().dev() {
        eprintln!("temporary directories use the same filesystem");
        return;
    }
    let source = input.path().join("call.wav");
    let destination = archive.path().join("call.wav");
    std::fs::write(&source, b"source audio").unwrap();
    std::fs::set_permissions(&source, std::fs::Permissions::from_mode(0o640)).unwrap();
    std::fs::write(&destination, b"older archive").unwrap();
    let signature = FileSignature::read(&source).unwrap();

    assert!(storage::move_file(&source, &destination, signature).is_err());
    assert_eq!(std::fs::read(&source).unwrap(), b"source audio");
    assert_eq!(std::fs::read(&destination).unwrap(), b"older archive");
    assert_eq!(std::fs::read_dir(archive.path()).unwrap().count(), 1);

    std::fs::remove_file(&destination).unwrap();
    storage::move_file(&source, &destination, signature).unwrap();
    assert!(!source.exists());
    assert_eq!(std::fs::read(&destination).unwrap(), b"source audio");
    assert_eq!(
        destination.metadata().unwrap().permissions().mode() & 0o777,
        0o640
    );
}
