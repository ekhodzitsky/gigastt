//! Batch and watch-folder transcription helpers for the offline CLI.
//!
//! Model-free by design: the engine is injected as a [`TranscribeFn`] closure,
//! so the traversal / rendering / retry / watch state machine is unit-tested
//! without loading ONNX models.

use anyhow::Context;
use gigastt_core::export::{ExportFormat, RenderOpts};
use gigastt_core::inference::TranscribeResult;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

/// Audio extensions accepted by the batch / watch walkers (case-insensitive).
/// Mirrors the symphonia-backed file support of `Engine::transcribe_file`.
pub const SUPPORTED_EXTENSIONS: &[&str] = &["wav", "mp3", "m4a", "ogg", "flac"];

/// Injectable transcription step: maps an audio file path to its result. The
/// production closure checks out a pool triplet and calls
/// `Engine::transcribe_file`; tests stub it out.
pub type TranscribeFn = Arc<dyn Fn(PathBuf) -> anyhow::Result<TranscribeResult> + Send + Sync>;

/// Shared knobs for one batch run.
pub struct BatchOptions {
    /// Root directory scanned (recursively) for audio files.
    pub input_dir: PathBuf,
    /// Directory the rendered outputs are written into (created if absent).
    pub output_dir: PathBuf,
    /// Formats rendered per input file (`{stem}.{ext}` per format).
    pub formats: Vec<ExportFormat>,
    /// Subtitle / Markdown rendering options.
    pub render_opts: RenderOpts,
    /// Optional directory successfully processed sources are moved into.
    pub move_to: Option<PathBuf>,
    /// Delete the source file after successful processing.
    pub delete_source: bool,
    /// Max files transcribed concurrently (bounded by the engine pool).
    pub concurrency: usize,
    /// Extra attempts after the first failure (per file).
    pub retries: u32,
}

/// Outcome counters of a [`run_batch`] run.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct BatchSummary {
    /// Files transcribed and written successfully.
    pub processed: usize,
    /// Files that failed after all retries.
    pub failed: usize,
    /// Files never started because the run was interrupted.
    pub skipped: usize,
    /// Whether the run was cut short by a shutdown request.
    pub interrupted: bool,
}

/// Outcome counters of a [`run_watch`] session.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct WatchSummary {
    /// Files transcribed and written successfully.
    pub processed: usize,
    /// Files that failed after all retries.
    pub failed: usize,
}

/// Watch-specific knobs on top of [`BatchOptions`].
pub struct WatchOptions {
    /// Processing settings shared with one-shot batches.
    pub batch: BatchOptions,
    /// Delay between directory scans.
    pub poll_interval: Duration,
    /// Consecutive identical observations (size + mtime) required before a
    /// file is considered fully written and scheduled.
    pub settle_polls: u32,
}

/// Whether `path` carries a supported audio extension (case-insensitive).
pub fn is_audio_file(path: &Path) -> bool {
    path.extension()
        .and_then(|e| e.to_str())
        .map(|e| SUPPORTED_EXTENSIONS.contains(&e.to_ascii_lowercase().as_str()))
        .unwrap_or(false)
}

/// Recursively collect supported audio files under `root`, sorted for a
/// deterministic processing order. Symlinked directories are not followed.
pub fn collect_audio_files(root: &Path) -> anyhow::Result<Vec<PathBuf>> {
    let mut out = Vec::new();
    collect_into(root, &mut out)?;
    out.sort();
    Ok(out)
}

fn collect_into(dir: &Path, out: &mut Vec<PathBuf>) -> anyhow::Result<()> {
    let entries = std::fs::read_dir(dir)
        .with_context(|| format!("failed to read directory {}", dir.display()))?;
    for entry in entries {
        let entry = entry.with_context(|| format!("failed to read entry in {}", dir.display()))?;
        let path = entry.path();
        // DirEntry::file_type does not follow symlinks, so a symlinked
        // directory can never send the walker into a loop.
        let file_type = entry.file_type()?;
        if file_type.is_dir() {
            collect_into(&path, out)?;
        } else if path.is_file() && is_audio_file(&path) {
            out.push(path);
        }
    }
    Ok(())
}

/// Output path for one input file and format: `<output_dir>/<stem>.<ext>`.
pub fn output_path_for(input: &Path, output_dir: &Path, extension: &str) -> PathBuf {
    let stem = input
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("transcript");
    output_dir.join(format!("{stem}.{extension}"))
}

/// Parse a comma-separated format list (`txt,json,md,srt,vtt`) into
/// [`ExportFormat`]s, de-duplicated, order preserved.
pub fn parse_formats(s: &str) -> Result<Vec<ExportFormat>, String> {
    let mut out: Vec<ExportFormat> = Vec::new();
    for part in s.split(',').map(str::trim).filter(|p| !p.is_empty()) {
        let fmt = part
            .parse::<ExportFormat>()
            .map_err(|_| format!("unsupported export format: {part}"))?;
        if !out.contains(&fmt) {
            out.push(fmt);
        }
    }
    if out.is_empty() {
        return Err("at least one export format is required".to_string());
    }
    Ok(out)
}

/// Move a file, falling back to copy + remove when `rename` cannot cross
/// filesystems.
fn move_file(source: &Path, dest: &Path) -> anyhow::Result<()> {
    if std::fs::rename(source, dest).is_ok() {
        return Ok(());
    }
    std::fs::copy(source, dest)
        .with_context(|| format!("failed to move {} to {}", source.display(), dest.display()))?;
    std::fs::remove_file(source)
        .with_context(|| format!("failed to remove {}", source.display()))?;
    Ok(())
}

/// Shared per-run state handed to every processing task.
struct ProcessCtx {
    output_dir: PathBuf,
    formats: Vec<ExportFormat>,
    render_opts: RenderOpts,
    move_to: Option<PathBuf>,
    delete_source: bool,
    transcribe: TranscribeFn,
}

/// One attempt: transcribe, render every format, then apply the post-success
/// source policy (move / delete). Runs on a blocking thread because both the
/// engine call and the file IO are synchronous.
async fn process_one_attempt(path: PathBuf, ctx: Arc<ProcessCtx>) -> anyhow::Result<()> {
    let task_path = path.clone();
    tokio::task::spawn_blocking(move || -> anyhow::Result<()> {
        let result = (ctx.transcribe)(task_path.clone())?;
        for fmt in &ctx.formats {
            let rendered = fmt.render(&result, &ctx.render_opts);
            let out = output_path_for(&task_path, &ctx.output_dir, fmt.extension());
            std::fs::write(&out, rendered)
                .with_context(|| format!("failed to write {}", out.display()))?;
        }
        if let Some(move_to) = &ctx.move_to {
            let file_name = task_path
                .file_name()
                .ok_or_else(|| anyhow::anyhow!("no file name: {}", task_path.display()))?;
            move_file(&task_path, &move_to.join(file_name))?;
        } else if ctx.delete_source {
            std::fs::remove_file(&task_path)
                .with_context(|| format!("failed to delete {}", task_path.display()))?;
        }
        Ok(())
    })
    .await
    .context("batch task panicked")?
}

/// Retry wrapper around [`process_one_attempt`]: `retries` extra attempts with
/// a short linear backoff between them.
async fn process_with_retries(
    path: PathBuf,
    ctx: Arc<ProcessCtx>,
    retries: u32,
) -> (PathBuf, anyhow::Result<()>) {
    let mut attempt = 0_u32;
    loop {
        match process_one_attempt(path.clone(), ctx.clone()).await {
            Ok(()) => return (path, Ok(())),
            Err(e) => {
                attempt += 1;
                if attempt > retries {
                    return (path, Err(e));
                }
                tracing::warn!(
                    attempt,
                    error = %format!("{e:#}"),
                    "retrying {}",
                    path.display()
                );
                tokio::time::sleep(Duration::from_millis(200 * u64::from(attempt))).await;
            }
        }
    }
}

/// Canonicalize the input directory and create the output / move-to
/// directories. Returns the canonical input root plus the canonical move-to
/// dir (so files already inside it can be excluded from scans).
fn prepare_dirs(opts: &BatchOptions) -> anyhow::Result<(PathBuf, Option<PathBuf>)> {
    let input_root = opts
        .input_dir
        .canonicalize()
        .with_context(|| format!("input directory not found: {}", opts.input_dir.display()))?;
    std::fs::create_dir_all(&opts.output_dir).with_context(|| {
        format!(
            "failed to create output directory {}",
            opts.output_dir.display()
        )
    })?;
    let move_to = match &opts.move_to {
        Some(dir) => {
            std::fs::create_dir_all(dir)
                .with_context(|| format!("failed to create move-to directory {}", dir.display()))?;
            Some(dir.canonicalize().unwrap_or_else(|_| dir.clone()))
        }
        None => None,
    };
    Ok((input_root, move_to))
}

/// Scan the input root, excluding anything under the move-to directory (files
/// moved there by earlier runs must not be picked up again).
fn scan_inputs(input_root: &Path, move_to: Option<&Path>) -> anyhow::Result<Vec<PathBuf>> {
    let mut files = collect_audio_files(input_root)?;
    if let Some(excluded) = move_to {
        files.retain(|p| !p.starts_with(excluded));
    }
    Ok(files)
}

/// Log a warning when two inputs would render to the same output file.
fn warn_duplicate_outputs(files: &[PathBuf], output_dir: &Path, formats: &[ExportFormat]) {
    let mut seen = std::collections::HashSet::new();
    for f in files {
        for fmt in formats {
            let out = output_path_for(f, output_dir, fmt.extension());
            if !seen.insert(out.clone()) {
                tracing::warn!(
                    "duplicate output {} — inputs with equal file stems overwrite each other",
                    out.display()
                );
            }
        }
    }
}

/// Process every supported audio file under `input_dir` once, up to
/// `concurrency` files in flight. On a shutdown request the scheduler stops
/// starting new files and waits for the in-flight ones to finish.
pub async fn run_batch(
    opts: &BatchOptions,
    transcribe: TranscribeFn,
    shutdown: tokio_util::sync::CancellationToken,
) -> anyhow::Result<BatchSummary> {
    let (input_root, move_to_canonical) = prepare_dirs(opts)?;
    let files = scan_inputs(&input_root, move_to_canonical.as_deref())?;
    if files.is_empty() {
        tracing::info!("no audio files found in {}", input_root.display());
        return Ok(BatchSummary::default());
    }
    warn_duplicate_outputs(&files, &opts.output_dir, &opts.formats);
    tracing::info!(
        total = files.len(),
        concurrency = opts.concurrency,
        "starting batch transcription"
    );

    let ctx = Arc::new(ProcessCtx {
        output_dir: opts.output_dir.clone(),
        formats: opts.formats.clone(),
        render_opts: opts.render_opts,
        move_to: opts.move_to.clone(),
        delete_source: opts.delete_source,
        transcribe,
    });

    let mut summary = BatchSummary::default();
    let mut set: tokio::task::JoinSet<(PathBuf, anyhow::Result<()>)> = tokio::task::JoinSet::new();
    let mut iter = files.into_iter();
    let concurrency = opts.concurrency.max(1);

    loop {
        // Top up the scheduler unless a shutdown was requested.
        while !shutdown.is_cancelled() && set.len() < concurrency {
            match iter.next() {
                Some(path) => {
                    set.spawn(process_with_retries(path, ctx.clone(), opts.retries));
                }
                None => break,
            }
        }
        if set.is_empty() {
            break;
        }
        let joined = tokio::select! {
            // Once cancelled, no new files are scheduled (the top-up loop above
            // is gated), so just drain whatever is still running.
            () = shutdown.cancelled() => set.join_next().await,
            r = set.join_next() => r,
        };
        let Some(joined) = joined else { break };
        let (path, result) = joined.context("batch task panicked")?;
        match result {
            Ok(()) => {
                summary.processed += 1;
                tracing::info!(
                    processed = summary.processed,
                    failed = summary.failed,
                    "done {}",
                    path.display()
                );
            }
            Err(e) => {
                summary.failed += 1;
                tracing::warn!(error = %format!("{e:#}"), "failed {}", path.display());
            }
        }
    }

    summary.skipped = iter.count();
    summary.interrupted = shutdown.is_cancelled();
    Ok(summary)
}

/// A file observation: size and modification time. Identical signatures across
/// `settle_polls` consecutive scans mean the writer is done.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct FileSignature {
    len: u64,
    mtime: Option<std::time::SystemTime>,
}

fn signature_of(path: &Path) -> Option<FileSignature> {
    let meta = std::fs::metadata(path).ok()?;
    Some(FileSignature {
        len: meta.len(),
        mtime: meta.modified().ok(),
    })
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum WatchStatus {
    /// Waiting for the signature to settle, then scheduled.
    Pending,
    /// Currently being transcribed.
    InFlight,
    /// Processed successfully for the current signature.
    Done,
    /// Failed all retries for the current signature.
    Failed,
}

#[derive(Debug, Clone, Copy)]
struct WatchEntry {
    sig: FileSignature,
    stable_polls: u32,
    status: WatchStatus,
    /// The signature changed while the file was InFlight — reprocess on completion.
    dirty: bool,
}

/// Watch `input_dir` for new or changed audio files and process them as they
/// settle. Files already present at startup are registered but not processed —
/// `transcribe-batch` covers the existing backlog. Returns after a shutdown
/// request once all in-flight files finish.
pub async fn run_watch(
    opts: &WatchOptions,
    transcribe: TranscribeFn,
    shutdown: tokio_util::sync::CancellationToken,
) -> anyhow::Result<WatchSummary> {
    let (input_root, move_to_canonical) = prepare_dirs(&opts.batch)?;
    let settle_polls = opts.settle_polls.max(1);
    let concurrency = opts.batch.concurrency.max(1);
    let ctx = Arc::new(ProcessCtx {
        output_dir: opts.batch.output_dir.clone(),
        formats: opts.batch.formats.clone(),
        render_opts: opts.batch.render_opts,
        move_to: opts.batch.move_to.clone(),
        delete_source: opts.batch.delete_source,
        transcribe,
    });

    let mut known: std::collections::HashMap<PathBuf, WatchEntry> =
        std::collections::HashMap::new();
    // Register the pre-existing backlog as Done so only genuinely new or
    // changed files are picked up.
    for f in scan_inputs(&input_root, move_to_canonical.as_deref())? {
        if let Some(sig) = signature_of(&f) {
            known.insert(
                f,
                WatchEntry {
                    sig,
                    stable_polls: 0,
                    status: WatchStatus::Done,
                    dirty: false,
                },
            );
        }
    }
    tracing::info!(
        backlog = known.len(),
        poll_ms = opts.poll_interval.as_millis() as u64,
        "watching {}",
        input_root.display()
    );

    let mut summary = WatchSummary::default();
    let mut set: tokio::task::JoinSet<(PathBuf, anyhow::Result<()>)> = tokio::task::JoinSet::new();
    let mut ticker = tokio::time::interval(opts.poll_interval);

    // Fold a finished task back into the watch state.
    fn handle_completion(
        known: &mut std::collections::HashMap<PathBuf, WatchEntry>,
        summary: &mut WatchSummary,
        joined: (PathBuf, anyhow::Result<()>),
    ) {
        let (path, result) = joined;
        let Some(entry) = known.get_mut(&path) else {
            return; // vanished from the directory mid-flight
        };
        if entry.status != WatchStatus::InFlight {
            return;
        }
        if entry.dirty {
            // Changed while processing: re-settle and process the new version.
            entry.dirty = false;
            entry.stable_polls = 0;
            entry.status = WatchStatus::Pending;
            return;
        }
        match result {
            Ok(()) => {
                summary.processed += 1;
                entry.status = WatchStatus::Done;
                tracing::info!(processed = summary.processed, "done {}", path.display());
            }
            Err(e) => {
                summary.failed += 1;
                entry.status = WatchStatus::Failed;
                tracing::warn!(error = %format!("{e:#}"), "failed {}", path.display());
            }
        }
    }

    while !shutdown.is_cancelled() {
        tokio::select! {
            () = shutdown.cancelled() => break,
            _ = ticker.tick() => {}
            // Guarded: an empty JoinSet resolves join_next immediately, which
            // would busy-spin the loop and starve the ticker.
            r = set.join_next(), if !set.is_empty() => {
                if let Some(joined) = r {
                    handle_completion(&mut known, &mut summary, joined.context("watch task panicked")?);
                }
                continue;
            }
        }

        let files = match scan_inputs(&input_root, move_to_canonical.as_deref()) {
            Ok(files) => files,
            Err(e) => {
                tracing::warn!(error = %format!("{e:#}"), "scan failed; retrying next poll");
                continue;
            }
        };
        // Forget files that disappeared, unless a task is still working on them.
        let present: std::collections::HashSet<&PathBuf> = files.iter().collect();
        known.retain(|p, e| e.status == WatchStatus::InFlight || present.contains(p));

        for f in files {
            let Some(sig) = signature_of(&f) else {
                continue;
            };
            match known.entry(f.clone()) {
                std::collections::hash_map::Entry::Vacant(v) => {
                    v.insert(WatchEntry {
                        sig,
                        stable_polls: 0,
                        status: WatchStatus::Pending,
                        dirty: false,
                    });
                }
                std::collections::hash_map::Entry::Occupied(mut o) => {
                    let entry = o.get_mut();
                    if entry.sig != sig {
                        entry.sig = sig;
                        entry.stable_polls = 0;
                        if entry.status == WatchStatus::InFlight {
                            entry.dirty = true;
                        } else {
                            entry.status = WatchStatus::Pending;
                        }
                    } else if entry.status == WatchStatus::Pending {
                        entry.stable_polls += 1;
                        if entry.stable_polls >= settle_polls && set.len() < concurrency {
                            entry.status = WatchStatus::InFlight;
                            set.spawn(process_with_retries(
                                f.clone(),
                                ctx.clone(),
                                opts.batch.retries,
                            ));
                        }
                    }
                }
            }
        }
    }

    // Graceful shutdown: no new work is scheduled from here on; wait for the
    // files already being transcribed.
    while let Some(r) = set.join_next().await {
        handle_completion(&mut known, &mut summary, r.context("watch task panicked")?);
    }
    Ok(summary)
}

#[cfg(test)]
mod tests {
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
}
