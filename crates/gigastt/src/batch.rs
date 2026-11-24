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

mod paths;
mod storage;
pub use paths::{
    SUPPORTED_EXTENSIONS, collect_audio_files, is_audio_file, output_path_for, parse_formats,
};
use storage::{Destinations, Directories};

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
async fn process_one_attempt(
    path: PathBuf,
    ctx: Arc<ProcessCtx>,
    signature: FileSignature,
) -> anyhow::Result<()> {
    tokio::task::spawn_blocking(move || -> anyhow::Result<()> {
        signature.ensure_unchanged(&path)?;
        let destinations =
            Destinations::plan(&path, &ctx.output_dir, &ctx.formats, ctx.move_to.as_deref())?;
        destinations.ensure_unused()?;
        let result = (ctx.transcribe)(path.clone())?;
        signature.ensure_unchanged(&path)?;

        let staged = ctx
            .formats
            .iter()
            .zip(&destinations.reports)
            .map(|(format, destination)| {
                let rendered = format.render(&result, &ctx.render_opts);
                storage::stage_report(destination, &rendered)
            })
            .collect::<anyhow::Result<Vec<_>>>()?;
        // Report any occupied name before publishing. Exclusive publication
        // is still required because another writer can commit after this check.
        destinations.ensure_unused()?;
        signature.ensure_unchanged(&path)?;
        for (file, destination) in staged.into_iter().zip(&destinations.reports) {
            storage::publish(file, destination)?;
        }
        if let Some(destination) = destinations.archive {
            storage::move_file(&path, &destination, signature)?;
        } else if ctx.delete_source {
            storage::remove_source(&path, signature)?;
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
    settled: Option<FileSignature>,
) -> (PathBuf, anyhow::Result<()>) {
    let signature = match settled.map_or_else(|| FileSignature::read(&path), Ok) {
        Ok(signature) => signature,
        Err(error) => return (path, Err(error)),
    };
    let mut attempt = 0_u32;
    loop {
        match process_one_attempt(path.clone(), ctx.clone(), signature).await {
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

/// Scan the input root, excluding anything under the move-to directory (files
/// moved there by earlier runs must not be picked up again).
fn scan_inputs(input_root: &Path, move_to: Option<&Path>) -> anyhow::Result<Vec<PathBuf>> {
    let mut files = collect_audio_files(input_root)?;
    if let Some(excluded) = move_to {
        files.retain(|p| !p.starts_with(excluded));
    }
    Ok(files)
}

/// Refuse a conflicting batch before any report or archive is published.
fn check_batch_destinations(
    files: &[PathBuf],
    dirs: &Directories,
    formats: &[ExportFormat],
) -> anyhow::Result<()> {
    let mut planned = std::collections::HashMap::new();
    for source in files {
        let destinations =
            Destinations::plan(source, &dirs.output, formats, dirs.archive.as_deref())?;
        for path in destinations.paths() {
            if let Some(other) = planned.insert(path.to_path_buf(), source) {
                anyhow::bail!(
                    "destination {} is shared by {} and {}",
                    path.display(),
                    other.display(),
                    source.display()
                );
            }
            storage::ensure_unused(path)
                .with_context(|| format!("destination conflict for {}", source.display()))?;
        }
    }
    Ok(())
}

/// Process every supported audio file under `input_dir` once, up to
/// `concurrency` files in flight. On a shutdown request the scheduler stops
/// starting new files and waits for the in-flight ones to finish.
pub async fn run_batch(
    opts: &BatchOptions,
    transcribe: TranscribeFn,
    shutdown: tokio_util::sync::CancellationToken,
) -> anyhow::Result<BatchSummary> {
    let dirs = Directories::prepare(opts)?;
    let files = scan_inputs(&dirs.input, dirs.archive.as_deref())?;
    if files.is_empty() {
        tracing::info!("no audio files found in {}", dirs.input.display());
        return Ok(BatchSummary::default());
    }
    check_batch_destinations(&files, &dirs, &opts.formats)?;
    tracing::info!(
        total = files.len(),
        concurrency = opts.concurrency,
        "starting batch transcription"
    );

    let ctx = Arc::new(ProcessCtx {
        output_dir: dirs.output,
        formats: opts.formats.clone(),
        render_opts: opts.render_opts,
        move_to: dirs.archive,
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
                    set.spawn(process_with_retries(path, ctx.clone(), opts.retries, None));
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
    #[cfg(unix)]
    identity: (u64, u64),
}

impl FileSignature {
    fn read(path: &Path) -> anyhow::Result<Self> {
        let metadata = std::fs::metadata(path)
            .with_context(|| format!("failed to inspect source {}", path.display()))?;
        Ok(Self {
            len: metadata.len(),
            mtime: metadata.modified().ok(),
            #[cfg(unix)]
            identity: {
                use std::os::unix::fs::MetadataExt;
                (metadata.dev(), metadata.ino())
            },
        })
    }

    fn ensure_unchanged(self, path: &Path) -> anyhow::Result<()> {
        anyhow::ensure!(
            Self::read(path)? == self,
            "source changed while processing {}",
            path.display()
        );
        Ok(())
    }
}

fn signature_of(path: &Path) -> Option<FileSignature> {
    FileSignature::read(path).ok()
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
    let dirs = Directories::prepare(&opts.batch)?;
    let settle_polls = opts.settle_polls.max(1);
    let concurrency = opts.batch.concurrency.max(1);
    let ctx = Arc::new(ProcessCtx {
        output_dir: dirs.output,
        formats: opts.batch.formats.clone(),
        render_opts: opts.batch.render_opts,
        move_to: dirs.archive,
        delete_source: opts.batch.delete_source,
        transcribe,
    });

    let mut known: std::collections::HashMap<PathBuf, WatchEntry> =
        std::collections::HashMap::new();
    // Register the pre-existing backlog as Done so only genuinely new or
    // changed files are picked up.
    for f in scan_inputs(&dirs.input, ctx.move_to.as_deref())? {
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
        dirs.input.display()
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

        let files = match scan_inputs(&dirs.input, ctx.move_to.as_deref()) {
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
                                Some(sig),
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
mod tests;
