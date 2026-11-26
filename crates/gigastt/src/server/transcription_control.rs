//! Cooperative cancellation and no-progress supervision for transcription tasks.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

/// Sets a shared abort flag when dropped. Held in the REST handler's async
/// scope so that a client disconnect — which drops the handler future before it
/// returns — flips the flag, and the detached blocking decode observes it at the
/// next decode step and releases its pooled triplet. On the normal return
/// path the flag is set after the result is already in hand, so it is a no-op.
pub(crate) struct AbortOnDrop(pub Arc<AtomicBool>);

impl Drop for AbortOnDrop {
    fn drop(&mut self) {
        self.0.store(true, Ordering::Relaxed);
    }
}

/// Link a blocking SSE producer to shutdown and receiver disconnect. Dropping
/// the returned guard after the producer finishes also retires the watcher.
pub(crate) fn stream_abort<T: Send + 'static>(
    tx: &tokio::sync::mpsc::Sender<T>,
    shutdown: tokio_util::sync::CancellationToken,
    tracker: &tokio_util::task::TaskTracker,
) -> (Arc<AtomicBool>, tokio_util::sync::DropGuard) {
    let abort = Arc::new(AtomicBool::new(false));
    let flag = abort.clone();
    let finished = tokio_util::sync::CancellationToken::new();
    let complete = finished.clone();
    let tx = tx.clone();
    tracker.spawn(async move {
        tokio::select! {
            biased;
            _ = complete.cancelled() => return,
            _ = shutdown.cancelled() => {},
            _ = tx.closed() => {},
        }
        flag.store(true, Ordering::Relaxed);
    });
    (abort, finished.drop_guard())
}

/// Outcome of awaiting a blocking transcription under the no-progress watchdog.
pub(crate) enum WatchdogOutcome<T> {
    /// The blocking task finished; carries the raw join result.
    Joined(Result<T, tokio::task::JoinError>),
    /// No window completed within the inference timeout. `abort` was set so the
    /// run cancels at its next decode step and frees the triplet.
    TimedOut,
}

/// Await a blocking transcription `handle`, redefining `inference_timeout_secs`
/// from a total wall-clock cap into a **no-progress watchdog**: the deadline
/// resets every time `progress` advances (i.e. a window completes), so a long
/// file that keeps making progress never trips, while a stalled run exhausts
/// its no-progress budget. `timeout_secs == 0` disables the
/// watchdog. Progress is sampled every 100 ms: with prompt runtime scheduling,
/// a stall is detected between `timeout` and `timeout + 100 ms` after the last
/// increment. Atomic counters carry no write timestamp, so the deadline uses
/// the observation time. A fired `shutdown` also flips `abort`, so SIGTERM
/// cancels the run at its next window instead of draining the whole file.
pub(crate) async fn await_transcription_watchdog<T>(
    mut handle: tokio::task::JoinHandle<T>,
    progress: &AtomicU64,
    abort: &AtomicBool,
    timeout_secs: u64,
    shutdown: &tokio_util::sync::CancellationToken,
) -> WatchdogOutcome<T> {
    let mut shutdown_fired = false;

    if timeout_secs == 0 {
        // No watchdog: still link shutdown -> abort so a drain cancels promptly.
        loop {
            tokio::select! {
                joined = &mut handle => return WatchdogOutcome::Joined(joined),
                _ = shutdown.cancelled(), if !shutdown_fired => {
                    shutdown_fired = true;
                    abort.store(true, Ordering::Relaxed);
                }
            }
        }
    }

    let timeout = std::time::Duration::from_secs(timeout_secs);
    let mut last_progress = progress.load(Ordering::Relaxed);
    let mut deadline = tokio::time::Instant::now() + timeout;
    let poll_interval = std::time::Duration::from_millis(100);
    loop {
        let next_poll = deadline.min(tokio::time::Instant::now() + poll_interval);
        tokio::select! {
            joined = &mut handle => return WatchdogOutcome::Joined(joined),
            _ = shutdown.cancelled(), if !shutdown_fired => {
                shutdown_fired = true;
                abort.store(true, Ordering::Relaxed);
            }
            _ = tokio::time::sleep_until(next_poll) => {
                let cur = progress.load(Ordering::Relaxed);
                if cur > last_progress {
                    // A window completed since the last check: reset the deadline.
                    last_progress = cur;
                    deadline = tokio::time::Instant::now() + timeout;
                } else if tokio::time::Instant::now() >= deadline {
                    abort.store(true, Ordering::Relaxed);
                    return WatchdogOutcome::TimedOut;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test(start_paused = true)]
    async fn test_stream_abort_tracks_shutdown_disconnect_and_completion() {
        enum End {
            Shutdown,
            Disconnect,
            Complete,
        }

        for (end, expected_abort) in [
            (End::Shutdown, true),
            (End::Disconnect, true),
            (End::Complete, false),
        ] {
            let (tx, rx) = tokio::sync::mpsc::channel::<()>(1);
            let shutdown = tokio_util::sync::CancellationToken::new();
            let tracker = tokio_util::task::TaskTracker::new();
            let (abort, guard) = stream_abort(&tx, shutdown.clone(), &tracker);
            tracker.close();
            match end {
                End::Shutdown => shutdown.cancel(),
                End::Disconnect => drop(rx),
                End::Complete => drop(guard),
            }
            tokio::time::timeout(std::time::Duration::from_secs(1), tracker.wait())
                .await
                .expect("watcher must retire after stream ends");
            assert_eq!(abort.load(Ordering::Relaxed), expected_abort);
        }
    }

    #[test]
    fn test_abort_on_drop_sets_flag() {
        let flag = Arc::new(AtomicBool::new(false));
        {
            let _guard = AbortOnDrop(flag.clone());
            assert!(!flag.load(Ordering::Relaxed), "flag is clear while held");
        }
        assert!(
            flag.load(Ordering::Relaxed),
            "drop must flip the abort flag"
        );
    }

    #[tokio::test(start_paused = true)]
    async fn test_watchdog_times_out_and_aborts_without_progress() {
        // A run that never advances `progress` past the timeout must trip and
        // flip `abort` (so the real decode would cancel at its next window).
        let progress = Arc::new(AtomicU64::new(0));
        let abort = Arc::new(AtomicBool::new(false));
        let shutdown = tokio_util::sync::CancellationToken::new();
        let handle = tokio::task::spawn(async {
            tokio::time::sleep(std::time::Duration::from_secs(3600)).await;
            0u32
        });
        let start = tokio::time::Instant::now();
        let outcome = await_transcription_watchdog(handle, &progress, &abort, 1, &shutdown).await;
        assert!(matches!(outcome, WatchdogOutcome::TimedOut));
        assert_eq!(start.elapsed(), std::time::Duration::from_secs(1));
        assert!(abort.load(Ordering::Relaxed), "a trip must flip abort");
    }

    #[tokio::test(start_paused = true)]
    async fn test_watchdog_resets_on_progress_and_joins() {
        // A run that reports a window every 0.5 s never exhausts the 1 s
        // no-progress budget, so it joins normally and is not aborted.
        let progress = Arc::new(AtomicU64::new(0));
        let abort = Arc::new(AtomicBool::new(false));
        let shutdown = tokio_util::sync::CancellationToken::new();
        let handle = tokio::task::spawn({
            let progress = progress.clone();
            async move {
                for window in 1..=4u64 {
                    tokio::time::sleep(std::time::Duration::from_millis(500)).await;
                    progress.store(window * 16_000, Ordering::Relaxed);
                }
                42u32
            }
        });
        let outcome = await_transcription_watchdog(handle, &progress, &abort, 1, &shutdown).await;
        match outcome {
            WatchdogOutcome::Joined(Ok(v)) => assert_eq!(v, 42),
            other => panic!("expected Joined(Ok(42)), got {}", label(&other)),
        }
        assert!(
            !abort.load(Ordering::Relaxed),
            "a steadily-progressing run must not be aborted"
        );
    }

    #[tokio::test(start_paused = true)]
    async fn test_watchdog_disabled_never_times_out() {
        // `timeout_secs == 0` disables the watchdog: even a long, silent run
        // joins without tripping and without abort.
        let progress = Arc::new(AtomicU64::new(0));
        let abort = Arc::new(AtomicBool::new(false));
        let shutdown = tokio_util::sync::CancellationToken::new();
        let handle = tokio::task::spawn(async {
            tokio::time::sleep(std::time::Duration::from_secs(3600)).await;
            5u32
        });
        let outcome = await_transcription_watchdog(handle, &progress, &abort, 0, &shutdown).await;
        assert!(matches!(outcome, WatchdogOutcome::Joined(Ok(5))));
        assert!(!abort.load(Ordering::Relaxed));
    }

    #[tokio::test(start_paused = true)]
    async fn test_watchdog_shutdown_flips_abort() {
        // A fired shutdown must flip `abort` so the run cancels at its next
        // window; the watchdog then joins the (now-cancelled) task.
        let progress = Arc::new(AtomicU64::new(0));
        let abort = Arc::new(AtomicBool::new(false));
        let shutdown = tokio_util::sync::CancellationToken::new();
        let handle = tokio::task::spawn({
            let abort = abort.clone();
            async move {
                loop {
                    if abort.load(Ordering::Relaxed) {
                        break 9u32;
                    }
                    tokio::time::sleep(std::time::Duration::from_millis(50)).await;
                }
            }
        });
        shutdown.cancel();
        // A generous timeout that must not be what ends the run.
        let outcome = await_transcription_watchdog(handle, &progress, &abort, 600, &shutdown).await;
        assert!(matches!(outcome, WatchdogOutcome::Joined(Ok(9))));
        assert!(abort.load(Ordering::Relaxed), "shutdown must flip abort");
    }

    #[tokio::test(start_paused = true)]
    async fn test_watchdog_stall_detection_is_bounded_after_early_progress() {
        let progress = Arc::new(AtomicU64::new(0));
        let abort = AtomicBool::new(false);
        let shutdown = tokio_util::sync::CancellationToken::new();
        let handle = tokio::spawn({
            let progress = progress.clone();
            async move {
                tokio::time::sleep(std::time::Duration::from_millis(10)).await;
                progress.store(1, Ordering::Relaxed);
                tokio::time::sleep(std::time::Duration::from_secs(3600)).await;
            }
        });
        let start = tokio::time::Instant::now();
        let outcome = await_transcription_watchdog(handle, &progress, &abort, 1, &shutdown).await;
        assert!(matches!(outcome, WatchdogOutcome::TimedOut));
        let elapsed = start.elapsed();
        assert!(
            elapsed >= std::time::Duration::from_millis(1010),
            "{elapsed:?}"
        );
        assert!(
            elapsed <= std::time::Duration::from_millis(1110),
            "{elapsed:?}"
        );
        assert!(abort.load(Ordering::Relaxed));
    }

    #[tokio::test(start_paused = true)]
    async fn test_watchdog_channel_transition_preserves_liveness() {
        let progress = Arc::new(AtomicU64::new(0));
        let abort = AtomicBool::new(false);
        let shutdown = tokio_util::sync::CancellationToken::new();
        let handle = tokio::spawn({
            let progress = progress.clone();
            async move {
                for channel in 0..2 {
                    for local_samples in 1..=3 {
                        tokio::time::sleep(std::time::Duration::from_millis(500)).await;
                        progress.store(channel * 3 + local_samples, Ordering::Relaxed);
                    }
                }
            }
        });
        let start = tokio::time::Instant::now();
        let result = await_transcription_watchdog(handle, &progress, &abort, 1, &shutdown).await;
        assert!(matches!(result, WatchdogOutcome::Joined(Ok(()))));
        assert_eq!(start.elapsed(), std::time::Duration::from_secs(3));
        assert!(!abort.load(Ordering::Relaxed));
    }

    #[tokio::test(start_paused = true)]
    async fn test_watchdog_disabled_still_cancels_on_shutdown() {
        let progress = AtomicU64::new(0);
        let abort = Arc::new(AtomicBool::new(false));
        let shutdown = tokio_util::sync::CancellationToken::new();
        let handle = tokio::spawn({
            let abort = abort.clone();
            let shutdown = shutdown.clone();
            async move {
                tokio::time::sleep(std::time::Duration::from_secs(5)).await;
                shutdown.cancel();
                while !abort.load(Ordering::Relaxed) {
                    tokio::task::yield_now().await;
                }
            }
        });
        let start = tokio::time::Instant::now();
        let result = await_transcription_watchdog(handle, &progress, &abort, 0, &shutdown).await;
        assert!(matches!(result, WatchdogOutcome::Joined(Ok(()))));
        assert_eq!(start.elapsed(), std::time::Duration::from_secs(5));
        assert!(abort.load(Ordering::Relaxed));
    }

    /// Human-readable tag for a `WatchdogOutcome` in test panics.
    fn label<T>(outcome: &WatchdogOutcome<T>) -> &'static str {
        match outcome {
            WatchdogOutcome::Joined(Ok(_)) => "Joined(Ok)",
            WatchdogOutcome::Joined(Err(_)) => "Joined(Err)",
            WatchdogOutcome::TimedOut => "TimedOut",
        }
    }
}
