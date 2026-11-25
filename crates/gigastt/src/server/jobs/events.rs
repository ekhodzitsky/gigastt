//! Bounded delivery of asynchronous job progress and terminal events.

use serde::Serialize;
use tokio::sync::mpsc::{self, error::TrySendError};

/// Server-sent event emitted by `GET /v1/jobs/{id}/events`.
#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum JobEvent {
    /// Progress estimate while processing.
    Progress {
        /// Approximate fraction complete, 0–100.
        percent: u32,
        /// Seconds of audio considered processed so far.
        processed_seconds: f64,
    },
    /// Job completed successfully.
    Done,
    /// Job failed.
    Failed {
        /// Sanitized error message (no paths or model internals).
        error: String,
    },
    /// Job was cancelled.
    Cancelled,
}

impl JobEvent {
    /// Whether this event ends the stream.
    pub fn is_terminal(&self) -> bool {
        matches!(
            self,
            JobEvent::Done | JobEvent::Failed { .. } | JobEvent::Cancelled
        )
    }
}

/// Maximum buffered progress events for each reader. One additional channel
/// slot is reserved for its terminal event.
pub const MAX_JOB_EVENT_PROGRESS: usize = 8;

/// Sender for a single job event subscription. Only progress notifications
/// can be shed when the reader is slow; store-owned, serialized transitions
/// reserve room for the terminal event.
#[derive(Debug, Clone)]
pub struct JobEventSender(mpsc::Sender<JobEvent>);

/// Create a bounded subscription channel with space for a terminal event.
pub fn job_event_channel() -> (JobEventSender, mpsc::Receiver<JobEvent>) {
    let (tx, rx) = mpsc::channel(MAX_JOB_EVENT_PROGRESS + 1);
    (JobEventSender(tx), rx)
}

impl JobEventSender {
    /// Returns whether the receiver is no longer listening.
    pub fn is_closed(&self) -> bool {
        self.0.is_closed()
    }

    /// Publish without waiting for the reader. Returns false if the receiver
    /// has disconnected (or the terminal slot invariant was violated).
    /// A full progress backlog drops new progress and returns true.
    ///
    /// Serialize publications with the job's lifecycle transitions; terminal
    /// events must be published only once and must end the subscription.
    pub fn publish(&self, event: JobEvent) -> bool {
        if event.is_terminal() {
            return match self.0.try_send(event) {
                Ok(()) => true,
                Err(TrySendError::Closed(_)) => false,
                Err(TrySendError::Full(_)) => {
                    tracing::error!("Job event terminal slot unexpectedly full");
                    false
                }
            };
        }

        // Acquire two slots without waiting, use one, and immediately release
        // the other. Progress can never consume the last available slot.
        match self.0.try_reserve_many(2) {
            Ok(mut permits) => {
                if let Some(permit) = permits.next() {
                    permit.send(event);
                }
                true
            }
            Err(TrySendError::Full(())) => true,
            Err(TrySendError::Closed(())) => false,
        }
    }
}
