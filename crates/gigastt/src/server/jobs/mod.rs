//! Asynchronous job queue for long-file and batch transcription.
//!
//! The queue is intentionally decoupled from the HTTP surface via the
//! [`JobStore`] trait so a persistent backend (e.g. SQLite) can plug in
//! later without touching the handlers.

mod events;
mod executor;
mod queue;
mod store;

pub use events::{JobEvent, JobEventSender, MAX_JOB_EVENT_PROGRESS, job_event_channel};
pub use executor::RealJobExecutor;
pub use queue::{JobExecution, JobQueue};
pub use store::{InMemoryJobStore, Job, JobStatus, JobStatusResponse, JobStore, JobStoreFuture};

#[cfg(test)]
pub(crate) use queue::broadcast_event;
#[cfg(test)]
pub(crate) use store::job_status_response;
pub use store::{JobStoreFull, JobTransition, TransitionOutcome};

#[cfg(test)]
mod tests;
