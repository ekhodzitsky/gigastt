//! Keep reading the socket while a blocking decode owns the streaming state.

use axum::extract::ws::{Message, WebSocket};
use futures_util::{StreamExt, stream::SplitStream};
use std::sync::{Arc, atomic::AtomicBool};
use tokio_util::sync::CancellationToken;

pub(super) struct Incoming {
    pub rx: tokio::sync::mpsc::Receiver<Result<Message, axum::Error>>,
    pub closed: CancellationToken,
    reader: tokio::task::JoinHandle<()>,
}

impl Incoming {
    pub fn new(mut source: SplitStream<WebSocket>, abort: Arc<AtomicBool>) -> Self {
        // One pending message preserves backpressure and bounds extra memory
        // to two maximum-sized frames (queued + currently being sent).
        let (tx, rx) = tokio::sync::mpsc::channel(1);
        let closed = CancellationToken::new();
        let disconnected = closed.clone();
        let reader = tokio::spawn(async move {
            let _abort = super::super::file_transcribe::AbortOnDrop(abort.clone());
            while let Some(message) = source.next().await {
                let terminal = matches!(&message, Ok(Message::Close(_)) | Err(_));
                if terminal {
                    abort.store(true, std::sync::atomic::Ordering::Relaxed);
                    disconnected.cancel();
                }
                if tx.send(message).await.is_err() || terminal {
                    break;
                }
            }
            disconnected.cancel();
        });
        Self { rx, closed, reader }
    }
}

impl Drop for Incoming {
    fn drop(&mut self) {
        self.reader.abort();
    }
}
