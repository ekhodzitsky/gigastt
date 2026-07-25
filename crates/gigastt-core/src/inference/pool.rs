//! Session pool: checkout/checkin of inference session triplets.

use std::ops::{Deref, DerefMut};

use crate::runtime::session::RuntimeSession;
use crate::runtime::tensor::Tensor;

/// A set of ONNX sessions for one inference pipeline (encoder + decoder + joiner).
///
/// Moved out of the pool on checkout and returned on checkin.
/// Each triplet is independent and can run inference concurrently with others.
///
/// The RNN-T heads populate all three sessions. The encoder-only CTC heads leave
/// `decoder` / `joiner` as `None` — the CTC branch in `run_inference` decodes
/// straight from the encoder output and never touches them, so loading them would
/// only waste encoder-sized RAM.
pub struct SessionTriplet {
    pub(crate) encoder: Box<dyn RuntimeSession>,
    pub(crate) decoder: Option<Box<dyn RuntimeSession>>,
    pub(crate) joiner: Option<Box<dyn RuntimeSession>>,
    /// Reusable encoder input tensors: `[audio_signal [1, N_MELS, num_frames], length [1]]`.
    /// Resized and overwritten in `run_inference` to avoid per-call allocations.
    pub(crate) encoder_inputs: Vec<Tensor>,
}

/// Errors returned by [`Pool::checkout`].
#[derive(Debug)]
pub enum PoolError {
    /// The pool was closed (graceful shutdown). All current and future
    /// waiters resolve to this variant; the caller should respond with a
    /// 503 / `pool_closed` to the client.
    Closed,
}

impl std::fmt::Display for PoolError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PoolError::Closed => write!(f, "session pool is closed"),
        }
    }
}

impl std::error::Error for PoolError {}

/// Pool of pre-loaded items of type `T`.
///
/// `SessionPool = Pool<SessionTriplet>` is the only public instantiation
/// outside this module. Generic `T` exists so the pool semantics can be
/// unit-tested without ONNX models.
///
/// Checkout = pop from the queue, checkin = push back via the
/// [`PoolGuard`] returned by [`checkout`](Self::checkout). The pool size acts
/// as the concurrency limit — no separate semaphore needed. FIFO ordering is
/// preserved because waiters are stored in a queue and served in order.
pub struct Pool<T> {
    inner: std::sync::Arc<PoolInner<T>>,
}

struct PoolInner<T> {
    items: parking_lot::Mutex<std::collections::VecDeque<T>>,
    waiters: parking_lot::Mutex<std::collections::VecDeque<Waiter<T>>>,
    closed: std::sync::atomic::AtomicBool,
    total: usize,
}

enum Waiter<T> {
    #[cfg(feature = "async-pool")]
    Async(tokio::sync::oneshot::Sender<T>),
    Blocking(std::sync::mpsc::Sender<T>),
}

/// Public alias for the production pool: holds [`SessionTriplet`] instances.
pub type SessionPool = Pool<SessionTriplet>;

impl<T: Send> Pool<T> {
    /// Create a pool pre-filled with the given items.
    pub fn new(items: Vec<T>) -> Self {
        let total = items.len();
        Self {
            inner: std::sync::Arc::new(PoolInner {
                items: parking_lot::Mutex::new(std::collections::VecDeque::from(items)),
                waiters: parking_lot::Mutex::new(std::collections::VecDeque::new()),
                closed: std::sync::atomic::AtomicBool::new(false),
                total,
            }),
        }
    }

    /// Checkout an item from the pool. Awaits FIFO if none available.
    ///
    /// Returns [`PoolError::Closed`] if the pool was shut down via
    /// [`close`](Self::close) before an item became available.
    #[cfg(feature = "async-pool")]
    pub async fn checkout(&self) -> Result<PoolGuard<T>, PoolError> {
        // Fast path
        {
            let mut items = self.inner.items.lock();
            if self.inner.closed.load(std::sync::atomic::Ordering::SeqCst) {
                return Err(PoolError::Closed);
            }
            if let Some(item) = items.pop_front() {
                return Ok(PoolGuard::new(self.inner.clone(), item));
            }
        }

        // Slow path: register as an async waiter
        let (tx, rx) = tokio::sync::oneshot::channel();
        {
            let mut waiters = self.inner.waiters.lock();
            if self.inner.closed.load(std::sync::atomic::Ordering::SeqCst) {
                return Err(PoolError::Closed);
            }
            // Re-check items under the waiters lock to prevent the lost-wakeup
            // race: between releasing items.lock() and acquiring waiters.lock(),
            // another thread may have checked in an item and pushed it back to
            // items because there were no waiters yet.
            let mut items = self.inner.items.lock();
            if let Some(item) = items.pop_front() {
                drop(items);
                drop(waiters);
                return Ok(PoolGuard::new(self.inner.clone(), item));
            }
            waiters.push_back(Waiter::Async(tx));
        }

        match rx.await {
            Ok(item) => Ok(PoolGuard::new(self.inner.clone(), item)),
            Err(_) => Err(PoolError::Closed),
        }
    }

    /// Synchronous (blocking) checkout. Used by FFI and other synchronous callers.
    pub fn checkout_blocking(&self) -> Result<PoolGuard<T>, PoolError> {
        // Fast path
        {
            let mut items = self.inner.items.lock();
            if self.inner.closed.load(std::sync::atomic::Ordering::SeqCst) {
                return Err(PoolError::Closed);
            }
            if let Some(item) = items.pop_front() {
                return Ok(PoolGuard::new(self.inner.clone(), item));
            }
        }

        // Slow path: register as a blocking waiter
        let (tx, rx) = std::sync::mpsc::channel();
        {
            let mut waiters = self.inner.waiters.lock();
            if self.inner.closed.load(std::sync::atomic::Ordering::SeqCst) {
                return Err(PoolError::Closed);
            }
            // Same lost-wakeup guard as the async variant.
            let mut items = self.inner.items.lock();
            if let Some(item) = items.pop_front() {
                drop(items);
                drop(waiters);
                return Ok(PoolGuard::new(self.inner.clone(), item));
            }
            waiters.push_back(Waiter::Blocking(tx));
        }

        match rx.recv() {
            Ok(item) => Ok(PoolGuard::new(self.inner.clone(), item)),
            Err(_) => Err(PoolError::Closed),
        }
    }

    /// Close the pool: all current and future [`checkout`](Self::checkout)
    /// callers resolve to [`PoolError::Closed`]. Used by graceful shutdown.
    /// Idempotent.
    pub fn close(&self) {
        self.inner
            .closed
            .store(true, std::sync::atomic::Ordering::SeqCst);
        // Drain all pending waiters so their receivers get Canceled / RecvError.
        let mut waiters = self.inner.waiters.lock();
        waiters.clear();
    }

    /// Total number of items the pool was created with.
    pub fn total(&self) -> usize {
        self.inner.total
    }

    /// Number of currently available (not checked-out) items. O(1).
    pub fn available(&self) -> usize {
        let items = self.inner.items.lock();
        items.len()
    }

    /// Number of waiters currently blocked on checkout. O(1).
    pub fn waiters(&self) -> usize {
        let waiters = self.inner.waiters.lock();
        waiters.len()
    }
}

impl<T> PoolInner<T> {
    fn checkin(&self, mut item: T) {
        if self.closed.load(std::sync::atomic::Ordering::SeqCst) {
            return;
        }
        // Retry loop: if the waiter at the front of the queue was abandoned
        // (its receiver was dropped because the checkout future was cancelled
        // via timeout, select!, or abort), we must skip it and try the next
        // one, or return the item to the pool. Without this retry a cancelled
        // waiter permanently leaks a pool slot.
        loop {
            let mut waiters = self.waiters.lock();
            if let Some(waiter) = waiters.pop_front() {
                drop(waiters);
                match waiter {
                    #[cfg(feature = "async-pool")]
                    Waiter::Async(tx) => {
                        if let Err(returned_item) = tx.send(item) {
                            item = returned_item;
                            continue;
                        }
                    }
                    Waiter::Blocking(tx) => {
                        if let Err(std::sync::mpsc::SendError(returned_item)) = tx.send(item) {
                            item = returned_item;
                            continue;
                        }
                    }
                }
            } else {
                drop(waiters);
                let mut items = self.items.lock();
                items.push_back(item);
            }
            break;
        }
    }
}

/// RAII guard that auto-checks-in an item when dropped.
///
/// Returned by [`Pool::checkout`]. Deref to access the inner item.
/// On drop (including panic unwind) the item is returned to the pool;
/// if the pool was closed in the meantime the item is silently dropped.
pub struct PoolGuard<T> {
    inner: Option<std::sync::Arc<PoolInner<T>>>,
    item: Option<T>,
}

impl<T> PoolGuard<T> {
    fn new(inner: std::sync::Arc<PoolInner<T>>, item: T) -> Self {
        Self {
            inner: Some(inner),
            item: Some(item),
        }
    }

    /// Strip the lifetime so the guard can be moved into a `'static`
    /// context (e.g. `tokio::task::spawn_blocking`). Returns an
    /// [`OwnedReservation`] that owns the item and automatically returns it
    /// to the pool on drop. Call [`OwnedReservation::checkin`] to return the
    /// item explicitly before the reservation is dropped.
    pub fn into_owned(mut self) -> OwnedReservation<T> {
        let item = self
            .item
            .take()
            .unwrap_or_else(|| unreachable!("PoolGuard::into_owned called after drop"));
        let inner = self.inner.take().unwrap();
        OwnedReservation {
            inner,
            item: Some(item),
        }
    }
}

impl<T> Deref for PoolGuard<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        self.item
            .as_ref()
            .unwrap_or_else(|| unreachable!("PoolGuard accessed after item taken"))
    }
}

impl<T> DerefMut for PoolGuard<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.item
            .as_mut()
            .unwrap_or_else(|| unreachable!("PoolGuard accessed after item taken"))
    }
}

impl<T> Drop for PoolGuard<T> {
    fn drop(&mut self) {
        if let (Some(inner), Some(item)) = (self.inner.take(), self.item.take()) {
            inner.checkin(item);
        }
    }
}

/// Owned counterpart to [`PoolGuard`] for `'static` contexts (e.g.
/// `spawn_blocking`). The item is returned to the pool automatically on drop.
///
/// Call [`Self::checkin`] to return the item explicitly and invalidate the
/// guard. If the reservation is dropped without calling `checkin`, the item
/// is still returned to the pool via the [`Drop`] impl. This guarantees that
/// the pool does not leak slots when a `spawn_blocking` task panics or is
/// cancelled.
pub struct OwnedReservation<T> {
    inner: std::sync::Arc<PoolInner<T>>,
    item: Option<T>,
}

impl<T> std::ops::Deref for OwnedReservation<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        self.item
            .as_ref()
            .unwrap_or_else(|| unreachable!("OwnedReservation accessed after checkin"))
    }
}

impl<T> std::ops::DerefMut for OwnedReservation<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.item
            .as_mut()
            .unwrap_or_else(|| unreachable!("OwnedReservation accessed after checkin"))
    }
}

impl<T> OwnedReservation<T> {
    /// Return the item to the pool explicitly. After this call the reservation
    /// is empty and its [`Drop`] is a no-op.
    pub fn checkin(mut self) {
        if let Some(item) = self.item.take() {
            self.inner.checkin(item);
        }
    }
}

impl<T> Drop for OwnedReservation<T> {
    fn drop(&mut self) {
        if let Some(item) = self.item.take() {
            self.inner.checkin(item);
        }
    }
}
