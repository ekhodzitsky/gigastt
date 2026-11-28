//! Integer-id handle table for C-ABI pointers.
//!
//! Returned `*mut GigasttEngine` / `*mut GigasttStream` values are **not**
//! dereferenced: they encode a table key. `free` removes the key; an in-flight
//! call that already cloned the `Arc` keeps the object alive. A later call
//! with a freed id is a failed lookup, not a use-after-free.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

use gigastt_core::inference::{Engine, OwnedReservation, SessionTriplet, StreamingState};

/// Opaque C type. Never constructed — returned pointers are table ids.
#[allow(dead_code)]
pub struct GigasttEngine {
    _private: (),
}

/// Opaque C type. Never constructed — returned pointers are table ids.
#[allow(dead_code)]
pub struct GigasttStream {
    _private: (),
}

pub(crate) struct EngineSlot {
    pub engine: Engine,
}

pub(crate) struct StreamSlot {
    pub state: StreamingState,
    pub reservation: Option<OwnedReservation<SessionTriplet>>,
    pub engine: Arc<EngineSlot>,
}

impl Drop for StreamSlot {
    fn drop(&mut self) {
        if let Some(reservation) = self.reservation.take() {
            reservation.checkin();
        }
    }
}

static ENGINES: OnceLock<Mutex<HashMap<u64, Arc<EngineSlot>>>> = OnceLock::new();
static STREAMS: OnceLock<Mutex<HashMap<u64, Arc<Mutex<StreamSlot>>>>> = OnceLock::new();
static NEXT_ID: AtomicU64 = AtomicU64::new(1);

fn engines() -> &'static Mutex<HashMap<u64, Arc<EngineSlot>>> {
    ENGINES.get_or_init(|| Mutex::new(HashMap::new()))
}

fn streams() -> &'static Mutex<HashMap<u64, Arc<Mutex<StreamSlot>>>> {
    STREAMS.get_or_init(|| Mutex::new(HashMap::new()))
}

fn lock_map<T>(m: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

fn next_id() -> u64 {
    NEXT_ID.fetch_add(1, Ordering::Relaxed)
}

fn as_ptr<T>(id: u64) -> *mut T {
    id as *mut T
}

fn as_id<T>(ptr: *mut T) -> Option<u64> {
    if ptr.is_null() {
        None
    } else {
        Some(ptr as u64)
    }
}

pub(crate) fn insert_engine(engine: Engine) -> *mut GigasttEngine {
    let id = next_id();
    lock_map(engines()).insert(id, Arc::new(EngineSlot { engine }));
    as_ptr(id)
}

pub(crate) fn get_engine(ptr: *mut GigasttEngine) -> Option<Arc<EngineSlot>> {
    let id = as_id(ptr)?;
    lock_map(engines()).get(&id).cloned()
}

pub(crate) fn take_engine(ptr: *mut GigasttEngine) -> Option<Arc<EngineSlot>> {
    let id = as_id(ptr)?;
    lock_map(engines()).remove(&id)
}

pub(crate) fn insert_stream(slot: StreamSlot) -> *mut GigasttStream {
    let id = next_id();
    lock_map(streams()).insert(id, Arc::new(Mutex::new(slot)));
    as_ptr(id)
}

pub(crate) fn get_stream(ptr: *mut GigasttStream) -> Option<Arc<Mutex<StreamSlot>>> {
    let id = as_id(ptr)?;
    lock_map(streams()).get(&id).cloned()
}

pub(crate) fn take_stream(ptr: *mut GigasttStream) -> Option<Arc<Mutex<StreamSlot>>> {
    let id = as_id(ptr)?;
    lock_map(streams()).remove(&id)
}

/// Opaque v4 C type. Never constructed; its pointers encode table IDs.
#[allow(dead_code)]
pub struct GigasttV4Engine {
    _private: (),
}

#[cfg(all(
    feature = "v4-mkl",
    target_os = "linux",
    target_arch = "x86_64",
    target_env = "gnu"
))]
pub(crate) struct V4EngineSlot {
    pub engine: gigastt_core::inference::v4::V4Engine,
    pub style: gigastt_core::model::TranscriptStyle,
}

#[cfg(all(
    feature = "v4-mkl",
    target_os = "linux",
    target_arch = "x86_64",
    target_env = "gnu"
))]
fn v4_engines() -> &'static Mutex<HashMap<u64, Arc<V4EngineSlot>>> {
    static V4_ENGINES: OnceLock<Mutex<HashMap<u64, Arc<V4EngineSlot>>>> = OnceLock::new();
    V4_ENGINES.get_or_init(|| Mutex::new(HashMap::new()))
}

#[cfg(all(
    feature = "v4-mkl",
    target_os = "linux",
    target_arch = "x86_64",
    target_env = "gnu"
))]
pub(crate) fn insert_v4_engine(
    engine: gigastt_core::inference::v4::V4Engine,
    style: gigastt_core::model::TranscriptStyle,
) -> *mut GigasttV4Engine {
    let id = next_id();
    lock_map(v4_engines()).insert(id, Arc::new(V4EngineSlot { engine, style }));
    as_ptr(id)
}

#[cfg(all(
    feature = "v4-mkl",
    target_os = "linux",
    target_arch = "x86_64",
    target_env = "gnu"
))]
pub(crate) fn get_v4_engine(ptr: *mut GigasttV4Engine) -> Option<Arc<V4EngineSlot>> {
    lock_map(v4_engines()).get(&as_id(ptr)?).cloned()
}

#[cfg(all(
    feature = "v4-mkl",
    target_os = "linux",
    target_arch = "x86_64",
    target_env = "gnu"
))]
pub(crate) fn take_v4_engine(ptr: *mut GigasttV4Engine) -> Option<Arc<V4EngineSlot>> {
    lock_map(v4_engines()).remove(&as_id(ptr)?)
}
