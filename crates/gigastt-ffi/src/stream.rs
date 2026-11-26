//! Streaming session lifecycle and audio delivery at the C ABI boundary.

use std::ffi::{CString, c_char};
use std::ptr;

use gigastt_core::inference::audio;

use crate::boundary::catch_ffi_panic;
use crate::handles::{
    GigasttEngine, GigasttStream, StreamSlot, get_engine, get_stream, insert_stream, take_stream,
};

/// Create a new streaming session.
///
/// Checks out a `SessionTriplet` from the engine pool and creates a fresh
/// `StreamingState`. The triplet is held for the lifetime of the stream and
/// returned to the pool by `gigastt_stream_free`.
///
/// # Safety
/// `engine` must be a live opaque handle returned by an engine constructor.
/// Returns a pointer to a `GigasttStream` on success, or `NULL` on failure.
#[allow(unsafe_code)] // Exported C ABI boundary.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn gigastt_stream_new(engine: *mut GigasttEngine) -> *mut GigasttStream {
    let Some(slot) = get_engine(engine) else {
        tracing::error!("gigastt_stream_new: engine is null or freed");
        eprintln!("gigastt_stream_new: engine is null or freed");
        return ptr::null_mut();
    };

    catch_ffi_panic("gigastt_stream_new", || {
        let guard = match slot.engine.pool.checkout_blocking() {
            Ok(g) => g,
            Err(e) => {
                tracing::error!("gigastt_stream_new: pool checkout failed: {e}");
                eprintln!("gigastt_stream_new: pool checkout failed: {e}");
                return ptr::null_mut();
            }
        };

        let reservation = guard.into_owned();
        let state = slot.engine.create_state(false);
        insert_stream(StreamSlot {
            state,
            reservation: Some(reservation),
            engine: slot.clone(),
        })
    })
    .unwrap_or(ptr::null_mut())
}

/// Process a chunk of PCM16 audio and return any partial/final segments.
///
/// # Safety
/// - `stream` must be a live opaque handle returned by `gigastt_stream_new`.
///   The engine parameter is retained for ABI compatibility and is not dereferenced.
/// - `pcm16_bytes` must be NULL or point to `len` initialized, readable bytes
///   in one allocation (little-endian mono PCM16), with `len <= isize::MAX`.
///   The bytes must not be mutated during the call.
/// - Concurrent free during this call is safe. A call after free returns `NULL`.
///
/// Returns a newly allocated JSON array string on success, or `NULL` on failure.
/// The caller **must** free the returned string with `gigastt_string_free`.
#[allow(unsafe_code)] // Exported C ABI boundary.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn gigastt_stream_process_chunk(
    engine: *mut GigasttEngine,
    stream: *mut GigasttStream,
    pcm16_bytes: *const u8,
    len: usize,
    sample_rate: u32,
) -> *mut c_char {
    let _ = engine; // engine id is optional; the stream holds its own Arc
    let Some(stream_arc) = get_stream(stream) else {
        tracing::error!("gigastt_stream_process_chunk: stream is null or freed");
        return ptr::null_mut();
    };
    if pcm16_bytes.is_null() {
        tracing::error!("gigastt_stream_process_chunk: pcm16_bytes is null");
        return ptr::null_mut();
    }

    let mut stream_guard = stream_arc.lock().unwrap_or_else(|e| e.into_inner());
    let stream_ref = &mut *stream_guard;

    // Convert PCM16 LE bytes → f32 samples.
    // SAFETY: non-null checked above; caller guarantees the readable buffer,
    // allocation bounds and shared access for the duration of this call.
    let bytes = unsafe { std::slice::from_raw_parts(pcm16_bytes, len) };
    let pcm16: Vec<i16> = bytes
        .as_chunks::<2>()
        .0
        .iter()
        .map(|c| i16::from_le_bytes(*c))
        .collect();
    let mut samples_f32: Vec<f32> = pcm16.iter().map(|&s| s as f32 / 32768.0).collect();

    // Resample to 16 kHz if needed.
    if sample_rate != 16000 {
        if let Err(e) = audio::resample_with_cache(
            samples_f32,
            audio::SampleRate(sample_rate),
            audio::SampleRate(16000),
            &mut stream_ref.state.resampler,
            &mut stream_ref.state.resample_output_buf,
        ) {
            tracing::error!("gigastt_stream_process_chunk: resample failed: {e}");
            return ptr::null_mut();
        }
        samples_f32 = std::mem::take(&mut stream_ref.state.resample_output_buf);
    }

    let engine_slot = stream_ref.engine.clone();
    let Some(reservation) = stream_ref.reservation.as_mut() else {
        tracing::error!("gigastt_stream_process_chunk: stream already freed");
        return ptr::null_mut();
    };
    let segments = match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        engine_slot
            .engine
            .process_chunk(&samples_f32, &mut stream_ref.state, reservation)
    })) {
        Ok(Ok(segs)) => segs,
        Ok(Err(e)) => {
            tracing::error!("gigastt_stream_process_chunk: inference failed: {e}");
            return ptr::null_mut();
        }
        Err(_) => {
            tracing::error!("gigastt_stream_process_chunk: panic during inference");
            return ptr::null_mut();
        }
    };

    let json = serde_json::to_string(&segments).unwrap_or_else(|_| "[]".into());
    match CString::new(json) {
        Ok(cstr) => cstr.into_raw(),
        Err(_) => ptr::null_mut(),
    }
}

/// Decode pending audio and return the final segment(s).
/// Successful flushes allow further chunks. A failed flush requires a new
/// stream; previously returned text remains incomplete.
///
/// # Safety
/// `stream` must be a pointer returned by `gigastt_stream_new`. Concurrent
/// free during this call is safe. A call after free returns `NULL`.
///
/// Returns a newly allocated JSON array string (possibly `[]`) on success,
/// or `NULL` on failure. The caller **must** free the returned string with
/// `gigastt_string_free`.
#[allow(unsafe_code)] // Exported C ABI boundary.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn gigastt_stream_flush(
    engine: *mut GigasttEngine,
    stream: *mut GigasttStream,
) -> *mut c_char {
    let _ = engine;
    let Some(stream_arc) = get_stream(stream) else {
        tracing::error!("gigastt_stream_flush: stream is null or freed");
        return ptr::null_mut();
    };

    catch_ffi_panic("gigastt_stream_flush", || {
        let mut stream_ref = stream_arc.lock().unwrap_or_else(|e| e.into_inner());
        let engine_slot = stream_ref.engine.clone();
        let StreamSlot {
            state, reservation, ..
        } = &mut *stream_ref;
        let Some(reservation) = reservation.as_mut() else {
            return ptr::null_mut();
        };
        let segments: Vec<_> = match engine_slot.engine.try_finish_stream(state, reservation) {
            Ok(segment) => segment.into_iter().collect(),
            Err(error) => {
                tracing::error!(%error, "gigastt_stream_flush failed");
                return ptr::null_mut();
            }
        };

        let json = serde_json::to_string(&segments).unwrap_or_else(|_| "[]".into());
        match CString::new(json) {
            Ok(cstr) => cstr.into_raw(),
            Err(_) => ptr::null_mut(),
        }
    })
    .unwrap_or(ptr::null_mut())
}

/// Free a streaming session and return its triplet to the pool.
///
/// # Safety
/// `stream` must be a pointer returned by `gigastt_stream_new` and not yet freed,
/// or `NULL` (in which case this is a no-op). Concurrent free during an
/// in-flight call is safe.
#[allow(unsafe_code)] // Exported C ABI boundary.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn gigastt_stream_free(stream: *mut GigasttStream) {
    let _dropped = take_stream(stream);
}
