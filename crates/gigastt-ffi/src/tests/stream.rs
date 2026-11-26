use std::ffi::CStr;
use std::ptr;

use crate::handles::{get_stream, insert_engine};
use crate::*;

use super::shared_test_engine;

#[test]
fn test_stream_new_null_engine() {
    let stream = unsafe { gigastt_stream_new(ptr::null_mut()) };
    assert!(stream.is_null());
}

#[test]
fn test_stream_new_unknown_handle_is_null() {
    let bogus = 0xFFFF_FFFFu64 as *mut GigasttEngine;
    let stream = unsafe { gigastt_stream_new(bogus) };
    assert!(stream.is_null());
}

#[test]
fn test_stream_process_chunk_null_args() {
    let r = unsafe {
        gigastt_stream_process_chunk(ptr::null_mut(), ptr::null_mut(), ptr::null(), 0, 16000)
    };
    assert!(r.is_null());
}

#[test]
fn test_stream_flush_null_args() {
    let r = unsafe { gigastt_stream_flush(ptr::null_mut(), ptr::null_mut()) };
    assert!(r.is_null());
}

#[test]
fn test_stream_free_null() {
    // Should be a no-op, not a crash.
    unsafe { gigastt_stream_free(ptr::null_mut()) };
}

#[test]
#[ignore = "requires model"]
fn test_stream_new_success() {
    let engine = shared_test_engine();
    let stream = unsafe { gigastt_stream_new(engine) };
    assert!(!stream.is_null());
    unsafe { gigastt_stream_free(stream) };
}

#[test]
#[ignore = "requires model"]
fn test_stream_process_chunk_success() {
    let engine = shared_test_engine();
    let stream = unsafe { gigastt_stream_new(engine) };
    assert!(!stream.is_null());
    // 0.2 s of silence at 16 kHz mono PCM16 = 6400 bytes
    let pcm16: Vec<u8> = vec![0u8; 3200 * 2];
    let r =
        unsafe { gigastt_stream_process_chunk(engine, stream, pcm16.as_ptr(), pcm16.len(), 16000) };
    assert!(!r.is_null());
    let s = unsafe { CStr::from_ptr(r) }.to_str().unwrap();
    assert!(s.starts_with('['));
    unsafe { gigastt_string_free(r) };
    unsafe { gigastt_stream_free(stream) };
}

#[test]
#[ignore = "requires model"]
fn test_stream_process_chunk_resample() {
    let engine = shared_test_engine();
    let stream = unsafe { gigastt_stream_new(engine) };
    assert!(!stream.is_null());
    // 0.2 s of silence at 48 kHz mono PCM16 = 19200 bytes
    let pcm16: Vec<u8> = vec![0u8; 9600 * 2];
    let r =
        unsafe { gigastt_stream_process_chunk(engine, stream, pcm16.as_ptr(), pcm16.len(), 48000) };
    assert!(!r.is_null());
    unsafe { gigastt_string_free(r) };
    unsafe { gigastt_stream_free(stream) };
}

#[test]
#[ignore = "requires model"]
fn test_stream_flush_success() {
    let engine = shared_test_engine();
    let stream = unsafe { gigastt_stream_new(engine) };
    assert!(!stream.is_null());
    let pcm16: Vec<u8> = vec![0u8; 3200 * 2];
    let r =
        unsafe { gigastt_stream_process_chunk(engine, stream, pcm16.as_ptr(), pcm16.len(), 16000) };
    assert!(!r.is_null());
    unsafe { gigastt_string_free(r) };
    let r = unsafe { gigastt_stream_flush(engine, stream) };
    assert!(!r.is_null());
    let s = unsafe { CStr::from_ptr(r) }.to_str().unwrap();
    assert!(s.starts_with('['));
    unsafe { gigastt_string_free(r) };
    unsafe { gigastt_stream_free(stream) };
}

/// With a live engine, a NULL stream pointer must be rejected by the
/// stream-null branch of `process_chunk`. Reaching that branch requires a
/// non-null engine, so the test is model-gated.
#[test]
#[ignore = "requires model"]
fn test_stream_process_chunk_null_stream_with_engine() {
    let engine = shared_test_engine();
    let pcm16: Vec<u8> = vec![0u8; 16];
    let r = unsafe {
        gigastt_stream_process_chunk(engine, ptr::null_mut(), pcm16.as_ptr(), pcm16.len(), 16000)
    };
    assert!(r.is_null());
}

/// With a live engine and stream, a NULL PCM buffer must be rejected by the
/// pcm16-null branch of `process_chunk`.
#[test]
#[ignore = "requires model"]
fn test_stream_process_chunk_null_pcm_with_engine() {
    let engine = shared_test_engine();
    let stream = unsafe { gigastt_stream_new(engine) };
    assert!(!stream.is_null());
    let r = unsafe { gigastt_stream_process_chunk(engine, stream, ptr::null(), 0, 16000) };
    assert!(r.is_null());
    unsafe { gigastt_stream_free(stream) };
}

/// A flush against a NULL stream (with a live engine) hits the stream-null
/// branch and returns NULL.
#[test]
#[ignore = "requires model"]
fn test_stream_flush_null_stream_with_engine() {
    let engine = shared_test_engine();
    let r = unsafe { gigastt_stream_flush(engine, ptr::null_mut()) };
    assert!(r.is_null());
}

#[test]
fn test_stream_flush_tail_failure_returns_null_and_preserves_state() {
    for joiner in [false, true] {
        for panic in [false, true] {
            let tmp = tempfile::tempdir().unwrap();
            gigastt_core::test_support::write_rnnt_layout(tmp.path()).unwrap();
            let factory = gigastt_core::test_support::FailingStreamFactory::new(joiner, panic);
            let core = gigastt_core::test_support::load_rnnt_engine_with_factory(
                tmp.path(),
                1,
                Box::new(factory.clone()),
            )
            .unwrap();
            factory.arm(1);
            let engine = insert_engine(core);
            let stream = unsafe { gigastt_stream_new(engine) };
            let pcm = [0u8; 3200];
            let result = unsafe {
                gigastt_stream_process_chunk(engine, stream, pcm.as_ptr(), pcm.len(), 16000)
            };
            assert!(!result.is_null());
            unsafe { gigastt_string_free(result) };
            assert!(unsafe { gigastt_stream_flush(engine, stream) }.is_null());
            assert!(unsafe { gigastt_stream_flush(engine, stream) }.is_null());
            assert!(
                get_stream(stream)
                    .unwrap()
                    .lock()
                    .unwrap_or_else(|e| e.into_inner())
                    .state
                    .is_failed()
            );
            unsafe {
                gigastt_stream_free(stream);
                gigastt_engine_free(engine);
            }
        }
    }
}
