//! Version-pinned UniFFI binding generator. Run, e.g.:
//!
//!   cargo run -p gigastt-uniffi --bin uniffi-bindgen -- generate \
//!     --library target/debug/libgigastt_uniffi.dylib \
//!     --language python --out-dir bindings/python
fn main() {
    uniffi::uniffi_bindgen_main()
}
