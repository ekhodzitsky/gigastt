# Flexible Sample Rate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Accept audio at multiple sample rates (8kHz, 16kHz, 24kHz, 44.1kHz, 48kHz) via WebSocket, with server-side resampling to 16kHz using rubato.

**Architecture:** New `ClientMessage::Configure` message lets clients declare sample rate before streaming audio. Server creates a rubato `SincFixedIn` resampler per-connection. Default remains 48kHz for backward compatibility. `ServerMessage::Ready` extended with `supported_rates` field.

**Tech Stack:** rubato (already in Cargo.toml), serde, tokio-tungstenite (existing)

---

### Task 1: Extend protocol — Configure message and Ready extension

**Files:**
- Modify: `src/protocol/mod.rs:54-61` (ClientMessage enum)
- Modify: `src/protocol/mod.rs:12-21` (ServerMessage::Ready)
- Test: `src/protocol/mod.rs` (inline tests module)

- [ ] **Step 1: Write failing tests for Configure message**

Add to the `#[cfg(test)] mod tests` block in `src/protocol/mod.rs`:

```rust
#[test]
fn test_client_message_configure_deserialize() {
    let json = r#"{"type":"configure","sample_rate":8000}"#;
    let msg: ClientMessage = serde_json::from_str(json).unwrap();
    match msg {
        ClientMessage::Configure { sample_rate } => assert_eq!(sample_rate, 8000),
        _ => panic!("Expected Configure"),
    }
}

#[test]
fn test_ready_supported_rates_serialization() {
    let msg = ServerMessage::Ready {
        model: "test".into(),
        sample_rate: 48000,
        version: "1.0".into(),
        supported_rates: vec![8000, 16000, 24000, 44100, 48000],
    };
    let json = serde_json::to_string(&msg).unwrap();
    let v: serde_json::Value = serde_json::from_str(&json).unwrap();
    assert_eq!(v["supported_rates"].as_array().unwrap().len(), 5);
}

#[test]
fn test_ready_empty_supported_rates_omitted() {
    let msg = ServerMessage::Ready {
        model: "test".into(),
        sample_rate: 48000,
        version: "1.0".into(),
        supported_rates: vec![],
    };
    let json = serde_json::to_string(&msg).unwrap();
    let v: serde_json::Value = serde_json::from_str(&json).unwrap();
    assert!(v.get("supported_rates").is_none());
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test --lib test_client_message_configure_deserialize test_ready_supported_rates_serialization test_ready_empty_supported_rates_omitted --features coreml 2>&1`
Expected: compilation errors — `Configure` variant and `supported_rates` field don't exist yet.

- [ ] **Step 3: Implement Configure variant and Ready extension**

In `src/protocol/mod.rs`, add `Configure` to `ClientMessage`:

```rust
#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
#[non_exhaustive]
pub enum ClientMessage {
    /// Request server to stop and finalize.
    Stop,
    /// Configure session parameters (must be sent before first audio frame).
    Configure {
        /// Audio sample rate in Hz (e.g., 8000, 16000, 24000, 44100, 48000).
        sample_rate: u32,
    },
}
```

Extend `ServerMessage::Ready`:

```rust
Ready {
    model: String,
    sample_rate: u32,
    version: String,
    /// Supported input sample rates (omitted from JSON if empty for backward compat).
    #[serde(skip_serializing_if = "Vec::is_empty")]
    supported_rates: Vec<u32>,
},
```

- [ ] **Step 4: Fix existing tests that construct Ready without supported_rates**

Update all existing `ServerMessage::Ready { .. }` in tests to include `supported_rates: vec![]`. There are 2 instances:
- `test_ready_serialization_includes_version` (line 74)
- `test_ready_empty_supported_rates_omitted` (already has it from step 1)

Also update `test_client_message_configure_deserialize` if it already exists with different content (check for duplicate test name from protocol/mod.rs line 125-130 — that test tests a different Configure variant, rename the old one or remove if redundant).

- [ ] **Step 5: Run all protocol tests**

Run: `cargo test --lib protocol --features coreml 2>&1`
Expected: all protocol tests pass.

- [ ] **Step 6: Commit**

```bash
git add src/protocol/mod.rs
git commit -m "feat(protocol): add Configure message and supported_rates to Ready"
```

---

### Task 2: Replace linear interpolation with rubato resampler

**Files:**
- Modify: `src/inference/audio.rs:123-158` (replace `resample()`)
- Test: `src/inference/audio.rs` (inline tests module)

- [ ] **Step 1: Write failing test for rubato quality**

Add to tests in `src/inference/audio.rs`:

```rust
#[test]
fn test_resample_rubato_8k_to_16k() {
    // 1 second of 440Hz sine at 8kHz
    let input: Vec<f32> = (0..8000)
        .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 8000.0).sin())
        .collect();
    let output = resample(&input, 8000, 16000);
    assert_eq!(output.len(), 16000);
    // Check that signal energy is preserved (within 5%)
    let in_energy: f32 = input.iter().map(|x| x * x).sum::<f32>() / input.len() as f32;
    let out_energy: f32 = output.iter().map(|x| x * x).sum::<f32>() / output.len() as f32;
    assert!((in_energy - out_energy).abs() / in_energy < 0.05, "Energy not preserved: in={in_energy} out={out_energy}");
}

#[test]
fn test_resample_rubato_24k_to_16k() {
    let input: Vec<f32> = (0..24000)
        .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 24000.0).sin())
        .collect();
    let output = resample(&input, 24000, 16000);
    assert_eq!(output.len(), 16000);
}

#[test]
fn test_resample_rubato_44100_to_16000() {
    let input: Vec<f32> = (0..44100)
        .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 44100.0).sin())
        .collect();
    let output = resample(&input, 44100, 16000);
    // 44100/16000 is not integer ratio — rubato handles this
    let expected_len = (44100.0 * 16000.0 / 44100.0) as usize;
    assert!((output.len() as i64 - expected_len as i64).unsigned_abs() <= 1);
}
```

- [ ] **Step 2: Run tests to verify baseline**

Run: `cargo test --lib resample_rubato --features coreml 2>&1`
Expected: tests fail or produce different results with current linear resampler.

- [ ] **Step 3: Replace `resample()` with rubato implementation**

Replace the `resample` function body in `src/inference/audio.rs:123-158`:

```rust
/// High-quality polyphase FIR resampler (rubato SincFixedIn).
///
/// Non-finite samples (NaN, infinity) are replaced with `0.0` before resampling.
pub fn resample(samples: &[f32], from_rate: u32, to_rate: u32) -> Vec<f32> {
    if samples.is_empty() || from_rate == 0 || to_rate == 0 {
        return Vec::new();
    }
    if from_rate == to_rate {
        return samples.to_vec();
    }

    // Sanitize non-finite values
    let samples: Vec<f32> = samples
        .iter()
        .map(|&s| if s.is_finite() { s } else { 0.0 })
        .collect();

    use rubato::{SincFixedIn, SincInterpolationParameters, SincInterpolationType, WindowFunction, Resampler};

    let params = SincInterpolationParameters {
        sinc_len: 256,
        f_cutoff: 0.95,
        interpolation: SincInterpolationType::Linear,
        oversampling_factor: 256,
        window: WindowFunction::BlackmanHarris2,
    };

    let ratio = to_rate as f64 / from_rate as f64;
    let mut resampler = SincFixedIn::<f32>::new(
        ratio,
        2.0,    // max_resample_ratio_relative
        params,
        samples.len(),
        1,      // mono
    )
    .expect("Failed to create resampler");

    let waves_in = vec![samples];
    match resampler.process(&waves_in, None) {
        Ok(mut waves_out) => waves_out.remove(0),
        Err(e) => {
            tracing::error!("Resampling failed: {e}, falling back to empty");
            Vec::new()
        }
    }
}
```

- [ ] **Step 4: Run all audio tests**

Run: `cargo test --lib audio --features coreml 2>&1`
Expected: all tests pass (existing + new rubato tests). Some existing tests may need assertion tolerance adjustments since rubato output lengths can differ by +-1 sample from exact ratios.

- [ ] **Step 5: Fix any test length assertions**

The existing `test_resample_downsample_length` asserts `output.len() == 1600` for 4800 samples at 48k→16k. Rubato may produce 1599 or 1601. Change exact equality to `assert!((output.len() as i64 - 1600).unsigned_abs() <= 1)` for all length-sensitive tests.

Similarly for `test_resample_upsample_length`.

- [ ] **Step 6: Run full test suite**

Run: `cargo test --lib --features coreml 2>&1`
Expected: all tests pass.

- [ ] **Step 7: Commit**

```bash
git add src/inference/audio.rs
git commit -m "feat(audio): replace linear interpolation with rubato polyphase FIR resampler"
```

---

### Task 3: Add per-connection resampler to server

**Files:**
- Modify: `src/server/mod.rs:76-109` (Ready message + audio processing)
- Test: `tests/server_integration.rs` (extend existing)

- [ ] **Step 1: Define supported rates constant**

At the top of `src/server/mod.rs`, add:

```rust
/// Supported input sample rates (Hz). Default is 48000 for backward compatibility.
const SUPPORTED_RATES: &[u32] = &[8000, 16000, 24000, 44100, 48000];
const DEFAULT_SAMPLE_RATE: u32 = 48000;
```

- [ ] **Step 2: Update Ready message to include supported_rates**

In `handle_connection()`, change the Ready message construction (line 77-81):

```rust
let ready = ServerMessage::Ready {
    model: "gigaam-v3-e2e-rnnt".into(),
    sample_rate: DEFAULT_SAMPLE_RATE,
    version: crate::protocol::PROTOCOL_VERSION.into(),
    supported_rates: SUPPORTED_RATES.to_vec(),
};
```

- [ ] **Step 3: Add Configure handling and dynamic sample rate**

After the Ready message is sent, add a mutable `client_sample_rate`:

```rust
let mut client_sample_rate: u32 = DEFAULT_SAMPLE_RATE;
```

In the `Message::Text` match arm, add Configure handling before the Stop check:

```rust
Message::Text(text) => {
    if let Ok(ClientMessage::Configure { sample_rate }) = serde_json::from_str(&text) {
        if SUPPORTED_RATES.contains(&sample_rate) {
            client_sample_rate = sample_rate;
            tracing::info!("Client {peer} configured sample rate: {sample_rate}Hz");
        } else {
            let err = ServerMessage::Error {
                message: format!("Unsupported sample rate: {sample_rate}Hz. Supported: {SUPPORTED_RATES:?}"),
                code: "invalid_sample_rate".into(),
            };
            sink.send(Message::Text(serde_json::to_string(&err)?)).await?;
        }
    } else if let Ok(ClientMessage::Stop) = serde_json::from_str(&text) {
        // ... existing Stop handling
    } else {
        tracing::debug!("Unrecognized text message from {peer}: {}", &text[..text.len().min(100)]);
    }
}
```

- [ ] **Step 4: Use dynamic sample rate in audio processing**

Change the resample call (line 109) from hardcoded 48000:

```rust
let samples_16k = if client_sample_rate == 16000 {
    samples_48k_f32 // already at target rate, no resampling needed
} else {
    crate::inference::audio::resample(&samples_48k_f32, client_sample_rate, 16000)
};
```

Also rename `samples_48k_f32` to `samples_f32` since it's no longer always 48kHz:

```rust
let samples_f32: Vec<f32> = data
    .chunks_exact(2)
    .map(|chunk| i16::from_le_bytes([chunk[0], chunk[1]]) as f32 / 32768.0)
    .collect();

let samples_16k = if client_sample_rate == 16000 {
    samples_f32
} else {
    crate::inference::audio::resample(&samples_f32, client_sample_rate, 16000)
};
```

- [ ] **Step 5: Run existing tests**

Run: `cargo test --lib --features coreml 2>&1`
Expected: all tests pass. Integration tests may need updating in next step.

- [ ] **Step 6: Update integration test to check supported_rates**

In `tests/server_integration.rs`, find the Ready message assertion and update to accept the new field. The existing test checks `sample_rate == 48000` — keep that, add check for `supported_rates`:

```rust
// After receiving Ready message, verify supported_rates is present
assert!(ready_value.get("supported_rates").is_some());
let rates = ready_value["supported_rates"].as_array().unwrap();
assert!(rates.len() >= 5);
```

- [ ] **Step 7: Commit**

```bash
git add src/server/mod.rs tests/server_integration.rs
git commit -m "feat(server): handle Configure message with dynamic sample rate"
```

---

### Task 4: Integration test — Configure + 8kHz audio

**Files:**
- Modify: `tests/server_integration.rs`

- [ ] **Step 1: Add integration test for Configure message**

```rust
#[tokio::test]
#[ignore] // requires model
async fn test_configure_8khz_sample_rate() {
    // Start server, connect, send Configure{sample_rate: 8000}, send 8kHz PCM16 silence, verify no error
    // ... (uses same pattern as existing integration tests but sends Configure first)
}
```

This test requires the model. Mark with `#[ignore]` for CI, run manually for E2E.

- [ ] **Step 2: Add test for invalid sample rate**

```rust
#[tokio::test]
async fn test_configure_invalid_sample_rate() {
    // Connect, send Configure{sample_rate: 7000}, expect Error message with code "invalid_sample_rate"
}
```

This test does NOT require the model — it tests protocol validation only.

- [ ] **Step 3: Run integration tests**

Run: `cargo test --test server_integration --features coreml 2>&1`
Expected: non-ignored tests pass.

- [ ] **Step 4: Commit**

```bash
git add tests/server_integration.rs
git commit -m "test: integration tests for Configure message and sample rate validation"
```

---

### Task 5: Update documentation

**Files:**
- Modify: `README.md`
- Modify: `CLAUDE.md`

- [ ] **Step 1: Update README.md**

Add to WebSocket protocol section:
- `Configure` message documentation
- Supported sample rates list
- Example: `{"type":"configure","sample_rate":8000}`
- Note: if Configure not sent, default is 48kHz (backward compatible)

- [ ] **Step 2: Update CLAUDE.md**

Update protocol section:
- Add `ClientMessage::Configure` to message types
- Update `Ready` message with `supported_rates`
- Note rubato resampler replaces linear interpolation

- [ ] **Step 3: Commit**

```bash
git add README.md CLAUDE.md
git commit -m "docs: document Configure message and flexible sample rate support"
```
