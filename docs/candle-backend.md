# Candle / Metal inference backend (experimental, opt-in)

gigastt's default inference backend is ONNX Runtime (`ort`). An **optional** pure-Rust
backend built on [Candle](https://github.com/huggingface/candle) is available behind the
`candle` Cargo feature. On Apple Silicon it runs the GigaAM v3 encoder/decoder/joiner on
the **Metal GPU**; elsewhere it falls back to Candle's CPU backend.

It is **additive and opt-in** — the default build is unchanged and still uses `ort`.

## Status

- Targets the default **`rnnt`** head (char vocab). FP32 (no quantization yet).
- **Byte-for-byte parity with the ort backend** is verified at every stage on the Golos
  fixtures: encoder `max_abs_diff ≈ 4e-6`, decoder (LSTM) `≈ 1e-6`, joiner `≈ 3e-6`, and
  whole-file + streaming transcripts are **identical** to ort.
- Experimental: only validated on Apple Silicon with short clips so far. Metal stability on
  very long audio / specific GPUs is not yet characterized.
- `candle` is mutually exclusive with `coreml`/`cuda` (compile-time error if combined).
  Auxiliary models (VAD, punctuation) continue to run on the CPU `ort` path.

## 1. Convert the model weights

The Candle backend loads weights from `safetensors` files derived from the ONNX models you
already have (`gigastt download` populates `~/.gigastt/models/`). No PyTorch or extra
download is required — the converter reads the local `v3_rnnt_*.onnx` files.

```sh
uv run --python 3.13 --with onnx --with numpy --with safetensors \
    python scripts/convert_gigaam_candle.py
```

This writes, next to the ONNX models:

```
~/.gigastt/models/candle/encoder.safetensors
~/.gigastt/models/candle/decoder.safetensors
~/.gigastt/models/candle/joiner.safetensors
```

## 2. Build with the feature

```sh
# server binary (Apple Silicon → Metal, else Candle CPU)
cargo build --release --features candle

# or just the core library
cargo build -p gigastt-core --release --features candle
```

Do **not** combine with `--features coreml` or `--features cuda` (mutually exclusive).
The default `ort` build remains `cargo build --release` (unchanged).

## 3. Run

The server and CLI use the Candle backend automatically when built with `--features candle`
(both `default_factory` and `production_factory` route to Candle under the feature). Usage is
otherwise identical to the default build.

## Notes on packaging

The implementation compiles against **upstream `candle` 0.9** with no API changes, so
publishing remains possible. (RustASR — the source of the vendored conformer encoder — uses a
candle fork carrying extra Metal-kernel patches; those differ only at the runtime/kernel
level, not the Rust API. If a Metal stability issue surfaces on specific hardware or long
audio, evaluate adopting/​upstreaming those patches.)

The vendored conformer encoder originates from
[askidmobile/RustASR](https://github.com/askidmobile/RustASR) (`crates/model-gigaam`),
dual-licensed MIT OR Apache-2.0; attribution is preserved in
`crates/gigastt-core/src/runtime/candle/conformer.rs`.
