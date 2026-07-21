// Electron main-process integration: in-process Russian speech-to-text via the
// `gigastt` npm package — no sidecar server, no loopback port, no version gate,
// no process supervision. The engine lives inside the main process.
//
// Pattern: one shared `Engine` (main process only — never in a renderer), one
// `Stream` per audio channel. The dual-channel recorder case below keeps a mic
// stream and a system-audio stream side by side; each holds one pool session
// for its lifetime, so the pool size must cover the number of live streams.
//
// Prerequisites:
//   npm install gigastt electron
//   gigastt download --prequantized   # INT8 bundle -> ~/.gigastt/models
//
// This is a teaching file, not a runnable app: renderer-side audio capture is
// sketched in comments at the bottom. The same Engine/Stream calls are exercised
// against a real model by crates/gigastt-node/__test__/smoke.cjs (`npm run smoke`).

import os from 'node:os';
import path from 'node:path';
import { app, BrowserWindow, ipcMain } from 'electron';
import gigastt from 'gigastt';

const { Engine, Stream } = gigastt;

// The model directory is side-loaded, not bundled: fetch the pre-quantized
// INT8 bundle once (`gigastt download --prequantized`, ~215 MB) or ship the
// directory inside your installer. Keep it OUT of the asar archive — native
// code memory-maps the weights, which asar's virtual filesystem does not support.
const modelDir =
  process.env.GIGASTT_MODEL_DIR || path.join(os.homedir(), '.gigastt', 'models');

// Errors surface as thrown JS `Error`s whose message starts with a stable code
// (`ModelNotFound`, `InvalidAudio`, `PoolExhausted`, `Inference`) — branch on
// the prefix, show the rest to the user. A bad model path throws here, at boot,
// instead of on the first transcription attempt.
const engine = new Engine(modelDir, 2); // pool of 2: one session per channel

// One Stream per channel: mic (local speaker) and system (remote speaker).
// Each Stream holds one pool session until it is garbage-collected; creating a
// third Stream on a pool of 2 throws `PoolExhausted`.
const streams = {
  mic: new Stream(engine),
  system: new Stream(engine),
};

// Audio contract: little-endian mono PCM16 (`Int16Array` bytes) at 16 kHz.
// Capture at 16 kHz mono if you can — `processChunk` resamples 8/24/44.1/48 kHz
// internally, but feeding 16 kHz skips that work entirely.
ipcMain.handle('stt:chunk', async (_event, channel, pcm16) => {
  const stream = streams[channel];
  if (!stream) throw new Error(`unknown channel: ${channel}`);
  // Await each chunk before feeding the next one: chunks are decoded in order,
  // and concurrent calls on one Stream would interleave audio.
  const segments = await stream.processChunk(new Uint8Array(pcm16), 16000);
  // Segments arrive as interim hypotheses (isFinal === false) — render them as
  // pending text, they can still be revised — and as finals (isFinal === true)
  // when an endpoint is detected mid-stream; finals are safe to persist.
  return segments;
});

ipcMain.handle('stt:flush', async (_event, channel) => {
  const stream = streams[channel];
  if (!stream) throw new Error(`unknown channel: ${channel}`);
  // Flush on recording stop: drains the buffered tail into the last segment(s).
  // It may legitimately return an empty array when every segment already
  // finalized via processChunk.
  return stream.flush();
});

// Inference runs on libuv's worker pool (default 4 threads, shared with
// fs/crypto). For N channels transcribing simultaneously, start Electron with
// UV_THREADPOOL_SIZE >= N, e.g. `UV_THREADPOOL_SIZE=4 electron .`

function createWindow() {
  const win = new BrowserWindow({ width: 900, height: 600 });
  win.loadFile('index.html'); // renderer: capture + IPC forwarding (see below)
}

app.whenReady().then(() => {
  createWindow();
  app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) createWindow();
  });
});

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') app.quit();
});

// --- renderer sketch (index.html) -------------------------------------------
// Mic channel: getUserMedia({ audio: { channelCount: 1, sampleRate: 16000 } })
// into an AudioContext + AudioWorklet; the worklet posts Float32 frames, the
// renderer converts to little-endian Int16 (`Math.max(-1, Math.min(1, s)) *
// 0x7fff`) and forwards via `ipcRenderer.invoke('stt:chunk', 'mic', bytes)`.
// System channel: same pipeline fed from `desktopCapturer` /
// `getDisplayMedia({ audio: true })` with channel 'system'.
// On stop: `ipcRenderer.invoke('stt:flush', channel)` and persist the finals.
