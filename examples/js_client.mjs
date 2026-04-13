/**
 * WebSocket client for gigastt — streams a WAV file and prints transcription.
 * Usage: node js_client.mjs <audio.wav> [ws://host:port]
 *
 * Requires: npm install ws
 */

import { readFileSync } from "fs";
import { WebSocket } from "ws";

const wavPath = process.argv[2];
const server = process.argv[3] || "ws://127.0.0.1:9876/ws";

if (!wavPath) {
  console.log("Usage: node js_client.mjs <audio.wav> [ws://host:port]");
  process.exit(1);
}

// Read WAV file (skip 44-byte header for raw PCM16)
const wav = readFileSync(wavPath);
const pcm = wav.subarray(44); // Skip WAV header

const ws = new WebSocket(server);

ws.on("open", () => {
  console.log("Connected to", server);
});

ws.on("message", (data) => {
  const msg = JSON.parse(data.toString());
  switch (msg.type) {
    case "ready":
      console.log(`Server ready: ${msg.model} @ ${msg.sample_rate}Hz\n`);
      // Send audio in 0.5s chunks
      const chunkSize = 16000; // 0.5s at 16kHz, 2 bytes per sample
      let offset = 0;
      const interval = setInterval(() => {
        if (offset >= pcm.length) {
          clearInterval(interval);
          setTimeout(() => ws.close(), 1000);
          return;
        }
        const chunk = pcm.subarray(offset, offset + chunkSize);
        ws.send(chunk);
        offset += chunkSize;
      }, 100);
      break;
    case "partial":
      process.stdout.write(`\r  ... ${msg.text}`);
      break;
    case "final":
      console.log(`\r  >>> ${msg.text}`);
      break;
    case "error":
      console.error(`  ERR: ${msg.message}`);
      break;
  }
});

ws.on("close", () => {
  console.log("\nDone.");
  process.exit(0);
});
