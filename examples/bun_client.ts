// WebSocket client for gigastt — streams a WAV file and prints transcription.
// Usage: bun examples/bun_client.ts <audio.wav> [ws://host:port]

const WAV_HEADER_BYTES = 44;
const CHUNK_BYTES = 32768; // ~1s at 16kHz PCM16

const wavPath = Bun.argv[2];
const server = Bun.argv[3] ?? "ws://127.0.0.1:9876/v1/ws";

if (!wavPath) {
  console.error(`Usage: bun ${Bun.argv[1]} <audio.wav> [ws://host:port]`);
  process.exit(1);
}

const buf = await Bun.file(wavPath).arrayBuffer();
const pcm = buf.slice(WAV_HEADER_BYTES);

let done: () => void;
const finished = new Promise<void>((resolve) => { done = resolve; });

const ws = new WebSocket(server);
ws.binaryType = "arraybuffer";

ws.onopen = () => {
  // Ready message is received first; audio is sent after onmessage handles it
};

let audioSent = false;

ws.onmessage = async (event) => {
  const msg = JSON.parse(event.data as string);

  if (msg.type === "ready") {
    console.log(`Connected: ${msg.model} @ ${msg.sample_rate}Hz\n`);

    // Send PCM16 in chunks
    for (let offset = 0; offset < pcm.byteLength; offset += CHUNK_BYTES) {
      ws.send(pcm.slice(offset, offset + CHUNK_BYTES));
    }
    audioSent = true;

    // Signal end of audio
    ws.send(JSON.stringify({ type: "stop" }));
  } else if (msg.type === "partial") {
    process.stdout.write(`\r  ... ${msg.text}`);
  } else if (msg.type === "final") {
    process.stdout.write(`\r  >>> ${msg.text}\n`);
    ws.close();
    done();
  } else if (msg.type === "error") {
    if (msg.retry_after_ms) {
      console.error(`\n  ERR: ${msg.message} (retry after ${msg.retry_after_ms}ms)`);
    } else {
      console.error(`\n  ERR: ${msg.message}`);
    }
    ws.close();
    done();
  }
};

ws.onerror = (err) => {
  console.error("WebSocket error:", err);
  done();
};

await finished;
