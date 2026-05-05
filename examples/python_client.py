#!/usr/bin/env python3
"""WebSocket client for gigastt — streams a WAV file and prints transcription."""

import asyncio
import json
import struct
import sys
import wave

try:
    import websockets
except ImportError:
    print("Install: pip install websockets")
    sys.exit(1)


async def transcribe(wav_path: str, server: str = "ws://127.0.0.1:9876/v1/ws"):
    async with websockets.connect(server) as ws:
        # Wait for ready
        msg = json.loads(await ws.recv())
        assert msg["type"] == "ready", f"Expected ready, got {msg}"
        print(f"Connected: {msg['model']} @ {msg['sample_rate']}Hz")

        # Read WAV
        with wave.open(wav_path, "rb") as wf:
            assert wf.getnchannels() == 1, "Expected mono audio"
            assert wf.getsampwidth() == 2, "Expected 16-bit audio"
            rate = wf.getframerate()
            frames = wf.readframes(wf.getnframes())

        if rate != 16000:
            print(f"Warning: audio is {rate}Hz, server expects 16000Hz")

        # Send audio in chunks (0.5s each)
        chunk_samples = 8000  # 0.5s at 16kHz
        chunk_bytes = chunk_samples * 2
        for i in range(0, len(frames), chunk_bytes):
            await ws.send(frames[i : i + chunk_bytes])
            await asyncio.sleep(0.05)  # Small delay between chunks

        # Signal end of audio and close
        await ws.send(json.dumps({"type": "stop"}))
        await ws.close()

        # Print results (collect during streaming)
        print("Done.")


async def stream_and_print(wav_path: str, server: str = "ws://127.0.0.1:9876/v1/ws"):
    async with websockets.connect(server) as ws:
        msg = json.loads(await ws.recv())
        assert msg["type"] == "ready", f"Expected ready, got {msg}"
        print(f"Connected: {msg['model']} @ {msg['sample_rate']}Hz\n")

        # Start receiver task
        async def receiver():
            async for raw in ws:
                msg = json.loads(raw)
                if msg["type"] == "partial":
                    print(f"\r  ... {msg['text']}", end="", flush=True)
                elif msg["type"] == "final":
                    print(f"\r  >>> {msg['text']}")
                elif msg["type"] == "error":
                    retry = msg.get("retry_after_ms")
                    if retry:
                        print(f"\n  ERR: {msg['message']} (retry after {retry}ms)")
                    else:
                        print(f"\n  ERR: {msg['message']}")

        recv_task = asyncio.create_task(receiver())

        # Send audio
        with wave.open(wav_path, "rb") as wf:
            frames = wf.readframes(wf.getnframes())

        chunk_bytes = 16000  # 0.5s chunks
        for i in range(0, len(frames), chunk_bytes):
            await ws.send(frames[i : i + chunk_bytes])
            await asyncio.sleep(0.1)

        # Signal end of audio
        await ws.send(json.dumps({"type": "stop"}))
        await asyncio.sleep(1)  # Wait for final results
        recv_task.cancel()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <audio.wav> [ws://host:port]")
        sys.exit(1)

    wav = sys.argv[1]
    server = sys.argv[2] if len(sys.argv) > 2 else "ws://127.0.0.1:9876/v1/ws"
    asyncio.run(stream_and_print(wav, server))
