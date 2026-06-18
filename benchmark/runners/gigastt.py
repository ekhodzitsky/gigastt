"""Runner for gigastt using WebSocket streaming (server stays up across samples)."""

import asyncio
import json
import subprocess
import time
import urllib.request
import wave
from pathlib import Path


try:
    import websockets
except ImportError as _e:
    websockets = None


class GigasttRunner:
    name = "gigastt"

    def __init__(self, model_dir: str | None = None, use_int8: bool = True, port: int = 9877):
        self.model_dir = model_dir
        self.use_int8 = use_int8
        self.port = port
        self._binary: str | None = None
        self._proc: subprocess.Popen | None = None

    @property
    def cache_config(self) -> str:
        """Stable config string for result caching.

        Intentionally excludes the resolved binary path: that path is discovered
        lazily after ``is_available()`` and would change the key between the
        first (cache-miss) and second (cache-lookup) runs.
        """
        return f"{self.model_dir}:{self.use_int8}:v2.2.0"

    def _find_binary(self) -> bool:
        """Locate the gigastt binary and cache the path."""
        if self._binary:
            return True
        candidates = [
            str(Path(__file__).parent.parent.parent / "target/release/gigastt"),
            "gigastt",
        ]
        for c in candidates:
            try:
                subprocess.run([c, "--version"], capture_output=True, check=True)
                self._binary = c
                return True
            except Exception:
                continue
        return False

    def is_available(self) -> bool:
        if not self._find_binary():
            return False
        if websockets is None:
            print("[gigastt] websockets not installed; run: pip install websockets")
            return False
        self._start_server()
        return True

    def _start_server(self):
        if self._proc is not None:
            return
        if not self._binary and not self._find_binary():
            raise RuntimeError("gigastt binary not found")
        cmd = [self._binary, "serve", "--port", str(self.port)]
        if self.model_dir:
            cmd.extend(["--model-dir", self.model_dir])
        # Suppress server logs for clean benchmark output
        env = {**dict(subprocess.os.environ), "RUST_LOG": "error"}
        self._proc = subprocess.Popen(
            cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=env,
        )
        # Wait for readiness
        for _ in range(60):
            try:
                with urllib.request.urlopen(
                    f"http://127.0.0.1:{self.port}/ready", timeout=1,
                ) as resp:
                    if resp.status == 200:
                        return
            except Exception:
                pass
            time.sleep(0.5)
        raise RuntimeError("gigastt server failed to start")

    def _stop_server(self):
        if self._proc:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._proc.kill()
                self._proc.wait()
            self._proc = None

    async def _stream_transcribe(self, wav_path: str) -> tuple[str, float]:
        """Stream a WAV file over WebSocket and return the final transcript."""
        url = f"ws://127.0.0.1:{self.port}/v1/ws"

        with wave.open(wav_path, "rb") as wf:
            channels = wf.getnchannels()
            width = wf.getsampwidth()
            rate = wf.getframerate()
            if channels != 1 or width != 2:
                raise ValueError(f"{wav_path}: expected mono 16-bit WAV")

            frames_per_chunk = int(rate * 0.2)  # 200 ms chunks
            chunk_bytes = frames_per_chunk * width
            audio_bytes = wf.readframes(wf.getnframes())

        final_text = ""
        started_at = None

        async with websockets.connect(url) as ws:
            # Wait for ready and configure stream
            ready = json.loads(await ws.recv())
            if ready.get("type") != "ready":
                raise RuntimeError(f"Unexpected ready message: {ready}")
            await ws.send(json.dumps({"type": "configure", "sample_rate": rate}))

            # Start reader task to collect partials/final
            final_future: asyncio.Future[str] = asyncio.get_event_loop().create_future()

            async def _reader():
                nonlocal final_text
                async for raw in ws:
                    msg = json.loads(raw)
                    kind = msg.get("type")
                    if kind == "final":
                        final_text = msg.get("text", "").strip()
                        if not final_future.done():
                            final_future.set_result(final_text)
                        return
                    elif kind == "error":
                        if not final_future.done():
                            final_future.set_exception(RuntimeError(msg.get("message", "WS error")))
                        return

            reader_task = asyncio.create_task(_reader())

            # Stream audio in real-time paced chunks
            started_at = time.perf_counter()
            for i in range(0, len(audio_bytes), chunk_bytes):
                chunk = audio_bytes[i : i + chunk_bytes]
                await ws.send(chunk)
                # Pace at ~real-time to mimic live microphone input
                await asyncio.sleep(len(chunk) / (rate * width))

            await ws.send(json.dumps({"type": "stop"}))

            try:
                await asyncio.wait_for(final_future, timeout=30.0)
            except asyncio.TimeoutError:
                reader_task.cancel()
                raise RuntimeError("Timeout waiting for final transcription")
            finally:
                if not reader_task.done():
                    reader_task.cancel()
                    try:
                        await reader_task
                    except asyncio.CancelledError:
                        pass

        elapsed = time.perf_counter() - started_at
        return final_text, elapsed

    def transcribe(self, wav_path: str) -> tuple[str, float]:
        if not self._binary:
            raise RuntimeError("gigastt not available")
        if websockets is None:
            raise RuntimeError("websockets package not installed")
        return asyncio.run(self._stream_transcribe(wav_path))

    def __enter__(self):
        self._start_server()
        return self

    def __exit__(self, exc_type, exc, tb):
        self._stop_server()
        return False


class GigasttCoreMLRunner(GigasttRunner):
    """gigastt built with ``--features coreml`` (macOS arm64 / Neural Engine).

    Only available on Apple Silicon; elsewhere ``is_available()`` is ``False``.
    Point it at a CoreML-built binary via ``BENCHMARK_GIGASTT_COREML_BINARY``;
    otherwise it falls back to ``target/release/gigastt`` (which must itself be a
    coreml build). Intended for the footprint / latency comparisons, not the
    cross-WER table (the transcription is identical to ``gigastt``).
    """

    name = "gigastt-coreml"

    def __init__(self, model_dir: str | None = None, port: int = 9878):
        super().__init__(model_dir=model_dir, use_int8=True, port=port)

    @staticmethod
    def _is_apple_silicon() -> bool:
        import platform

        return platform.system() == "Darwin" and platform.machine() == "arm64"

    def _find_binary(self) -> bool:
        import os

        if not self._is_apple_silicon():
            return False
        override = os.environ.get("BENCHMARK_GIGASTT_COREML_BINARY")
        if override:
            try:
                subprocess.run([override, "--version"], capture_output=True, check=True)
                self._binary = override
                return True
            except Exception:
                return False
        return super()._find_binary()

    def is_available(self) -> bool:
        if not self._is_apple_silicon():
            print("[gigastt-coreml] Not available: requires macOS arm64")
            return False
        return super().is_available()
