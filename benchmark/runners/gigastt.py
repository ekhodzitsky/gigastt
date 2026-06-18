"""Runner for gigastt using the REST API (server stays up across samples)."""

import json
import subprocess
import time
import urllib.request
from pathlib import Path


# Bumped whenever the runner's transcription behavior changes in a way that
# invalidates previously cached results (model format, preprocessing, API).
GIGASTT_CACHE_SCHEMA_VERSION = "v2.2.0"


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
        return f"{self.model_dir}:{self.use_int8}:{GIGASTT_CACHE_SCHEMA_VERSION}"

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

    def transcribe(self, wav_path: str) -> tuple[str, float]:
        if not self._binary:
            raise RuntimeError("gigastt not available")
        with open(wav_path, "rb") as f:
            data = f.read()

        req = urllib.request.Request(
            f"http://127.0.0.1:{self.port}/v1/transcribe",
            data=data,
            headers={"Content-Type": "application/octet-stream"},
            method="POST",
        )
        start = time.perf_counter()
        with urllib.request.urlopen(req, timeout=300) as resp:
            body = resp.read().decode("utf-8")
        elapsed = time.perf_counter() - start
        result = json.loads(body)
        text = result.get("text", "").strip()
        return text, elapsed

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
