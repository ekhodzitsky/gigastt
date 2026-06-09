"""Runner for gigastt using the REST API (server stays up across samples)."""

import json
import subprocess
import time
import urllib.request
from pathlib import Path


class GigasttRunner:
    name = "gigastt"

    def __init__(self, model_dir: str | None = None, use_int8: bool = True, port: int = 9877):
        self.model_dir = model_dir
        self.use_int8 = use_int8
        self.port = port
        self._binary: str | None = None
        self._proc: subprocess.Popen | None = None

    def is_available(self) -> bool:
        # Try cargo-built release binary first, then PATH
        candidates = [
            str(Path(__file__).parent.parent.parent / "target/release/gigastt"),
            "gigastt",
        ]
        for c in candidates:
            try:
                subprocess.run([c, "--version"], capture_output=True, check=True)
                self._binary = c
                break
            except Exception:
                continue
        if not self._binary:
            return False

        self._start_server()
        return True

    def _start_server(self):
        cmd = [self._binary, "serve", "--port", str(self.port), "--bind-all"]
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

    def __del__(self):
        self._stop_server()
