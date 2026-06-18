"""Runner for whisper.cpp using the HTTP server (whisper-server)."""

import mimetypes
import os
import platform
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path


class WhisperCppRunner:
    name = "whisper.cpp"

    def __init__(
        self,
        model_name: str = "ggml-large-v3.bin",
        download_dir: str | None = None,
        source_tag: str = "v1.7.5",
        port: int | None = None,
    ):
        self.model_name = model_name
        self.source_tag = source_tag
        self.download_dir = Path(download_dir) if download_dir else Path.home() / ".cache" / "whisper.cpp"
        self.download_dir.mkdir(parents=True, exist_ok=True)
        self.port = port if port is not None else int(os.environ.get("WHISPER_CPP_PORT", "8191"))
        self._binary: Path | None = None
        self._model_path: Path | None = None
        self._source_dir: Path = self.download_dir / f"whisper.cpp-{source_tag}"
        self._proc: subprocess.Popen | None = None

    @property
    def cache_config(self) -> str:
        return f"{self.model_name}:{self.source_tag}"

    def _build(self) -> Path:
        """Clone and build whisper-server from source."""
        # Check for pre-existing local build (e.g. /tmp/whisper.cpp)
        local_build = Path("/tmp/whisper.cpp/build/bin/whisper-server")
        if local_build.exists():
            return local_build

        if not self._source_dir.exists():
            print(f"[whisper.cpp] Cloning source (tag {self.source_tag}) ...")
            subprocess.run(
                [
                    "git", "clone", "--depth", "1",
                    "--branch", self.source_tag,
                    "https://github.com/ggerganov/whisper.cpp.git",
                    str(self._source_dir),
                ],
                check=True,
                capture_output=True,
            )
        build_dir = self._source_dir / "build"
        binary = build_dir / "bin" / "whisper-server"
        if binary.exists():
            return binary

        print("[whisper.cpp] Building whisper-server ...")
        cmake_args = ["cmake", "-B", str(build_dir), "-S", str(self._source_dir)]
        sysname = platform.system().lower()
        if sysname == "darwin" and platform.machine().lower() in ("arm64", "aarch64"):
            cmake_args.append("-DWHISPER_METAL=ON")
        elif sysname == "linux":
            cmake_args.append("-DWHISPER_OPENBLAS=ON")

        subprocess.run(cmake_args, check=True, capture_output=True)
        subprocess.run(
            ["cmake", "--build", str(build_dir), "--config", "Release", "-j", "--target", "whisper-server"],
            check=True,
            capture_output=True,
        )
        return binary

    def _download_model(self) -> Path:
        model_path = self.download_dir / self.model_name
        if not model_path.exists():
            url = f"https://huggingface.co/ggerganov/whisper.cpp/resolve/main/{self.model_name}"
            print(f"[whisper.cpp] Downloading model {self.model_name} ...")
            urllib.request.urlretrieve(url, model_path)
        return model_path

    def is_available(self) -> bool:
        try:
            # Check if whisper-server is already on PATH
            for name in ("whisper-server",):
                try:
                    subprocess.run([name, "-h"], capture_output=True, check=True)
                    self._binary = Path(name)
                    self._model_path = self._download_model()
                    self._start_server()
                    return True
                except Exception:
                    continue
            # Check for local build
            local_build = Path("/tmp/whisper.cpp/build/bin/whisper-server")
            if local_build.exists():
                self._binary = local_build
                self._model_path = self._download_model()
                self._start_server()
                return True
            # Build from source
            self._binary = self._build()
            self._model_path = self._download_model()
            self._start_server()
            return True
        except Exception as e:
            print(f"[whisper.cpp] Not available: {e}")
            return False

    def _start_server(self):
        if not self._binary or not self._model_path:
            raise RuntimeError("whisper.cpp binary or model not initialized")
        cmd = [
            str(self._binary),
            "-m", str(self._model_path),
            "--host", "127.0.0.1",
            "--port", str(self.port),
            "-l", "ru",
        ]
        # Suppress server logs for clean benchmark output
        self._proc = subprocess.Popen(
            cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        # Wait for readiness
        health_url = f"http://127.0.0.1:{self.port}/health"
        for _ in range(60):
            try:
                with urllib.request.urlopen(health_url, timeout=1) as resp:
                    if resp.status == 200:
                        return
            except Exception:
                pass
            time.sleep(0.5)
        self._stop_server()
        raise RuntimeError("whisper.cpp server failed to start")

    def _stop_server(self):
        if self._proc:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._proc.kill()
                self._proc.wait()
            self._proc = None

    def _encode_multipart(self, fields: dict[str, str], files: dict[str, tuple[str, bytes]]) -> tuple[str, bytes]:
        boundary = "----WebKitFormBoundary" + os.urandom(16).hex()
        body = b""
        for name, value in fields.items():
            body += f"--{boundary}\r\n".encode()
            body += f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode()
            body += value.encode()
            body += b"\r\n"
        for name, (filename, data) in files.items():
            ctype = mimetypes.guess_type(filename)[0] or "application/octet-stream"
            body += f"--{boundary}\r\n".encode()
            body += f'Content-Disposition: form-data; name="{name}"; filename="{filename}"\r\n'.encode()
            body += f"Content-Type: {ctype}\r\n\r\n".encode()
            body += data
            body += b"\r\n"
        body += f"--{boundary}--\r\n".encode()
        return f"multipart/form-data; boundary={boundary}", body

    def transcribe(self, wav_path: str) -> tuple[str, float]:
        if not self._proc:
            raise RuntimeError("whisper.cpp server not running")
        wav_path = Path(wav_path)
        with open(wav_path, "rb") as f:
            audio_data = f.read()

        content_type, body = self._encode_multipart(
            fields={"response_format": "text", "language": "ru"},
            files={"file": (wav_path.name, audio_data)},
        )
        req = urllib.request.Request(
            f"http://127.0.0.1:{self.port}/inference",
            data=body,
            headers={"Content-Type": content_type},
            method="POST",
        )
        start = time.perf_counter()
        try:
            with urllib.request.urlopen(req, timeout=300) as resp:
                text = resp.read().decode("utf-8").strip()
        except urllib.error.HTTPError as e:
            raise RuntimeError(f"whisper.cpp server returned {e.code}: {e.read().decode('utf-8', errors='ignore')}") from e
        elapsed = time.perf_counter() - start
        # Normalize segment newlines to spaces to match prior text assembly
        text = " ".join(text.split())
        return text, elapsed

    def __enter__(self):
        self._start_server()
        return self

    def __exit__(self, exc_type, exc, tb):
        self._stop_server()
        return False
