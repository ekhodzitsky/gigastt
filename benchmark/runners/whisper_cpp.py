"""Runner for whisper.cpp."""

import json
import os
import platform
import subprocess
import time
import urllib.request
from pathlib import Path


class WhisperCppRunner:
    name = "whisper.cpp"

    def __init__(
        self,
        model_name: str = "ggml-large-v3.bin",
        download_dir: str | None = None,
        source_tag: str = "v1.7.5",
    ):
        self.model_name = model_name
        self.source_tag = source_tag
        self.download_dir = Path(download_dir) if download_dir else Path.home() / ".cache" / "whisper.cpp"
        self.download_dir.mkdir(parents=True, exist_ok=True)
        self._binary: Path | None = None
        self._model_path: Path | None = None
        self._source_dir: Path = self.download_dir / f"whisper.cpp-{source_tag}"

    def _build(self) -> Path:
        """Clone and build whisper-cli from source."""
        # Check for pre-existing local build (e.g. /tmp/whisper.cpp)
        local_build = Path("/tmp/whisper.cpp/build/bin/whisper-cli")
        if local_build.exists():
            return local_build

        if not self._source_dir.exists():
            print(f"[whisper.cpp] Cloning source (tag {self.source_tag}) ...")
            subprocess.run(
                [
                    "git", "clone", "--depth", "1",
                    "--branch", self.source_tag,
                    "https://github.com/ggml-org/whisper.cpp.git",
                    str(self._source_dir),
                ],
                check=True,
                capture_output=True,
            )
        build_dir = self._source_dir / "build"
        binary = build_dir / "bin" / "whisper-cli"
        if binary.exists():
            return binary

        print("[whisper.cpp] Building whisper-cli ...")
        cmake_args = ["cmake", "-B", str(build_dir), "-S", str(self._source_dir)]
        sysname = platform.system().lower()
        if sysname == "darwin" and platform.machine().lower() in ("arm64", "aarch64"):
            cmake_args.append("-DWHISPER_METAL=ON")
        elif sysname == "linux":
            cmake_args.append("-DWHISPER_OPENBLAS=ON")

        subprocess.run(cmake_args, check=True, capture_output=True)
        subprocess.run(
            ["cmake", "--build", str(build_dir), "--config", "Release", "-j"],
            check=True,
            capture_output=True,
        )
        if not binary.exists():
            # older tags use "main" instead of "whisper-cli"
            binary = build_dir / "bin" / "main"
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
            # Check if whisper-cli is already on PATH
            for name in ("whisper-cli", "main"):
                try:
                    subprocess.run([name, "-h"], capture_output=True, check=True)
                    self._binary = Path(name)
                    self._model_path = self._download_model()
                    return True
                except Exception:
                    continue
            # Check for local build
            local_build = Path("/tmp/whisper.cpp/build/bin/whisper-cli")
            if local_build.exists():
                self._binary = local_build
                self._model_path = self._download_model()
                return True
            # Build from source
            self._binary = self._build()
            self._model_path = self._download_model()
            return True
        except Exception as e:
            print(f"[whisper.cpp] Not available: {e}")
            return False

    def transcribe(self, wav_path: str) -> tuple[str, float]:
        if not self._binary or not self._model_path:
            raise RuntimeError("whisper.cpp not initialized")
        cmd = [
            str(self._binary),
            "-m", str(self._model_path),
            "-f", wav_path,
            "-l", "ru",
            "--output-json",
            "-of", "/tmp/whisper_cpp_benchmark",
        ]
        start = time.perf_counter()
        result = subprocess.run(cmd, capture_output=True, text=True)
        elapsed = time.perf_counter() - start
        if result.returncode != 0:
            raise RuntimeError(f"whisper.cpp failed: {result.stderr}")
        # Parse JSON output
        json_path = Path("/tmp/whisper_cpp_benchmark.json")
        if json_path.exists():
            with open(json_path, encoding="utf-8") as f:
                data = json.load(f)
            text = " ".join(seg["text"].strip() for seg in data.get("transcription", []))
        else:
            # fallback: parse stdout lines
            text = result.stdout.strip()
        return text, elapsed
