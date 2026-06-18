"""Runner for faster-whisper (Python package)."""

import time
from pathlib import Path


class FasterWhisperRunner:
    name = "faster-whisper"

    def __init__(self, model_size: str = None, device: str = "cpu", compute_type: str = "int8"):
        import os
        if model_size is None:
            model_size = os.environ.get("BENCHMARK_FASTER_WHISPER_MODEL", "large-v3")
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self._model = None

    @property
    def cache_config(self) -> str:
        return f"{self.model_size}:{self.device}:{self.compute_type}"

    def is_available(self) -> bool:
        try:
            from faster_whisper import WhisperModel
            return True
        except Exception as e:
            print(f"[faster-whisper] Not available: {e}")
            return False

    def _load_model(self):
        if self._model is None:
            from faster_whisper import WhisperModel
            print(f"[faster-whisper] Loading model {self.model_size} ({self.compute_type}) ...")
            self._model = WhisperModel(
                self.model_size,
                device=self.device,
                compute_type=self.compute_type,
            )
        return self._model

    def transcribe(self, wav_path: str) -> tuple[str, float]:
        model = self._load_model()
        start = time.perf_counter()
        segments, info = model.transcribe(wav_path, language="ru", beam_size=5)
        text_parts = []
        for segment in segments:
            text_parts.append(segment.text.strip())
        elapsed = time.perf_counter() - start
        return " ".join(text_parts), elapsed
