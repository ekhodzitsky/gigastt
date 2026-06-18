"""Runner for Vosk."""

import json
import time
import urllib.request
import wave
from pathlib import Path


class VoskRunner:
    name = "vosk"

    def __init__(self, model_name: str = None, download_dir: str | None = None):
        import os
        if model_name is None:
            model_name = os.environ.get("BENCHMARK_VOSK_MODEL", "vosk-model-ru-0.42")
        self.model_name = model_name
        self.download_dir = Path(download_dir) if download_dir else Path.home() / ".cache" / "vosk"
        self.download_dir.mkdir(parents=True, exist_ok=True)
        self._model = None

    @property
    def cache_config(self) -> str:
        return self.model_name

    def _download_model(self) -> Path:
        model_dir = self.download_dir / self.model_name
        if model_dir.exists():
            return model_dir
        url = f"https://alphacephei.com/vosk/models/{self.model_name}.zip"
        zip_path = self.download_dir / f"{self.model_name}.zip"
        print(f"[vosk] Downloading model from {url} ...")
        urllib.request.urlretrieve(url, zip_path)
        import zipfile
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(self.download_dir)
        return model_dir

    def is_available(self) -> bool:
        try:
            import vosk
            return True
        except Exception as e:
            print(f"[vosk] Not available: {e}")
            return False

    def _load_model(self):
        if self._model is None:
            import vosk
            model_path = self._download_model()
            self._model = vosk.Model(str(model_path))
        return self._model

    def transcribe(self, wav_path: str) -> tuple[str, float]:
        import vosk
        model = self._load_model()
        rec = vosk.KaldiRecognizer(model, 16000)
        rec.SetWords(False)

        with wave.open(wav_path, "rb") as wf:
            if wf.getnchannels() != 1 or wf.getsampwidth() != 2 or wf.getframerate() != 16000:
                raise ValueError("Vosk requires 16kHz mono 16-bit WAV")
            start = time.perf_counter()
            while True:
                data = wf.readframes(4000)
                if len(data) == 0:
                    break
                rec.AcceptWaveform(data)
            elapsed = time.perf_counter() - start

        result = json.loads(rec.FinalResult())
        text = result.get("text", "").strip()
        return text, elapsed
