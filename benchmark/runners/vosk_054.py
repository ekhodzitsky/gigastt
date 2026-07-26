"""Runner for Vosk 0.54 (Zipformer2) via sherpa-onnx.

``vosk-model-ru-0.54`` is NOT a Kaldi model — it is a sherpa-onnx **offline
transducer** (``am-onnx/{encoder,decoder,joiner}.onnx`` + ``lang/tokens.txt``),
exactly as the bundle's own ``decode-onnx.py`` shows
(``sherpa_onnx.OfflineRecognizer.from_transducer(...)``). So this runner uses the
``sherpa-onnx`` package, not the vosk-api.

That encoder→decoder→joiner ONNX transducer is the same pattern the author's native
Rust engines implement (``voxrs/src/inference/zipformer.rs``, ``siamstt``); sherpa-onnx
is the quickest path for the Python benchmark. The decode config below mirrors the
model's own ``decode-onnx.py`` (modified beam search, max_active_paths=10).

Install: ``uv pip install sherpa-onnx``. Until present, ``is_available()`` returns
False and the suite skips Vosk 0.54.
"""

import os
import time
import urllib.request
import zipfile
from pathlib import Path


class Vosk054Runner:
    name = "vosk-0.54"

    def __init__(self, model_name: str | None = None, download_dir: str | None = None):
        self.model_name = model_name or os.environ.get("BENCHMARK_VOSK054_MODEL", "vosk-model-ru-0.54")
        self.download_dir = Path(download_dir) if download_dir else Path.home() / ".cache" / "vosk"
        self.download_dir.mkdir(parents=True, exist_ok=True)
        self._recognizer = None

    @property
    def cache_config(self) -> str:
        # Bump when decode/load path changes (float WAV / MP3 support).
        return f"{self.model_name}:audio-load-v2"

    def is_available(self) -> bool:
        try:
            import sherpa_onnx  # noqa: F401
            return True
        except Exception as e:
            print(f"[vosk-0.54] Not available (install: uv pip install sherpa-onnx): {e}")
            return False

    def _download_model(self) -> Path:
        model_dir = self.download_dir / self.model_name
        if model_dir.exists():
            return model_dir
        url = f"https://alphacephei.com/vosk/models/{self.model_name}.zip"
        zip_path = self.download_dir / f"{self.model_name}.zip"
        print(f"[vosk-0.54] Downloading {url} ...")
        urllib.request.urlretrieve(url, zip_path)
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(self.download_dir)
        return model_dir

    def _load(self):
        if self._recognizer is None:
            import sherpa_onnx

            d = self._download_model()
            am = d / "am-onnx"
            self._recognizer = sherpa_onnx.OfflineRecognizer.from_transducer(
                encoder=str(am / "encoder.onnx"),
                decoder=str(am / "decoder.onnx"),
                joiner=str(am / "joiner.onnx"),
                tokens=str(d / "lang" / "tokens.txt"),
                num_threads=1,
                provider="cpu",
                sample_rate=16000,
                dither=3e-5,
                max_active_paths=10,
                decoding_method="modified_beam_search",
            )
        return self._recognizer

    def _load_mono_16k(self, path: str):
        """Load any common audio file as float32 mono @ 16 kHz.

        The old path used ``wave`` (PCM16 WAV only). Held-out sets ship MP3
        (Common Voice) and IEEE-float WAV (FLEURS), so we decode via
        ``soundfile`` with an ``av`` fallback for containers soundfile skips.
        """
        import numpy as np

        try:
            import soundfile as sf

            data, sr = sf.read(path, dtype="float32", always_2d=True)
            samples = data.mean(axis=1)
        except Exception:
            import av

            container = av.open(path)
            resampler = av.audio.resampler.AudioResampler(
                format="flt", layout="mono", rate=16000
            )
            chunks = []
            for frame in container.decode(audio=0):
                for out in resampler.resample(frame):
                    chunks.append(out.to_ndarray().reshape(-1))
            # flush
            for out in resampler.resample(None):
                chunks.append(out.to_ndarray().reshape(-1))
            if not chunks:
                raise ValueError(f"no audio decoded from {path}")
            samples = np.concatenate(chunks).astype(np.float32)
            sr = 16000

        if sr != 16000:
            # linear resample (good enough for WER smoke; avoids extra deps)
            import numpy as np

            n_out = int(round(len(samples) * 16000 / sr))
            x_old = np.linspace(0.0, 1.0, num=len(samples), endpoint=False)
            x_new = np.linspace(0.0, 1.0, num=n_out, endpoint=False)
            samples = np.interp(x_new, x_old, samples).astype(np.float32)
        return samples

    def transcribe(self, wav_path: str) -> tuple[str, float]:
        recognizer = self._load()
        samples = self._load_mono_16k(wav_path)
        start = time.perf_counter()
        stream = recognizer.create_stream()
        stream.accept_waveform(16000, samples)
        recognizer.decode_stream(stream)
        elapsed = time.perf_counter() - start
        return stream.result.text.strip().lower(), elapsed
