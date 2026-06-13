"""Runner for T-one (voicekit-team/T-one) streaming CTC conformer.

Gracefully degrades to unavailable if transformers/torch or the model fail.
"""

import time
from pathlib import Path


class TOneRunner:
    name = "t-one"

    def __init__(self, model_id: str = None):
        import os
        if model_id is None:
            model_id = os.environ.get("BENCHMARK_TONE_MODEL", "voicekit-team/T-one")
        self.model_id = model_id
        self._processor = None
        self._model = None
        self._unavailable_reason = None

    def is_available(self) -> bool:
        try:
            import torch
            from transformers import AutoModelForCTC, AutoProcessor
            return True
        except Exception as e:
            self._unavailable_reason = f"dependencies missing: {e}"
            print(f"[t-one] Not available: {self._unavailable_reason}")
            return False

    def _load(self):
        if self._model is not None:
            return
        from transformers import AutoModelForCTC, AutoProcessor
        print(f"[t-one] Loading model {self.model_id} ...")
        self._processor = AutoProcessor.from_pretrained(self.model_id)
        self._model = AutoModelForCTC.from_pretrained(self.model_id)

    def transcribe(self, wav_path: str) -> tuple[str, float]:
        import torch
        import torchaudio
        self._load()

        waveform, sample_rate = torchaudio.load(wav_path)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(sample_rate, 16000)
            waveform = resampler(waveform)

        start = time.perf_counter()
        inputs = self._processor(waveform.squeeze().numpy(), sampling_rate=16000, return_tensors="pt")
        with torch.no_grad():
            logits = self._model(**inputs).logits
        predicted_ids = torch.argmax(logits, dim=-1)
        text = self._processor.batch_decode(predicted_ids)[0]
        elapsed = time.perf_counter() - start
        return text.strip(), elapsed
