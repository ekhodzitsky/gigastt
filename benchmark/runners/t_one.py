"""Runner for T-one (``t-tech/T-one``) — T-Bank's streaming CTC Conformer (Apache-2.0).

T-one is purpose-built for Russian telephony / call-center streaming — exactly the
niche gigastt targets, which is why it belongs in the comparison. It ships its **own**
inference package ``tone``; it is NOT a generic ``transformers`` model (the HF repo
``t-tech/T-one`` declares a custom ``ToneForCTC`` architecture). The documented offline
path is::

    from tone import StreamingCTCPipeline, read_audio
    pipeline = StreamingCTCPipeline.from_hugging_face()      # pulls t-tech/T-one
    text = pipeline.forward_offline(read_audio("clip.wav"))

Decoder is selected via ``BENCHMARK_TONE_DECODER`` (``greedy`` default, or
``beam_search`` for T-one's production config). Greedy uses only the 144 MB
``model.onnx``; beam_search additionally needs the optional **5.5 GB KenLM**. That LM
hangs on HF download, so fetch it separately (e.g. ``curl`` the
``t-tech/T-one/resolve/main/kenlm.bin``) and point ``BENCHMARK_TONE_KENLM`` at the
local file — it is loaded via ``BeamSearchCTCDecoder.from_local``. Beam+LM is T-one's
honest config; greedy is the always-works fallback. Flag which decoder was used when
reporting T-one numbers.

Install (``tone`` pulls torch; ``read_audio`` needs ``miniaudio``)::

    uv pip install "git+https://github.com/voicekit-team/T-one.git" miniaudio

Until ``tone`` is importable, ``is_available()`` returns ``False`` and the suite skips
T-one. All heavy imports are lazy so importing this module never fails.
"""

import time


class TOneRunner:
    name = "t-one"

    def __init__(self, device: str | None = None):
        self.device = device
        self._pipeline = None

    @property
    def cache_config(self) -> str:
        import os

        decoder = os.environ.get("BENCHMARK_TONE_DECODER", "greedy")
        kenlm = os.environ.get("BENCHMARK_TONE_KENLM", "")
        return f"{self.device or 'default'}:{decoder}:{kenlm}"

    def is_available(self) -> bool:
        try:
            import tone  # noqa: F401
            return True
        except Exception as e:
            print(
                "[t-one] Not available (install: "
                "uv pip install 'git+https://github.com/voicekit-team/T-one.git'): "
                f"{e}"
            )
            return False

    def _load(self):
        if self._pipeline is None:
            import os

            from tone import DecoderType, StreamingCTCPipeline

            # Decoder via BENCHMARK_TONE_DECODER: "greedy" (default — only the 144 MB
            # model.onnx) or "beam_search" (T-one's production config — needs the 5.5 GB
            # KenLM). The HF download of that LM hangs, so a locally-fetched kenlm.bin
            # can be passed via BENCHMARK_TONE_KENLM and is loaded with from_local().
            mode = os.environ.get("BENCHMARK_TONE_DECODER", "greedy").lower()
            kenlm = os.environ.get("BENCHMARK_TONE_KENLM")
            if mode in ("beam", "beam_search"):
                if kenlm and os.path.exists(kenlm):
                    from tone import (
                        BeamSearchCTCDecoder,
                        StreamingCTCModel,
                        StreamingLogprobSplitter,
                    )

                    print(f"[t-one] Loading beam+LM (local KenLM: {kenlm}) ...")
                    self._pipeline = StreamingCTCPipeline(
                        StreamingCTCModel.from_hugging_face(),
                        StreamingLogprobSplitter(),
                        BeamSearchCTCDecoder.from_local(kenlm),
                    )
                else:
                    print("[t-one] Loading beam+LM (downloads 5.5 GB KenLM from HF) ...")
                    self._pipeline = StreamingCTCPipeline.from_hugging_face(
                        decoder_type=DecoderType.BEAM_SEARCH
                    )
            else:
                print("[t-one] Loading greedy (no external LM) ...")
                self._pipeline = StreamingCTCPipeline.from_hugging_face(
                    decoder_type=DecoderType.GREEDY
                )
        return self._pipeline

    @staticmethod
    def _extract_text(result) -> str:
        """``forward_offline`` returns phrase objects (or strings); join defensively."""
        if isinstance(result, str):
            return result
        parts = []
        for item in result or []:
            if isinstance(item, str):
                parts.append(item)
            else:
                parts.append(
                    getattr(item, "text", None)
                    or getattr(item, "transcription", None)
                    or str(item)
                )
        return " ".join(p for p in parts if p).strip()

    def transcribe(self, wav_path: str) -> tuple[str, float]:
        from tone import read_audio

        pipeline = self._load()
        audio = read_audio(wav_path)
        start = time.perf_counter()
        result = pipeline.forward_offline(audio)
        elapsed = time.perf_counter() - start
        return self._extract_text(result).lower(), elapsed
