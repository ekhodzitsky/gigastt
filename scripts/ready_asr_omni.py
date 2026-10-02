"""Pinned official Omnilingual CTC1B v2, CPU FP32, without adaptation."""

import hashlib
import importlib.metadata
import json
from pathlib import Path

SOURCE_REVISION = "81f51e224ce9e74b02cc2a3eaf21b2d91d743455"
SOURCE_ARCHIVE_SHA256 = "67c484bba6502454bf192f549d0d0aee974f707179f5c35649679a44eaf0d435"
SOURCE_TREE_SHA256 = "947069e894e9adea2ba075346dd95eec3eeab7f64cae88eb4eb46600c12d9faa"
MODEL_CARD = "omniASR_CTC_1B_v2"
TOKENIZER_CARD = "omniASR_tokenizer_written_v2"
# Downloaded official artifacts are pinned before any benchmark inference.
FILES = {
    "omniASR-CTC-1B-v2.pt": (
        3902956068, "354f981756aa8f41591ea363e45b9c4eba1ec5144c2273af82e747efbb08919c"
    ),
    "omniASR_tokenizer_written_v2.model": (
        91481, "8aa11a1092142ef472537476ef6e76541123e2f0d789b79f3ebd119008240b1e"
    ),
}


def verify_files(model_dir):
    identities = []
    for filename, (size, expected) in FILES.items():
        path = model_dir / filename
        if path.stat().st_size != size:
            raise ValueError(f"Model artifact size mismatch: {filename}")
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if digest != expected:
            raise ValueError(f"Model artifact SHA256 mismatch: {filename}")
        identities.append(dict(filename=filename, bytes=size, sha256=digest))
    return identities


class ReadyASR:
    def __init__(self, model_dir: Path, language: str, threads: int = 3):
        if language not in {"kk", "ky", "uz"} or threads < 1:
            raise ValueError("Expected kk/ky/uz and positive threads")
        model_dir = Path(model_dir).expanduser().resolve()
        identities = verify_files(model_dir)
        distribution = importlib.metadata.distribution("omnilingual-asr")
        direct_url = json.loads(distribution.read_text("direct_url.json") or "{}")
        source_archive = model_dir / "omnilingual-asr-81f51e2.tar.gz"
        if (direct_url.get("url") != source_archive.as_uri()
                or hashlib.sha256(source_archive.read_bytes()).hexdigest() != SOURCE_ARCHIVE_SHA256):
            raise ValueError("Official source archive identity mismatch")
        package_root = Path(distribution.locate_file("omnilingual_asr"))
        source_hash = hashlib.sha256()
        for path in sorted(p for p in package_root.rglob("*") if p.suffix in {".py", ".yaml"}):
            source_hash.update(path.relative_to(package_root).as_posix().encode() + b"\0")
            source_hash.update(hashlib.sha256(path.read_bytes()).digest())
        if source_hash.hexdigest() != SOURCE_TREE_SHA256:
            raise ValueError("Installed official Python/card source differs from pinned archive")
        import torch
        from fairseq2.assets import AssetCard, get_asset_store
        from fairseq2.data.tokenizers.hub import load_tokenizer
        from fairseq2.models.hub import load_model
        from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline

        torch.set_num_threads(threads)
        torch.set_num_interop_threads(1)
        store = get_asset_store()
        official_model_card = store.retrieve_card(MODEL_CARD)
        official_tokenizer_card = store.retrieve_card(TOKENIZER_CARD)
        # Preserve all official settings; resolve only the two URLs to verified files.
        model_card = AssetCard(
            MODEL_CARD,
            dict(official_model_card.metadata,
                 checkpoint=(model_dir / "omniASR-CTC-1B-v2.pt").as_uri()),
            official_model_card.base,
        )
        tokenizer_card = AssetCard(
            TOKENIZER_CARD,
            dict(official_tokenizer_card.metadata,
                 tokenizer=(model_dir / "omniASR_tokenizer_written_v2.model").as_uri()),
            official_tokenizer_card.base,
        )
        model = load_model(model_card, device=torch.device("cpu"), dtype=torch.float32)
        parameter_count = sum(parameter.numel() for parameter in model.parameters())
        if any(parameter.dtype != torch.float32 or parameter.device.type != "cpu"
               for parameter in model.parameters()):
            raise ValueError("Expected every model parameter on CPU in float32")
        tokenizer = load_tokenizer(tokenizer_card)
        self.pipeline = ASRInferencePipeline(
            None, model=model, tokenizer=tokenizer, device="cpu", dtype=torch.float32
        )
        self.metadata = dict(
            model_card=MODEL_CARD, tokenizer_card=TOKENIZER_CARD,
            source_revision=SOURCE_REVISION, model_files=identities,
            source_archive_sha256=SOURCE_ARCHIVE_SHA256,
            installed_source_tree_sha256=SOURCE_TREE_SHA256,
            versions={name: importlib.metadata.version(name) for name in
                      ["omnilingual-asr", "fairseq2", "fairseq2n", "torch", "torchaudio", "numpy"]},
            device="cpu", dtype="float32", threads=threads, interop_threads=1,
            parameter_count=parameter_count,
            batch_size=1, decoder="official greedy CTC; no external language model",
            language_conditioning=False, corpus_language=language,
            preprocessing="official waveform normalization; supplied mono float32 16 kHz",
            maximum_audio_seconds=40, recommended_audio_seconds=30,
            long_audio_policy="process complete input up to 40 seconds; reject longer input",
            training=False, adaptation=False,
            declared_dependency_omissions={
                "kenlm": "Native build failed in this environment; absent from the official greedy CTC code path"
            },
        )

    def transcribe(self, audio):
        import numpy as np

        if not isinstance(audio, np.ndarray) or audio.dtype != np.float32:
            raise ValueError("Expected float32 numpy audio")
        if audio.ndim != 1 or not len(audio) or not np.isfinite(audio).all():
            raise ValueError("Expected nonempty finite mono audio")
        if len(audio) > 40 * 16000:
            raise ValueError("Official non-streaming pipeline limit is 40 seconds")
        result = self.pipeline.transcribe(
            [{"waveform": audio, "sample_rate": 16000}], batch_size=1
        )
        if len(result) != 1 or not isinstance(result[0], str):
            raise ValueError("Unexpected official pipeline result")
        return dict(hypothesis=result[0], input_samples=len(audio),
                    exceeds_recommended_30_seconds=len(audio) > 30 * 16000,
                    language_conditioning=False)
