#!/usr/bin/env python3
"""Pinned Whisper large-v3 CPU INT8 backend for fixed conversational benchmarks."""
import hashlib
import importlib.metadata
import json
from pathlib import Path

import numpy as np

REPOSITORY = 'Systran/faster-whisper-large-v3'
REVISION = 'edaa852ec7e145841d8ffdb056a99866b5f0a478'
WEIGHT_SHA256 = '69f74147e3334731bc3a76048724833325d2ec74642fb52620eda87352e3d4f1'
REQUIRED_FILES = {'model.bin', 'config.json', 'tokenizer.json',
                  'vocabulary.json', 'preprocessor_config.json', 'README.md'}

SOURCE_BLOBS = {
    'config.json': '75336feae814999bae6ccccdecf177639ffc6f9d',
    'tokenizer.json': '3a5e2ba63acdcac9a19ba56cf9bd27f185bfff61',
    'vocabulary.json': '0adcd01e7c237205d593b707e66dd5d7bc785d2d',
    'preprocessor_config.json': '931c77a740890c46365c7ae0c9d350ba3cca908f',
    'README.md': 'a84bfa7f20cac02ea5a99efa5eaf687ad58c1caf',
}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def verify_model(model_dir):
    model_dir = Path(model_dir)
    provenance = json.loads((model_dir / 'provenance.json').read_text())
    if provenance['repo'] != REPOSITORY or provenance['revision'] != REVISION:
        raise ValueError('Unexpected model repository or revision')
    if set(provenance['files']) != REQUIRED_FILES:
        raise ValueError('Incomplete or unexpected pinned model file set')
    if provenance['files']['model.bin']['sha256'] != WEIGHT_SHA256:
        raise ValueError('Unexpected model weight identity')
    for name, expected in provenance['files'].items():
        path = model_dir / name
        if path.stat().st_size != expected['size_bytes'] or digest(path) != expected['sha256']:
            raise ValueError(f'Model file verification failed: {name}')
        if name in SOURCE_BLOBS:
            data = path.read_bytes()
            git_blob = hashlib.sha1(f'blob {len(data)}\0'.encode() + data).hexdigest()
            if git_blob != SOURCE_BLOBS[name]:
                raise ValueError(f'Pinned Git blob mismatch: {name}')
    return provenance


def validate_audio(audio):
    if not isinstance(audio, np.ndarray) or audio.dtype != np.float32:
        raise ValueError('Audio must be an original float32 ndarray; no implicit integer conversion')
    if audio.ndim != 1 or not audio.size or not np.isfinite(audio).all():
        raise ValueError('Audio must be nonempty finite mono samples at 16000 Hz')
    return audio


class ReadyASR:
    def __init__(self, model_dir: Path, language: str, threads: int = 3):
        if language not in {'kk', 'uz'}:
            raise ValueError('This protocol supports kk/uz; Whisper has no ky language token')
        if threads < 1:
            raise ValueError('threads must be positive')
        provenance = verify_model(model_dir)
        import ctranslate2
        from faster_whisper import WhisperModel
        if 'int8' not in ctranslate2.get_supported_compute_types('cpu'):
            raise ValueError('CPU INT8 unavailable; refusing precision fallback')
        self.language = language
        self.options = dict(task='transcribe', beam_size=5, temperature=0.0,
                            condition_on_previous_text=False, vad_filter=False,
                            word_timestamps=False, initial_prompt=None, hotwords=None,
                            chunk_length=30, without_timestamps=False,
                            compression_ratio_threshold=2.4, log_prob_threshold=-1.0,
                            no_speech_threshold=0.6, patience=1.0, length_penalty=1.0,
                            repetition_penalty=1.0, no_repeat_ngram_size=0,
                            suppress_blank=True, suppress_tokens=[-1])
        self.model = WhisperModel(str(model_dir), device='cpu', compute_type='int8',
                                  cpu_threads=threads, num_workers=1, local_files_only=True)
        self.metadata = {
            'backend': 'faster-whisper', 'model': REPOSITORY, 'revision': REVISION,
            'model_files': provenance['files'],
            'provenance_sha256': digest(Path(model_dir) / 'provenance.json'),
            'device': 'cpu', 'requested_compute_type': 'int8',
            'actual_compute_type': self.model.model.compute_type,
            'cpu_threads': threads, 'num_workers': 1, 'language': language,
            'input': 'caller-supplied float32 mono 16000 Hz ndarray; file decoder bypassed',
            'transcription_options': self.options,
            'packages': {name: importlib.metadata.version(name) for name in
                         ('faster-whisper', 'ctranslate2', 'numpy', 'tokenizers', 'av',
                          'huggingface-hub', 'onnxruntime')},
            'backend_source_sha256': digest(__file__),
        }
        if self.model.model.compute_type not in {'int8', 'int8_float32'}:
            raise ValueError('Unexpected CPU compute precision')

    def transcribe(self, audio):
        audio = validate_audio(audio)
        segments, info = self.model.transcribe(audio, language=self.language, **self.options)
        texts = [segment.text for segment in segments]
        return {'hypothesis': ''.join(texts).strip(),
                'diagnostics': {'language': info.language,
                                'language_probability': info.language_probability,
                                'segments': len(texts)}}
