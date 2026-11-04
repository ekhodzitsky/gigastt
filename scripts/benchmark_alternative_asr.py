#!/usr/bin/env python3
"""Pinned Whisper+LoRA CPU control; restricted phone details stay outside repo.

Lab dependencies: torch (CPU), transformers==4.57.3, peft==0.18.0,
numpy, jiwer. Download the base/adapter snapshots first; loading is local-only.
"""
import argparse
import json
import os
from pathlib import Path
import resource
import sys
import platform
import subprocess
import tempfile
import time

from benchmark_multilingual_public import score, sha256, summarize


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--models', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--summary-only', action='store_true', help='Write aggregate metrics and metadata without source text, paths, or hypotheses')
    parser.add_argument('--conditions', nargs='+', default=['original'], choices=['original', 'simulated_telephone'])
    parser.add_argument('--threads', type=int, default=3)
    parser.add_argument('--beams', type=int, default=5)
    parser.add_argument('--dtype', choices=['float32', 'bfloat16', 'int8_dynamic', 'int8'], default='float32')
    parser.add_argument('--backend', choices=['torch', 'ctranslate2'], default='torch')
    parser.add_argument('--limit', type=int)
    parser.add_argument('--language', choices=['auto', 'kazakh'], default='auto')
    parser.add_argument('--silence-seconds', nargs='+', type=float, default=[])
    args = parser.parse_args()
    if args.threads < 1 or args.beams < 1 or (args.limit is not None and args.limit < 1):
        parser.error('threads, beams and limit must be positive')
    if any(not 0 < seconds <= 30 for seconds in args.silence_seconds):
        parser.error('silence durations must be in (0, 30] seconds')
    if args.backend == 'torch' and args.dtype == 'int8':
        parser.error('Use int8_dynamic for PyTorch, or int8 with CTranslate2')
    if args.backend == 'ctranslate2' and args.dtype not in ('float32', 'int8'):
        parser.error('CTranslate2 CPU requires float32 or int8')
    manifest = json.loads(args.manifest.read_text())
    restricted = 'ldc' in str(manifest.get('license', '')).lower() or 'no redistribution' in str(manifest.get('license', '')).lower() or 'ldc.upenn.edu' in str(manifest.get('source', ''))
    if restricted and args.output.resolve().is_relative_to(Path(__file__).resolve().parents[1]) and not args.summary_only:
        parser.error('Restricted telephone details must remain outside the repository; use --summary-only for aggregate publication')
    os.environ['OMP_NUM_THREADS'] = str(args.threads)
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    import numpy as np
    import torch
    import transformers
    import peft
    from transformers import WhisperProcessor, WhisperForConditionalGeneration
    from peft import PeftModel
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    weights = {
        'base/model.safetensors': '95235871856d690c02c663d83d7099e89650877fcf57b48e0d1e5e4a278fecdf',
        'adapter/adapter_model.safetensors': 'f49c02568e5d37300f736ef2a325f9f6b356936d4378e2a8d842a09c8194dc23',
    }
    for name, expected in weights.items():
        if sha256(args.models / name) != expected:
            raise ValueError(f'Unexpected model weights: {name}')
    started = time.perf_counter()
    processor = WhisperProcessor.from_pretrained(args.models / 'base', local_files_only=True, language=None, task='transcribe')
    backend_version = torch.__version__
    if args.backend == 'torch':
        model = WhisperForConditionalGeneration.from_pretrained(args.models / 'base', local_files_only=True, torch_dtype=torch.float32)
        print('Base loaded; merging LoRA', flush=True)
        model = PeftModel.from_pretrained(model, args.models / 'adapter', local_files_only=True).merge_and_unload().eval()
        if args.dtype == 'int8_dynamic':
            model = torch.ao.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
        else:
            model = model.to(dtype=getattr(torch, args.dtype))
        print('Model ready', round(time.perf_counter() - started, 2), flush=True)
        # Auto language selection follows the published model-card recipe. Task is ASR.
        model.generation_config.forced_decoder_ids = None
        component_seconds = {}
        component_started = {}
        def before_component(name):
            def before(module, inputs):
                component_started[name] = time.perf_counter()
            return before
        def after_component(name):
            def after(module, inputs, output):
                component_seconds[name] = component_seconds.get(name, 0.0) + time.perf_counter() - component_started[name]
            return after
        for name in ('encoder', 'decoder'):
            component = getattr(model.model, name)
            component.register_forward_pre_hook(before_component(name))
            component.register_forward_hook(after_component(name))

        def recognize(audio):
            component_seconds.clear()
            features = processor(audio, sampling_rate=16000, return_tensors='pt', return_attention_mask=True)
            with torch.inference_mode():
                ids = model.generate(features.input_features.to(model.dtype), attention_mask=features.attention_mask, task='transcribe', language=None if args.language == 'auto' else args.language, num_beams=args.beams, max_new_tokens=440, do_sample=False, return_timestamps=False, use_cache=True)
            return processor.batch_decode(ids, skip_special_tokens=True)[0].strip(), ids, dict(component_seconds)

    else:
        import ctranslate2
        if args.dtype not in ('float32', 'int8'):
            raise ValueError('CTranslate2 CPU requires float32 or int8')
        converted = args.models / 'ct2'
        conversion = json.loads((converted / 'conversion.json').read_text())
        if conversion['source_weights_sha256'] != weights:
            raise ValueError('Unexpected converted model provenance')
        for name, expected in conversion['files_sha256'].items():
            if sha256(converted / name) != expected:
                raise ValueError(f'Converted model checksum mismatch: {name}')
        model = ctranslate2.models.Whisper(str(converted), device='cpu', compute_type=args.dtype, inter_threads=1, intra_threads=args.threads)
        backend_version = ctranslate2.__version__
        suppression = json.loads((args.models / 'base/generation_config.json').read_text())['suppress_tokens']
        def recognize(audio):
            features = processor(audio, sampling_rate=16000, return_tensors='np').input_features
            tick = time.perf_counter()
            encoded = model.encode(ctranslate2.StorageView.from_array(features))
            encoder_s = time.perf_counter() - tick
            language = model.detect_language(encoded)[0][0][0] if args.language == 'auto' else '<|kk|>'
            prompt = processor.tokenizer.convert_tokens_to_ids(['<|startoftranscript|>', language, '<|transcribe|>', '<|notimestamps|>'])
            tick = time.perf_counter()
            result = model.generate(encoded, [prompt], beam_size=args.beams, max_length=440, suppress_tokens=suppression, suppress_blank=True, sampling_topk=1)[0]
            decoder_s = time.perf_counter() - tick
            ids = np.array([result.sequences_ids[0]], dtype=np.int64)
            return processor.batch_decode(ids, skip_special_tokens=True)[0].strip(), ids, {'encoder': encoder_s, 'decoder': decoder_s}
        print('CTranslate2 ready', round(time.perf_counter() - started, 2), flush=True)

    details = []
    output = {
        'base_model': 'abilmansplus/whisper-turbo-ksc2',
        'base_revision': '92847d02bbdb4311e4c85d9d3b4c7f82e31e59bf',
        'adapter_model': 'abilmansplus/whisper-turbo-kaz-rus-v1',
        'adapter_revision': '904d5d22339952909001d1abcbc1ecf29d26d706',
        'license': 'MIT (both model cards)', 'weights_sha256': weights,
        'python': sys.version, 'platform': platform.platform(),
        'torch': torch.__version__, 'transformers': transformers.__version__, 'peft': peft.__version__,
        'backend': args.backend, 'backend_version': backend_version, 'dtype': args.dtype, 'device': 'cpu', 'threads': args.threads, 'num_beams': args.beams,
        'language': args.language, 'max_new_tokens': 440, 'condition_on_previous_text': False, 'use_cache': True,
        'long_form': 'independent nonoverlapping 30-second chunks within each provided input; no VAD; source/oracle segmentation is per dataset and manifest',
        'decode': 'ffmpeg mono 16 kHz PCM16 then float32/32768',
        'load_seconds': time.perf_counter() - started,
        'scoring': 'shared wer_unicode.normalize / jiwer, single-reference micro WER and CER; punctuation removed',
        'timing_caveat': 'Exploratory concurrent-machine timing; not a comparative performance benchmark',
        'manifest': manifest, 'details': details, 'completed': False,
        'expected_requests': len(manifest['samples'][:args.limit]) * len(args.conditions),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    def persist():
        payload = {key: value for key, value in output.items() if key not in ('details', 'manifest', 'silence_controls')} if args.summary_only else output
        args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + '\n')
    with tempfile.TemporaryDirectory(prefix='gigastt-whisper-') as tmp:
        for index, sample in enumerate(manifest['samples'][:args.limit]):
            source = Path(sample['path'])
            if not source.is_absolute(): source = args.manifest.resolve().parent / source
            if sha256(source) != sample['sha256']: raise ValueError(f'Checksum mismatch: {source}')
            for condition in args.conditions:
                path = source
                if condition == 'simulated_telephone':
                    path = Path(tmp) / 'narrowband.wav'
                    subprocess.run(['ffmpeg', '-v', 'error', '-y', '-i', str(source), '-ac', '1', '-af', 'highpass=f=300,lowpass=f=3400', '-ar', '8000', '-c:a', 'pcm_mulaw', str(path)], check=True)
                raw = subprocess.check_output(['ffmpeg', '-v', 'error', '-i', str(path), '-ac', '1', '-ar', '16000', '-f', 's16le', '-'])
                audio = np.frombuffer(raw, dtype='<i2').astype(np.float32) / 32768
                before = time.perf_counter()
                texts, token_limits, chunks = [], 0, []
                for start in range(0, len(audio), 30 * 16000):
                    text, ids, components = recognize(audio[start:start + 30 * 16000])
                    texts.append(text)
                    token_limits += int(ids.shape[-1] >= 440)
                    chunks.append({'start_s': start / 16000, 'text': text, 'tokens': ids.shape[-1], 'component_seconds': components})
                hypothesis = ' '.join(texts)
                refs = {'verbatim': sample['reference'], 'normalized': sample.get('reference_normalized', sample['reference'])}
                details.append({**sample, 'condition': condition, 'input_sha256': sha256(path), 'duration_s': len(audio) / 16000, 'elapsed_s': time.perf_counter() - before, 'hypothesis': hypothesis, 'chunks': chunks, 'token_limit_chunks': token_limits, 'scores': {k: score(v, hypothesis) for k, v in refs.items()}})
                output['summary'] = summarize(details)
                output['max_rss_kib'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                persist()
                print(index + 1, sample['id'], condition, round(details[-1]['elapsed_s'], 2), flush=True)
    output['silence_controls'] = []
    for duration in args.silence_seconds:
        if not 0 < duration <= 30:
            raise ValueError('Silence duration must be in (0, 30] seconds')
        text, ids, components = recognize(np.zeros(round(duration * 16000), dtype=np.float32))
        output['silence_controls'].append({'duration_s': duration, 'hypothesis': text, 'tokens': ids.shape[-1], 'component_seconds': components})
        print('Silence control', duration, bool(text), flush=True)
    output['completed'] = True
    persist()
    print(json.dumps(output['summary'], ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
