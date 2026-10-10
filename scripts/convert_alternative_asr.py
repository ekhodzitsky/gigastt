#!/usr/bin/env python3
"""Merge pinned Kazakh/Russian Whisper LoRA and export an FP32 CTranslate2 lab model."""
import argparse
import json
from pathlib import Path
import tempfile

from benchmark_multilingual_public import sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--models', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    import torch
    import ctranslate2
    from peft import PeftModel
    from transformers import WhisperForConditionalGeneration, WhisperProcessor
    torch.set_num_threads(1)
    weights = {'base/model.safetensors': '95235871856d690c02c663d83d7099e89650877fcf57b48e0d1e5e4a278fecdf',
               'adapter/adapter_model.safetensors': 'f49c02568e5d37300f736ef2a325f9f6b356936d4378e2a8d842a09c8194dc23'}
    for path, expected in weights.items():
        if sha256(args.models / path) != expected:
            raise ValueError(f'Unexpected model weights: {path}')
    model = WhisperForConditionalGeneration.from_pretrained(args.models / 'base', local_files_only=True, dtype=torch.float32)
    model = PeftModel.from_pretrained(model, args.models / 'adapter', local_files_only=True).merge_and_unload().eval()
    print('Merged pinned model', flush=True)
    with tempfile.TemporaryDirectory(prefix='merged-', dir=args.models) as temporary:
        model.save_pretrained(temporary, safe_serialization=True)
        WhisperProcessor.from_pretrained(args.models / 'base', local_files_only=True).save_pretrained(temporary)
        del model
        converter = ctranslate2.converters.TransformersConverter(temporary, copy_files=['tokenizer.json', 'preprocessor_config.json'])
        converter.convert(str(args.output), quantization='float32')
    manifest = {'source_weights_sha256': weights, 'ctranslate2': ctranslate2.__version__,
                'conversion': 'Merge FP32 base and LoRA with peft, export FP32; no quantization at export',
                'files_sha256': {p.name: sha256(p) for p in args.output.iterdir() if p.is_file()}}
    (args.output / 'conversion.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print('Converted', args.output, flush=True)


if __name__ == '__main__':
    main()
