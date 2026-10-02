#!/usr/bin/env python3
"""Prepare all pinned Uzbek Telegram evaluation rows without audio augmentation.

Publisher claims conversational voice messages and manual labels. No per-row
spontaneity or speaker annotation exists; this script does not infer one.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import urllib.request

import numpy as np
import pyarrow as pa
import soundfile as sf

from prepare_multilingual_conversation import REVISION, HASHES
from benchmark_multilingual_1000 import atomic_json
from wer_unicode import normalize

SOURCE = 'https://huggingface.co/datasets/BoburAmirov/asr_evaluate_set'


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def acquire(cache, name, expected=None):
    path=cache/name
    if not path.exists():
        temporary=path.with_suffix(path.suffix+'.partial')
        if not temporary.exists():
            urllib.request.urlretrieve(f'{SOURCE}/resolve/{REVISION}/{name}?download=true',temporary)
        if expected and digest(temporary)!=expected:raise ValueError('Downloaded checksum mismatch: '+name)
        temporary.replace(path)
    if expected and digest(path)!=expected:raise ValueError('Cached checksum mismatch: '+name)
    return path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache',type=Path,default=Path.home()/'.cache/gigastt-conversation/uz')
    p.add_argument('--output',type=Path,default=Path('benchmark/results/multilingual_conversation_20261001'))
    args=p.parse_args();cache=args.cache.resolve();cache.mkdir(parents=True,exist_ok=True);args.output.mkdir(parents=True,exist_ok=True)
    audio_dir=cache/'audio';audio_dir.mkdir(exist_ok=True)
    card=acquire(cache,'README.md');info=acquire(cache,'dataset_info.json')
    sources=[];rows=[];rates=Counter();max_cast_delta=0.;exact_source_arrays=0;source_name_counts=Counter()
    empty_reference_ids=[];empty_normalized_ids=[]
    for shard,expected in enumerate(HASHES):
        path=acquire(cache,f'data-0000{shard}-of-00002.arrow',expected)
        sources.append(dict(url=f'{SOURCE}/resolve/{REVISION}/{path.name}',sha256=expected,bytes=path.stat().st_size))
        with pa.memory_map(str(path),'r') as mapped:
            stream=pa.ipc.open_stream(mapped)
            for batch in stream:
                for offset in range(batch.num_rows):
                    source=batch.slice(offset,1).to_pylist()[0]
                    index=len(rows);identifier=f'uz_telegram_{index:04d}'
                    reference=source['transcript'];audio=source['audio'];rate=audio['sampling_rate']
                    original=np.asarray(audio['array'],dtype=np.float64)
                    if original.ndim!=1 or not len(original) or not np.isfinite(original).all():raise ValueError('Invalid source audio: '+identifier)
                    if not isinstance(reference,str) or not isinstance(rate,int) or rate<=0:raise ValueError('Invalid row fields')
                    waveform=original.astype(np.float32)
                    if not np.isfinite(waveform).all():raise ValueError('Float32 overflow')
                    delta=float(np.max(np.abs(original-waveform.astype(np.float64))))
                    max_cast_delta=max(max_cast_delta,delta);exact_source_arrays+=delta==0
                    destination=audio_dir/(identifier+'.wav')
                    sf.write(destination,waveform,rate,subtype='FLOAT')
                    decoded,decoded_rate=sf.read(destination,dtype='float32')
                    if decoded_rate!=rate or not np.array_equal(decoded,waveform):raise ValueError('Float WAV roundtrip changed source samples')
                    rates[rate]+=1;source_name_counts[source['name']]+=1
                    if not reference.strip():empty_reference_ids.append(identifier)
                    if not normalize(reference):empty_normalized_ids.append(identifier)
                    rows.append(dict(id=identifier,path=str(destination),sha256=digest(destination),reference=reference,
                                     dataset='uzbek_telegram_publisher_conversational',language='uz',split='train',
                                     duration_s=len(original)/rate,row_index=index,source_shard=shard,source_name=source['name'],
                                     source_sample_rate=rate,num_samples=len(original),
                                     source_float64_pcm_sha256=hashlib.sha256(original.astype('<f8').tobytes()).hexdigest(),
                                     decoded_float32_pcm_sha256=hashlib.sha256(waveform.astype('<f4').tobytes()).hexdigest(),
                                     storage_subtype='FLOAT',float64_to_float32_max_absolute_difference=delta))
        print('Extracted',len(rows),'rows',flush=True)
    if len(rows)!=745 or len({r['id'] for r in rows})!=745:raise ValueError('Expected all745unique row IDs')
    audio_counts=Counter(r['sha256'] for r in rows);pcm_counts=Counter(r['decoded_float32_pcm_sha256'] for r in rows)
    ref_counts=Counter(r['reference'] for r in rows);norm_counts=Counter(' '.join(normalize(r['reference'])) for r in rows)
    provenance=dict(source=SOURCE,revision=REVISION,license='Apache-2.0 per pinned publisher README',
                    publisher_card_url=f'{SOURCE}/blob/{REVISION}/README.md',publisher_card_sha256=digest(card),
                    dataset_info_sha256=digest(info),arrow_sources=sources,
                    publisher_claims=dict(domain='Voice messages from open Telegram groups; conversational speech',annotations='Manually annotated, human-verified ground truth',speakers='Multiple speakers; no identities or speaker-disjoint split supplied'),
                    verified_structure=dict(count=745,columns=['name','transcript','audio'],published_split='train',
                                            card_schema_discrepancy='README calls the text column transcription; actual Arrow/dataset_info uses transcript.',
                                            sample_rates=dict(rates),audio_seconds=sum(r['duration_s'] for r in rows),
                                            unique_source_names=len(source_name_counts),unique_audio_files=len(audio_counts),unique_pcm_arrays=len(pcm_counts),
                                            unique_verbatim_references=len(ref_counts),unique_normalized_references=len(norm_counts),
                                            duplicate_audio_rows=sum(n-1 for n in audio_counts.values()),duplicate_pcm_rows=sum(n-1 for n in pcm_counts.values()),
                                            empty_reference_ids=empty_reference_ids,empty_normalized_reference_ids=empty_normalized_ids),
                    extraction=dict(audio='Original sample-rate mono arrays serialized as IEEE float32 WAV; no filtering, gain, resampling, segmentation, narrowband degradation or integer PCM quantization.',
                                    source_array_dtype='float64',native_decoder_dtype='float32',float32_roundtrip_exact=True,
                                    source_arrays_exactly_representable_float32=exact_source_arrays,max_float64_to_float32_absolute_difference=max_cast_delta),
                    selection='All745published rows in original shard/row order; no exclusions, deduplication, replacement, model-result selection or training.',
                    limitations='Conversational provenance and manual annotation are publisher claims, not independent sample-level validation. No per-row read/spontaneous/commercial labels, speaker IDs, telephone metadata or blind-holdout guarantee. Do not call745rows745independent speakers or conversations. Prior100-row pilot usedPCM16 serialization and is not asserted bit-identical.',
                    script_sha256=digest(__file__))
    atomic_json(args.output/'uz_manifest.json',{**provenance,'samples':rows})
    atomic_json(args.output/'uz_provenance.json',provenance)
    print(json.dumps(provenance['verified_structure'],ensure_ascii=False),flush=True)


if __name__=='__main__':main()
