#!/usr/bin/env python3
"""Audit available natural-speech corpora on the frozen standard large model.

One fixed primary reference per row; alternative reference and deduplication
scores are explicitly secondary and never replace the complete-corpus metric.
"""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

import soundfile as sf

from audit_multilingual_1000 import MODEL_HASHES, VOCAB_HASH, digest, require
from audit_multilingual_telephony import totals, verify_scores, verify_summary
from benchmark_multilingual_public import score
from wer_unicode import normalize

BINARY_SHA='bb30ca980374af9eb6e9fae9d239c3ce34a85e69e202bc8f8bac6c1e7ae36566'


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--directory',type=Path,required=True)
    p.add_argument('--languages',nargs='+',required=True,help='Corpus filename prefixes, e.g. kk uz kk_conversations; each corpus remains separate')
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();audits=[]
    require(len(set(args.languages))==len(args.languages),'Duplicate language request')
    overarching=json.loads((args.directory/'protocol.json').read_text())
    require(overarching['model_sha256']==MODEL_HASHES['ml_ctc_large'] and overarching['binary_sha256']==BINARY_SHA,'Unexpected overarching model')
    for corpus in args.languages:
        manifest_path=args.directory/(corpus+'_manifest.json')
        result_path=args.directory/(corpus+'_large.json')
        manifest=json.loads(manifest_path.read_text());samples=manifest['samples'];n=len(samples)
        require(n>0,'Empty corpus')
        language=manifest.get('language',samples[0]['language'])
        require(language in ('kk','ky','uz'),'Unexpected corpus language')
        require(n>0 and len({s['id'] for s in samples})==n,'Invalid/duplicate manifest IDs')
        require({s['language'] for s in samples}=={language},'Mixed language corpus')
        result=json.loads(result_path.read_text());rows=result['details']
        require(result['completed'] is True and result['n']==len(rows)==n,'Incomplete result')
        identity=result['metadata']['identity']
        require(identity['expected_samples_per_language']==n and identity['variant']=='ml_ctc_large','Wrong run population/model')
        require(identity['manifest_sha256']==digest(manifest_path) and identity['binary_sha256']==BINARY_SHA,'Input/binary identity mismatch')
        expected_model_paths={str((Path.home()/'.gigastt/models'/name).resolve()):sha for name,sha in [('multilingual_large_ctc.int8.onnx',MODEL_HASHES['ml_ctc_large']),('multilingual_vocab.txt',VOCAB_HASH)]}
        require(identity['model_sha256']==expected_model_paths,'Nonstandard model pack')
        for path,sha in expected_model_paths.items():require(digest(path)==sha,'Model/vocab bytes changed')
        for name in ('benchmark_multilingual_1000.py','benchmark_multilingual_public.py','wer_unicode.py'):
            require(identity['scorer_sha256'][name]==digest(Path(__file__).with_name(name)),'Scorer/runner changed')
        require(identity['jiwer']=='4.0.0' and identity['workers']==2,'Scoring/concurrency changed')
        hc=identity['health_configuration'];mc=identity['model_configuration']
        require(hc['variant']=='ml_ctc_large' and hc['punctuation'] is False and hc['itn'] is False and hc.get('vad') in (None,False),'Wrong health configuration')
        require(mc['variant']=='ml_ctc_large' and mc['execution_provider']=='cpu' and mc['pool_size']==2 and mc['encoder']=='int8' and mc['sample_rate']==16000 and mc['vocab_size']==71,'Wrong inference configuration')
        per_run=json.loads(result_path.with_suffix('.protocol.json').read_text())
        require(per_run['manifest_sha256']==digest(manifest_path) and per_run['n']==n and per_run['binary_sha256']==BINARY_SHA and per_run['encoder_sha256']==MODEL_HASHES['ml_ctc_large'] and per_run['vocab_sha256']==VOCAB_HASH,'Per-run protocol changed')
        require(per_run['script_sha256']==digest(Path(__file__).with_name('benchmark_conversational_languages.py')),'Wrapper changed')
        command=per_run['command'];require(digest(command[0])==BINARY_SHA,'Server binary changed')
        expected_flags={'--model-variant':'ml_ctc_large','--execution-provider':'cpu','--pool-size':'2','--encoder-intra-threads':'3','--punctuation':'off','--itn':'off'}
        for flag,value in expected_flags.items():require(flag in command and command[command.index(flag)+1]==value,'Server flags differ')
        require('--vad' not in command and '--offline' in command,'VAD/offline changed')
        require(json.loads(result_path.with_suffix('.meta.json').read_text())==result['metadata'],'Final metadata sidecar mismatch')
        ledger_bytes=result_path.with_suffix('.jsonl').read_bytes()
        require(ledger_bytes.endswith(b'\n'),'Uncommitted ledger tail')
        ledger=[json.loads(line) for line in ledger_bytes.splitlines()]
        require(len(ledger)==len({r['id'] for r in ledger})==n,'Incomplete/duplicate ledger')
        ledger_by_id={r['id']:r for r in ledger}
        for sample,row in zip(samples,rows):
            require(row==ledger_by_id.get(sample['id']),'Final/ledger/order mismatch')
            require(row['condition']=='original' and all(row[k]==sample[k] for k in ('id','path','sha256','reference','language','split','dataset')),'Input/reference changed')
            require(digest(sample['path'])==sample['sha256'],'Audio checksum mismatch')
            info=sf.info(sample['path'])
            require(abs(info.duration-row['duration_s'])<=1/info.samplerate+1e-9,'Audio duration mismatch')
            if 'duration_s' in sample:require(abs(sample['duration_s']-row['duration_s'])<=1/info.samplerate+1e-9,'Manifest duration changed')
        verify_scores(rows);verify_summary(result)
        primary=totals(rows)
        hashes=defaultdict(list)
        for row in rows:hashes[row['sha256']].append(row)
        deduplicated=[group[0] for group in hashes.values()]
        duplicates=[dict(audio_sha256=sha,ids=[r['id'] for r in group],distinct_verbatim_references=len({r['reference'] for r in group}),distinct_normalized_references=len({' '.join(normalize(r['reference'])) for r in group})) for sha,group in hashes.items() if len(group)>1]
        secondary=[]
        if len(deduplicated)!=n:
            secondary.append(dict(kind='first_row_per_audio_sha256',selection='Source order; keep first row for each exact audio hash regardless of transcript or result; not the primary metric',**totals(deduplicated)))
        secondary.append(dict(kind='digit_free',selection='Exclude supplied references containing str.isdigit characters; diagnostic only',**totals([r for r in rows if not any(c.isdigit() for c in r['reference'])])))
        normalized_pairs=[(s,r) for s,r in zip(samples,rows) if 'reference_normalized' in s]
        if normalized_pairs:
            require(len(normalized_pairs)==n,'Partial alternative reference coverage')
            alt=[{**r,'reference':s['reference_normalized'],'scores':score(s['reference_normalized'],r['hypothesis'])} for s,r in normalized_pairs]
            verify_scores(alt)
            secondary.append(dict(kind='publisher_normalized_written_reference',selection='Same unchanged hypotheses, separately scored publisher alternative for every row; never min/best of references',**totals(alt)))
        audits.append(dict(corpus=corpus,language=language,n=n,audio_s=sum(r['duration_s'] for r in rows),primary=primary,
                           primary_reference='Exact supplied manifest reference for every row; no alternate-reference selection',
                           unique_audio_sha256=len(hashes),duplicate_audio_rows=n-len(hashes),duplicate_groups=duplicates,
                           unique_verbatim_references=len({r['reference'] for r in rows}),unique_normalized_references=len({' '.join(normalize(r['reference'])) for r in rows}),
                           manifest_sha256=digest(manifest_path),result_sha256=digest(result_path),ledger_sha256=digest(result_path.with_suffix('.jsonl')),
                           all_word_distances_independently_verified=True,all_character_scores_recomputed=True,all_summary_groups_verified=True,
                           exact_ledger_final_manifest_pairing_verified=True,all_audio_model_binary_scorer_hashes_verified=True,
                           secondary=secondary))
        print('Verified',corpus,language,n,'primary WER',primary['wer_pct'],'CER',primary['cer_pct'],flush=True)
    output=dict(completed=True,evaluated_corpora=args.languages,evaluated_languages=sorted({r['language'] for r in audits}),not_evaluated_languages=sorted(set(('kk','ky','uz'))-{r['language'] for r in audits}),
                total_recognitions=sum(r['n'] for r in audits),model='standard ml_ctc_large',model_sha256=MODEL_HASHES['ml_ctc_large'],
                limitations='Corpora differ in source and speech type; no cross-language ranking or population estimate. Human/conversational provenance comes from publisher evidence, not inferred by this numeric audit. Duplicate rows remain in the primary metric. Missing languages are not scored as zero.',corpora=audits)
    args.output.write_text(json.dumps(output,ensure_ascii=False,indent=2)+'\n')


if __name__=='__main__':main()
