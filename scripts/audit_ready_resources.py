#!/usr/bin/env python3
"""Audit the final 22 matched resource jobs; never run inference or use preparatory jobs."""
import argparse
from collections import defaultdict
import json
import math
from pathlib import Path

import soundfile as sf

from audit_ready_asr import ROOT, DIRECTORY, verify_model, verify_pairing
from audit_multilingual_1000 import digest, require
from wer_unicode import normalize

SUBSET_SHA = '1b1e05702dfbf73dcc7d6560891a1cb6984eebde1ca081a3581b54e225c4df10'
BINARY_SHA = 'bb30ca980374af9eb6e9fae9d239c3ce34a85e69e202bc8f8bac6c1e7ae36566'
ENCODER_SHA = 'b2ad9c38fc04197ba758105d33f7404fd13d977958722e0f49e3f3e22521f1c6'
VOCAB_SHA = '4d130287892e1099fedfb3f93c4b4cf8a263151158801680b28977d1be4133f4'
CORPORA = ('kk_conversation', 'kk_codeswitch', 'uz_conversation', 'ky_read')
QUALITY_SUFFIX = dict(kk_conversation='kk_conversations_expanded', kk_codeswitch='kk',
                      uz_conversation='uz', ky_read='ky')
FIRST_ROUND = [(b,c) for b in ('gigastt','whisper','omni') for c in CORPORA
               if not (b == 'whisper' and c == 'ky_read')]
EXPECTED = [(b,c,1) for b,c in FIRST_ROUND] + [(b,c,2) for b in ('omni','whisper','gigastt')
           for c in CORPORA if not (b == 'whisper' and c == 'ky_read')]


def close(a, b):
    return math.isfinite(a) and math.isfinite(b) and math.isclose(a,b,rel_tol=1e-9,abs_tol=1e-9)


def verify_totals(result):
    rows=result['details']
    for row in rows:
        require(row['status'] in ('ok','failed'), 'Invalid resource row status')
        for key in ('duration_s','elapsed_s','cpu_s'):
            require(math.isfinite(row[key]) and row[key] >= 0, 'Invalid resource time')
        require(row['duration_s'] > 0, 'Zero audio duration')
    for field,row_key in [('audio_s','duration_s'),('wall_s','elapsed_s'),('cpu_s','cpu_s')]:
        require(close(result[field],sum(r[row_key] for r in rows)), 'Resource aggregate mismatch: '+field)
    require(result['failed'] == sum(r['status']=='failed' for r in rows), 'Failure count mismatch')
    require(close(result['rtf'],result['wall_s']/result['audio_s']), 'Wall RTF mismatch')
    require(close(result['cpu_rtf'],result['cpu_s']/result['audio_s']), 'CPU RTF mismatch')


def verify_memory(memory):
    require('measured' in memory, 'No measured-phase memory samples')
    for phase, values in memory.items():
        require(phase in ('load','warmup','measured') and values['samples'] > 0, 'Invalid memory sampling phase')
        for key in ('peak_rss_bytes','peak_pss_bytes','peak_uss_bytes','last_pss_bytes'):
            require(isinstance(values[key],int) and values[key] >= 0, 'Invalid sampled memory')
        # psutil reads USS/PSS from smaps before a separate RSS statm read.
        # These are not one atomic snapshot; statm RSS also uses approximate accounting.
        require(values['peak_uss_bytes'] <= values['peak_pss_bytes'],
                'Inconsistent same-source USS/PSS peaks')
        require(values['last_pss_bytes'] <= values['peak_pss_bytes'], 'Final PSS exceeds sampled peak')


def quality_rows(directory, backend, corpus, spec, allow_partial):
    if backend == 'gigastt':
        path=Path(spec['baseline'])
        require(digest(path)==spec['baseline_sha256'],'GigaAM baseline changed')
        data=path.read_text()
        rows=[json.loads(line) for line in data.splitlines()] if path.suffix=='.jsonl' else json.loads(data)['details']
        return rows,True
    path=directory/f'{backend}_{QUALITY_SUFFIX[corpus]}.json'
    if not path.exists():
        require(allow_partial,'Candidate quality final missing: '+str(path))
        return [],False
    final=json.loads(path.read_text())
    require(final.get('completed') is True and final['n']==len(final['details'])==spec['n'],
            'Candidate quality result incomplete')
    return final['details'],True


def verify_giga(metadata):
    require((metadata['binary_sha256'],metadata['encoder_sha256'],metadata['vocab_sha256'])
            ==(BINARY_SHA,ENCODER_SHA,VOCAB_SHA),'GigaAM resource identity changed')
    command=metadata['command']
    require(digest(Path(command[0]))==BINARY_SHA,'Frozen binary changed')
    model_dir=Path(command[command.index('--model-dir')+1])
    require(digest(model_dir/'multilingual_large_ctc.int8.onnx')==ENCODER_SHA
            and digest(model_dir/'multilingual_vocab.txt')==VOCAB_SHA,'GigaAM model bytes changed')
    for flag,value in {'--model-variant':'ml_ctc_large','--execution-provider':'cpu','--pool-size':'2',
                       '--encoder-intra-threads':'3','--punctuation':'off','--itn':'off'}.items():
        require(command[command.index(flag)+1]==value,'GigaAM resource option changed')
    require('--offline' in command and '--vad' not in command,'GigaAM offline/VAD changed')
    require(metadata['models']['pool_size']==2 and metadata['models']['execution_provider']=='cpu'
            and not metadata['health']['punctuation'] and not metadata['health']['itn'],
            'GigaAM reported resource configuration changed')


def minmax(values):
    return dict(min=min(values),max=max(values))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory',type=Path,default=DIRECTORY)
    parser.add_argument('--measurements',type=Path,help='Only final measurement directory; defaults to --directory')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--allow-partial',action='store_true')
    parser.add_argument('--whisper-model-dir',type=Path,default=Path.home()/'.cache/gigastt-ready-comparison/models/whisper-large-v3')
    parser.add_argument('--omni-model-dir',type=Path,default=Path.home()/'.cache/gigastt-ready-comparison/omni-model')
    args=parser.parse_args()
    measurements=args.measurements or args.directory
    require('resource-before-parallel-quality' not in str(measurements)
            and 'resource-harness-smoke' not in str(measurements),'Preparatory measurements forbidden')
    subset_path=args.directory/'resource_subset.json'
    require(digest(subset_path)==SUBSET_SHA,'Frozen resource subset changed')
    subset=json.loads(subset_path.read_text())
    specs={r['corpus']:r for r in json.loads((args.directory/'baseline_pairing_audit.json').read_text())['corpora']}
    selections={}
    for corpus in CORPORA:
        spec=specs[corpus];mp=Path(spec['manifest'])
        require(digest(mp)==spec['manifest_sha256'],'Corpus manifest changed')
        full=json.loads(mp.read_text())['samples']
        selected=[s for s in subset['samples'] if s['corpus']==corpus]
        warmup=[s for s in subset['warmup_samples'] if s['corpus']==corpus]
        require(selected==[dict(corpus=corpus,**s) for s in full[:5]],'Not frozen first-five selection')
        require(warmup==[dict(corpus=corpus,**full[5])],'Not frozen sixth-row warmup')
        for row in selected+warmup:
            require(digest(row['path'])==row['sha256'],'Resource source bytes changed')
        selections[corpus]=(selected,warmup[0])
    reports=[];missing=[];ordered=[];grouped=defaultdict(list)
    for backend,corpus,round_number in EXPECTED:
        path=measurements/f'resource_{backend}_{corpus}_r{round_number}.json'
        if not path.exists():
            require(args.allow_partial,'Missing final resource result: '+str(path))
            missing.append(path.name);continue
        result=json.loads(path.read_text());samples,warmup=selections[corpus]
        require(result.get('completed') is True,'Incomplete resource result')
        require((result['backend'],result['corpus'],result['round'])==(backend,corpus,round_number),'Wrong resource identity')
        require(result['subset_sha256']==SUBSET_SHA,'Resource subset hash changed')
        require(result['worker_sha256']==digest(ROOT/'scripts/measure_ready_asr.py'),'Resource worker changed')
        require(result['decoder_script_sha256']==digest(ROOT/'scripts/benchmark_ready_asr.py'),'Resource decoder changed')
        require(len(result['details'])==5,'Wrong measured population')
        verify_pairing(samples,result['details'],complete=True)
        verify_totals(result);verify_memory(result['memory'])
        require('sampled harness/process-tree' in result['scope'] and 'not an isolated' in result['scope'],
                'Memory/timing scope declaration missing')
        require(all(math.isfinite(result[k]) and result[k]>=0 for k in ('load_s','warmup_s')),
                'Invalid startup/warmup duration')
        for row in result['details']:
            info=sf.info(row['path'])
            require(abs(info.duration-row['duration_s'])<=1/info.samplerate+1e-9,'Resource duration differs from audio')
        if backend=='gigastt':
            verify_giga(result['metadata'])
        else:
            verify_model(result['metadata']['model'],backend,
                         args.whisper_model_dir if backend=='whisper' else args.omni_model_dir)
            require(result['metadata']['backend_source_sha256']==digest(ROOT/'scripts'/f'ready_asr_{backend}.py'),'Resource adapter changed')
        quality,quality_complete=quality_rows(args.directory,backend,corpus,specs[corpus],args.allow_partial)
        require(len(quality)==len({r['id'] for r in quality}),'Duplicate quality IDs')
        if backend != 'gigastt' and quality_complete:
            qfinal=json.loads((args.directory/f'{backend}_{QUALITY_SUFFIX[corpus]}.json').read_text())
            require(result['metadata']['model']==qfinal['metadata']['model'], 'Resource/quality backend configuration differs')
        by_id={r['id']:r for r in quality};parity=[]
        for row in result['details']:
            match=by_id.get(row['id'])
            if match is None:
                require(args.allow_partial,'Missing matching quality row');continue
            require(all(row[k]==match[k] for k in ('id','reference','sha256','path')),'Resource/quality input mismatch')
            parity.append(dict(id=row['id'],exact_hypothesis_match=row['hypothesis']==match['hypothesis'],
                               normalized_hypothesis_match=normalize(row['hypothesis'])==normalize(match['hypothesis']),
                               status_match=row['status']==match['status']))
        # Added execution fields make order and warmup independently reviewable.
        execution_fields={'worker_start_epoch_s','measurement_end_epoch_s','process_start_epoch_s','pid','warmup'}
        execution_verified=execution_fields.issubset(result)
        if execution_verified:
            require(result['warmup']['id']==warmup['id'] and result['warmup']['sha256']==warmup['sha256'],'Resource warmup changed')
            require(math.isfinite(result['worker_start_epoch_s']) and math.isfinite(result['measurement_end_epoch_s'])
                    and result['measurement_end_epoch_s']>=result['worker_start_epoch_s']>=result['process_start_epoch_s'] and result['pid']>0,'Invalid execution interval')
            ordered.append((result['worker_start_epoch_s'],result['measurement_end_epoch_s'],backend,corpus,round_number))
        require(args.allow_partial or execution_verified,'Execution/warmup evidence missing')
        report=dict(backend=backend,corpus=corpus,round=round_number,result_sha256=digest(path),
                    rtf=result['rtf'],cpu_rtf=result['cpu_rtf'],failed=result['failed'],
                    startup_including_validation_s=result['load_s'],warmup_s=result['warmup_s'],
                    memory=result['memory'],execution_verified=execution_verified,
                    non_atomic_memory_discrepancies=[dict(phase=phase,**values,
                        pss_minus_rss_bytes=values['peak_pss_bytes']-values['peak_rss_bytes'])
                        for phase,values in result['memory'].items()
                        if values['peak_pss_bytes']>values['peak_rss_bytes']],
                    quality_complete=quality_complete,hypothesis_parity=parity,
                    pid=result.get('pid'),worker_start_epoch_s=result.get('worker_start_epoch_s'),
                    measurement_end_epoch_s=result.get('measurement_end_epoch_s'),
                    all_five_hypotheses_exactly_match_quality=len(parity)==5 and all(r['exact_hypothesis_match'] for r in parity))
        reports.append(report);grouped[(backend,corpus)].append(report)
    ledger_path=measurements/'resource_order.jsonl'
    ledger_verified=False
    if ledger_path.exists():
        ledger_bytes=ledger_path.read_bytes()
        require(not ledger_bytes or ledger_bytes.endswith(b'\n'),'Uncommitted order ledger tail')
        events=[json.loads(line) for line in ledger_bytes.splitlines()]
        active=None;successful=[];attempts=[]
        expected_names={f'resource_{b}_{c}_r{r}' for b,c,r in EXPECTED}
        for event in events:
            require(event['job'] in expected_names,'Unknown job in order ledger')
            require(math.isfinite(event['epoch_s']),'Invalid order event timestamp')
            if event['event']=='start':
                require(active is None,'Overlapping worker starts in order ledger')
                command=event['command']
                require(Path(command[1]).resolve()==ROOT/'scripts/measure_ready_asr.py','Unexpected resource executable')
                for flag,value in {'--backend':event['backend'],'--corpus':event['corpus'],'--round':str(event['round'])}.items():
                    require(command[command.index(flag)+1]==value,'Order command/identity mismatch')
                require(Path(command[command.index('--subset')+1]).resolve()==subset_path.resolve(), 'Wrong order subset path')
                require(Path(command[command.index('--output')+1]).resolve()==(measurements/(event['job']+'.json')).resolve(), 'Wrong order output path')
                active=event
            else:
                require(event['event'] in ('exit','interrupted') and active is not None,'Unmatched order event')
                require((event['job'],event['pid'])==(active['job'],active['pid']),'Order job/PID mismatch')
                require(event['epoch_s']>=active['epoch_s'],'Invalid worker time interval')
                attempts.append(dict(start=active,end=event))
                if event['event']=='exit' and event['exit_code']==0:
                    successful.append(attempts[-1])
                active=None
        require(args.allow_partial or active is None,'Unfinished order-ledger job')
        require(len({x['end']['job'] for x in successful})==len(successful),'Duplicate successful resource jobs')
        completed_by_name={x['end']['job']:x for x in successful}
        for report in reports:
            name=f"resource_{report['backend']}_{report['corpus']}_r{report['round']}"
            event=completed_by_name.get(name)
            require(args.allow_partial or event is not None,'Result lacks successful ledger exit')
            if event is not None:
                require(event['start']['pid']==report['pid'],'Result/ledger process mismatch')
                require(event['start']['epoch_s']<=report['worker_start_epoch_s']<=report['measurement_end_epoch_s']<=event['end']['epoch_s'],
                        'Result timing falls outside orchestrator interval')
        expected_order=[f'resource_{b}_{c}_r{r}' for b,c,r in EXPECTED]
        require([x['end']['job'] for x in successful]==expected_order[:len(successful)],'Successful model/corpus order differs')
        ledger_verified=len(successful)==22 and active is None
    require(args.allow_partial or ledger_verified,'Missing/incomplete resource order ledger')
    order_verified=len(ordered)==22 and ledger_verified
    if order_verified:
        ordered.sort()
        require([(b,c,r) for _,_,b,c,r in ordered]==EXPECTED,'Two-round execution order differs')
        require(all(a[1]<=b[0] for a,b in zip(ordered,ordered[1:])),'Resource jobs overlapped')
    summaries=[]
    for (backend,corpus),rows in grouped.items():
        summaries.append(dict(backend=backend,corpus=corpus,rounds=len(rows),
                              rtf=minmax([r['rtf'] for r in rows]),cpu_rtf=minmax([r['cpu_rtf'] for r in rows]),
                              sampled_measured_peak_pss_bytes=minmax([r['memory']['measured']['peak_pss_bytes'] for r in rows]),
                              sampled_measured_peak_rss_bytes=minmax([r['memory']['measured']['peak_rss_bytes'] for r in rows]),
                              failed_requests=sum(r['failed'] for r in rows)))
    completed=len(reports)==22 and order_verified and all(r['quality_complete'] for r in reports)
    output=dict(completed=completed,partial_report=not completed,expected_jobs=22,audited_jobs=len(reports),
                missing=missing,round_order_and_nonoverlap_verified=order_verified,runs=reports,ranges=summaries,
                order_ledger_sha256=digest(ledger_path) if ledger_path.exists() and ledger_verified else None,
                orchestrator_current_sha256=digest(ROOT/'scripts/run_ready_resources.py'),
                limitations=['Only five fixed recordings per corpus per round; ranges are observed two-round ranges, not confidence intervals.',
                             'Other user workloads remain active; timing is confounded and does not establish an isolated speed ratio.',
                             'Memory is sampled harness/process-tree RSS/PSS/USS. RSS double-counts shared mappings; PSS is preferable. Short peaks may be missed.',
                             'psutil USS/PSS comes from smaps(_rollup), then RSS from statm: different non-atomic reads; statm RSS is approximate. Recorded PSS>RSS discrepancies are retained. Prefer sampled PSS; no values repaired.',
                             'GigaAM pool2 versus one candidate worker; local HTTP/adapter boundaries and model precision differ.',
                             'Startup includes validation and loading; filesystem caches are uncontrolled.',
                             'Hypothesis parity differences are reported, never used to tune or replace results.'],auditor_sha256=digest(Path(__file__)))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(output,ensure_ascii=False,indent=2)+'\n')
    require(args.allow_partial or completed,'Resource comparison incomplete')


if __name__=='__main__':
    main()
