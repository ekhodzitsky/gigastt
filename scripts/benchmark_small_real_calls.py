#!/usr/bin/env python3
"""Compare fixed model packs on public call demonstrations; keep text private."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import json
import hashlib
import os
from pathlib import Path
import socket
import subprocess
import time
import urllib.error
import urllib.request

from benchmark_multilingual_1000 import atomic_json, duration
from benchmark_multilingual_public import score, sha256
from audit_multilingual_telephony import totals, verify_scores
from wer_unicode import normalize

WORK = Path.home() / '.cache/gigastt-real-calls'
REPORT = Path('benchmark/results/multilingual_real_calls_20261001')
BINARY = Path.home() / '.cache/gigastt-multilingual-1000/gigastt-candidate'
BINARY_SHA = 'bb30ca980374af9eb6e9fae9d239c3ce34a85e69e202bc8f8bac6c1e7ae36566'
MODEL_DIR = Path.home() / '.gigastt/models'
MODELS = {
    'original_small': ('ml_ctc', MODEL_DIR, MODEL_DIR / 'multilingual_ctc.int8.onnx', 'e08e27ae5669b39f0c378fae101bbbb9a80505f74f9b66719c309bf5b894a480'),
    'candidate_small': ('ml_ctc', Path.home() / '.cache/gigastt-small-quantization/selected_product_pack', Path.home() / '.cache/gigastt-small-quantization/selected_product_pack/lab_multilingual_ctc.int8.onnx', '18fa5fab6ece0123d7813f7abc8da129530f6d1a7da17c85c4b3948f27b8e3e0'),
    'large': ('ml_ctc_large', MODEL_DIR, MODEL_DIR / 'multilingual_large_ctc.int8.onnx', 'b2ad9c38fc04197ba758105d33f7404fd13d977958722e0f49e3f3e22521f1c6'),
}


def get(url):
    with urllib.request.urlopen(url, timeout=10) as response:
        return json.load(response)


def prepare():
    sources = [
        ('babel', 'full_channel', Path.home() / '.cache/gigastt-official-telephony/babel/telephone_manifest.json'),
        ('material', 'full_channel', Path.home() / '.cache/gigastt-official-telephony/material/telephone_expansion_manifest.json'),
        ('babel', 'oracle_segments', WORK / 'babel_oracle/manifest.json'),
        ('material', 'oracle_segments', Path.home() / '.cache/gigastt-official-telephony/material/telephone_segments_manifest.json'),
    ]
    rows = []
    provenance = []
    for recording, group, path in sources:
        manifest = json.loads(path.read_text())
        provenance.append(dict(recording=recording, group=group, manifest_sha256=sha256(path), n=len(manifest['samples'])))
        for sample in manifest['samples']:
            audio = Path(sample['path'])
            if sha256(audio) != sample['sha256']:
                raise ValueError('Source audio mismatch')
            rows.append(dict(id=recording+'_'+group+'_'+sample['id'], recording=recording, group=group,
                             path=str(audio), audio_sha256=sample['sha256'], reference=sample['reference'],
                             reference_sha256=hashlib.sha256(sample['reference'].encode()).hexdigest(),
                             duration_s=duration(audio)))
    if len(rows) != 51 or len({r['id'] for r in rows}) != 51:
        raise ValueError('Expected three full channels and 48 oracle clips')
    identities = {}
    for name, (variant, directory, encoder, expected) in MODELS.items():
        if sha256(encoder) != expected:
            raise ValueError('Model changed')
        identities[name] = dict(variant=variant, model_dir=str(directory), encoder_sha256=expected,
                                vocab_sha256=sha256(directory/'multilingual_vocab.txt'))
    if sha256(BINARY) != BINARY_SHA:
        raise ValueError('Binary changed')
    private = {'rows': rows}
    private_path = WORK / 'cases.json'
    if private_path.exists() and json.loads(private_path.read_text()) != private:
        raise ValueError('Refusing to change frozen cases')
    atomic_json(private_path, private)
    protocol = dict(status='frozen_before_inference', unique_real_recordings=2, full_channels=3,
                    oracle_excerpts=48, inputs_per_model=51, measured_requests=153,
                    binary_sha256=BINARY_SHA, models=identities, cases_sha256=sha256(private_path),
                    manifests=provenance, source_urls=['https://catalog.ldc.upenn.edu/LDC2018S13','https://catalog.ldc.upenn.edu/LDC2025S03'],
                    inputs=[{k:v for k,v in r.items() if k not in ('reference','path')} for r in rows],
                    configuration=dict(pool_size=2,encoder_intra_threads=3,http_workers=2,punctuation=False,itn=False,vad=False,execution_provider='cpu'),
                    conditions='Original prepared input files only; no resampling/preprocessing interventions. Fixed existing reference-timestamp excerpts are separate oracle diagnostics.',
                    limitations='Two repeatedly inspected Kazakh conversations, three channels; not representative or certified independent speakers; not Uzbek real-call evidence. MATERIAL references cover only the retained first56.14s/channel. Oracle clips are not extra calls. No training, source-model API, gated data, purchase or outreach.',
                    privacy='Restricted audio, references and hypotheses remain in local cache; public results contain metrics and hashes only.',
                    scorer_sha256={n:sha256(Path(__file__).with_name(n)) for n in ('benchmark_multilingual_public.py','wer_unicode.py')})
    protocol_path = REPORT / 'protocol.json'
    if protocol_path.exists() and json.loads(protocol_path.read_text()) != protocol:
        raise ValueError('Refusing to change frozen protocol')
    atomic_json(protocol_path, protocol)
    print('Frozen 3 full channels +48 oracle excerpts ×3 models', flush=True)


@contextmanager
def server(name, port, vad=False):
    variant, directory, encoder, expected = MODELS[name]
    if sha256(BINARY) != BINARY_SHA or sha256(encoder) != expected:
        raise ValueError('Server/model identity changed')
    with socket.socket() as check:
        check.bind(('127.0.0.1', port))
    command = [str(BINARY),'--offline','serve','--host','127.0.0.1','--port',str(port),'--model-dir',str(directory),
               '--model-variant',variant,'--execution-provider','cpu','--pool-size','2','--encoder-intra-threads','3','--punctuation','off','--itn','off']
    if vad:
        command += ['--vad','--vad-model-dir',str(MODEL_DIR/'vad')]
    env = {k:v for k,v in os.environ.items() if not k.startswith('GIGASTT_')}
    env['RUST_LOG']='gigastt=info'
    with (WORK/(name+('_vad' if vad else '')+'_server.log')).open('w') as log:
        process = subprocess.Popen(command,env=env,stdout=log,stderr=subprocess.STDOUT)
        try:
            url='http://127.0.0.1:'+str(port)
            deadline=time.monotonic()+180
            while True:
                if process.poll() is not None:
                    raise RuntimeError('Server exited during startup')
                try:
                    health, model=get(url+'/health'),get(url+'/v1/models')
                    break
                except (urllib.error.URLError,TimeoutError):
                    if time.monotonic()>deadline:raise TimeoutError('Server readiness timeout')
                    time.sleep(.5)
            if health['variant']!=variant or health['punctuation'] or health['itn'] or model['pool_size']!=2 or model['execution_provider']!='cpu':
                raise ValueError('Wrong server configuration')
            if vad and 'VAD enabled (threshold 0.5, min_silence 500ms)' not in (WORK/(name+'_vad_server.log')).read_text():
                raise ValueError('VAD attachment not verified')
            yield url,dict(command=command,health=health,model_info=model)
        finally:
            if process.poll() is None:
                process.terminate()
                try:process.wait(timeout=30)
                except subprocess.TimeoutExpired:process.kill();process.wait()


def run(name, vad=False):
    protocol_path=REPORT/('vad_protocol.json' if vad else 'protocol.json')
    output_name=name+('_vad' if vad else '')
    protocol=json.loads(protocol_path.read_text())
    cases=WORK/'cases.json'
    if sha256(cases)!=protocol['cases_sha256']:raise ValueError('Cases changed')
    rows=json.loads(cases.read_text())['rows']
    if vad:rows=[r for r in rows if r['group']=='full_channel']
    expected_count=len(rows)
    results=[]
    with server(name,(19944 if vad else 19941) + list(MODELS).index(name),vad=vad) as (url,server_info):
        def request(row):
            audio=Path(row['path'])
            if sha256(audio)!=row['audio_sha256']:raise ValueError('Audio changed')
            start=time.monotonic();text='';error=None
            try:
                req=urllib.request.Request(url+'/v1/transcribe',data=audio.read_bytes(),headers={'Content-Type':'application/octet-stream'},method='POST')
                with urllib.request.urlopen(req,timeout=600) as response:text=json.load(response)['text']
                if not isinstance(text,str):raise ValueError('Invalid text')
            except Exception as exc:error=type(exc).__name__+': '+str(exc);text=''
            return {**row,'hypothesis':text,'scores':score(row['reference'],text),'elapsed_s':time.monotonic()-start,'status':'failed' if error else 'ok','error':error}
        warm=request(rows[0])
        if warm['status']!='ok':raise RuntimeError('Warmup failed')
        metadata=dict(model=name,model_identity=protocol['models'][name],binary_sha256=BINARY_SHA,protocol_sha256=sha256(protocol_path),script_sha256=sha256(__file__),server=server_info,warmup_excluded=True,timing_caveat='Concurrent host workload; not a resource benchmark')
        with ThreadPoolExecutor(max_workers=2) as pool:
            for row in pool.map(request,rows):
                results.append(row)
                atomic_json(WORK/(output_name+'_results.json'),dict(metadata=metadata,complete=len(results)==expected_count,rows=results))
                print(output_name,len(results),'/',expected_count,row['status'],flush=True)
    verify_scores(results)
    groups=[]
    for group in (('full_channel',) if vad else ('full_channel','oracle_segments')):
        for recording in ('babel','material','all'):
            selected=[r for r in results if r['group']==group and (recording=='all' or r['recording']==recording)]
            groups.append(dict(group=group,recording=recording,**totals(selected)))
    public=[{k:v for k,v in r.items() if k not in ('path','reference','hypothesis','error')} for r in results]
    if vad:
        log_path=WORK/(name+'_vad_server.log')
        log_text=log_path.read_text()
        fallback_lines=[line for line in log_text.splitlines() if 'request{' in line and 'VAD' in line and ('fallback' in line.lower() or 'falling back' in line.lower() or 'failed' in line.lower())]
        metadata['vad_evidence']=dict(command_enabled=True,attached_engines=log_text.count('VAD enabled (threshold 0.5, min_silence 500ms)'),fallback_warning_count=len(fallback_lines),log_sha256=sha256(log_path),health_exposes_vad=False)
        if fallback_lines:raise ValueError('VAD fallback occurred; inspect private log before interpreting results')
    atomic_json(REPORT/(output_name+'.json'),dict(metadata=metadata,complete=True,n=expected_count,groups=groups,rows=public))
    print(name,'complete',flush=True)


def prepare_vad():
    model = MODEL_DIR / 'vad/silero_vad.onnx'
    expected = '2623a2953f6ff3d2c1e61740c6cdb7168133479b267dfef114a4a3cc5bdd788f'
    if sha256(model) != expected:raise ValueError('Cached VAD model mismatch')
    base = json.loads((REPORT/'protocol.json').read_text())
    protocol = {**base, 'status':'frozen_before_vad_inference', 'inputs_per_model':3,'measured_requests':9,'oracle_excerpts':0,
                'inputs':[r for r in base['inputs'] if r['group']=='full_channel'],
                'configuration':{**base['configuration'],'vad':True,'vad_threshold':0.5,'vad_min_silence_ms':500},
                'vad_model_sha256':expected,'vad_model_source':'https://github.com/snakers4/silero-vad/tree/v5.1.2',
                'conditions':'Separate operational VAD control: same three full channels, existing Silero defaults; no tuning or oracle clips.',
                'vad_verification':'Frozen health/model endpoints omit VAD; require explicit flag, attachment logs and absence of fallback warnings.'}
    path = REPORT/'vad_protocol.json'
    if path.exists() and json.loads(path.read_text()) != protocol:raise ValueError('VAD protocol changed')
    atomic_json(path,protocol)
    print('Frozen nine full-channel VAD controls',flush=True)


def audit():
    base = json.loads((REPORT/'protocol.json').read_text())
    cases_path = WORK/'cases.json'
    if sha256(cases_path) != base['cases_sha256']:raise ValueError('Frozen cases changed')
    all_cases = json.loads(cases_path.read_text())['rows']
    for row in all_cases:
        if sha256(row['path']) != row['audio_sha256']:raise ValueError('Audio changed')
    checks=[]
    include_vad=all((REPORT/(name+'_vad.json')).exists() for name in MODELS)
    for vad in ([False,True] if include_vad else [False]):
        protocol_path=REPORT/('vad_protocol.json' if vad else 'protocol.json')
        protocol=json.loads(protocol_path.read_text())
        cases=[r for r in all_cases if not vad or r['group']=='full_channel']
        for name, (_,directory,encoder,expected_sha) in MODELS.items():
            if sha256(BINARY)!=BINARY_SHA or sha256(encoder)!=expected_sha:raise ValueError('Model/binary changed')
            if sha256(directory/'multilingual_vocab.txt')!=protocol['models'][name]['vocab_sha256']:raise ValueError('Vocab changed')
            output_name=name+('_vad' if vad else '')
            private=json.loads((WORK/(output_name+'_results.json')).read_text())
            public=json.loads((REPORT/(output_name+'.json')).read_text())
            if not private['complete'] or not public['complete'] or public['n']!=len(cases):raise ValueError('Incomplete results')
            if len(private['rows'])!=len(cases) or len(public['rows'])!=len(cases):raise ValueError('Wrong result counts')
            public_metadata={k:v for k,v in public['metadata'].items() if k!='vad_evidence'}
            if private['metadata']!=public_metadata or public_metadata['protocol_sha256']!=sha256(protocol_path):raise ValueError('Metadata changed')
            if public_metadata['model_identity']!=protocol['models'][name]:raise ValueError('Model identity changed')
            for actual,published,case in zip(private['rows'],public['rows'],cases):
                if any(actual[k]!=value for k,value in case.items()):raise ValueError('Case identity changed')
                if {k:v for k,v in actual.items() if k not in ('path','reference','hypothesis','error')}!=published:raise ValueError('Public/private mismatch')
            verify_scores(private['rows'])
            for summary in public['groups']:
                selected=[r for r in private['rows'] if r['group']==summary['group'] and (summary['recording']=='all' or r['recording']==summary['recording'])]
                if any(summary[k]!=value for k,value in totals(selected).items()):raise ValueError('Summary mismatch')
            if vad:
                if sha256(MODEL_DIR/'vad/silero_vad.onnx')!=protocol['vad_model_sha256']:raise ValueError('VAD model changed')
                log=WORK/(name+'_vad_server.log');text=log.read_text();evidence=public['metadata']['vad_evidence']
                if sha256(log)!=evidence['log_sha256'] or '--vad' not in public_metadata['server']['command']:raise ValueError('VAD provenance mismatch')
                if 'VAD enabled (threshold 0.5, min_silence 500ms)' not in text:raise ValueError('VAD attachment missing')
                requests=[line for line in text.splitlines() if 'request{' in line]
                if sum('transcribe complete (streaming windows, vad)' in line for line in requests)!=4:raise ValueError('Actual VAD use not confirmed for3calls+warmup')
                if any('VAD' in line and ('falling back' in line.lower() or 'failed' in line.lower()) for line in requests):raise ValueError('Request VAD fallback')
            checks.append(dict(model=name,vad=vad,n=len(cases),failed_requests=sum(r['status']=='failed' for r in private['rows']),result_sha256=sha256(REPORT/(output_name+'.json')),all_word_distances_independently_verified=True,all_character_scores_recomputed=True,reference_and_input_identity_verified=True,public_restricted_text_removed=True))
    atomic_json(REPORT/'validation.json',dict(completed=True,measured_requests=sum(r['n'] for r in checks),unique_real_recordings=2,primary_channel_requests=9,oracle_excerpt_requests=144,vad_channel_requests=9 if include_vad else 0,checks=checks,limitations='Saved hypotheses independently rescored with word DP; inference not rerun. Two repeatedly inspected recordings; no representative-call claim. VAD startup synthetic-silence fallback warnings excluded; actualrequests verified through VAD completion logs.'))
    print('Validated allresults; restricted text remains private',flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',choices=('prepare','run','audit','prepare-vad','run-vad'),required=True)
    p.add_argument('--model',choices=list(MODELS))
    args=p.parse_args();WORK.mkdir(parents=True,exist_ok=True);REPORT.mkdir(parents=True,exist_ok=True)
    if args.mode=='prepare':prepare()
    elif args.mode=='audit':audit()
    elif args.mode=='prepare-vad':prepare_vad()
    else:
        if args.model is None:p.error('--model required for run')
        run(args.model,vad=args.mode=='run-vad')


if __name__=='__main__':main()
