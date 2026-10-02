#!/usr/bin/env python3
"""Measure frozen small resource subsets in one fresh process per model/corpus/round."""
import argparse
import importlib
import json
import os
from pathlib import Path
import platform
import subprocess
import threading
import time
import urllib.error
import urllib.request

import psutil

from benchmark_ready_asr import decode
from benchmark_multilingual_1000 import atomic_json
from benchmark_multilingual_public import sha256
from benchmark_conversational_languages import BINARY, BINARY_SHA, MODELS, MODEL_SHA, VOCAB_SHA


class Monitor:
    def __init__(self):
        self.root = psutil.Process()
        self.phase = 'load'
        self.rows = []
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self.poll, daemon=True)

    def poll(self):
        while not self.stop.is_set():
            rss = pss = uss = 0
            processes = [self.root] + self.root.children(recursive=True)
            for process in processes:
                try:
                    memory = process.memory_full_info()
                    rss += memory.rss
                    pss += getattr(memory, 'pss', 0)
                    uss += getattr(memory, 'uss', 0)
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
            self.rows.append(dict(t=time.monotonic(), phase=self.phase,
                                  rss_bytes=rss, pss_bytes=pss, uss_bytes=uss))
            self.stop.wait(.1)

    def finish(self):
        self.stop.set()
        self.thread.join()
        return {phase:dict(samples=len(rows), peak_rss_bytes=max(r['rss_bytes'] for r in rows),
                            peak_pss_bytes=max(r['pss_bytes'] for r in rows),
                            peak_uss_bytes=max(r['uss_bytes'] for r in rows),
                            last_pss_bytes=rows[-1]['pss_bytes'])
                for phase in ('load','warmup','measured')
                if (rows := [r for r in self.rows if r['phase'] == phase])}


def cpu_seconds():
    # os.times includes waited-for FFmpeg children; live server CPU is added separately.
    values = os.times()
    return values.user + values.system + values.children_user + values.children_system


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backend', required=True, choices=['gigastt','whisper','omni'])
    parser.add_argument('--model-dir', type=Path)
    parser.add_argument('--subset', type=Path, required=True)
    parser.add_argument('--corpus', required=True)
    parser.add_argument('--round', type=int, required=True, choices=[1,2])
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--port', type=int, default=20041)
    args = parser.parse_args()
    process_start_epoch_s = psutil.Process().create_time()
    worker_start_epoch_s = time.time()
    if args.output.exists():
        raise ValueError('Existing measurement must not be overwritten')
    if sha256(args.subset) != '1b1e05702dfbf73dcc7d6560891a1cb6984eebde1ca081a3581b54e225c4df10':
        raise ValueError('Resource subset differs from frozen selection')
    subset = json.loads(args.subset.read_text())
    samples = [r for r in subset['samples'] if r['corpus'] == args.corpus]
    warmups = [r for r in subset['warmup_samples'] if r['corpus'] == args.corpus]
    if len(samples) != 5 or len(warmups) != 1:
        raise ValueError('Expected frozen five measured rows and sixth warmup')
    for row in samples + warmups:
        if sha256(Path(row['path'])) != row['sha256']:
            raise ValueError('Resource input identity changed')
    os.environ['OMP_NUM_THREADS'] = '3'
    os.environ['MKL_NUM_THREADS'] = '3'
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    args.output.parent.mkdir(parents=True, exist_ok=True)
    monitor = Monitor()
    monitor.thread.start()
    process = None
    log = None
    metadata = {}
    before_load = time.monotonic()
    try:
        if args.backend == 'gigastt':
            if (sha256(BINARY), sha256(MODELS/'multilingual_large_ctc.int8.onnx'),
                    sha256(MODELS/'multilingual_vocab.txt')) != (BINARY_SHA, MODEL_SHA, VOCAB_SHA):
                raise ValueError('GigaAM frozen identity mismatch')
            # A bind preflight prevents accidentally probing another server.
            import socket
            with socket.socket() as sock:
                sock.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
                sock.bind(('127.0.0.1',args.port))
            command = [str(BINARY),'--offline','serve','--model-dir',str(MODELS),
                       '--model-variant','ml_ctc_large','--execution-provider','cpu','--pool-size','2',
                       '--encoder-intra-threads','3','--punctuation','off','--itn','off','--port',str(args.port)]
            env = {k:v for k,v in os.environ.items() if not k.startswith('GIGASTT_')}
            env['RUST_LOG'] = 'warn'
            log = args.output.with_suffix('.server.log').open('w')
            process = subprocess.Popen(command,env=env,stdout=log,stderr=subprocess.STDOUT)
            url = f'http://127.0.0.1:{args.port}'
            deadline = time.monotonic()+180
            while True:
                if process.poll() is not None:
                    raise RuntimeError('GigaAM server exited before readiness')
                try:
                    with urllib.request.urlopen(url+'/v1/models',timeout=5) as response:
                        models = json.load(response)
                    with urllib.request.urlopen(url+'/health',timeout=5) as response:
                        health = json.load(response)
                    break
                except (urllib.error.URLError,TimeoutError):
                    if time.monotonic()>deadline:
                        raise TimeoutError('Server readiness timeout')
                    time.sleep(.1)
            if (models['variant']!='ml_ctc_large' or models['pool_size']!=2
                    or models['execution_provider']!='cpu' or health.get('punctuation') or health.get('itn')):
                raise ValueError('Server configuration differs')
            metadata = dict(command=command, binary_sha256=BINARY_SHA, encoder_sha256=MODEL_SHA,
                            vocab_sha256=VOCAB_SHA, models=models, health=health)
            def transcribe(row):
                request = urllib.request.Request(url+'/v1/transcribe',data=Path(row['path']).read_bytes(),
                           headers={'Content-Type':'application/octet-stream'},method='POST')
                with urllib.request.urlopen(request,timeout=600) as response:
                    return json.load(response)['text']
            server_process = psutil.Process(process.pid)
            def cpu():
                values=server_process.cpu_times()
                return cpu_seconds()+values.user+values.system
        else:
            backend=importlib.import_module('ready_asr_'+args.backend)
            model=backend.ReadyASR(args.model_dir,samples[0]['language'],threads=3)
            metadata=dict(model=model.metadata,backend_source_sha256=sha256(Path(backend.__file__)))
            def transcribe(row):
                return model.transcribe(decode(Path(row['path'])))['hypothesis']
            cpu=cpu_seconds
        load_s=time.monotonic()-before_load
        monitor.phase='warmup'
        before=time.monotonic()
        transcribe(warmups[0])
        warmup_s=time.monotonic()-before
        monitor.phase='measured'
        rows=[]
        for sample in samples:
            cpu_before=cpu()
            before=time.monotonic()
            status,error,hypothesis='ok',None,''
            try:
                hypothesis=transcribe(sample)
            except Exception as exc:
                status,error='failed',f'{type(exc).__name__}: {exc}'
            rows.append(dict(sample,status=status,error=error,hypothesis=hypothesis,
                             elapsed_s=time.monotonic()-before,cpu_s=cpu()-cpu_before))
        memory=monitor.finish()
        audio=sum(r['duration_s'] for r in rows)
        payload=dict(completed=True,backend=args.backend,corpus=args.corpus,round=args.round,
                     pid=os.getpid(),process_start_epoch_s=process_start_epoch_s,
                     worker_start_epoch_s=worker_start_epoch_s,measurement_end_epoch_s=time.time(),
                     warmup=dict(id=warmups[0]['id'],sha256=warmups[0]['sha256']),
                     subset_sha256=sha256(args.subset),worker_sha256=sha256(Path(__file__)),
                     decoder_script_sha256=sha256(Path(__file__).with_name('benchmark_ready_asr.py')),
                     metadata=metadata,platform=platform.platform(),load_s=load_s,warmup_s=warmup_s,
                     memory=memory,details=rows,audio_s=audio,wall_s=sum(r['elapsed_s'] for r in rows),
                     cpu_s=sum(r['cpu_s'] for r in rows),failed=sum(r['status']=='failed' for r in rows),
                     load_average=os.getloadavg(),
                     scope='Fresh process per model/corpus/round; ready-to-final-text includes decoding, preprocessing and inference. GigaAM additionally includes local HTTP. Memory is sampled harness/process-tree RSS/PSS/USS, not model weights alone. Other user workloads may contend; this is not an isolated hardware benchmark.')
        payload['rtf']=payload['wall_s']/audio
        payload['cpu_rtf']=payload['cpu_s']/audio
        atomic_json(args.output,payload)
        print(args.backend,args.corpus,args.round,'RTF',payload['rtf'],flush=True)
    finally:
        if monitor.thread.is_alive():
            monitor.finish()
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        if log is not None:
            log.close()


if __name__=='__main__':
    main()
