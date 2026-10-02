#!/usr/bin/env python3
"""Run the frozen resource experiment sequentially and retain an order ledger."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
DIRECTORY = ROOT / 'benchmark/results/multilingual_model_comparison_20261001'
HOME = Path.home()
PYTHONS = {'gigastt': HOME/'.cache/gigastt-alternative/venv312/bin/python',
           'whisper': HOME/'.cache/gigastt-alternative/venv312/bin/python',
           'omni': HOME/'.cache/gigastt-ready-comparison/omni-venv/bin/python'}
MODELS = {'whisper': HOME/'.cache/gigastt-ready-comparison/models/whisper-large-v3',
          'omni': HOME/'.cache/gigastt-ready-comparison/omni-model'}
CORPORA = ('kk_conversation','kk_codeswitch','uz_conversation','ky_read')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory',type=Path,default=DIRECTORY)
    args=parser.parse_args()
    directory=args.directory.resolve()
    directory.mkdir(parents=True,exist_ok=True)
    with (directory/'resource_order.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX | fcntl.LOCK_NB)
        ledger_path=directory/'resource_order.jsonl'
        events=[]
        if ledger_path.exists():
            data=ledger_path.read_bytes()
            if data and not data.endswith(b'\n'):
                raise ValueError('Uncommitted order ledger tail')
            events=[json.loads(line) for line in data.splitlines()]
        with ledger_path.open('a') as ledger:
            def append(event):
                ledger.write(json.dumps(event)+'\n');ledger.flush();os.fsync(ledger.fileno())
                events.append(event)
            for round_,backends in ((1,('gigastt','whisper','omni')),(2,('omni','whisper','gigastt'))):
                for backend in backends:
                    for corpus in CORPORA:
                        if backend=='whisper' and corpus=='ky_read':
                            continue
                        name=f'resource_{backend}_{corpus}_r{round_}'
                        output=directory/(name+'.json')
                        if output.exists():
                            if not any(e.get('job')==name and e.get('event')=='exit' and e.get('exit_code')==0 for e in events):
                                raise ValueError('Existing output has no successful order-ledger commit')
                            if json.loads(output.read_text()).get('completed') is not True:
                                raise ValueError('Existing measurement incomplete')
                            continue
                        command=[str(PYTHONS[backend]),str(ROOT/'scripts/measure_ready_asr.py'),
                                 '--backend',backend,'--subset',str(directory/'resource_subset.json'),
                                 '--corpus',corpus,'--round',str(round_),'--output',str(output)]
                        if backend in MODELS:
                            command+=['--model-dir',str(MODELS[backend])]
                        with (directory/(name+'.worker.log')).open('a') as log:
                            process=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
                            append(dict(event='start',job=name,pid=process.pid,epoch_s=time.time(),
                                        backend=backend,corpus=corpus,round=round_,command=command))
                            try:
                                code=process.wait()
                            except BaseException:
                                process.terminate()
                                try:process.wait(timeout=30)
                                except subprocess.TimeoutExpired:
                                    process.kill();process.wait()
                                append(dict(event='interrupted',job=name,pid=process.pid,epoch_s=time.time()))
                                raise
                        append(dict(event='exit',job=name,pid=process.pid,epoch_s=time.time(),exit_code=code))
                        if code:
                            raise RuntimeError(f'Resource job failed: {name}')
                        print(name,'completed',flush=True)


if __name__=='__main__':
    main()
