#!/usr/bin/env python3
"""Synthetic local HTTP regression smoke; run with the benchmark Python environment."""
import io,json,pathlib,sys,subprocess,threading,time,wave,hashlib,unittest,tempfile
from http.server import ThreadingHTTPServer,BaseHTTPRequestHandler
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parent))
import benchmark_multilingual_1000 as runner

class BenchmarkResumeSmoke(unittest.TestCase):
    def test_http_interruption_resume_and_identity(self):
        temporary=tempfile.TemporaryDirectory(prefix='gigastt-benchmark1000-smoke-')
        self.addCleanup(temporary.cleanup)
        root=pathlib.Path(temporary.name)
        for name in ['result.json','result.meta.json','result.jsonl','result.lock']:
         (root/name).unlink(missing_ok=True)
        refs=['hello world','hello world','hello 2','қазақ','hello world'];samples=[]
        for i in range(5):
         p=root/f'{i}.wav'
         with wave.open(str(p),'wb') as w:
          w.setnchannels(1);w.setsampwidth(2);w.setframerate(8000);w.writeframes((i+1).to_bytes(2,'little')*8000*(31 if i==1 else 1))
         samples.append({'id':f'kk_test_{i}','path':str(p),'reference':refs[i],'dataset':'synthetic','language':'kk','split':'test','sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
        samples[1]['condition']='telephone_alaw'
        (root/'manifest.json').write_text(json.dumps({'samples':samples}));(root/'binary').write_bytes(b'binary');(root/'model').write_bytes(b'model')
        requests=[]
        class Handler(BaseHTTPRequestHandler):
         def log_message(self,*args):pass
         def do_GET(self):
          x={'variant':'ml_ctc','punctuation':False,'itn':False}
          if self.path=='/v1/models':x.update(execution_provider='cpu',encoder='int8',pool_size=2,sample_rate=16000,version='test',vocab_size=71)
          b=json.dumps(x).encode();self.send_response(200);self.end_headers();self.wfile.write(b)
         def do_POST(self):
          body=self.rfile.read(int(self.headers['Content-Length']))
          with wave.open(io.BytesIO(body),'rb') as w:i=int.from_bytes(w.readframes(1),'little')-1
          requests.append(i)
          if i==2:time.sleep(.8)
          self.send_response(500 if i==4 else 200);self.end_headers()
          try:self.wfile.write(json.dumps({'text':refs[i]}).encode())
          except BrokenPipeError:pass
        server=ThreadingHTTPServer(('127.0.0.1',0),Handler);threading.Thread(target=server.serve_forever,daemon=True).start()
        kwargs={'manifest':str(root/'manifest.json'),'output':str(root/'result.json'),'binary':str(root/'binary'),'model':[str(root/'model')],'variant':'ml_ctc','url':f'http://127.0.0.1:{server.server_port}','workers':1,'timeout':5,'fsync':True}
        code="import sys,json,pathlib,argparse;sys.path.insert(0,sys.argv[2]);import benchmark_multilingual_1000 as m;a=json.loads(sys.argv[1]);a.update({k:pathlib.Path(a[k]) for k in ('manifest','output','binary')});a['model']=[pathlib.Path(p) for p in a['model']];raise SystemExit(m.run(argparse.Namespace(**a),expected_per_language=5))"
        def start():return subprocess.Popen([sys.executable,'-c',code,json.dumps(kwargs),str(pathlib.Path(__file__).resolve().parent)],stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
        p=start()
        for _ in range(500):
         ledger=root/'result.jsonl'
         if ledger.exists() and len(ledger.read_bytes().splitlines())>=2:break
         if p.poll() is not None:raise AssertionError(p.communicate())
         time.sleep(.01)
        else:raise AssertionError('No ledger progress')
        p.kill();p.communicate()
        with ledger.open('ab') as f:f.write(b'{"id": "interrupted')
        p=start();out,err=p.communicate(timeout=30);assert p.returncode==1,(out,err)
        x=json.loads((root/'result.json').read_text());assert x['completed'] and x['n']==5 and x['failed_requests']==1
        assert [d['condition'] for d in x['details']]==['original','telephone_alaw','original','original','original']
        assert len(ledger.read_bytes().splitlines())==5
        assert sum(d['status']=='failed' for d in x['details'])==1
        failed=x['details'][-1];assert failed['hypothesis']=='' and failed['scores']['deletions']==2
        ss={s['subset']:s for s in x['summary'] if s['split'] is None}
        assert [ss[k]['n'] for k in ['full','digit_free','eligible_official_style']]==[5,4,3]
        assert 'Recovered interrupted final JSONL line' in err
        before=len(requests);p=start();out,err=p.communicate(timeout=30);assert p.returncode==1 and len(requests)==before+1
        saved_ledger=ledger.read_bytes()
        rows=[json.loads(line) for line in saved_ledger.splitlines()]
        rows[1]['condition']='original'
        ledger.write_text(''.join(json.dumps(row)+'\n' for row in rows))
        p=start();out,err=p.communicate(timeout=30);assert p.returncode!=0 and 'condition' in err
        ledger.write_bytes(saved_ledger)
        (root/'model').write_bytes(b'changed-model');p=start();out,err=p.communicate(timeout=30);assert p.returncode!=0 and 'Resume identity mismatch' in err
        (root/'model').write_bytes(b'model')
        (root/'binary').write_bytes(b'changed-binary');p=start();out,err=p.communicate(timeout=30);assert p.returncode!=0 and 'Resume identity mismatch' in err
        (root/'binary').write_bytes(b'binary')
        with (root/'0.wav').open('ab') as f:f.write(b'changed')
        p=start();out,err=p.communicate(timeout=30);assert p.returncode!=0 and 'Audio checksum mismatch' in err
        try:runner.preflight(root/'manifest.json')
        except ValueError as e:assert 'Expected 1000' in str(e)
        else:raise AssertionError('1000 preflight not enforced')
        server.shutdown()
        server.server_close()
        print('PASS: real HTTP, interrupted process+truncated tail resume, persistent failure accounting, subsets, completed resume warmup-only, model/binary/audio identity rejection, strict1000 preflight')

if __name__ == '__main__':
    unittest.main()
