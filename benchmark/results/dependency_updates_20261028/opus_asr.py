import json, pathlib, signal, sys
from types import SimpleNamespace
sys.path.insert(0, '/home/ekhodzitsky/Projects/gigastt-oct-deps/scripts')
import benchmark_stitch_case as b
root=pathlib.Path(__file__).parent
args=SimpleNamespace(models=pathlib.Path('/home/ekhodzitsky/.cache/gigastt/overlap-fix-20261025/models'))
bins={'baseline':pathlib.Path('/home/ekhodzitsky/.cache/gigastt/security-fix-20261025/gigastt-candidate'),'candidate':root/'gigastt-final'}
report=[]
ref='шестьдесят тысяч тенге сколько будет стоить'
for head in b.ENCODERS:
 for arm,binary in bins.items():
  with (root/f'opus-asr-{head}-{arm}.log').open('w') as log:
   proc,cmd,start=b.start_server(binary,19879,head,args,log)
   try:
    for path in sorted((root/'opus-fixtures').glob('*.ogg')):
     for label,audio in [('opus',path),('libopus',root/'opus-results'/(path.stem+'-libopus.wav'))]:
      body,seconds=b.request(19879,audio.read_bytes())
      report.append({'head':head,'arm':arm,'sample':path.stem,'decoder':label,'response':body,'normalized_sdi':b.edits(b.normalize(ref),b.normalize(body['text']))})
      (root/'opus-asr.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
   finally:
    proc.send_signal(signal.SIGTERM);proc.wait(timeout=20)
print([(r['head'],r['arm'],r['sample'],r['decoder'],r['normalized_sdi']) for r in report])
