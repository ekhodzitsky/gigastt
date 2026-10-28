import array, hashlib, json, math, pathlib, statistics, subprocess
root=pathlib.Path(__file__).parent
fixtures=root/'opus-fixtures'
out=root/'opus-results';out.mkdir(exist_ok=True)
report={'rounds':[], 'quality':{}}
for file in sorted(fixtures.glob('*.ogg')):
 ref=out/(file.stem+'-libopus.wav')
 subprocess.run(['ffmpeg','-nostdin','-y','-v','error','-i',str(file),'-ar','48000','-acodec','pcm_f32le',str(ref)],check=True)
 subprocess.run([str(root/'decode-baseline'),str(ref),str(out/(file.stem+'-reference.pcm'))],capture_output=True,check=True)
 for i in range(7):
  for arm in (('baseline','candidate') if i%2==0 else ('candidate','baseline')):
   dest=out/(file.stem+'-'+arm+'.pcm')
   result=subprocess.run(['taskset','-c','2,3',str(root/('decode-'+arm)),str(file),str(dest)],capture_output=True,text=True,check=True)
   report['rounds'].append({'sample':file.stem,'arm':arm,'round':i,**json.loads(result.stdout)})
 def pcm(path):
  data=array.array('f');data.frombytes(path.read_bytes());return data
 reference=pcm(out/(file.stem+'-reference.pcm'))
 quality={}
 for arm in ('baseline','candidate'):
  actual=pcm(out/(file.stem+'-'+arm+'.pcm'))
  # Search fixed +/- 1024 samples using a strided interior, then score all aligned samples.
  def mse(lag):
   return statistics.mean((actual[k+lag]-reference[k])**2 for k in range(1024,min(len(reference),len(actual))-1024,32))
  lag=min(range(-1024,1025),key=mse)
  begin=max(0,-lag);end=min(len(reference),len(actual)-lag)
  errors=sum((actual[k+lag]-reference[k])**2 for k in range(begin,end))
  power=sum(reference[k]**2 for k in range(begin,end))
  quality[arm]={'samples':len(actual),'reference_samples':len(reference),'lag_samples':lag,'aligned_snr_db':10*math.log10(power/max(errors,1e-30)),'sha256':hashlib.sha256((out/(file.stem+'-'+arm+'.pcm')).read_bytes()).hexdigest()}
 report['quality'][file.stem]=quality
 (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
 print(file.stem,quality,flush=True)
