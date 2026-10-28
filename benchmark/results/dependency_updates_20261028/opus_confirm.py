import json,os,pathlib,statistics,subprocess
root=pathlib.Path(__file__).parent
rows=[]
for file in sorted((root/'opus-fixtures').glob('*.ogg')):
 for i in range(7):
  for arm in (('baseline','candidate') if i%2==0 else ('candidate','baseline')):
   result=subprocess.run(['taskset','-c','2,3',str(root/('decode-'+arm)),str(file),str(root/'opus-results'/('confirm-'+arm+'.pcm'))],capture_output=True,text=True,check=True)
   rows.append({'sample':file.stem,'arm':arm,'round':i,'system_loadavg':os.getloadavg(),**json.loads(result.stdout)})
   (root/'opus-confirm.json').write_text(json.dumps(rows,indent=2)+'\n')
 a=[x['seconds_100'] for x in rows if x['sample']==file.stem and x['arm']=='baseline'];b=[x['seconds_100'] for x in rows if x['sample']==file.stem and x['arm']=='candidate']
 print(file.stem,statistics.median(y/x for x,y in zip(a,b)),flush=True)
