#!/usr/bin/env python3
"""Freeze failure-enriched diagnostic cases from the completed paired benchmark."""
import json,hashlib
from pathlib import Path
root=Path(__file__).resolve().parents[1];base=root/'benchmark/results/multilingual_1000_20261001';phone=root/'benchmark/results/multilingual_telephony_20261001';out=root/'benchmark/results/multilingual_small_diagnosis_20261001'
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
failure_ids=set(); triggers={};results={};manifests={}
for lang in ['ru','en','kk','ky','uz']:
 for condition in ['original','alaw','mulaw']:
  directory=base if condition=='original' else phone
  suffix=lang if condition=='original' else f'{lang}_{condition}'
  p=directory/f'ml_ctc_{suffix}.json';d=json.loads(p.read_text());results[lang,condition]={r['id']:r for r in d['details']}
  manifests[lang,condition]=json.loads((directory/f'{suffix}_manifest.json').read_text())['samples']
  if condition=='original': continue
  for r in d['details']:
   s=r['scores'];n=s['reference_words']
   if not r['hypothesis'].strip() or (n>0 and 5*s['deletions']>=4*n):
    failure_ids.add((lang,r['id']));triggers.setdefault(r['id'],[]).append(condition)
selected=[]
for lang in ['ru','en','kk','ky','uz']:
 failures=[r for r in manifests[lang,'original'] if (lang,r['id']) in failure_ids]
 controls=[r for r in manifests[lang,'original'] if (lang,r['id']) not in failure_ids][:3]
 for label,items in [('failure',failures),('control',controls)]:
  for r in items:
   selected.append({'language':lang,'id':r['id'],'selection':label,'trigger_conditions':triggers.get(r['id'],[])})
rows=[]
for item in selected:
 for condition in ['original','alaw','mulaw']:
  source=next(r for r in manifests[item['language'],condition] if r['id']==item['id'])
  expected=results[item['language'],condition][item['id']]
  assert sha(source['path'])==source['sha256']
  rows.append({**item,'condition':condition,'path':source['path'],'audio_sha256':source['sha256'],'reference':source['reference'],'baseline_hypothesis':expected['hypothesis'],'baseline_scores':expected['scores'],'duration_s':source['duration_s']})
assert len(selected)==37 and len(rows)==111 and len({(r['id'],r['condition']) for r in rows})==111
payload={'protocol':'All 22 unique recordings with empty small output or >=80% reference deletions in either completed telephone condition; plus first three baseline-ordered nonfailure recordings per language (15 controls). All original/A-law/mu-law versions retained. Failure-enriched diagnostics, not a representative WER benchmark. Frozen before new interventions.','unique_recordings':37,'inputs':111,'failure_recordings':22,'control_recordings':15,'baseline_binary_sha256':'bb30ca980374af9eb6e9fae9d239c3ce34a85e69e202bc8f8bac6c1e7ae36566','rows':rows}
path=out/'cases.json';path.write_text(json.dumps(payload,ensure_ascii=False,indent=2)+'\n');print('Frozen',len(rows),'inputs; SHA',sha(path))
