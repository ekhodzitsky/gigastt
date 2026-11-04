#!/usr/bin/env python3
"""Audit the selected candidate against all 15,000 paired baseline recognitions.

Outside-screening recordings are not a blind holdout: baseline results were
already inspected. This audit does not authorize product replacement.
"""
import argparse
import json
import tomllib
from pathlib import Path

import numpy as np

from audit_multilingual_1000 import LANGUAGES, MODEL_HASHES, VOCAB_HASH, digest, require
from audit_multilingual_telephony import load_result, totals
from wer_unicode import normalize


def paired(before, after, seed=42):
    original, candidate = totals(before), totals(after)
    changed = [b['scores']['word_errors'] - a['scores']['word_errors'] for a,b in zip(before,after)]
    old_empty = [not normalize(r['hypothesis']) for r in before]
    new_empty = [not normalize(r['hypothesis']) for r in after]
    old_cat = [5*r['scores']['deletions'] >= 4*r['scores']['reference_words'] for r in before]
    new_cat = [5*r['scores']['deletions'] >= 4*r['scores']['reference_words'] for r in after]
    result = dict(original=original,candidate=candidate,
                  wer_delta_pp=candidate['wer_pct']-original['wer_pct'] if before else None,
                  cer_delta_pp=candidate['cer_pct']-original['cer_pct'] if before else None,
                  word_error_delta=sum(changed),deletion_delta=candidate['deletions']-original['deletions'],
                  improved=sum(v<0 for v in changed),worsened=sum(v>0 for v in changed),unchanged=sum(v==0 for v in changed),
                  original_catastrophic=sum(old_cat),candidate_catastrophic=sum(new_cat),
                  newly_empty_ids=[b['id'] for a,b,oe,ne in zip(before,after,old_empty,new_empty) if ne and not oe],
                  recovered_empty_ids=[b['id'] for a,b,oe,ne in zip(before,after,old_empty,new_empty) if oe and not ne],
                  new_catastrophic_ids=[b['id'] for b,oc,nc in zip(after,old_cat,new_cat) if nc and not oc],
                  recovered_catastrophic_ids=[b['id'] for b,oc,nc in zip(after,old_cat,new_cat) if oc and not nc])
    if before:
        rng=np.random.default_rng(seed)
        indices=rng.integers(0,len(before),size=(1000,len(before)))
        words=np.array([r['scores']['reference_words'] for r in before])
        deltas=np.array(changed)
        values=100*deltas[indices].sum(axis=1)/words[indices].sum(axis=1)
        result['paired_utterance_bootstrap_wer_delta_95_pp']=np.quantile(values,[.025,.975]).tolist()
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--directory',type=Path,required=True)
    p.add_argument('--results',type=Path,required=True)
    p.add_argument('--original',type=Path,required=True)
    p.add_argument('--telephone',type=Path,required=True)
    p.add_argument('--cases',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--filename',default='{language}_{condition}.json')
    args=p.parse_args()
    selection=json.loads((args.directory/'screen_validation.json').read_text())
    require(selection['completed'] is True and selection['selected_candidate'],'No independently selected candidate')
    name=selection['selected_candidate']
    model=json.loads((args.directory/(name+'.json')).read_text())
    require(digest(model['path'])==model['sha256'],'Candidate model changed')
    pack=json.loads((args.directory/'product_pack.json').read_text())
    require(pack['candidate']==name and pack['encoder_sha256']==model['sha256'] and pack['vocab_sha256']==VOCAB_HASH,'Product pack selection differs')
    pack_dir=Path(pack['model_dir'])
    manifest_file=pack_dir/'manifest.toml'
    require(digest(manifest_file)==pack['manifest_sha256'],'Pack manifest changed')
    custom=tomllib.loads(manifest_file.read_text())
    require(custom==dict(architecture='ml_ctc',files=dict(encoder='lab_multilingual_ctc.int8.onnx',encoder_int8='lab_multilingual_ctc.int8.onnx',vocab='multilingual_vocab.txt')),'Undeclared pack manifest')
    expected_models={str((pack_dir/custom['files'][k]).resolve()):sha for k,sha in [('encoder',model['sha256']),('vocab',VOCAB_HASH)]}
    expected_models[str(manifest_file.resolve())]=pack['manifest_sha256']
    for path,sha in expected_models.items():require(digest(path)==sha,'Pack bytes changed')
    require(digest(pack['binary'])==pack['binary_sha256'],'Pack binary changed')
    screen_private=Path.home()/'.cache/gigastt-small-quantization'/name/'screen/private_results.json'
    screen_rows={(r['id'],r['condition']):r for r in json.loads(screen_private.read_text())['rows']}
    require(len(screen_rows)==111,'Incomplete private screening result')
    cases=json.loads(args.cases.read_text())
    protocol=json.loads((args.directory/'protocol.json').read_text())
    require(digest(args.cases)==protocol['cases_sha256'],'Cases changed')
    case_rows={(r['id'],r['condition']):r for r in cases['rows']}
    screened={r['id'] for r in cases['rows']}
    require(len(screened)==37,'Wrong screening population')
    audits=[]; identities=[]; seen_unique=set(); screened_inputs=0; exact_screen_join=0
    for language in sorted(LANGUAGES):
        for condition in ('original','alaw','mulaw'):
            original=condition=='original'
            baseline_dir=args.original if original else args.telephone
            manifest_path=baseline_dir/(f'{language}_manifest.json' if original else f'{language}_{condition}_manifest.json')
            manifest=json.loads(manifest_path.read_text())
            baseline_path=baseline_dir/(f'ml_ctc_{language}.json' if original else f'ml_ctc_{language}_{condition}.json')
            candidate_path=args.results/args.filename.format(language=language,condition=condition,candidate=name)
            row_condition='original' if original else 'simulated_telephone_'+condition
            before=load_result(baseline_path,manifest,manifest_path,row_condition)
            after=load_result(candidate_path,manifest,manifest_path,row_condition)
            bi,ai=before['metadata']['identity'],after['metadata']['identity']
            for key in ('variant','binary_sha256','jiwer','health_configuration'):
                require(ai[key]==bi[key],f'Changed baseline inference identity {key}')
            require(ai['variant']=='ml_ctc' and ai['binary_sha256']==pack['binary_sha256'],'Wrong variant/binary')
            old_config,new_config=bi['model_configuration'],ai['model_configuration']
            require(old_config.keys()==new_config.keys(),'Model configuration fields changed')
            for key in old_config:
                if key=='encoder':
                    require(new_config[key] in (old_config[key],custom['files']['encoder']),'Undeclared encoder label')
                else:require(new_config[key]==old_config[key],f'Changed model setting {key}')
            require(set(bi['model_sha256'].values())=={MODEL_HASHES['ml_ctc'],VOCAB_HASH},'Baseline model changed')
            require(ai['model_sha256']==expected_models,'Wrong candidate pack identity or resolved paths')
            for scorer in ('benchmark_multilingual_public.py','wer_unicode.py'):
                require(ai['scorer_sha256'][scorer]==bi['scorer_sha256'][scorer]==digest(Path(__file__).with_name(scorer)),'Changed scorer')
            require(ai['scorer_sha256']['benchmark_multilingual_1000.py']==digest(Path(__file__).with_name('benchmark_multilingual_1000.py')),'Changed benchmark runner')
            require([r['id'] for r in before['details']]==[r['id'] for r in after['details']],'Unpaired IDs')
            for sample in manifest['samples']:
                require(digest(sample['path'])==sample['sha256'],'Audio hash mismatch')
            a,b=before['details'],after['details']
            for row in b:
                if row['id'] in screened:
                    key=(row['id'],condition)
                    require(row['sha256']==case_rows[key]['audio_sha256'],'Screen source audio mismatch')
                    require(row['reference']==screen_rows[key]['reference'],'Screen reference mismatch')
                    require(row['hypothesis'].strip()==screen_rows[key]['hypothesis'].strip(),'Independent full HTTP/native screen parity mismatch')
                    require(all(row['scores'][k]==screen_rows[key][k] for k in ('reference_words','word_errors','substitutions','deletions','insertions')),'Screen score mismatch')
                    exact_screen_join+=1
            subsets=[]
            for exposure in ('all','screened','outside_screening'):
                for numeric in ('full','digit_free','digit_bearing'):
                    pairs=[(x,y) for x,y in zip(a,b) if
                           (exposure=='all' or ((x['id'] in screened)==(exposure=='screened'))) and
                           (numeric=='full' or (any(c.isdigit() for c in x['reference'])==(numeric=='digit_bearing')))]
                    subsets.append(dict(exposure=exposure,subset=numeric,**paired([x for x,y in pairs],[y for x,y in pairs])))
            screened_inputs+=sum(r['id'] in screened for r in a)
            seen_unique.update(r['id'] for r in a)
            audits.append(dict(language=language,condition=condition,n=1000,
                               candidate_result_sha256=digest(candidate_path),baseline_result_sha256=digest(baseline_path),
                               manifest_sha256=digest(manifest_path),subsets=subsets))
            identities.append(ai)
            print('Verified',language,condition,flush=True)
    require(len(audits)==15 and len(seen_unique)==5000 and screened_inputs==111 and exact_screen_join==111,'Wrong full benchmark coverage')
    require(len({i['binary_sha256'] for i in identities})==1,'Mixed binaries')
    result=dict(completed=True,candidate=name,candidate_sha256=model['sha256'],candidate_requests=15000,
                baseline_recognitions_reaudited=15000,independent_full_http_screen_exact_text_matches=exact_screen_join,unique_recordings=5000,screened_unique_recordings=37,
                outside_screening_unique_recordings=4963,screened_inputs=111,outside_screening_inputs=14889,
                all_word_distances_independently_verified=True,all_character_distances_recomputed=True,
                all_model_audio_manifest_binary_scorer_identities_verified=True,
                bootstrap=dict(seed=42,resamples=1000,unit='paired utterance',coverage='95%',
                               caveat='Descriptive only; repeated sentences/speakers and baseline-based screening violate blind independent-speaker interpretation.'),
                limitations='Screened and outside-screening sets reported separately; all baseline data were inspected before this experiment. No product replacement decision is made by this audit.',runs=audits)
    args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')


if __name__=='__main__':
    main()
