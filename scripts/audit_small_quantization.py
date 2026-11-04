#!/usr/bin/env python3
"""Independently apply frozen nontraining quantization screening gates.

Requires all declared candidate results; never promotes or changes models.
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import subprocess

import jiwer
import numpy as np

from audit_multilingual_1000 import MODEL_HASHES, VOCAB_HASH, digest, require
from wer_unicode import normalize, word_edit_distance

GRAPH_AUDIT = r'''
import json,sys
import numpy as np
import onnx
from onnx import numpy_helper
source=onnx.load(sys.argv[1]); candidate=onnx.load(sys.argv[2])
original={t.name:t for t in source.graph.initializer}
initializers={t.name:t for t in candidate.graph.initializer}
weight_names={n.input[1] for n in candidate.graph.node if n.op_type in ('MatMulInteger','ConvInteger')}
quantized=[]
for name in weight_names:
 t=initializers[name]
 assert t.data_type in (onnx.TensorProto.INT8,onnx.TensorProto.UINT8)
 base=name.removesuffix('_quantized')
 assert base in original and tuple(t.dims)==tuple(original[base].dims)
 quantized.append(t)
all_tensors=list(candidate.graph.initializer)
for n in candidate.graph.node:
 for a in n.attribute:
  if a.type==onnx.AttributeProto.TENSOR:all_tensors.append(a.t)
  elif a.type==onnx.AttributeProto.TENSORS:all_tensors.extend(a.tensors)
assert not any(t.data_location==onnx.TensorProto.EXTERNAL for t in all_tensors)
quantized_bytes=sum(numpy_helper.to_array(t).nbytes for t in quantized)
stored_bytes=sum(numpy_helper.to_array(t).nbytes for t in all_tensors)
original_elements=sum(numpy_helper.to_array(t).size for t in original.values())
quantized_elements=sum(numpy_helper.to_array(t).size for t in quantized)
for name,t in initializers.items():
 if name in original:
  assert np.array_equal(numpy_helper.to_array(t),numpy_helper.to_array(original[name]))
print(json.dumps(dict(quantized_matrix_count=len(quantized),quantized_original_elements=quantized_elements,original_initializer_elements=original_elements,quantized_original_element_fraction=quantized_elements/original_elements,quantized_stored_weight_bytes=quantized_bytes,total_initializer_and_constant_tensor_bytes=stored_bytes,quantized_stored_weight_byte_fraction=quantized_bytes/stored_bytes,source_named_tensors_unchanged=True,denominator='All source initializer elements; all candidate initializer and Constant tensor bytes, including biases, scales, zero points and bookkeeping tensors (conservative).')))
'''


def counts(rows):
    groups = {}
    for r in rows:
        g = groups.setdefault(r['selection'] + '/' + r['condition'], dict(n=0, reference_words=0, word_errors=0, deletions=0, empty=0))
        g['n'] += 1
        for k in ('reference_words', 'word_errors', 'deletions', 'empty'):
            g[k] += r[k]
    return groups


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--work', type=Path, required=True)
    parser.add_argument('--onnx-python', type=Path, required=True)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--prepared', type=Path, required=True)
    parser.add_argument('--vocab', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    protocol_path = args.directory / 'protocol.json'
    protocol = json.loads(protocol_path.read_text())
    require(protocol['training'] is False and len(protocol['candidates']) == 3, 'Unexpected protocol')
    require(digest(args.source) == '8bc803289f9cb5147ee95451fd9bdba219b1ecf1ddcd59a3651177c103c9eeec', 'Source weights changed')
    require(digest(args.vocab) == VOCAB_HASH, 'Vocab changed')
    vocab = [line.rsplit(' ', 1)[0] for line in args.vocab.read_text().splitlines()]
    prepared = json.loads(args.prepared.read_text())
    require(prepared['metadata']['cases_sha256'] == protocol['cases_sha256'], 'Cases identity changed')
    samples = {(r['id'], r['condition']): r for r in prepared['rows'] if r['variant'] == 'decoded_float'}
    require(len(samples) == 111, 'Wrong input count')
    baseline = counts([{**s, **s['baseline_scores'], 'empty': not bool(s['baseline_hypothesis'].strip())} for s in samples.values()])
    probe = Path.home() / '.cache/gigastt-small-diagnosis/bin/native_ctc_probe_5c2f64d2cb09c50e'
    probe_hash = digest(probe)
    original_path = args.vocab.with_name('multilingual_ctc.int8.onnx')
    require(digest(original_path) == MODEL_HASHES['ml_ctc'], 'Original model changed')
    reports = []
    for candidate in protocol['candidates']:
        public_path = args.directory / (candidate + '_screen.json')
        public = json.loads(public_path.read_text())
        private = json.loads((args.work / candidate / 'screen/private_results.json').read_text())
        build = json.loads((args.directory / (candidate + '.json')).read_text())
        require(public['complete'] is True and len(public['rows']) == len(private['rows']) == 111, 'Incomplete screening')
        require(public['metadata'] == private['metadata'], 'Public/private identity mismatch')
        meta = public['metadata']
        require(meta['probe_sha256'] == probe_hash, 'Probe changed')
        require(build['original_bytes'] == original_path.stat().st_size, 'Wrong original size denominator')
        require(meta['candidate'] == candidate and meta['model_sha256'] == build['sha256'] == digest(build['path']), 'Model mismatch')
        require(meta['protocol_sha256'] == digest(protocol_path) and meta['prepared_sha256'] == digest(args.prepared), 'Protocol/features changed')
        require(meta['script_sha256'] == digest(Path(__file__).with_name('screen_small_quantization.py')), 'Screen runner changed')
        require(build['original_sha256'] == MODEL_HASHES['ml_ctc'] and build['calibration_data'] is None, 'Model provenance not original/data-free')
        resource = json.loads(subprocess.check_output([str(args.onnx_python), '-c', GRAPH_AUDIT, str(args.source), build['path']], text=True))
        require(Path(build['path']).stat().st_size == build['bytes'], 'File size mismatch')
        keys, new_empty, new_catastrophic = set(), [], []
        for row, pub in zip(private['rows'], public['rows']):
            require({k:v for k,v in row.items() if k not in ('reference','hypothesis')} == pub, 'Public/private row mismatch')
            key = row['id'], row['condition']
            require(key in samples and key not in keys, 'Wrong/duplicate case')
            keys.add(key)
            sample = samples[key]
            require(all(row[k] == sample[k] for k in ('id','language','selection','condition','reference','feature_sha256','baseline_scores')), 'Case changed')
            prefix = args.work / candidate / 'screen/native' / '_'.join(key)
            native = json.loads(Path(str(prefix)+'.json').read_text())
            require(digest(str(prefix)+'.features.f32') == digest(sample['feature_path']) == sample['feature_sha256'], 'Features changed')
            require(native['feature_length'] == sample['frames'] and native['intra_threads'] == 3 and native['inter_threads'] == 1 and native['factory'] == 'production', 'Runtime configuration changed')
            logits = np.fromfile(str(prefix)+'.logits.f32',dtype='<f4').reshape(native['logit_shape'])
            length = native['output_lengths'][0]
            require(np.isfinite(logits).all() and logits.shape[0] == 1 and logits.shape[2] == 71 and 0 < length <= logits.shape[1], 'Invalid logits')
            ids = np.argmax(logits[0,:length],axis=1)
            collapsed = ids[np.r_[True,ids[1:] != ids[:-1]]]
            text = ' '.join(''.join(vocab[int(i)] for i in collapsed if i != 70).replace('▁',' ').split())
            require(text == row['hypothesis'], 'Decode mismatch')
            ref, hyp = normalize(row['reference']), normalize(text)
            alignment = jiwer.process_words(' '.join(ref),' '.join(hyp))
            score = dict(reference_words=len(ref),word_errors=word_edit_distance(ref,hyp),substitutions=alignment.substitutions,deletions=alignment.deletions,insertions=alignment.insertions,empty=not bool(text))
            require(all(row[k] == v for k,v in score.items()), 'Independent metric mismatch')
            empty = score['empty'] and bool(sample['baseline_hypothesis'].strip())
            catastrophic = 5*score['deletions'] >= 4*len(ref) and 5*sample['baseline_scores']['deletions'] < 4*len(ref)
            require(row['new_empty'] == empty and row['new_catastrophic'] == catastrophic, 'Incorrect failure flags')
            if empty:new_empty.append('_'.join(key))
            if catastrophic:new_catastrophic.append('_'.join(key))
        require(keys == samples.keys(), 'Missing cases')
        groups = counts(private['rows'])
        require(groups == public['groups'] == private['groups'], 'Aggregate mismatch')
        require(new_empty == public['new_empty'] and new_catastrophic == public['new_catastrophic'], 'Failure-list mismatch')
        gates = dict(complete_finite=True,no_new_empty=not new_empty,no_new_catastrophic=not new_catastrophic,
                     model_size=build['bytes'] <= 1.5*build['original_bytes'],
                     quantized_elements=resource['quantized_original_element_fraction'] > .5,
                     quantized_stored_bytes=resource['quantized_stored_weight_byte_fraction'] > .5,
                     failure_original=groups['failure/original']['word_errors'] <= 80,
                     controls=all(groups['control/'+c]['word_errors'] <= n for c,n in zip(('original','alaw','mulaw'),(22,29,26))),
                     failure_phone_errors=all(groups['failure/'+c]['word_errors'] < baseline['failure/'+c]['word_errors'] for c in ('alaw','mulaw')),
                     failure_phone_deletions=all(groups['failure/'+c]['deletions'] < baseline['failure/'+c]['deletions'] for c in ('alaw','mulaw')),
                     failure_phone_deletions_25pct=sum(groups['failure/'+c]['deletions'] for c in ('alaw','mulaw')) <= 562,
                     all_phone_errors_and_deletions=all(sum(groups[s+'/'+c][k] for s in ('failure','control')) < sum(baseline[s+'/'+c][k] for s in ('failure','control')) for c in ('alaw','mulaw') for k in ('word_errors','deletions')))
        reports.append(dict(candidate=candidate,passed=all(gates.values()),gates=gates,resource=resource,groups=groups,new_empty=new_empty,new_catastrophic=new_catastrophic,result_sha256=digest(public_path),model_bytes=build['bytes']))
        print('Audited',candidate,'PASS' if all(gates.values()) else 'FAIL',[k for k,v in gates.items() if not v],flush=True)
    eligible = [r for r in reports if r['passed']]
    eligible.sort(key=lambda r:(sum(r['groups']['failure/'+c]['word_errors'] for c in ('alaw','mulaw')),sum(r['groups']['control/'+c]['word_errors'] for c in ('alaw','mulaw')),r['model_bytes'],r['candidate']))
    args.output.write_text(json.dumps(dict(completed=True,protocol_sha256=digest(protocol_path),candidate_count=3,executions=333,baseline=baseline,candidates=reports,selected_candidate=eligible[0]['candidate'] if eligible else None,limitations='111 previously seen failure-enriched inputs; acceptance authorizes full benchmark only, not production replacement.'),indent=2)+'\n')


if __name__ == '__main__':
    main()
