#!/usr/bin/env python3
"""Build bounded, data-free multilingual quantization diagnostics.

No training, calibration corpus, inference, or installed-model mutation. The
conversion environment is separate from the native runtime used for evaluation.
"""
import argparse
from collections import Counter
import hashlib
import importlib.metadata
import json
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper


SOURCE_SHA = '8bc803289f9cb5147ee95451fd9bdba219b1ecf1ddcd59a3651177c103c9eeec'
ORIGINAL_SHA = 'e08e27ae5669b39f0c378fae101bbbb9a80505f74f9b66719c309bf5b894a480'
CANDIDATES = ('original_u8_conv_fp32', 's8_channel_full_conv_fp32',
              's8_channel_reduced_conv_fp32', 'reproduce_u8_tensor')


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.partial')
    temporary.write_text(json.dumps(data, indent=2) + '\n')
    temporary.replace(path)


def tensor_signature(tensor):
    array = numpy_helper.to_array(tensor)
    return (str(array.dtype), tuple(array.shape), hashlib.sha256(array.tobytes()).hexdigest())


def graph_summary(model):
    initializers = list(model.graph.initializer)
    quant = [x for x in initializers if x.name.endswith('_quantized')]
    return {
        'op_counts': dict(sorted(Counter(n.op_type for n in model.graph.node).items())),
        'initializer_types': dict(Counter(onnx.TensorProto.DataType.Name(t.data_type) for t in initializers)),
        'quantized_weight_elements': sum(int(np.prod(t.dims)) for t in quant),
        'float_weight_elements': sum(int(np.prod(t.dims)) for t in initializers if t.data_type == onnx.TensorProto.FLOAT and len(t.dims) >= 2),
        'quantized_weights': len(quant),
        'scale_shapes': dict(Counter(str(list(t.dims)) for t in initializers if t.name.endswith('_scale'))),
        'producer': [model.producer_name, model.producer_version],
        'opsets': [[x.domain, x.version] for x in model.opset_import],
    }


def restore_convolutions(original, source):
    """Replace Conv terminal producers, then prune only unreachable graph nodes."""
    original_initializers = {t.name: t for t in original.graph.initializer}
    source_initializers = {t.name: t for t in source.graph.initializer}
    convs = [n for n in source.graph.node if n.op_type == 'Conv']
    replacements = {n.output[0]: n for n in convs}
    nodes = []
    replaced = set()
    for node in original.graph.node:
        match = [name for name in node.output if name in replacements]
        if match:
            if len(match) != 1 or match[0] in replaced:
                raise ValueError('Ambiguous Conv output producer')
            nodes.append(replacements[match[0]])
            replaced.add(match[0])
        else:
            nodes.append(node)
    if replaced != set(replacements):
        raise ValueError('Missing original Conv output')
    needed = {o.name for o in original.graph.output}
    kept = []
    for node in reversed(nodes):
        if needed.intersection(node.output):
            kept.append(node)
            needed.update(node.input)
    del original.graph.node[:]
    original.graph.node.extend(reversed(kept))
    for conv in convs:
        for name in conv.input[1:]:
            if name:
                original_initializers[name] = source_initializers[name]
    del original.graph.initializer[:]
    original.graph.initializer.extend(t for name, t in original_initializers.items() if name in needed)
    # Drop stale inferred intermediate types; the graph remains in the same opset.
    del original.graph.value_info[:]
    return original


def check_identity(candidate, source_signatures, original_signatures, name):
    tensors = {t.name: t for t in candidate.graph.initializer}
    checks = {'unchanged_source_float_tensors': 0, 'identical_original_quantization_tensors': 0}
    for tensor in tensors.values():
        if tensor.name in source_signatures:
            if tensor_signature(tensor) != source_signatures[tensor.name]:
                raise ValueError('Source tensor changed: ' + tensor.name)
            checks['unchanged_source_float_tensors'] += 1
    if name == 'original_u8_conv_fp32':
        matmul_weights = [n.input[1] for n in candidate.graph.node if n.op_type == 'MatMulInteger']
        if len(matmul_weights) != 128:
            raise ValueError('Unexpected MatMulInteger count')
        for weight in matmul_weights:
            base = weight.removesuffix('_quantized')
            for suffix in ['_quantized', '_scale', '_zero_point']:
                key = base + suffix
                if tensor_signature(tensors[key]) != original_signatures[key]:
                    raise ValueError('Original quantization changed: ' + key)
                checks['identical_original_quantization_tensors'] += 1
    checks['source_weight_identity_passed'] = True
    return checks


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, default=Path.home() / '.cache/gigastt-precision/models/multilingual_ctc.fp32.onnx')
    p.add_argument('--original', type=Path, default=Path.home() / '.gigastt/models/multilingual_ctc.int8.onnx')
    p.add_argument('--work', type=Path, default=Path.home() / '.cache/gigastt-small-quantization')
    p.add_argument('--report', type=Path, default=Path('benchmark/results/multilingual_small_quantization_20261001'))
    p.add_argument('--candidate', choices=CANDIDATES, required=True)
    args = p.parse_args()
    if digest(args.source) != SOURCE_SHA or digest(args.original) != ORIGINAL_SHA:
        raise ValueError('Pinned pretrained source/model mismatch')
    output = args.work / args.candidate / 'multilingual_ctc.int8.onnx'
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError('Refusing to overwrite an existing candidate')
    source = onnx.load(args.source)
    original = onnx.load(args.original)
    source_signatures = {t.name: tensor_signature(t) for t in source.graph.initializer}
    original_signatures = {t.name: tensor_signature(t) for t in original.graph.initializer}
    audit = {'source_sha256': SOURCE_SHA, 'original_sha256': ORIGINAL_SHA,
             'source': graph_summary(source), 'original': graph_summary(original),
             'reference': 'https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html',
             'restriction': 'No training, gradients, calibration samples, or quality inference in this converter.'}
    audit_path = args.report / 'graph_audit.json'
    if audit_path.exists() and json.loads(audit_path.read_text()) != audit:
        raise ValueError('Existing graph audit differs')
    save(audit_path, audit)
    config = {'candidate': args.candidate, 'source_sha256': SOURCE_SHA, 'original_sha256': ORIGINAL_SHA,
              'script_sha256': digest(__file__), 'calibration_data': None,
              'package_versions': {k: importlib.metadata.version(k) for k in ['onnx', 'onnxruntime', 'numpy', 'ml_dtypes', 'protobuf']}}
    temporary = output.with_suffix('.partial.onnx')
    if args.candidate == 'original_u8_conv_fp32':
        result = restore_convolutions(original, source)
        config.update({'conversion': 'Restore exact source Conv nodes/weights into original quantized graph; backward reachability prune',
                       'matmul_weight_type': 'original UINT8', 'per_channel': False, 'reduce_range': False,
                       'restored_conv_count': 51})
        del source
    else:
        del source, original
        from onnxruntime.quantization import QuantType, quantize_dynamic
        reproduction = args.candidate == 'reproduce_u8_tensor'
        config.update({'conversion': 'onnxruntime.quantization.quantize_dynamic',
                       'weight_type': 'QUInt8' if reproduction else 'QInt8',
                       'per_channel': not reproduction,
                       'reduce_range': args.candidate == 's8_channel_reduced_conv_fp32',
                       'op_types_to_quantize': ['MatMul', 'Conv'] if reproduction else ['MatMul'],
                       'extra_options': {'MatMulConstBOnly': True, 'WeightSymmetric': not reproduction},
                       'graph_optimization_before_conversion': False})
        quantize_dynamic(str(args.source), str(temporary),
                         op_types_to_quantize=config['op_types_to_quantize'],
                         per_channel=config['per_channel'], reduce_range=config['reduce_range'],
                         weight_type=QuantType.QUInt8 if reproduction else QuantType.QInt8,
                         extra_options=config['extra_options'])
        result = onnx.load(temporary)
    onnx.checker.check_model(result)
    config['identity_checks'] = check_identity(result, source_signatures, original_signatures, args.candidate)
    config['graph'] = graph_summary(result)
    if args.candidate == 'reproduce_u8_tensor':
        actual = {t.name: tensor_signature(t) for t in result.graph.initializer}
        config['original_initializer_comparison'] = {
            'original_count': len(original_signatures), 'candidate_count': len(actual),
            'identical_named_tensors': sum(actual.get(k) == v for k, v in original_signatures.items()),
            'all_identical': actual == original_signatures}
    onnx.save(result, temporary)
    temporary.replace(output)
    config.update({'path': str(output.resolve()), 'sha256': digest(output), 'bytes': output.stat().st_size,
                   'original_bytes': args.original.stat().st_size,
                   'size_ratio': output.stat().st_size / args.original.stat().st_size,
                   'within_1_5_size_budget': output.stat().st_size <= 1.5 * args.original.stat().st_size,
                   'onnx_checker_passed': True})
    config['majority_weight_elements_quantized'] = config['graph']['quantized_weight_elements'] > config['graph']['float_weight_elements']
    save(args.report / (args.candidate + '.json'), config)
    print(json.dumps({k: config[k] for k in ['candidate', 'path', 'sha256', 'bytes', 'size_ratio', 'within_1_5_size_budget']}), flush=True)


if __name__ == '__main__':
    main()
