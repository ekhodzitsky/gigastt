#!/usr/bin/env python3
"""Convert the GigaAM v3 rnnt Conformer ENCODER weights from ONNX to safetensors.

The output safetensors uses tensor keys that EXACTLY match the ``VarBuilder``
paths consumed by the vendored encoder in
``crates/gigastt-core/src/runtime/candle/conformer.rs``.

Run (local python 3.14 is broken; always use this uv invocation):

    uv run --python 3.13 --with onnx --with numpy --with safetensors \
        python scripts/convert_gigaam_candle.py

Naming facts (verified against the on-disk ONNX):

* Conv weights/biases keep their PyTorch names and are copied verbatim
  (``pre_encode.conv.{0,2}.{weight,bias}``,
  ``layers.N.conv.{pointwise_conv1,depthwise_conv,batch_norm,pointwise_conv2}.*``).
  NOTE: ``conv.batch_norm`` is a LayerNorm despite the name.
* LayerNorm weights/biases keep their names
  (``layers.N.{norm_feed_forward1,norm_self_att,norm_conv,norm_feed_forward2,norm_out}.*``).
* Linear layers: the ONNX export kept only the ``*.bias`` initializer under the
  PyTorch name. The weight became an anonymous top-level initializer named
  ``onnx::MatMul_NNNN``. We recover the weight<->bias pairing by tracing the graph:
  for each ``Add`` node whose one input is a named ``*.bias`` initializer, the
  other input is produced by a ``MatMul`` node whose weight is an
  ``onnx::MatMul_NNNN`` initializer. We emit that weight under
  ``<bias-name-without-.bias>.weight``.
* TRANSPOSE (parity-critical): the ONNX MatMul weight has shape ``[in, out]``
  (computes ``x @ W``); ``candle_nn::linear`` expects ``[out, in]`` (computes
  ``x @ W^T``). So every recovered Linear weight is transposed ``[in,out] ->
  [out,in]`` before saving. Conv weights are NOT transposed.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper
from safetensors.numpy import load_file, save_file

ONNX_PATH = Path("/Users/ekhodzitsky/.gigastt/models/v3_rnnt_encoder.onnx")
OUT_DIR = Path("/Users/ekhodzitsky/.gigastt/models/candle")
OUT_PATH = OUT_DIR / "encoder.safetensors"

N_LAYERS = 16
D_MODEL = 768
D_FF = 3072  # d_model * ff_expansion_factor (4)


def build_expected_shapes() -> dict[str, tuple[int, ...]]:
    """The full set of keys + shapes the candle encoder VarBuilder expects."""
    exp: dict[str, tuple[int, ...]] = {
        # Strided subsampling (conv.0, conv.2 — ReLU sits at 1, 3).
        "pre_encode.conv.0.weight": (768, 64, 5),
        "pre_encode.conv.0.bias": (768,),
        "pre_encode.conv.2.weight": (768, 768, 5),
        "pre_encode.conv.2.bias": (768,),
    }
    for n in range(N_LAYERS):
        p = f"layers.{n}."
        exp.update(
            {
                # FFN1 (Macaron)
                p + "norm_feed_forward1.weight": (768,),
                p + "norm_feed_forward1.bias": (768,),
                p + "feed_forward1.linear1.weight": (3072, 768),
                p + "feed_forward1.linear1.bias": (3072,),
                p + "feed_forward1.linear2.weight": (768, 3072),
                p + "feed_forward1.linear2.bias": (768,),
                # Self-attention
                p + "norm_self_att.weight": (768,),
                p + "norm_self_att.bias": (768,),
                p + "self_attn.linear_q.weight": (768, 768),
                p + "self_attn.linear_q.bias": (768,),
                p + "self_attn.linear_k.weight": (768, 768),
                p + "self_attn.linear_k.bias": (768,),
                p + "self_attn.linear_v.weight": (768, 768),
                p + "self_attn.linear_v.bias": (768,),
                p + "self_attn.linear_out.weight": (768, 768),
                p + "self_attn.linear_out.bias": (768,),
                # Convolution module
                p + "norm_conv.weight": (768,),
                p + "norm_conv.bias": (768,),
                p + "conv.pointwise_conv1.weight": (1536, 768, 1),
                p + "conv.pointwise_conv1.bias": (1536,),
                p + "conv.depthwise_conv.weight": (768, 1, 5),
                p + "conv.depthwise_conv.bias": (768,),
                p + "conv.batch_norm.weight": (768,),  # LayerNorm despite the name
                p + "conv.batch_norm.bias": (768,),
                p + "conv.pointwise_conv2.weight": (768, 768, 1),
                p + "conv.pointwise_conv2.bias": (768,),
                # FFN2 (Macaron)
                p + "norm_feed_forward2.weight": (768,),
                p + "norm_feed_forward2.bias": (768,),
                p + "feed_forward2.linear1.weight": (3072, 768),
                p + "feed_forward2.linear1.bias": (3072,),
                p + "feed_forward2.linear2.weight": (768, 3072),
                p + "feed_forward2.linear2.bias": (768,),
                # Output norm
                p + "norm_out.weight": (768,),
                p + "norm_out.bias": (768,),
            }
        )
    return exp


def main() -> int:
    if not ONNX_PATH.is_file():
        print(f"FAIL: ONNX not found: {ONNX_PATH}", file=sys.stderr)
        return 1

    print(f"Loading ONNX: {ONNX_PATH}")
    model = onnx.load(str(ONNX_PATH))
    graph = model.graph

    inits = {i.name: i for i in graph.initializer}
    print(f"  {len(inits)} initializers")

    # producer map: output tensor name -> producing node
    producer: dict[str, onnx.NodeProto] = {}
    for node in graph.node:
        for out in node.output:
            producer[out] = node

    bias_names = {n for n in inits if n.endswith(".bias")}
    anon_names = {n for n in inits if n.startswith("onnx::MatMul")}

    tensors: dict[str, np.ndarray] = {}

    # 1. Recover Linear weights via Add -> MatMul -> onnx::MatMul tracing.
    #    Track which biases got paired so the rest can be copied directly.
    paired_bias: set[str] = set()
    n_linear = 0
    for node in graph.node:
        if node.op_type != "Add":
            continue
        bias_in = None
        other_in = None
        for inp in node.input:
            if inp in bias_names:
                bias_in = inp
            else:
                other_in = inp
        if bias_in is None or other_in is None:
            continue
        prod = producer.get(other_in)
        if prod is None or prod.op_type != "MatMul":
            continue
        weight_init = next((x for x in prod.input if x in anon_names), None)
        if weight_init is None:
            continue

        w = numpy_helper.to_array(inits[weight_init]).astype(np.float32)
        # ONNX MatMul weight is [in, out] (x @ W); candle_nn::linear wants [out, in].
        w_t = np.ascontiguousarray(w.T)
        key = bias_in[: -len(".bias")] + ".weight"
        tensors[key] = w_t
        paired_bias.add(bias_in)
        n_linear += 1

    print(f"  recovered {n_linear} Linear weights (transposed [in,out]->[out,in])")

    # 2. Copy every named conv/LayerNorm/bias initializer verbatim (key = onnx name).
    #    This covers: all *.bias (linear biases + conv/norm biases) and all named
    #    *.weight (conv weights + LayerNorm weights). Anonymous onnx::MatMul
    #    initializers are NOT copied — they were handled (transposed) in step 1.
    n_copied = 0
    for name, init in inits.items():
        if name in anon_names:
            continue
        arr = numpy_helper.to_array(init).astype(np.float32)
        tensors[name] = np.ascontiguousarray(arr)
        n_copied += 1
    print(f"  copied {n_copied} named conv/LayerNorm/bias initializers verbatim")

    # 3. Save.
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(OUT_PATH))
    size = OUT_PATH.stat().st_size
    print(f"Saved {len(tensors)} tensors -> {OUT_PATH} ({size / 1e6:.1f} MB)")

    # 4. Reload + assert the full expected key set is present with correct shapes.
    expected = build_expected_shapes()
    reloaded = load_file(str(OUT_PATH))

    missing: list[str] = []
    mismatched: list[str] = []
    for key, shape in expected.items():
        if key not in reloaded:
            missing.append(key)
            continue
        got = tuple(reloaded[key].shape)
        if got != shape:
            mismatched.append(f"{key}: expected {shape}, got {got}")

    extra = sorted(set(reloaded) - set(expected))

    print()
    print(f"Expected keys: {len(expected)}; saved keys: {len(reloaded)}")
    if extra:
        print(f"  WARNING: {len(extra)} unexpected extra keys: {extra[:10]}")

    if missing or mismatched:
        print("FAIL")
        for k in missing:
            print(f"  MISSING: {k}")
        for k in mismatched:
            print(f"  MISMATCH: {k}")
        return 1

    if extra:
        print("FAIL: unexpected extra keys present (key set must match exactly)")
        return 1

    print(f"PASS: all {len(expected)} expected keys present with correct shapes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
