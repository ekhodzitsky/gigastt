#!/usr/bin/env python3
"""Convert the GigaAM v3 rnnt ENCODER + DECODER + JOINER weights ONNX->safetensors.

The output safetensors use tensor keys that EXACTLY match the ``VarBuilder``
paths consumed by the vendored encoder in
``crates/gigastt-core/src/runtime/candle/conformer.rs`` and the RNN-T decoder /
joiner sessions in ``crates/gigastt-core/src/runtime/candle/session.rs``.

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

MODELS_DIR = Path("/Users/ekhodzitsky/.gigastt/models")
ONNX_PATH = MODELS_DIR / "v3_rnnt_encoder.onnx"
DECODER_ONNX_PATH = MODELS_DIR / "v3_rnnt_decoder.onnx"
JOINER_ONNX_PATH = MODELS_DIR / "v3_rnnt_joint.onnx"
OUT_DIR = MODELS_DIR / "candle"
OUT_PATH = OUT_DIR / "encoder.safetensors"
DECODER_OUT_PATH = OUT_DIR / "decoder.safetensors"
JOINER_OUT_PATH = OUT_DIR / "joiner.safetensors"

N_LAYERS = 16
D_MODEL = 768
D_FF = 3072  # d_model * ff_expansion_factor (4)
PRED_HIDDEN = 320  # LSTM hidden size / decoder output dim
VOCAB = 34  # rnnt char vocab (incl. blank)
ENC_DIM = 768  # encoder output dim


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


def convert_encoder() -> int:
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


def convert_decoder() -> int:
    """Emit decoder.safetensors from the ONNX RNN-T prediction network.

    Source initializers (verified on disk):
      * ``embed.weight``     [34, 320]   -> copied verbatim (Gather table)
      * ``onnx::LSTM_93``    [1, 1280, 320] = W_ih -> squeeze dir -> [1280, 320]
      * ``onnx::LSTM_94``    [1, 1280, 320] = W_hh -> squeeze dir -> [1280, 320]
      * ``onnx::LSTM_95``    [1, 2560]      = bias -> squeeze -> [2560] =
                              concat(b_ih[1280], b_hh[1280])

    ONNX LSTM gate order is **iofc** (input, output, forget, cell): the 1280 rows
    are 4 blocks of 320 in THAT order. We keep the rows as-is (the Rust cell reads
    the iofc blocks directly) and do NOT transpose the LSTM weights — ONNX stores
    them as [4*hidden, in] = [out, in] already.
    """
    if not DECODER_ONNX_PATH.is_file():
        print(f"FAIL: ONNX not found: {DECODER_ONNX_PATH}", file=sys.stderr)
        return 1

    print(f"Loading ONNX: {DECODER_ONNX_PATH}")
    model = onnx.load(str(DECODER_ONNX_PATH))
    inits = {i.name: numpy_helper.to_array(i).astype(np.float32) for i in model.graph.initializer}

    def need(name: str) -> np.ndarray:
        if name not in inits:
            raise SystemExit(f"FAIL: decoder initializer missing: {name}")
        return inits[name]

    embed = need("embed.weight")  # [34, 320]
    w_ih = need("onnx::LSTM_93")  # [1, 1280, 320]
    w_hh = need("onnx::LSTM_94")  # [1, 1280, 320]
    bias = need("onnx::LSTM_95")  # [1, 2560]

    # Squeeze the num_directions dim (always 1 here; assert to catch bidirectional).
    assert w_ih.shape[0] == 1, f"unexpected LSTM W_ih dir dim: {w_ih.shape}"
    assert w_hh.shape[0] == 1, f"unexpected LSTM W_hh dir dim: {w_hh.shape}"
    assert bias.shape[0] == 1, f"unexpected LSTM bias dir dim: {bias.shape}"
    w_ih = w_ih[0]  # [1280, 320]
    w_hh = w_hh[0]  # [1280, 320]
    bias = bias[0]  # [2560]

    # ONNX B = concat(Wb[iofc], Rb[iofc]); first 1280 = input bias, last 1280 = recurrent.
    b_ih = bias[:4 * PRED_HIDDEN]   # [1280]
    b_hh = bias[4 * PRED_HIDDEN:]   # [1280]

    tensors = {
        "embed.weight": np.ascontiguousarray(embed),
        "lstm.w_ih": np.ascontiguousarray(w_ih),
        "lstm.w_hh": np.ascontiguousarray(w_hh),
        "lstm.b_ih": np.ascontiguousarray(b_ih),
        "lstm.b_hh": np.ascontiguousarray(b_hh),
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(DECODER_OUT_PATH))
    print(f"Saved {len(tensors)} tensors -> {DECODER_OUT_PATH}")

    expected = {
        "embed.weight": (VOCAB, PRED_HIDDEN),
        "lstm.w_ih": (4 * PRED_HIDDEN, PRED_HIDDEN),
        "lstm.w_hh": (4 * PRED_HIDDEN, PRED_HIDDEN),
        "lstm.b_ih": (4 * PRED_HIDDEN,),
        "lstm.b_hh": (4 * PRED_HIDDEN,),
    }
    reloaded = load_file(str(DECODER_OUT_PATH))
    if _check_keys("decoder", expected, reloaded) != 0:
        return 1
    print(f"PASS: all {len(expected)} decoder keys present with correct shapes")
    return 0


def convert_joiner() -> int:
    """Emit joiner.safetensors from the ONNX RNN-T joint network.

    Source initializers (verified on disk):
      * ``enc.bias``         [320]      -> enc_proj.bias
      * ``pred.bias``        [320]      -> dec_proj.bias
      * ``joint_net.1.bias`` [34]       -> out.bias
      * ``onnx::MatMul_26``  [768, 320] = enc_proj W  [in, out]  -> T -> [320, 768]
      * ``onnx::MatMul_27``  [320, 320] = dec_proj W  [in, out]  -> T -> [320, 320]
      * ``onnx::MatMul_28``  [320, 34]  = out W       [in, out]  -> T -> [34, 320]

    The three MatMul weights are identified by SHAPE (not just name) so the
    enc/dec/out assignment is robust. MatMul weights are [in, out]; candle_nn
    ``linear`` (and the Rust matmul in session.rs) expect [out, in] -> TRANSPOSE.
    """
    if not JOINER_ONNX_PATH.is_file():
        print(f"FAIL: ONNX not found: {JOINER_ONNX_PATH}", file=sys.stderr)
        return 1

    print(f"Loading ONNX: {JOINER_ONNX_PATH}")
    model = onnx.load(str(JOINER_ONNX_PATH))
    inits = {i.name: numpy_helper.to_array(i).astype(np.float32) for i in model.graph.initializer}

    def need(name: str) -> np.ndarray:
        if name not in inits:
            raise SystemExit(f"FAIL: joiner initializer missing: {name}")
        return inits[name]

    enc_bias = need("enc.bias")            # [320]
    dec_bias = need("pred.bias")           # [320]
    out_bias = need("joint_net.1.bias")    # [34]

    # Identify the three MatMul weights by shape (defensive: confirm names too).
    matmuls = {n: a for n, a in inits.items() if n.startswith("onnx::MatMul")}
    if len(matmuls) != 3:
        raise SystemExit(f"FAIL: expected 3 onnx::MatMul weights, got {sorted(matmuls)}")

    def find_by_shape(shape: tuple[int, int]) -> np.ndarray:
        hits = [a for a in matmuls.values() if a.shape == shape]
        if len(hits) != 1:
            raise SystemExit(
                f"FAIL: expected exactly 1 MatMul weight of shape {shape}, "
                f"got {[a.shape for a in matmuls.values()]}"
            )
        return hits[0]

    enc_w = find_by_shape((ENC_DIM, PRED_HIDDEN))      # [768, 320] enc_proj
    dec_w = find_by_shape((PRED_HIDDEN, PRED_HIDDEN))  # [320, 320] dec_proj
    out_w = find_by_shape((PRED_HIDDEN, VOCAB))        # [320, 34]  out

    tensors = {
        "enc_proj.weight": np.ascontiguousarray(enc_w.T),  # [320, 768]
        "enc_proj.bias": np.ascontiguousarray(enc_bias),
        "dec_proj.weight": np.ascontiguousarray(dec_w.T),  # [320, 320]
        "dec_proj.bias": np.ascontiguousarray(dec_bias),
        "out.weight": np.ascontiguousarray(out_w.T),       # [34, 320]
        "out.bias": np.ascontiguousarray(out_bias),
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(JOINER_OUT_PATH))
    print(f"Saved {len(tensors)} tensors -> {JOINER_OUT_PATH}")

    expected = {
        "enc_proj.weight": (PRED_HIDDEN, ENC_DIM),
        "enc_proj.bias": (PRED_HIDDEN,),
        "dec_proj.weight": (PRED_HIDDEN, PRED_HIDDEN),
        "dec_proj.bias": (PRED_HIDDEN,),
        "out.weight": (VOCAB, PRED_HIDDEN),
        "out.bias": (VOCAB,),
    }
    reloaded = load_file(str(JOINER_OUT_PATH))
    if _check_keys("joiner", expected, reloaded) != 0:
        return 1
    print(f"PASS: all {len(expected)} joiner keys present with correct shapes")
    return 0


def _check_keys(label: str, expected: dict, reloaded: dict) -> int:
    """Assert reloaded safetensors keys+shapes match `expected` exactly."""
    missing, mismatched = [], []
    for key, shape in expected.items():
        if key not in reloaded:
            missing.append(key)
            continue
        got = tuple(reloaded[key].shape)
        if got != shape:
            mismatched.append(f"{key}: expected {shape}, got {got}")
    extra = sorted(set(reloaded) - set(expected))
    if missing or mismatched or extra:
        print(f"FAIL ({label})")
        for k in missing:
            print(f"  MISSING: {k}")
        for k in mismatched:
            print(f"  MISMATCH: {k}")
        for k in extra:
            print(f"  EXTRA: {k}")
        return 1
    return 0


def main() -> int:
    rc = convert_encoder()
    if rc != 0:
        return rc
    print()
    rc = convert_decoder()
    if rc != 0:
        return rc
    print()
    rc = convert_joiner()
    if rc != 0:
        return rc
    print()
    print("PASS: encoder + decoder + joiner conversion complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
