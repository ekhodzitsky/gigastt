#!/usr/bin/env python3
"""Production ANE conversion pipeline for the GigaAM v3 ConformerEncoder.

For each requested bucket (a fixed mel-frame length ``N``) this script builds a
mask-free, fixed-window re-implementation of the encoder, traces it, converts it
to a Core ML ``mlprogram`` (FP16, ``CPU_AND_NE``, ``minimum_deployment_target``
macOS 15), then palettizes the weights (k-means, ``per_grouped_channel``) and
writes ``gigaam_v3_encoder_<N>.mlpackage`` to the output directory. The size and
a deterministic SHA-256 are printed for each artifact.

Why per-bucket FIXED windows
----------------------------
The spike established that a per-bucket *fixed* shape keeps ~99.8 % of ops
resident on the Apple Neural Engine for buckets >= 288 mel frames, whereas a
single ``EnumeratedShapes`` package evicts dynamic-shape partitions to the CPU.
The Rust side selects the smallest bucket >= the clip's mel length, pads the mel
up to that bucket, runs the matching package, and trims the encoded output back
to the real subsampled length.

Static-shape tracing
---------------------
The stock RoPE helpers compute ``x.shape[-1] // 2`` and ``half + x.shape[1]``,
which ``torch.jit.trace`` records as data-dependent ``aten::Int`` chains that the
coremltools torch frontend rejects. At a fixed window these are constants, so we
replace the rotate-half / apply-rotary helpers with shape-free ``torch.chunk``
ops and slice the positional table with Python-int literals (numerically
identical -- verified against the stock forward per bucket at convert time).

SHA-256 scheme
--------------
Each ``.mlpackage`` (a directory) is also written to disk as a *deterministic
tar* (``gigaam_v3_encoder_<N>.mlpackage.tar``): entries are emitted in sorted
path order with mtime, owner/group and permissions normalised to fixed values,
so the bytes depend only on the file contents and layout (the model spec + the
palettized ``weight.bin``), not on filesystem metadata. That ``.tar`` IS the
published + downloaded artifact, so its SHA-256 is simultaneously the
content-identity fingerprint AND the value the Rust downloader pins — one
artifact, one digest (no zip, no separate ``SHA256SUMS`` digest). It is
reproducible across runs and machines for a given coremltools build.

Run from the spike directory (which has the GigaAM clone + downloaded model):

  uv run --python 3.12 --with torch --with coremltools --with gigaam \\
      --with soundfile --with scikit-learn --with "numpy<2" \\
      python convert_gigaam_ane.py --buckets 768
"""
import argparse
import hashlib
import os
import shutil
import sys
import tarfile
import time

import numpy as np
import torch
import torch.nn as nn

# The GigaAM source tree (clone) must be importable; the spike keeps it here.
GIGAAM_REPO = os.environ.get("GIGAAM_REPO", "/tmp/gigaam-ane-spike/GigaAM")
if os.path.isdir(GIGAAM_REPO):
    sys.path.insert(0, GIGAAM_REPO)

import gigaam  # noqa: E402
import coremltools as ct  # noqa: E402
import coremltools.optimize.coreml as cto  # noqa: E402

import gigaam.utils as _gu  # noqa: E402
import gigaam.encoder as _ge  # noqa: E402


# ---- Static-shape monkeypatches for clean Core ML tracing (from the spike) ----
def _rtt_half_static(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = torch.chunk(x, 2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def _apply_rotary_static(q, k, cos, sin, offset: int = 0):
    cos = cos.to(dtype=q.dtype)
    sin = sin.to(dtype=q.dtype)
    return (q * cos) + (_rtt_half_static(q) * sin), (k * cos) + (_rtt_half_static(k) * sin)


_gu.rtt_half = _rtt_half_static
_gu.apply_rotary_pos_emb = _apply_rotary_static
_ge.apply_rotary_pos_emb = _apply_rotary_static  # encoder imported the symbol directly


def _patch_rope_emb_forward(pos_enc, t_subsampled: int) -> None:
    """Replace ``RotaryPositionalEmbedding.forward`` with a static-length slice.

    The stock forward slices ``self.pe[half : half + x.shape[1]]``, a
    data-dependent end index that the Core ML torch frontend cannot fold. With a
    fixed window ``x.shape[1]`` is the constant ``t_subsampled``, so we slice with
    Python-int literals (identical result).
    """
    half_pe = pos_enc.pe.shape[0] // 2
    t = int(t_subsampled)

    def _static_forward(x):
        cos_emb = pos_enc.pe[0:t]
        sin_emb = pos_enc.pe[half_pe:half_pe + t]
        return x, [cos_emb, sin_emb]

    pos_enc.forward = _static_forward


class FixedEncoderWrapper(nn.Module):
    """Mask-free, fixed-window re-implementation of ``ConformerEncoder.forward``.

    Takes a single ``mel`` input ``[1, 64, N]`` (the length tensor is dropped).
    At batch=1 with ``length == N`` the stock forward computes ``att_mask=None``
    and an all-False ``pad_mask``, so a mask-free forward is numerically
    identical (verified per bucket at convert time).

    Note: the conv-stage ``_mask_time`` and the all-False ``pad_mask`` are no-ops
    ONLY at ``length == mel_frames`` (a full bucket). The pad-up-then-trim
    equivalence for short clips (pad the mel up to the bucket, run, trim the
    encoded output back to the real subsampled length) is a Phase-1b *runtime*
    invariant this script does not and cannot prove; a padded-input parity test
    belongs in the Rust runtime, not here.
    """

    def __init__(self, encoder):
        super().__init__()
        self.enc = encoder

    def _subsample(self, mel: torch.Tensor) -> torch.Tensor:
        # mel is [1, 64, T]. encoder.forward feeds pre_encode mel.transpose(1,2)
        # == [1, T, 64], which the conv1d pre_encode transposes back to [1, 64, T];
        # passing [1, 64, T] straight in matches after that internal transpose.
        x = mel
        for module in self.enc.pre_encode.conv:
            x = module(x)
        return x.transpose(1, 2)  # [1, T/4, d_model]

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        x = self._subsample(mel)            # [1, T', 768]
        x, pos_emb = self.enc.pos_enc(x=x)  # RoPE cos/sin for the fixed T'
        for layer in self.enc.layers:
            x = layer(x=x, pos_emb=pos_emb, att_mask=None, pad_mask=None)
        return x.transpose(1, 2)            # [1, 768, T']


def write_deterministic_tar(pkg_path: str, tar_path: str) -> str:
    """Write the whole ``.mlpackage`` to ``tar_path`` as a deterministic tar and
    return the SHA-256 hex digest of the on-disk tar.

    Members are emitted in sorted-path order with mtime, uid/gid, owner/group
    names and mode normalised, so the tar bytes reflect only file contents +
    layout (reproducible across runs and machines). Each member is streamed from
    disk by ``tarfile`` (no in-memory BytesIO), and the resulting tar is hashed
    in 1 MiB chunks. The ``.tar`` IS the published + downloaded artifact, so this
    digest is both the content-identity fingerprint and the Rust download pin.
    """
    base = os.path.dirname(pkg_path.rstrip("/"))
    members = []
    for root, _, files in os.walk(pkg_path):
        for f in files:
            members.append(os.path.join(root, f))
    with tarfile.open(tar_path, "w") as tar:
        for full in sorted(members):
            info = tarfile.TarInfo(name=os.path.relpath(full, base))
            info.size = os.path.getsize(full)
            info.mtime = 0
            info.uid = info.gid = 0
            info.uname = info.gname = ""
            info.mode = 0o644
            with open(full, "rb") as fh:
                tar.addfile(info, fh)  # tarfile streams the file contents

    h = hashlib.sha256()
    with open(tar_path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def build_encoder(variant: str):
    """Load the GigaAM model and return its ``encoder`` ready for tracing."""
    print(f"== load {variant} (cpu, fp32) ==")
    t0 = time.time()
    model = gigaam.load_model(variant, fp16_encoder=False, use_flash=False, device="cpu")
    enc = model.encoder.eval()
    enc.pos_enc.extend_pe(enc.pos_emb_max_len, torch.device("cpu"))
    print(f"loaded in {time.time() - t0:.1f}s; subsampling={enc.pre_encode.subsampling_type}; "
          f"pe={tuple(enc.pos_enc.pe.shape)}")
    return enc


def convert_bucket(enc, mel_frames: int, nbits: int, group_size: int, out_dir: str) -> str:
    """Convert + palettize one fixed bucket; return the saved ``.mlpackage`` path."""
    t_sub = int(enc.pre_encode.calc_output_length(torch.tensor(mel_frames)).item())
    print(f"\n== bucket mel={mel_frames} -> subsampled T'={t_sub} ==")

    ctx = enc.onnx_export_mode()  # plain matmul/softmax attention
    ctx.__enter__()
    try:
        mel = torch.randn(1, 64, mel_frames, dtype=torch.float32)

        # Numerical equivalence vs the stock forward, captured BEFORE patching
        # pos_enc.forward (compares the static-slice wrapper to the stock one).
        with torch.inference_mode():
            ref_full, _ = enc(mel, torch.tensor([mel_frames], dtype=torch.int64))
        _patch_rope_emb_forward(enc.pos_enc, t_sub)
        wrapper = FixedEncoderWrapper(enc).eval()
        with torch.inference_mode():
            ref = wrapper(mel)
        eq = (ref - ref_full).abs()
        max_abs = eq.max().item()
        print(f"  wrapper vs stock forward: max_abs={max_abs:.3e} "
              f"mean_abs={eq.mean().item():.3e} out={tuple(ref.shape)}")
        # FP32-vs-FP32: the static wrapper must reproduce the stock forward
        # (expect ~1e-6). Hard-fail rather than silently shipping a divergent
        # graph.
        if max_abs > 1e-3:
            raise SystemExit(
                f"bucket {mel_frames}: static wrapper diverges from stock forward "
                f"(max_abs={max_abs:.3e} > 1e-3)"
            )

        with torch.inference_mode():
            traced = torch.jit.trace(wrapper, mel, check_trace=False)
            traced = torch.jit.freeze(traced.eval())
            torch.jit.run_frozen_optimizations(traced)
            # Explicitly verify the (frozen, optimized) traced graph against the
            # eager reference: check_trace=False skips torch's own re-run, so this
            # closes the silent-bad-trace gap.
            traced_out = traced(mel)
        traced_abs = (traced_out - ref).abs().max().item()
        if traced_abs > 1e-3:
            raise SystemExit(
                f"bucket {mel_frames}: traced graph diverges from wrapper "
                f"(max_abs={traced_abs:.3e} > 1e-3)"
            )

        print("  ct.convert (FP16, CPU_AND_NE, macOS15, mlprogram)")
        t1 = time.time()
        mlmodel = ct.convert(
            traced,
            inputs=[ct.TensorType(name="mel", shape=(1, 64, mel_frames), dtype=np.float32)],
            outputs=[ct.TensorType(name="encoded", dtype=np.float32)],
            compute_precision=ct.precision.FLOAT16,
            compute_units=ct.ComputeUnit.CPU_AND_NE,
            minimum_deployment_target=ct.target.macOS15,
            convert_to="mlprogram",
        )
        print(f"  converted in {time.time() - t1:.1f}s")
    finally:
        ctx.__exit__(None, None, None)

    print(f"  palettize (kmeans, {nbits}-bit, per_grouped_channel, group_size={group_size})")
    t2 = time.time()
    op_cfg = cto.OpPalettizerConfig(
        mode="kmeans",
        nbits=nbits,
        granularity="per_grouped_channel",
        group_size=group_size,
    )
    pal = cto.palettize_weights(mlmodel, config=cto.OptimizationConfig(global_config=op_cfg))
    print(f"  palettized in {time.time() - t2:.1f}s")

    out_pkg = os.path.join(out_dir, f"gigaam_v3_encoder_{mel_frames}.mlpackage")
    if os.path.exists(out_pkg):
        shutil.rmtree(out_pkg)
    pal.save(out_pkg)  # keep the unpacked .mlpackage for local use too

    out_tar = out_pkg + ".tar"
    digest = write_deterministic_tar(out_pkg, out_tar)
    size = os.path.getsize(out_tar) / 1e6
    print(f"  saved {out_pkg}")
    print(f"  wrote {out_tar}")
    print(f"  size={size:.1f} MB  sha256={digest}")
    return out_pkg


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Convert + palettize the GigaAM v3 encoder to per-bucket ANE Core ML packages.",
    )
    p.add_argument("--buckets", default="512,768,1536,3000",
                   help="Comma-separated mel-frame bucket lengths (default: 512,768,1536,3000).")
    p.add_argument("--nbits", type=int, default=6, help="Palettization bit-width (default: 6).")
    p.add_argument("--group-size", type=int, default=32,
                   help="per_grouped_channel group size (default: 32).")
    p.add_argument("--out-dir", default=os.path.expanduser("~/.gigastt/models/ane"),
                   help="Output directory for the .mlpackage artifacts.")
    p.add_argument("--variant", default="v3_rnnt", help="GigaAM model variant (default: v3_rnnt).")
    args = p.parse_args(argv)
    args.buckets = [int(b) for b in args.buckets.split(",") if b.strip()]
    return args


def main(argv=None) -> None:
    args = parse_args(argv)
    os.makedirs(args.out_dir, exist_ok=True)
    print(f"buckets={args.buckets} nbits={args.nbits} group_size={args.group_size} "
          f"variant={args.variant} out_dir={args.out_dir}")

    enc = build_encoder(args.variant)
    saved = [
        convert_bucket(enc, n, args.nbits, args.group_size, args.out_dir)
        for n in args.buckets
    ]

    print("\n== artifacts ==")
    for pkg in saved:
        tar_path = pkg + ".tar"
        h = hashlib.sha256()
        with open(tar_path, "rb") as fh:
            for chunk in iter(lambda: fh.read(1024 * 1024), b""):
                h.update(chunk)
        size_mb = os.path.getsize(tar_path) / 1e6
        print(f"  {os.path.basename(tar_path)}  {size_mb:.1f} MB  {h.hexdigest()}")
    print("DONE_CONVERT")


if __name__ == "__main__":
    main()
