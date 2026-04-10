#!/usr/bin/env python3
"""Quantize GigaAM v3 encoder to INT8 for faster inference and smaller model size.

Usage:
    pip install onnxruntime
    python scripts/quantize.py [--model-dir ~/.gigastt/models]

Produces: v3_e2e_rnnt_encoder_int8.onnx (~210MB vs 844MB original)
Decoder and Joiner are too small to benefit from quantization.
"""

import argparse
import os
import sys


def main():
    parser = argparse.ArgumentParser(description="Quantize GigaAM encoder to INT8")
    parser.add_argument(
        "--model-dir",
        default=os.path.expanduser("~/.gigastt/models"),
        help="Path to model directory (default: ~/.gigastt/models)",
    )
    args = parser.parse_args()

    model_dir = args.model_dir
    input_path = os.path.join(model_dir, "v3_e2e_rnnt_encoder.onnx")
    output_path = os.path.join(model_dir, "v3_e2e_rnnt_encoder_int8.onnx")

    if not os.path.exists(input_path):
        print(f"Error: {input_path} not found. Run `gigastt download` first.", file=sys.stderr)
        sys.exit(1)

    if os.path.exists(output_path):
        print(f"INT8 model already exists: {output_path}")
        print("Delete it to re-quantize.")
        sys.exit(0)

    try:
        from onnxruntime.quantization import QuantType, quantize_dynamic
    except ImportError:
        print("Error: onnxruntime not installed. Run: pip install onnxruntime", file=sys.stderr)
        sys.exit(1)

    input_size = os.path.getsize(input_path) / (1024 * 1024)
    print(f"Quantizing encoder ({input_size:.0f} MB) to INT8...")
    print(f"  Input:  {input_path}")
    print(f"  Output: {output_path}")

    quantize_dynamic(
        model_input=input_path,
        model_output=output_path,
        weight_type=QuantType.QInt8,
        per_channel=True,  # Better accuracy on ARM64 (SDOT/UDOT)
    )

    output_size = os.path.getsize(output_path) / (1024 * 1024)
    ratio = input_size / output_size
    print(f"Done! {input_size:.0f} MB -> {output_size:.0f} MB ({ratio:.1f}x smaller)")
    print(f"\ngigastt will automatically use the INT8 model when available.")


if __name__ == "__main__":
    main()
