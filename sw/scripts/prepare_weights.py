#!/usr/bin/env python3
"""
Prepare weight/bias binary files for the DPU bare-metal application.

Reads the quantized weights (from load_yolov4_tiny_weights.py or golden model)
and exports them as flat binary files ready for the Zynq application:
  - weights.bin:  All conv layer weights concatenated (cin-contiguous layout)
  - biases.bin:   All conv layer biases as int32 LE
  - scales.bin:   Per-layer requant scales as uint16 LE (36 entries)
  - input.bin:    Input image as INT8 CHW (optional)

Usage:
  python prepare_weights.py                     # Use real weights
  python prepare_weights.py --input-image dog.jpg
  python prepare_weights.py --output-dir /path  # Custom output dir
"""
import sys
import struct
import argparse
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT / "tests"))

# Layer definitions matching the 36-layer golden model
CONV_LAYER_INDICES = [0, 1, 2, 4, 5, 7, 10, 12, 13, 15,
                      18, 20, 21, 23, 26, 27, 28, 29, 31, 34, 35]

H0, W0 = 32, 32


def load_real_weights():
    """Load quantized weights from npz."""
    npz_path = PROJECT_ROOT / "image_sim_out" / "dpu_top_real" / "quantized_weights.npz"
    if not npz_path.exists():
        print(f"Weights not found at {npz_path}")
        print("Run: python tests/load_yolov4_tiny_weights.py")
        sys.exit(1)
    return np.load(str(npz_path))


def prepare_input_image(image_path):
    """Load and preprocess an image to INT8 CHW."""
    from PIL import Image
    img = Image.open(image_path).convert("RGB").resize((W0, H0))
    arr = np.array(img, dtype=np.float32)
    # Normalize to INT8: pixel/255 * 254 - 127
    arr_int8 = np.clip(np.round(arr / 255.0 * 254.0 - 127.0), -128, 127).astype(np.int8)
    # HWC -> CHW
    arr_chw = arr_int8.transpose(2, 0, 1)
    return arr_chw


def main():
    ap = argparse.ArgumentParser(description="Prepare DPU weight binaries")
    ap.add_argument("--output-dir", type=str, default=None,
                    help="Output directory (default: sw/data/)")
    ap.add_argument("--input-image", type=str, default=None,
                    help="Input image to preprocess")
    args = ap.parse_args()

    out_dir = Path(args.output_dir) if args.output_dir else PROJECT_ROOT / "sw" / "data"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading real YOLOv4-tiny weights...")
    data = load_real_weights()

    # Export weights (concatenated, cin-contiguous layout)
    weight_parts = []
    bias_parts = []
    total_w = 0
    total_b = 0

    for layer_idx in CONV_LAYER_INDICES:
        w_key = f"layer{layer_idx}_weights"
        b_key = f"layer{layer_idx}_bias"

        if w_key not in data:
            print(f"  WARNING: {w_key} not in npz, skipping")
            continue

        w = data[w_key]   # Already in cin-contiguous layout (int8)
        b = data[b_key]   # int32 biases

        weight_parts.append(w.astype(np.int8).tobytes())
        # Bias as int32 little-endian
        for bval in b.flatten():
            bias_parts.append(struct.pack('<i', int(bval)))

        total_w += w.size
        total_b += b.size
        print(f"  Layer {layer_idx:2d}: weights={w.size:8d} bytes, biases={b.size:4d}")

    # Write weights.bin
    w_path = out_dir / "weights.bin"
    with open(w_path, 'wb') as f:
        for part in weight_parts:
            f.write(part)
    print(f"\nWeights: {w_path} ({total_w} bytes)")

    # Write biases.bin
    b_path = out_dir / "biases.bin"
    with open(b_path, 'wb') as f:
        for part in bias_parts:
            f.write(part)
    print(f"Biases:  {b_path} ({total_b * 4} bytes)")

    # Write scales.bin (36 entries x 2 bytes, LE uint16)
    scale_path = out_dir / "scales.bin"
    scales_npz = PROJECT_ROOT / "image_sim_out" / "dpu_top_real" / "output_scales.npz"
    with open(scale_path, 'wb') as f:
        if scales_npz.exists():
            sdata = np.load(str(scales_npz))
            for i in range(36):
                key = f"scale_{i}"
                s = int(sdata[key]) if key in sdata else 655
                f.write(struct.pack('<H', s & 0xFFFF))
            print(f"Scales:  {scale_path} (36 x 2 bytes, from calibration)")
        else:
            for i in range(36):
                f.write(struct.pack('<H', 655))
            print(f"Scales:  {scale_path} (36 x 2 bytes, default 655)")

    # Write input.bin (optional)
    if args.input_image:
        img_int8 = prepare_input_image(args.input_image)
        in_path = out_dir / "input.bin"
        img_int8.tofile(str(in_path))
        print(f"Input:   {in_path} ({img_int8.size} bytes, {img_int8.shape})")

    print(f"\nAll files written to: {out_dir}")
    print("Copy these to the SD card or embed in the ELF.")


if __name__ == "__main__":
    main()
