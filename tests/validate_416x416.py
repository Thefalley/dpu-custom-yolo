#!/usr/bin/env python3
"""
Validate YOLOv4-tiny golden model with full 416x416 input size.

This verifies that the DPU golden model produces correct shapes and
reasonable outputs at the real YOLOv4-tiny input resolution.

Python-only (RTL simulation at 416x416 would take hours in Icarus).

Usage:
  python tests/validate_416x416.py
  python tests/validate_416x416.py --real-weights
  python tests/validate_416x416.py --real-weights --input-image photo.jpg
"""

import sys
import argparse
import time
from pathlib import Path
import numpy as np

SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from phase3_dpu_functional_model import (
    conv2d_3x3,
    conv2d_1x1,
    leaky_relu_hardware,
    requantize_fixed_point,
    maxpool_2x2,
    route_split,
    route_concat,
)

# --------------------------------------------------------------------------
# Full YOLOv4-tiny resolution
# --------------------------------------------------------------------------
H0 = 416
W0 = 416
SCALE_RTL = 655
SHIFT_RTL = 16

LAYER_DEFS = [
    ('conv3x3',      3,   32, 2, 3),   # 0: 416x416 -> 208x208
    ('conv3x3',     32,   64, 2, 3),   # 1: 208x208 -> 104x104
    ('conv3x3',     64,   64, 1, 3),   # 2: 104x104 -> 104x104
    ('route_split', 64,   32, 0, 0),   # 3: split -> 32ch
    ('conv3x3',     32,   32, 1, 3),   # 4: 104x104 -> 104x104
    ('conv3x3',     32,   32, 1, 3),   # 5: 104x104 -> 104x104
    ('route_concat',32,   64, 0, 0),   # 6: concat L5+L4 -> 64ch
    ('conv1x1',     64,   64, 1, 1),   # 7: 104x104 -> 104x104
    ('route_concat',64,  128, 0, 0),   # 8: concat L2_save+L7 -> 128ch
    ('maxpool',    128,  128, 2, 0),   # 9: 104x104 -> 52x52
    ('conv3x3',    128,  128, 1, 3),   # 10: 52x52 -> 52x52
    ('route_split',128,   64, 0, 0),   # 11: split -> 64ch
    ('conv3x3',     64,   64, 1, 3),   # 12: 52x52 -> 52x52
    ('conv3x3',     64,   64, 1, 3),   # 13: 52x52 -> 52x52
    ('route_concat',64,  128, 0, 0),   # 14: concat L13+L12_save -> 128ch
    ('conv1x1',    128,  128, 1, 1),   # 15: 52x52 -> 52x52
    ('route_concat',128, 256, 0, 0),   # 16: concat L10_save+L15 -> 256ch
    ('maxpool',    256,  256, 2, 0),   # 17: 52x52 -> 26x26
]


def calibrate_scale(input_fm, weights, bias, stride=1, kernel_size=3):
    if kernel_size == 3:
        conv_out = conv2d_3x3(input_fm, weights, bias, stride=stride)
    else:
        conv_out = conv2d_1x1(input_fm, weights, bias)
    relu_out = leaky_relu_hardware(conv_out)
    max_abs = int(np.max(np.abs(relu_out)))
    if max_abs == 0:
        return SCALE_RTL
    scale = int(127 * (1 << SHIFT_RTL) / max_abs)
    return max(1, min(scale, 65535))


def conv_bn_leaky_hw(input_fm, weights, bias, stride=1, kernel_size=3, scale=SCALE_RTL):
    if kernel_size == 3:
        conv_out = conv2d_3x3(input_fm, weights, bias, stride=stride)
    else:
        conv_out = conv2d_1x1(input_fm, weights, bias)
    relu_out = leaky_relu_hardware(conv_out)
    return requantize_fixed_point(relu_out, np.int32(scale), SHIFT_RTL)


def load_input_image(image_path):
    from PIL import Image
    img = Image.open(image_path).convert('RGB')
    img = img.resize((W0, H0), Image.BILINEAR)
    arr = np.array(img, dtype=np.float32)
    arr = np.round(arr / 255.0 * 254.0 - 127.0)
    arr = np.clip(arr, -128, 127).astype(np.int8)
    return arr.transpose(2, 0, 1)  # HWC -> CHW


def main():
    parser = argparse.ArgumentParser(description='Validate 416x416 golden model')
    parser.add_argument('--real-weights', action='store_true')
    parser.add_argument('--input-image', type=str, default=None)
    args = parser.parse_args()

    print("=" * 70)
    print(f"  YOLOv4-tiny Golden Model Validation — {H0}x{W0} Input")
    print("=" * 70)

    np.random.seed(42)
    use_per_layer_scale = args.real_weights

    # Input
    if args.input_image:
        print(f"\n[INPUT] Loading image: {args.input_image}")
        input_img = load_input_image(args.input_image)
    else:
        print(f"\n[INPUT] Generating random input: 3x{H0}x{W0}")
        input_img = np.random.randint(-50, 50, (3, H0, W0), dtype=np.int8)
    print(f"  Shape: {input_img.shape}, range=[{input_img.min()}, {input_img.max()}]")

    # Weights
    weights = {}
    biases = {}

    if args.real_weights:
        npz_path = PROJECT_ROOT / "image_sim_out" / "dpu_top_real" / "quantized_weights.npz"
        if npz_path.exists():
            print("[WEIGHTS] Loading real YOLOv4-tiny weights")
            data = np.load(npz_path)
            conv_layers = [0, 1, 2, 4, 5, 7, 10, 12, 13, 15]
            for idx in conv_layers:
                weights[idx] = data[f'w{idx}']
                biases[idx] = data[f'b{idx}']
        else:
            print(f"[WARNING] Real weights not found at {npz_path}, using synthetic")
            args.real_weights = False

    if not args.real_weights:
        print("[WEIGHTS] Using synthetic random weights")
        for i, (ltype, c_in, c_out, stride, kernel) in enumerate(LAYER_DEFS):
            if ltype in ('conv3x3', 'conv1x1'):
                weights[i] = np.random.randint(-20, 20, (c_out, c_in, kernel, kernel), dtype=np.int8)
                biases[i] = np.random.randint(-200, 200, c_out, dtype=np.int32)

    # Expected shapes
    expected_shapes = [
        (32, 208, 208), (64, 104, 104), (64, 104, 104), (32, 104, 104),
        (32, 104, 104), (32, 104, 104), (64, 104, 104), (64, 104, 104),
        (128, 104, 104), (128, 52, 52), (128, 52, 52), (64, 52, 52),
        (64, 52, 52), (64, 52, 52), (128, 52, 52), (128, 52, 52),
        (256, 52, 52), (256, 26, 26),
    ]

    # Run
    print(f"\n{'Layer':>6s} {'Type':>14s} {'Cin':>4s} {'Cout':>4s} "
          f"{'Output Shape':>18s} {'Expected':>18s} {'Match':>6s} "
          f"{'Range':>12s} {'Time':>8s}")
    print("-" * 100)

    layer_outputs = [None] * 18
    save_l2 = save_l4 = save_l10 = save_l12 = None
    current_fmap = input_img
    layer_scales = [SCALE_RTL] * 18
    total_time = 0.0
    all_pass = True

    for i, (ltype, c_in, c_out, stride, kernel) in enumerate(LAYER_DEFS):
        t0 = time.time()

        if ltype == 'conv3x3':
            if use_per_layer_scale:
                layer_scales[i] = calibrate_scale(current_fmap, weights[i], biases[i],
                                                  stride=stride, kernel_size=3)
            out = conv_bn_leaky_hw(current_fmap, weights[i], biases[i],
                                   stride=stride, kernel_size=3, scale=layer_scales[i])
        elif ltype == 'conv1x1':
            if use_per_layer_scale:
                layer_scales[i] = calibrate_scale(current_fmap, weights[i], biases[i],
                                                  stride=1, kernel_size=1)
            out = conv_bn_leaky_hw(current_fmap, weights[i], biases[i],
                                   stride=1, kernel_size=1, scale=layer_scales[i])
        elif ltype == 'route_split':
            out = route_split(current_fmap, groups=2, group_id=1)
        elif ltype == 'route_concat':
            if i == 6:
                out = route_concat([layer_outputs[5], save_l4])
            elif i == 8:
                out = route_concat([save_l2, layer_outputs[7]])
            elif i == 14:
                out = route_concat([layer_outputs[13], save_l12])
            elif i == 16:
                out = route_concat([save_l10, layer_outputs[15]])
        elif ltype == 'maxpool':
            out = maxpool_2x2(current_fmap)

        dt = time.time() - t0
        total_time += dt

        layer_outputs[i] = out
        current_fmap = out

        if i == 2:  save_l2 = out.copy()
        elif i == 4:  save_l4 = out.copy()
        elif i == 10: save_l10 = out.copy()
        elif i == 12: save_l12 = out.copy()

        shape_match = (out.shape == expected_shapes[i])
        status = "PASS" if shape_match else "FAIL"
        if not shape_match:
            all_pass = False

        print(f"  {i:4d}  {ltype:>14s} {c_in:4d} {c_out:4d}  "
              f"{str(out.shape):>18s} {str(expected_shapes[i]):>18s} {status:>6s}  "
              f"[{out.min():4d},{out.max():4d}]  {dt:7.2f}s")

    # Summary
    print("-" * 100)
    print(f"\nFinal output shape: {current_fmap.shape}")
    print(f"Total inference time: {total_time:.1f}s (Python/NumPy)")
    print(f"Output range: [{current_fmap.min()}, {current_fmap.max()}]")
    print(f"Non-zero outputs: {np.count_nonzero(current_fmap)}/{current_fmap.size}")

    # Estimate DPU cycle count at 416x416
    # Using measured per-pixel cost from 16x16 validation
    # Layer 0: 3*32*9 = 864 MACs/pixel, 208*208 = 43264 pixels
    total_macs = 0
    for i, (ltype, c_in, c_out, stride, kernel) in enumerate(LAYER_DEFS):
        if ltype in ('conv3x3', 'conv1x1'):
            h_out, w_out = expected_shapes[i][1], expected_shapes[i][2]
            macs = c_in * c_out * kernel * kernel * h_out * w_out
            total_macs += macs

    print(f"\nTotal MAC operations: {total_macs:,}")
    print(f"At 1024 MACs/cycle (32x32 array): {total_macs // 1024:,} ideal cycles")
    print(f"At ~10% utilization (measured): {total_macs // 100:,} estimated cycles")
    print(f"At 100 MHz: {total_macs / 100 / 100e6 * 1000:.1f} ms estimated")

    # Save intermediate outputs for post-processing
    out_dir = PROJECT_ROOT / "image_sim_out" / "dpu_top_416"
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(out_dir / "backbone_outputs.npz",
             layer8=layer_outputs[8],
             layer15=layer_outputs[15],
             layer17=layer_outputs[17])
    print(f"\nBackbone outputs saved to: {out_dir / 'backbone_outputs.npz'}")

    if all_pass:
        print("\n*** ALL 18 LAYERS PASS (shape validation) ***")
    else:
        print("\n*** SOME LAYERS FAILED ***")

    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
