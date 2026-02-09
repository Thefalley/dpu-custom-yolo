#!/usr/bin/env python3
"""
Golden model for dpu_top 18-layer YOLOv4-tiny (H0=16, W0=16 for fast sim).
Runs all 18 layers using phase3_dpu_functional_model primitives,
exports hex files for RTL TB comparison.

IMPORTANT: Uses hardware-accurate requantize_fixed_point to match RTL exactly.

Output directory: image_sim_out/dpu_top/
  - input_image.hex          (3*16*16 = 768 bytes)
  - layerN_weights.hex       (per conv layer)
  - layerN_bias.hex          (per conv layer, 4 bytes LE per channel)
  - layerN_expected.hex      (output fmap for each layer)
  - final_output_expected.hex

Usage: python tests/dpu_top_18layer_golden.py
"""

import sys
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
    INT8_MIN,
    INT8_MAX,
)

# Dimensions
H0 = 16
W0 = 16
SCALE_RTL = 655  # 0.01 * 65536
SHIFT_RTL = 16

# Layer descriptors: (type, c_in, c_out, stride, kernel)
LAYER_DEFS = [
    ('conv3x3',      3,   32, 2, 3),   # 0
    ('conv3x3',     32,   64, 2, 3),   # 1
    ('conv3x3',     64,   64, 1, 3),   # 2
    ('route_split', 64,   32, 0, 0),   # 3  (groups=2, group_id=1)
    ('conv3x3',     32,   32, 1, 3),   # 4
    ('conv3x3',     32,   32, 1, 3),   # 5
    ('route_concat',32,   64, 0, 0),   # 6  (L5 + L4)
    ('conv1x1',     64,   64, 1, 1),   # 7
    ('route_concat',64,  128, 0, 0),   # 8  (L2_save + L7)
    ('maxpool',    128,  128, 2, 0),   # 9
    ('conv3x3',    128,  128, 1, 3),   # 10
    ('route_split',128,   64, 0, 0),   # 11 (groups=2, group_id=1)
    ('conv3x3',     64,   64, 1, 3),   # 12
    ('conv3x3',     64,   64, 1, 3),   # 13
    ('route_concat',64,  128, 0, 0),   # 14 (L13 + L12_save)
    ('conv1x1',    128,  128, 1, 1),   # 15
    ('route_concat',128, 256, 0, 0),   # 16 (L10_save + L15)
    ('maxpool',    256,  256, 2, 0),   # 17
]


def conv_bn_leaky_hw(input_fm, weights, bias, stride=1, kernel_size=3):
    """Hardware-accurate conv+BN+LeakyReLU+requantize using fixed-point scale."""
    if kernel_size == 3:
        conv_out = conv2d_3x3(input_fm, weights, bias, stride=stride)
    else:
        conv_out = conv2d_1x1(input_fm, weights, bias)

    relu_out = leaky_relu_hardware(conv_out)
    output = requantize_fixed_point(relu_out, np.int32(SCALE_RTL), SHIFT_RTL)
    return output


def to_hex8(x):
    return f"{(int(x) & 0xff):02x}"


def write_hex8_file(path, data_flat):
    with open(path, 'w') as f:
        for v in data_flat:
            f.write(to_hex8(v) + '\n')


def write_bias_hex(path, bias_array):
    with open(path, 'w') as f:
        for b in bias_array:
            val = int(b) & 0xffffffff
            f.write(f"{(val >> 0) & 0xff:02x}\n")
            f.write(f"{(val >> 8) & 0xff:02x}\n")
            f.write(f"{(val >> 16) & 0xff:02x}\n")
            f.write(f"{(val >> 24) & 0xff:02x}\n")


def main():
    np.random.seed(42)

    out_dir = PROJECT_ROOT / "image_sim_out" / "dpu_top"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Generate input image
    input_img = np.random.randint(-50, 50, (3, H0, W0), dtype=np.int8)
    write_hex8_file(out_dir / "input_image.hex", input_img.flatten())

    # Storage
    layer_outputs = [None] * 18
    save_l2 = None
    save_l4 = None
    save_l10 = None
    save_l12 = None

    # Generate weights for conv layers
    weights = {}
    biases = {}

    for i, (ltype, c_in, c_out, stride, kernel) in enumerate(LAYER_DEFS):
        if ltype in ('conv3x3', 'conv1x1'):
            w = np.random.randint(-20, 20, (c_out, c_in, kernel, kernel), dtype=np.int8)
            b = np.random.randint(-200, 200, c_out, dtype=np.int32)
            weights[i] = w
            biases[i] = b

    # Run all layers
    current_fmap = input_img

    for i, (ltype, c_in, c_out, stride, kernel) in enumerate(LAYER_DEFS):
        if ltype == 'conv3x3':
            out = conv_bn_leaky_hw(current_fmap, weights[i], biases[i],
                                   stride=stride, kernel_size=3)
            write_hex8_file(out_dir / f"layer{i}_weights.hex", weights[i].flatten())
            write_bias_hex(out_dir / f"layer{i}_bias.hex", biases[i])

        elif ltype == 'conv1x1':
            out = conv_bn_leaky_hw(current_fmap, weights[i], biases[i],
                                   stride=1, kernel_size=1)
            write_hex8_file(out_dir / f"layer{i}_weights.hex", weights[i].flatten())
            write_bias_hex(out_dir / f"layer{i}_bias.hex", biases[i])

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
            else:
                raise ValueError(f"Unknown concat layer {i}")

        elif ltype == 'maxpool':
            out = maxpool_2x2(current_fmap)

        else:
            raise ValueError(f"Unknown layer type {ltype}")

        layer_outputs[i] = out
        current_fmap = out

        if i == 2:
            save_l2 = out.copy()
        elif i == 4:
            save_l4 = out.copy()
        elif i == 10:
            save_l10 = out.copy()
        elif i == 12:
            save_l12 = out.copy()

        write_hex8_file(out_dir / f"layer{i}_expected.hex", out.flatten())

        print(f"Layer {i:2d} ({ltype:14s}): {c_in:3d} -> {c_out:3d}  "
              f"shape={out.shape}  range=[{out.min():4d}, {out.max():3d}]")

    write_hex8_file(out_dir / "final_output_expected.hex", current_fmap.flatten())

    with open(out_dir / "scale.hex", 'w') as f:
        f.write(f"{SCALE_RTL & 0xff:02x}\n")
        f.write(f"{(SCALE_RTL >> 8) & 0xff:02x}\n")

    print(f"\nFinal output shape: {current_fmap.shape}")
    print(f"Files written to: {out_dir}")
    print("GOLDEN COMPLETE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
