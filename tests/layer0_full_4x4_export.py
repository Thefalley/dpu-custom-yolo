#!/usr/bin/env python3
"""
Export data for RTL full layer 0 — first 4×4 output region.
Output: padded input patch 3×9×9 (243 bytes), weights 32×27, bias 32,
expected 4×4×32 (512 bytes) in image_sim_out/layer0_full4x4_*.hex.

Run after run_image_to_detection.py and tests/layer0_full_golden.py
(or run_layer0_full_check.py which runs them).
"""
import sys
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from phase3_dpu_functional_model import (
    mac_array,
    leaky_relu_hardware,
    requantize_fixed_point,
    INT8_MIN,
    INT8_MAX,
)

SCALE_RTL = 655
SHIFT_RTL = 16
STRIDE = 2
PAD = 1
C_OUT = 32
H4, W4 = 4, 4  # 4×4 output region


def find_sim_out():
    for name in ("image_sim_out", "sim_out"):
        d = PROJECT_ROOT / name
        if d.is_dir() and (d / "image_input_layer0.npy").exists():
            return d
    return None


def to_hex8(x):
    return f"{(int(x) & 0xff):02x}"


def to_hex32(x):
    return f"{(int(x) & 0xffffffff):08x}"


def main():
    sim_out = find_sim_out()
    if sim_out is None:
        print("Run run_image_to_detection.py first.")
        return 1
    hex_dir = PROJECT_ROOT / "image_sim_out"
    hex_dir.mkdir(parents=True, exist_ok=True)

    img = np.load(sim_out / "image_input_layer0.npy")
    w0 = np.load(sim_out / "layer0_weights.npy")
    b0 = np.load(sim_out / "layer0_bias.npy")

    img_pad = np.pad(img, ((0, 0), (PAD, PAD), (PAD, PAD)), mode="constant", constant_values=0)
    # For output (0,0)..(3,3) we need input rows 0..8, cols 0..8 (stride 2: (3,3) needs [6:9,6:9])
    patch_h, patch_w = 9, 9
    padded_patch = img_pad[:, :patch_h, :patch_w]  # (3, 9, 9)
    with open(hex_dir / "layer0_full4x4_padded.hex", "w") as f:
        for x in padded_patch.flatten().tolist():
            f.write(to_hex8(x) + "\n")

    with open(hex_dir / "layer0_full4x4_weights.hex", "w") as f:
        for c in range(C_OUT):
            for x in w0[c].flatten().tolist():
                f.write(to_hex8(x) + "\n")

    with open(hex_dir / "layer0_full4x4_bias.hex", "w") as f:
        for c in range(C_OUT):
            f.write(to_hex32(b0[c]) + "\n")

    expected = []
    for oh in range(H4):
        for ow in range(W4):
            h_in = oh * STRIDE
            w_in = ow * STRIDE
            window = img_pad[:, h_in : h_in + 3, w_in : w_in + 3]
            flat_a = window.flatten()
            for c in range(C_OUT):
                flat_w = w0[c].flatten()
                acc = int(mac_array(flat_w, flat_a)) + int(b0[c])
                leaky = int(leaky_relu_hardware(np.array([acc], dtype=np.int32))[0])
                q = int(requantize_fixed_point(np.array([leaky], dtype=np.int32), np.int32(SCALE_RTL), SHIFT_RTL)[0])
                q = max(INT8_MIN, min(INT8_MAX, q))
                expected.append(q)

    with open(hex_dir / "layer0_full4x4_expected.hex", "w") as f:
        for x in expected:
            f.write(to_hex8(x) + "\n")

    print("Exported layer0_full4x4_*.hex: padded 243, weights 864, bias 32, expected 512")
    return 0


if __name__ == "__main__":
    sys.exit(main())
