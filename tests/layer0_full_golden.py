#!/usr/bin/env python3
"""
Full layer 0 golden: recompute entire layer 0 with same primitives as RTL
(mac_array, leaky_relu_hardware, requantize_fixed_point) and compare with
layer0_output_ref.npy (from conv_bn_leaky with float requantize).
Saves layer0_output_ref_fp.npy (fixed-point reference) for RTL comparison.

Run from project root after run_image_to_detection.py.
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


def find_sim_out():
    for name in ("image_sim_out", "sim_out"):
        d = PROJECT_ROOT / name
        if d.is_dir() and (d / "image_input_layer0.npy").exists():
            return d
    return None


def run_full_layer0_fixed_point(img, w0, b0):
    """Layer 0: conv 3x3 stride 2 pad 1, bias, leaky_relu_hw, requantize_fp. Same as RTL."""
    img_pad = np.pad(img, ((0, 0), (PAD, PAD), (PAD, PAD)), mode="constant", constant_values=0)
    _, H, W = img.shape
    H_out = (H + 2 * PAD - 3) // STRIDE + 1
    W_out = (W + 2 * PAD - 3) // STRIDE + 1
    out = np.zeros((C_OUT, H_out, W_out), dtype=np.int8)
    for c in range(C_OUT):
        for oh in range(H_out):
            for ow in range(W_out):
                h_in = oh * STRIDE
                w_in = ow * STRIDE
                window = img_pad[:, h_in : h_in + 3, w_in : w_in + 3]
                flat_a = window.flatten()
                flat_w = w0[c].flatten()
                acc = int(mac_array(flat_w, flat_a)) + int(b0[c])
                leaky = int(leaky_relu_hardware(np.array([acc], dtype=np.int32))[0])
                q = int(requantize_fixed_point(np.array([leaky], dtype=np.int32), np.int32(SCALE_RTL), SHIFT_RTL)[0])
                out[c, oh, ow] = np.clip(q, INT8_MIN, INT8_MAX)
    return out


def main():
    sim_out = find_sim_out()
    if sim_out is None:
        print("Run run_image_to_detection.py first.")
        return 1
    img = np.load(sim_out / "image_input_layer0.npy")
    w0 = np.load(sim_out / "layer0_weights.npy")
    b0 = np.load(sim_out / "layer0_bias.npy")
    out_fp = run_full_layer0_fixed_point(img, w0, b0)
    np.save(sim_out / "layer0_output_ref_fp.npy", out_fp)
    print(f"Full layer 0 (fixed-point): shape {out_fp.shape} -> layer0_output_ref_fp.npy")

    ref_path = sim_out / "layer0_output_ref.npy"
    if ref_path.exists():
        ref_float = np.load(ref_path)
        diff = np.abs(out_fp.astype(np.int32) - ref_float.astype(np.int32))
        max_diff = int(np.max(diff))
        match = np.array_equal(out_fp, ref_float)
        print(f"  vs float ref: max_diff={max_diff}  exact={match}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
