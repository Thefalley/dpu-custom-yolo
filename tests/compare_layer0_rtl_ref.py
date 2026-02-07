#!/usr/bin/env python3
"""
Compare RTL layer0 output (layer0_rtl_output.hex) with Python reference (layer0_output_ref.npy).
RTL uses fixed-point requantize (655/65536); ref may be float (0.01) so allow small diff.
"""
import sys
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent

H_OUT, W_OUT = 208, 208
NUM_CH = 32
TOLERANCE = 1  # allow +/-1 for float vs fixed-point


def find_sim_out():
    for name in ("image_sim_out", "sim_out"):
        d = PROJECT_ROOT / name
        if d.is_dir():
            return d
    return None


def main():
    sim_out = find_sim_out()
    if sim_out is None:
        print("No image_sim_out/ or sim_out/ found.")
        return 1
    rtl_hex = sim_out / "layer0_rtl_output.hex"
    ref_npy = sim_out / "layer0_output_ref.npy"
    if not rtl_hex.exists():
        print(f"RTL output not found: {rtl_hex}")
        return 1
    if not ref_npy.exists():
        print(f"Reference not found: {ref_npy}. Run run_image_to_detection.py first.")
        return 1

    # Read RTL output: one hex byte per line, order (oh, ow, ch)
    rtl_vals = []
    with open(rtl_hex, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # hex to signed int8
            v = int(line, 16)
            if v >= 128:
                v -= 256
            rtl_vals.append(v)
    rtl_vals = np.array(rtl_vals, dtype=np.int8)
    expected_len = H_OUT * W_OUT * NUM_CH
    if len(rtl_vals) != expected_len:
        print(f"RTL output length {len(rtl_vals)} != expected {expected_len}")
        return 1
    # Reshape: TB order (oh, ow, ch) -> (32, 208, 208)
    rtl_3d = rtl_vals.reshape((H_OUT, W_OUT, NUM_CH))
    rtl_3d = np.transpose(rtl_3d, (2, 0, 1))  # (32, 208, 208)

    ref = np.load(ref_npy)
    if ref.shape != (NUM_CH, H_OUT, W_OUT):
        print(f"Reference shape {ref.shape} != ({NUM_CH}, {H_OUT}, {W_OUT})")
        return 1
    ref = ref.astype(np.int32)

    diff = np.abs(rtl_3d.astype(np.int32) - ref)
    max_diff = int(np.max(diff))
    exact = np.sum(diff == 0)
    within_tol = np.sum(diff <= TOLERANCE)
    total = ref.size
    fail = np.sum(diff > TOLERANCE)

    print("=== Compare RTL layer0 vs Python reference ===")
    print(f"  RTL output: {rtl_hex}")
    print(f"  Reference:  {ref_npy}")
    print(f"  Shape: {NUM_CH} x {H_OUT} x {W_OUT}")
    print(f"  Exact match: {exact} / {total}")
    print(f"  Within tolerance +/-{TOLERANCE}: {within_tol} / {total}")
    print(f"  Max diff: {max_diff}")
    print(f"  Mismatch (diff > {TOLERANCE}): {fail}")

    if fail == 0:
        print("RESULT: PASS (RTL matches reference within tolerance)")
        return 0
    else:
        print("RESULT: FAIL (some pixels differ beyond tolerance)")
        return 1


if __name__ == "__main__":
    sys.exit(main())
