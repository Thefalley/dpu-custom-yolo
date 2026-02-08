#!/usr/bin/env python3
"""
Golden for Layer 13: Conv 3x3, 64->64, stride 1 (576 MACs per pixel)
Loads layer12 output + layer13 weights/bias, computes output pixel (0,0) for channels 0..N-1
using Phase 3 primitives, and writes vectors for RTL comparison.

Run from project root after run_image_to_detection.py --layers 14
Output: image_sim_out/layer13_patch_*.hex, layer13_patch_golden.json
"""
NUM_CHANNELS = 4

import sys
import json
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


def find_sim_out():
    for name in ("image_sim_out", "sim_out"):
        d = PROJECT_ROOT / name
        if d.is_dir() and (d / "layer12_output_ref.npy").exists():
            return d
    return None


def main():
    sim_out = find_sim_out()
    if sim_out is None:
        print("Run 'run_image_to_detection.py --layers 14' first")
        return 1

    inp = np.load(sim_out / "layer12_output_ref.npy")   # (64, 52, 52)
    w = np.load(sim_out / "layer13_weights.npy")          # (64, 64, 3, 3)
    b = np.load(sim_out / "layer13_bias.npy")             # (64,)

    print(f"Layer13 patch golden (one pixel (0,0), {NUM_CHANNELS} channels)")
    print(f"  Input shape: {inp.shape} ({inp.shape[0]} channels)")
    print(f"  Weight shape: {w.shape}")
    print(f"  MACs per output pixel: {inp.shape[0]} * 9 = {inp.shape[0] * 9}")

    inp_pad = np.pad(inp, ((0, 0), (1, 1), (1, 1)), mode="constant", constant_values=0)
    window = inp_pad[:, 0:3, 0:3]
    flat_a = window.flatten()

    hex_dir = PROJECT_ROOT / "image_sim_out"
    hex_dir.mkdir(parents=True, exist_ok=True)

    def to_hex8(x):
        return f"{(int(x) & 0xff):02x}"

    with open(hex_dir / "layer13_patch_a.hex", "w") as f:
        for x in flat_a.tolist():
            f.write(to_hex8(x) + "\n")

    golden_channels = []
    bias_lines = []
    expected_lines = []

    for ch in range(NUM_CHANNELS):
        flat_w = w[ch].flatten()
        bias_ch = int(b[ch])
        conv_sum = int(mac_array(flat_w, flat_a)) + bias_ch
        leaky_out = int(leaky_relu_hardware(np.array([conv_sum], dtype=np.int32))[0])
        expected_int8 = int(requantize_fixed_point(np.array([leaky_out]), np.int32(SCALE_RTL), SHIFT_RTL)[0])
        expected_int8 = max(INT8_MIN, min(INT8_MAX, expected_int8))

        with open(hex_dir / f"layer13_patch_w{ch}.hex", "w") as f:
            for x in flat_w.tolist():
                f.write(to_hex8(x) + "\n")
        bias_lines.append(f"{(bias_ch & 0xffffffff):08x}\n")
        expected_lines.append(to_hex8(expected_int8) + "\n")
        golden_channels.append({
            "channel": ch,
            "conv_sum": conv_sum,
            "leaky_out": leaky_out,
            "expected_int8": expected_int8,
        })
        print(f"  ch{ch}: conv_sum={conv_sum}, expected_int8={expected_int8}")

    with open(hex_dir / "layer13_patch_bias.hex", "w") as f:
        f.writelines(bias_lines)
    with open(hex_dir / "layer13_patch_expected.hex", "w") as f:
        f.writelines(expected_lines)

    golden = {
        "layer": 13,
        "input_channels": int(inp.shape[0]),
        "macs_per_pixel": int(inp.shape[0]) * 9,
        "activations_count": len(flat_a),
        "num_channels": NUM_CHANNELS,
        "scale_rtl": SCALE_RTL,
        "shift_rtl": SHIFT_RTL,
        "channels": golden_channels,
    }
    with open(sim_out / "layer13_patch_golden.json", "w") as f:
        json.dump(golden, f, indent=2)

    print(f"  Written: layer13_patch_golden.json, layer13_patch_*.hex")
    return 0


if __name__ == "__main__":
    sys.exit(main())
