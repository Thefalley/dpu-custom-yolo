#!/usr/bin/env python3
"""
Golden for one pixel of layer 0 (real image data) — one or more output channels.
Loads saved image + layer0 weights/bias, computes output pixel (0,0) for channels 0..N-1
using Phase 3 primitives, and writes vectors for RTL comparison.

Run from project root after run_image_to_detection.py (so image_sim_out/ or sim_out/ has the .npy files).
Output: image_sim_out/layer0_patch_*.hex, layer0_patch_golden.json
"""
# Number of output channels to export for the same pixel (0,0)
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

# Same as run_image_to_detection: layer0 stride 2, padding 1
SCALE_RTL = 655  # 0.01 * 65536 for RTL requantize (16-bit fixed point)
SHIFT_RTL = 16


def find_sim_out():
    for name in ("image_sim_out", "sim_out"):
        d = PROJECT_ROOT / name
        if d.is_dir():
            inp = d / "image_input_layer0.npy"
            if inp.exists():
                return d
    return None


def main():
    sim_out = find_sim_out()
    if sim_out is None:
        print("Run run_image_to_detection.py first to create image_sim_out/ or sim_out/ with image_input_layer0.npy")
        return 1

    img = np.load(sim_out / "image_input_layer0.npy")
    w0 = np.load(sim_out / "layer0_weights.npy")
    b0 = np.load(sim_out / "layer0_bias.npy")

    # Pad input (same as conv2d_3x3, padding=1)
    img_pad = np.pad(img, ((0, 0), (1, 1), (1, 1)), mode="constant", constant_values=0)

    # One output pixel (0,0): same input window for all channels. Stride 2 -> window (0:3, 0:3)
    window = img_pad[:, 0:3, 0:3]   # (3, 3, 3)
    flat_a = window.flatten()       # 27

    hex_dir = PROJECT_ROOT / "image_sim_out"
    hex_dir.mkdir(parents=True, exist_ok=True)

    def to_hex8(x):
        return f"{(x & 0xff):02x}"

    # Activations same for all channels
    with open(hex_dir / "layer0_patch_a.hex", "w") as f:
        for x in flat_a.tolist():
            f.write(to_hex8(x) + "\n")

    golden_channels = []
    bias_lines = []
    expected_lines = []

    for ch in range(NUM_CHANNELS):
        flat_w = w0[ch].flatten()
        bias_ch = int(b0[ch])
        conv_sum = int(mac_array(flat_w, flat_a)) + bias_ch
        leaky_out = int(leaky_relu_hardware(np.array([conv_sum], dtype=np.int32))[0])
        expected_int8 = int(requantize_fixed_point(np.array([leaky_out]), np.int32(SCALE_RTL), SHIFT_RTL)[0])
        expected_int8 = max(INT8_MIN, min(INT8_MAX, expected_int8))

        with open(hex_dir / f"layer0_patch_w{ch}.hex", "w") as f:
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

    with open(hex_dir / "layer0_patch_bias.hex", "w") as f:
        f.writelines(bias_lines)
    with open(hex_dir / "layer0_patch_expected.hex", "w") as f:
        f.writelines(expected_lines)

    golden = {
        "activations": flat_a.tolist(),
        "num_channels": NUM_CHANNELS,
        "scale_rtl": SCALE_RTL,
        "shift_rtl": SHIFT_RTL,
        "channels": golden_channels,
    }
    with open(sim_out / "layer0_patch_golden.json", "w") as f:
        json.dump(golden, f, indent=2)

    print(f"Layer0 patch golden (one pixel (0,0), {NUM_CHANNELS} channels)")
    for g in golden_channels:
        print(f"  ch{g['channel']}: conv_sum={g['conv_sum']}, expected_int8={g['expected_int8']}")
    print(f"  Written: {sim_out}/layer0_patch_golden.json, {hex_dir}/layer0_patch_*.hex")
    return 0


if __name__ == "__main__":
    sys.exit(main())
