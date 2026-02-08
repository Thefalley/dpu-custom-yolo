#!/usr/bin/env python3
"""
Golden for one pixel of Layer 5 (real image data) - one or more output channels.
Loads layer4 output + layer5 weights/bias, computes output pixel (0,0) for channels 0..N-1
using Phase 3 primitives, and writes vectors for RTL comparison.

Layer 5: 32 input channels, 32 output channels, 3x3 kernel, stride 1
Each output pixel: 32 * 3 * 3 = 288 MACs

Run from project root after run_image_to_detection.py --layers 6
Output: image_sim_out/layer5_patch_*.hex, layer5_patch_golden.json
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

# Same scale as other layers for RTL requantize (16-bit fixed point)
SCALE_RTL = 655  # 0.01 * 65536
SHIFT_RTL = 16


def find_sim_out():
    for name in ("image_sim_out", "sim_out"):
        d = PROJECT_ROOT / name
        if d.is_dir():
            inp = d / "layer4_output_ref.npy"
            if inp.exists():
                return d
    return None


def main():
    sim_out = find_sim_out()
    if sim_out is None:
        print("Run 'run_image_to_detection.py --layers 6' first to create layer4_output_ref.npy")
        return 1

    # Layer 5 input is layer 4 output
    layer4_out = np.load(sim_out / "layer4_output_ref.npy")  # (32, 104, 104)
    w5 = np.load(sim_out / "layer5_weights.npy")              # (32, 32, 3, 3)
    b5 = np.load(sim_out / "layer5_bias.npy")                 # (32,)

    print(f"Layer5 patch golden (one pixel (0,0), {NUM_CHANNELS} channels)")
    print(f"  Input shape: {layer4_out.shape} (32 channels)")
    print(f"  Weight shape: {w5.shape} (32 filters, 32 channels, 3x3)")
    print(f"  MACs per output pixel: 32 * 9 = 288")

    # Pad input (same as conv2d_3x3, padding=1)
    layer4_pad = np.pad(layer4_out, ((0, 0), (1, 1), (1, 1)), mode="constant", constant_values=0)

    # One output pixel (0,0): stride 1 -> window starts at (0,0)
    # Window: all 32 input channels, 3x3 spatial
    window = layer4_pad[:, 0:3, 0:3]   # (32, 3, 3)
    flat_a = window.flatten()           # 288

    hex_dir = PROJECT_ROOT / "image_sim_out"
    hex_dir.mkdir(parents=True, exist_ok=True)

    def to_hex8(x):
        return f"{(int(x) & 0xff):02x}"

    # Activations same for all output channels
    with open(hex_dir / "layer5_patch_a.hex", "w") as f:
        for x in flat_a.tolist():
            f.write(to_hex8(x) + "\n")

    golden_channels = []
    bias_lines = []
    expected_lines = []

    for ch in range(NUM_CHANNELS):
        flat_w = w5[ch].flatten()  # (32*3*3) = 288
        bias_ch = int(b5[ch])
        conv_sum = int(mac_array(flat_w, flat_a)) + bias_ch
        leaky_out = int(leaky_relu_hardware(np.array([conv_sum], dtype=np.int32))[0])
        expected_int8 = int(requantize_fixed_point(np.array([leaky_out]), np.int32(SCALE_RTL), SHIFT_RTL)[0])
        expected_int8 = max(INT8_MIN, min(INT8_MAX, expected_int8))

        with open(hex_dir / f"layer5_patch_w{ch}.hex", "w") as f:
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

    with open(hex_dir / "layer5_patch_bias.hex", "w") as f:
        f.writelines(bias_lines)
    with open(hex_dir / "layer5_patch_expected.hex", "w") as f:
        f.writelines(expected_lines)

    golden = {
        "layer": 5,
        "input_channels": 32,
        "macs_per_pixel": 288,
        "activations_count": len(flat_a),
        "num_channels": NUM_CHANNELS,
        "scale_rtl": SCALE_RTL,
        "shift_rtl": SHIFT_RTL,
        "channels": golden_channels,
    }
    with open(sim_out / "layer5_patch_golden.json", "w") as f:
        json.dump(golden, f, indent=2)

    print(f"  Written: {sim_out}/layer5_patch_golden.json, {hex_dir}/layer5_patch_*.hex")
    return 0


if __name__ == "__main__":
    sys.exit(main())
