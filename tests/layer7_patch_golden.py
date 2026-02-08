#!/usr/bin/env python3
"""
Golden for one pixel of Layer 7 (real image data) - one or more output channels.
Loads layer6 output + layer7 weights/bias, computes output pixel (0,0) for channels 0..N-1
using Phase 3 primitives, and writes vectors for RTL comparison.

Layer 7: 64 input channels, 64 output channels, 1x1 kernel, stride 1
Each output pixel: 64 * 1 * 1 = 64 MACs (pointwise convolution)

Run from project root after run_image_to_detection.py --layers 8
Output: image_sim_out/layer7_patch_*.hex, layer7_patch_golden.json
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
            inp = d / "layer6_output_ref.npy"
            if inp.exists():
                return d
    return None


def main():
    sim_out = find_sim_out()
    if sim_out is None:
        print("Run 'run_image_to_detection.py --layers 8' first to create layer6_output_ref.npy")
        return 1

    # Layer 7 input is layer 6 output
    layer6_out = np.load(sim_out / "layer6_output_ref.npy")  # (64, 104, 104)
    w7 = np.load(sim_out / "layer7_weights.npy")              # (64, 64, 1, 1)
    b7 = np.load(sim_out / "layer7_bias.npy")                 # (64,)

    print(f"Layer7 patch golden (one pixel (0,0), {NUM_CHANNELS} channels)")
    print(f"  Input shape: {layer6_out.shape} (64 channels)")
    print(f"  Weight shape: {w7.shape} (64 filters, 64 channels, 1x1)")
    print(f"  MACs per output pixel: 64 * 1 = 64 (pointwise conv)")

    # For 1x1 conv, no padding needed - just take the single pixel at (0,0)
    # Window: all 64 input channels, 1x1 spatial
    window = layer6_out[:, 0:1, 0:1]   # (64, 1, 1)
    flat_a = window.flatten()           # 64

    hex_dir = PROJECT_ROOT / "image_sim_out"
    hex_dir.mkdir(parents=True, exist_ok=True)

    def to_hex8(x):
        return f"{(int(x) & 0xff):02x}"

    # Activations same for all output channels
    with open(hex_dir / "layer7_patch_a.hex", "w") as f:
        for x in flat_a.tolist():
            f.write(to_hex8(x) + "\n")

    golden_channels = []
    bias_lines = []
    expected_lines = []

    for ch in range(NUM_CHANNELS):
        flat_w = w7[ch].flatten()  # (64*1*1) = 64
        bias_ch = int(b7[ch])
        conv_sum = int(mac_array(flat_w, flat_a)) + bias_ch
        leaky_out = int(leaky_relu_hardware(np.array([conv_sum], dtype=np.int32))[0])
        expected_int8 = int(requantize_fixed_point(np.array([leaky_out]), np.int32(SCALE_RTL), SHIFT_RTL)[0])
        expected_int8 = max(INT8_MIN, min(INT8_MAX, expected_int8))

        with open(hex_dir / f"layer7_patch_w{ch}.hex", "w") as f:
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

    with open(hex_dir / "layer7_patch_bias.hex", "w") as f:
        f.writelines(bias_lines)
    with open(hex_dir / "layer7_patch_expected.hex", "w") as f:
        f.writelines(expected_lines)

    golden = {
        "layer": 7,
        "type": "conv1x1",
        "input_channels": 64,
        "macs_per_pixel": 64,
        "activations_count": len(flat_a),
        "num_channels": NUM_CHANNELS,
        "scale_rtl": SCALE_RTL,
        "shift_rtl": SHIFT_RTL,
        "channels": golden_channels,
    }
    with open(sim_out / "layer7_patch_golden.json", "w") as f:
        json.dump(golden, f, indent=2)

    print(f"  Written: {sim_out}/layer7_patch_golden.json, {hex_dir}/layer7_patch_*.hex")
    return 0


if __name__ == "__main__":
    sys.exit(main())
