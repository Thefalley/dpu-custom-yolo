#!/usr/bin/env python3
"""
Golden for Layer 10: Conv 3x3, 128->128, stride 1 (1152 MACs per pixel)
Run from project root after run_image_to_detection.py --layers 11
"""
NUM_CHANNELS = 4

import sys
import json
from pathlib import Path
import numpy as np

SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from phase3_dpu_functional_model import mac_array, leaky_relu_hardware, requantize_fixed_point, INT8_MIN, INT8_MAX

SCALE_RTL = 655
SHIFT_RTL = 16

def find_sim_out():
    for name in ("image_sim_out", "sim_out"):
        d = PROJECT_ROOT / name
        if d.is_dir() and (d / "layer9_output_ref.npy").exists():
            return d
    return None

def main():
    sim_out = find_sim_out()
    if sim_out is None:
        print("Run 'run_image_to_detection.py --layers 11' first")
        return 1

    layer9_out = np.load(sim_out / "layer9_output_ref.npy")
    w10 = np.load(sim_out / "layer10_weights.npy")
    b10 = np.load(sim_out / "layer10_bias.npy")

    print(f"Layer10 patch golden (one pixel (0,0), {NUM_CHANNELS} channels)")
    print(f"  Input shape: {layer9_out.shape} (128 channels)")
    print(f"  Weight shape: {w10.shape}")
    print(f"  MACs per output pixel: 128 * 9 = 1152")

    layer9_pad = np.pad(layer9_out, ((0, 0), (1, 1), (1, 1)), mode="constant", constant_values=0)
    window = layer9_pad[:, 0:3, 0:3]
    flat_a = window.flatten()

    hex_dir = PROJECT_ROOT / "image_sim_out"
    hex_dir.mkdir(parents=True, exist_ok=True)

    def to_hex8(x):
        return f"{(int(x) & 0xff):02x}"

    with open(hex_dir / "layer10_patch_a.hex", "w") as f:
        for x in flat_a.tolist():
            f.write(to_hex8(x) + "\n")

    golden_channels = []
    bias_lines = []
    expected_lines = []

    for ch in range(NUM_CHANNELS):
        flat_w = w10[ch].flatten()
        bias_ch = int(b10[ch])
        conv_sum = int(mac_array(flat_w, flat_a)) + bias_ch
        leaky_out = int(leaky_relu_hardware(np.array([conv_sum], dtype=np.int32))[0])
        expected_int8 = int(requantize_fixed_point(np.array([leaky_out]), np.int32(SCALE_RTL), SHIFT_RTL)[0])
        expected_int8 = max(INT8_MIN, min(INT8_MAX, expected_int8))

        with open(hex_dir / f"layer10_patch_w{ch}.hex", "w") as f:
            for x in flat_w.tolist():
                f.write(to_hex8(x) + "\n")
        bias_lines.append(f"{(bias_ch & 0xffffffff):08x}\n")
        expected_lines.append(to_hex8(expected_int8) + "\n")
        golden_channels.append({"channel": ch, "conv_sum": conv_sum, "expected_int8": expected_int8})
        print(f"  ch{ch}: conv_sum={conv_sum}, expected_int8={expected_int8}")

    with open(hex_dir / "layer10_patch_bias.hex", "w") as f:
        f.writelines(bias_lines)
    with open(hex_dir / "layer10_patch_expected.hex", "w") as f:
        f.writelines(expected_lines)

    golden = {"layer": 10, "input_channels": 128, "macs_per_pixel": 1152, "channels": golden_channels}
    with open(sim_out / "layer10_patch_golden.json", "w") as f:
        json.dump(golden, f, indent=2)

    print(f"  Written: layer10_patch_golden.json")
    return 0

if __name__ == "__main__":
    sys.exit(main())
