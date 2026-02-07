#!/usr/bin/env python3
"""
Export data for RTL full layer 0 (208x208x32).
Output: padded input 3x418x418 (523254 bytes), weights 864, bias 32.
Reference: layer0_output_ref.npy (compare in Python after RTL run).
"""
import sys
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

STRIDE = 2
PAD = 1
C_OUT = 32
H_OUT, W_OUT = 208, 208
PAD_H, PAD_W = 418, 418  # 416 + 2


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
    assert img_pad.shape == (3, PAD_H, PAD_W)

    with open(hex_dir / "layer0_full208_padded.hex", "w") as f:
        for x in img_pad.flatten().tolist():
            f.write(to_hex8(x) + "\n")

    with open(hex_dir / "layer0_full208_weights.hex", "w") as f:
        for c in range(C_OUT):
            for x in w0[c].flatten().tolist():
                f.write(to_hex8(x) + "\n")

    with open(hex_dir / "layer0_full208_bias.hex", "w") as f:
        for c in range(C_OUT):
            f.write(to_hex32(b0[c]) + "\n")

    print("Exported layer0_full208_*.hex: padded 523254, weights 864, bias 32")
    print("Reference: use layer0_output_ref.npy for comparison after RTL run.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
