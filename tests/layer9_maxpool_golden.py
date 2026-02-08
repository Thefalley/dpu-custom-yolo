#!/usr/bin/env python3
"""
Golden for Layer 9 (MaxPool 2x2) - validates max pooling operation.
Layer 9: MaxPool 2x2, stride 2 -> (128, 104, 104) -> (128, 52, 52)

This is a simple max pooling operation. We validate it by checking
sample windows and comparing with reference output.

Run from project root after run_image_to_detection.py --layers 10
"""
import sys
import json
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from phase3_dpu_functional_model import maxpool_2x2


def find_sim_out():
    for name in ("image_sim_out", "sim_out"):
        d = PROJECT_ROOT / name
        if d.is_dir():
            inp = d / "layer8_output_ref.npy"
            if inp.exists():
                return d
    return None


def main():
    sim_out = find_sim_out()
    if sim_out is None:
        print("Run 'run_image_to_detection.py --layers 10' first to create layer8_output_ref.npy")
        return 1

    # Load layer 8 output
    layer8_out = np.load(sim_out / "layer8_output_ref.npy")  # (128, 104, 104)

    # Load layer 9 output if it exists
    layer9_ref_path = sim_out / "layer9_output_ref.npy"
    if layer9_ref_path.exists():
        layer9_ref = np.load(layer9_ref_path)
    else:
        layer9_ref = None

    print(f"Layer9 maxpool 2x2 golden")
    print(f"  Input shape: {layer8_out.shape}")
    print(f"  Operation: MaxPool 2x2, stride 2")
    print(f"  Expected output: (128, 52, 52)")

    # Compute expected output using maxpool_2x2
    expected = maxpool_2x2(layer8_out, stride=2)

    print(f"  Computed output shape: {expected.shape}")

    # Verify output shape
    if expected.shape != (128, 52, 52):
        print(f"  [FAIL] Output shape mismatch: expected (128, 52, 52), got {expected.shape}")
        return 1
    print(f"  [PASS] Output shape correct")

    # Compare with saved reference if available
    if layer9_ref is not None:
        if np.array_equal(expected, layer9_ref):
            print(f"  [PASS] Matches saved layer9_output_ref.npy")
        else:
            print(f"  [FAIL] Does NOT match saved layer9_output_ref.npy")
            return 1

    # Verify max pooling by checking a few sample windows
    print(f"\n  Sample window verification:")
    passed = True
    for ch in [0, 63, 127]:
        for h in [0, 25, 51]:
            for w in [0, 25, 51]:
                # Get 2x2 window from input
                h_in = h * 2
                w_in = w * 2
                window = layer8_out[ch, h_in:h_in+2, w_in:w_in+2]
                max_val = np.max(window)
                out_val = expected[ch, h, w]
                if max_val != out_val:
                    print(f"    [FAIL] ch={ch} h={h} w={w}: window max={max_val}, output={out_val}")
                    passed = False

    if passed:
        print(f"    [PASS] All sample windows verified")
    else:
        return 1

    # Save golden
    golden = {
        "layer": 9,
        "type": "maxpool",
        "size": 2,
        "stride": 2,
        "input_shape": list(layer8_out.shape),
        "output_shape": list(expected.shape),
    }
    with open(sim_out / "layer9_maxpool_golden.json", "w") as f:
        json.dump(golden, f, indent=2)

    print(f"\n  Written: {sim_out}/layer9_maxpool_golden.json")
    print(f"\nRESULT: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
