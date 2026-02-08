#!/usr/bin/env python3
"""
Golden for Layer 8 (route concat) - validates channel concatenation operation.
Layer 8: route concat (layer2 + layer7) -> 64+64 = 128 channels

This is a simple data routing operation (no MACs), but we validate it to ensure
the channel ordering is correct.

Run from project root after run_image_to_detection.py --layers 9
"""
import sys
import json
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from phase3_dpu_functional_model import route_concat


def find_sim_out():
    for name in ("image_sim_out", "sim_out"):
        d = PROJECT_ROOT / name
        if d.is_dir():
            if (d / "layer2_output_ref.npy").exists() and (d / "layer7_output_ref.npy").exists():
                return d
    return None


def main():
    sim_out = find_sim_out()
    if sim_out is None:
        print("Run 'run_image_to_detection.py --layers 9' first to create layer2/7_output_ref.npy")
        return 1

    # Load layer 2 and 7 outputs
    layer2_out = np.load(sim_out / "layer2_output_ref.npy")  # (64, 104, 104)
    layer7_out = np.load(sim_out / "layer7_output_ref.npy")  # (64, 104, 104)

    # Load layer 8 output if it exists
    layer8_ref_path = sim_out / "layer8_output_ref.npy"
    if layer8_ref_path.exists():
        layer8_ref = np.load(layer8_ref_path)
    else:
        layer8_ref = None

    print(f"Layer8 route concat golden")
    print(f"  Input shapes: {layer2_out.shape} + {layer7_out.shape}")
    print(f"  Operation: route concat (layer2 first, then layer7)")
    print(f"  Expected output: (128, 104, 104)")

    # Compute expected output using route_concat
    # Order: layer2 (channels 0-63) + layer7 (channels 64-127)
    expected = route_concat([layer2_out, layer7_out])

    print(f"  Computed output shape: {expected.shape}")

    # Verify the concatenation is correct
    manual_concat = np.concatenate([layer2_out, layer7_out], axis=0)

    if np.array_equal(expected, manual_concat):
        print(f"  [PASS] route_concat matches np.concatenate")
    else:
        print(f"  [FAIL] route_concat does NOT match np.concatenate")
        return 1

    # Compare with saved reference if available
    if layer8_ref is not None:
        if np.array_equal(expected, layer8_ref):
            print(f"  [PASS] Matches saved layer8_output_ref.npy")
        else:
            print(f"  [FAIL] Does NOT match saved layer8_output_ref.npy")
            return 1

    # Verify specific values at corners
    print(f"\n  Sample values (verifying channel ordering):")
    print(f"    layer2_out[0, 0, 0] = {layer2_out[0, 0, 0]} -> expected[0, 0, 0] = {expected[0, 0, 0]}")
    print(f"    layer2_out[63, 0, 0] = {layer2_out[63, 0, 0]} -> expected[63, 0, 0] = {expected[63, 0, 0]}")
    print(f"    layer7_out[0, 0, 0] = {layer7_out[0, 0, 0]} -> expected[64, 0, 0] = {expected[64, 0, 0]}")
    print(f"    layer7_out[63, 0, 0] = {layer7_out[63, 0, 0]} -> expected[127, 0, 0] = {expected[127, 0, 0]}")

    checks = [
        layer2_out[0, 0, 0] == expected[0, 0, 0],
        layer2_out[63, 0, 0] == expected[63, 0, 0],
        layer7_out[0, 0, 0] == expected[64, 0, 0],
        layer7_out[63, 0, 0] == expected[127, 0, 0],
    ]
    if all(checks):
        print(f"  [PASS] Channel ordering verified")
    else:
        print(f"  [FAIL] Channel ordering mismatch")
        return 1

    # Save golden
    golden = {
        "layer": 8,
        "type": "route_concat",
        "input_layers": [-6, -1],
        "input_shapes": [list(layer2_out.shape), list(layer7_out.shape)],
        "output_shape": list(expected.shape),
    }
    with open(sim_out / "layer8_route_golden.json", "w") as f:
        json.dump(golden, f, indent=2)

    print(f"\n  Written: {sim_out}/layer8_route_golden.json")
    print(f"\nRESULT: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
