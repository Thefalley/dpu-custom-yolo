#!/usr/bin/env python3
"""
Golden for Layer 6 (route concat) - validates channel concatenation operation.
Layer 6: route concat (layer5 + layer4) -> 32+32 = 64 channels

This is a simple data routing operation (no MACs), but we validate it to ensure
the channel ordering is correct.

Run from project root after run_image_to_detection.py --layers 7
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
            if (d / "layer5_output_ref.npy").exists() and (d / "layer4_output_ref.npy").exists():
                return d
    return None


def main():
    sim_out = find_sim_out()
    if sim_out is None:
        print("Run 'run_image_to_detection.py --layers 7' first to create layer4/5_output_ref.npy")
        return 1

    # Load layer 4 and 5 outputs
    layer4_out = np.load(sim_out / "layer4_output_ref.npy")  # (32, 104, 104)
    layer5_out = np.load(sim_out / "layer5_output_ref.npy")  # (32, 104, 104)

    # Load layer 6 output if it exists
    layer6_ref_path = sim_out / "layer6_output_ref.npy"
    if layer6_ref_path.exists():
        layer6_ref = np.load(layer6_ref_path)
    else:
        layer6_ref = None

    print(f"Layer6 route concat golden")
    print(f"  Input shapes: {layer5_out.shape} + {layer4_out.shape}")
    print(f"  Operation: route concat (layer5 first, then layer4)")
    print(f"  Expected output: (64, 104, 104)")

    # Compute expected output using route_concat
    # Order: layer5 (channels 0-31) + layer4 (channels 32-63)
    expected = route_concat([layer5_out, layer4_out])

    print(f"  Computed output shape: {expected.shape}")

    # Verify the concatenation is correct
    manual_concat = np.concatenate([layer5_out, layer4_out], axis=0)

    if np.array_equal(expected, manual_concat):
        print(f"  [PASS] route_concat matches np.concatenate")
    else:
        print(f"  [FAIL] route_concat does NOT match np.concatenate")
        return 1

    # Compare with saved reference if available
    if layer6_ref is not None:
        if np.array_equal(expected, layer6_ref):
            print(f"  [PASS] Matches saved layer6_output_ref.npy")
        else:
            print(f"  [FAIL] Does NOT match saved layer6_output_ref.npy")
            return 1

    # Verify specific values at corners
    print(f"\n  Sample values (verifying channel ordering):")
    print(f"    layer5_out[0, 0, 0] = {layer5_out[0, 0, 0]} -> expected[0, 0, 0] = {expected[0, 0, 0]}")
    print(f"    layer5_out[31, 0, 0] = {layer5_out[31, 0, 0]} -> expected[31, 0, 0] = {expected[31, 0, 0]}")
    print(f"    layer4_out[0, 0, 0] = {layer4_out[0, 0, 0]} -> expected[32, 0, 0] = {expected[32, 0, 0]}")
    print(f"    layer4_out[31, 0, 0] = {layer4_out[31, 0, 0]} -> expected[63, 0, 0] = {expected[63, 0, 0]}")

    checks = [
        layer5_out[0, 0, 0] == expected[0, 0, 0],
        layer5_out[31, 0, 0] == expected[31, 0, 0],
        layer4_out[0, 0, 0] == expected[32, 0, 0],
        layer4_out[31, 0, 0] == expected[63, 0, 0],
    ]
    if all(checks):
        print(f"  [PASS] Channel ordering verified")
    else:
        print(f"  [FAIL] Channel ordering mismatch")
        return 1

    # Save golden
    golden = {
        "layer": 6,
        "type": "route_concat",
        "input_layers": [-1, -2],
        "input_shapes": [list(layer5_out.shape), list(layer4_out.shape)],
        "output_shape": list(expected.shape),
    }
    with open(sim_out / "layer6_route_golden.json", "w") as f:
        json.dump(golden, f, indent=2)

    print(f"\n  Written: {sim_out}/layer6_route_golden.json")
    print(f"\nRESULT: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
