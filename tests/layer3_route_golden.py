#!/usr/bin/env python3
"""
Golden for Layer 3 (route split) - validates channel slicing operation.
Layer 3: route split (groups=2, group_id=1) -> takes second half of 64 channels = 32 channels

This is a simple data routing operation (no MACs), but we validate it to ensure
the channel indices are correct.

Run from project root after run_image_to_detection.py --layers 4
"""
import sys
import json
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from phase3_dpu_functional_model import route_split


def find_sim_out():
    for name in ("image_sim_out", "sim_out"):
        d = PROJECT_ROOT / name
        if d.is_dir():
            inp = d / "layer2_output_ref.npy"
            if inp.exists():
                return d
    return None


def main():
    sim_out = find_sim_out()
    if sim_out is None:
        print("Run 'run_image_to_detection.py --layers 4' first to create layer2_output_ref.npy")
        return 1

    # Load layer 2 output (input to layer 3)
    layer2_out = np.load(sim_out / "layer2_output_ref.npy")  # (64, 104, 104)

    # Load layer 3 output if it exists
    layer3_ref_path = sim_out / "layer3_output_ref.npy"
    if layer3_ref_path.exists():
        layer3_ref = np.load(layer3_ref_path)
    else:
        layer3_ref = None

    print(f"Layer3 route split golden")
    print(f"  Input shape: {layer2_out.shape} (64 channels)")
    print(f"  Operation: route split (groups=2, group_id=1)")
    print(f"  Expected output: (32, 104, 104) - second half of channels")

    # Compute expected output using route_split
    groups = 2
    group_id = 1
    expected = route_split(layer2_out, groups=groups, group_id=group_id)

    print(f"  Computed output shape: {expected.shape}")

    # Verify the slicing is correct
    # group_id=1 means channels 32..63 (second half)
    channels_per_group = layer2_out.shape[0] // groups
    start_ch = group_id * channels_per_group
    end_ch = start_ch + channels_per_group

    manual_slice = layer2_out[start_ch:end_ch]

    if np.array_equal(expected, manual_slice):
        print(f"  [PASS] route_split matches manual slice [32:64]")
    else:
        print(f"  [FAIL] route_split does NOT match manual slice")
        return 1

    # Compare with saved reference if available
    if layer3_ref is not None:
        if np.array_equal(expected, layer3_ref):
            print(f"  [PASS] Matches saved layer3_output_ref.npy")
        else:
            print(f"  [FAIL] Does NOT match saved layer3_output_ref.npy")
            return 1

    # Verify specific values at corners
    print(f"\n  Sample values (verifying channel indices):")
    print(f"    layer2_out[32, 0, 0] = {layer2_out[32, 0, 0]} -> expected[0, 0, 0] = {expected[0, 0, 0]}")
    print(f"    layer2_out[63, 0, 0] = {layer2_out[63, 0, 0]} -> expected[31, 0, 0] = {expected[31, 0, 0]}")

    if layer2_out[32, 0, 0] == expected[0, 0, 0] and layer2_out[63, 0, 0] == expected[31, 0, 0]:
        print(f"  [PASS] Channel indices verified")
    else:
        print(f"  [FAIL] Channel indices mismatch")
        return 1

    # Save golden
    golden = {
        "layer": 3,
        "type": "route_split",
        "groups": groups,
        "group_id": group_id,
        "input_shape": list(layer2_out.shape),
        "output_shape": list(expected.shape),
        "start_channel": start_ch,
        "end_channel": end_ch,
    }
    with open(sim_out / "layer3_route_golden.json", "w") as f:
        json.dump(golden, f, indent=2)

    print(f"\n  Written: {sim_out}/layer3_route_golden.json")
    print(f"\nRESULT: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
