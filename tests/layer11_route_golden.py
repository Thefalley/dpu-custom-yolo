#!/usr/bin/env python3
"""Golden for Layer 11: route split (groups=2, group_id=1) -> 128->64 channels"""
import sys, json
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))
from phase3_dpu_functional_model import route_split

def main():
    for name in ("image_sim_out", "sim_out"):
        d = PROJECT_ROOT / name
        if d.is_dir() and (d / "layer10_output_ref.npy").exists():
            sim_out = d
            break
    else:
        print("Run 'run_image_to_detection.py --layers 12' first")
        return 1

    layer10_out = np.load(sim_out / "layer10_output_ref.npy")
    expected = route_split(layer10_out, groups=2, group_id=1)

    print(f"Layer11 route split golden")
    print(f"  Input: {layer10_out.shape} -> Output: {expected.shape}")

    layer11_ref = sim_out / "layer11_output_ref.npy"
    if layer11_ref.exists():
        ref = np.load(layer11_ref)
        if np.array_equal(expected, ref):
            print("  [PASS] Matches saved reference")
        else:
            print("  [FAIL] Mismatch")
            return 1

    print("RESULT: PASS")
    return 0

if __name__ == "__main__":
    sys.exit(main())
