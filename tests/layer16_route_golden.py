#!/usr/bin/env python3
"""Golden for Layer 16: route concat (layer10 + layer15) -> 256 channels"""
import sys
from pathlib import Path
import numpy as np
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))
from phase3_dpu_functional_model import route_concat

def main():
    for name in ("image_sim_out", "sim_out"):
        d = PROJECT_ROOT / name
        if d.is_dir() and (d / "layer15_output_ref.npy").exists():
            sim_out = d
            break
    else:
        print("Run 'run_image_to_detection.py --layers 17' first")
        return 1

    l10 = np.load(sim_out / "layer10_output_ref.npy")
    l15 = np.load(sim_out / "layer15_output_ref.npy")
    expected = route_concat([l10, l15])

    print(f"Layer16 route concat: {l10.shape} + {l15.shape} -> {expected.shape}")

    ref_path = sim_out / "layer16_output_ref.npy"
    if ref_path.exists():
        ref = np.load(ref_path)
        if np.array_equal(expected, ref):
            print("  [PASS] Matches saved reference")
        else:
            print("  [FAIL] Mismatch")
            return 1
    print("RESULT: PASS")
    return 0

if __name__ == "__main__":
    sys.exit(main())
