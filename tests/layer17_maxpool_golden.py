#!/usr/bin/env python3
"""Golden for Layer 17: MaxPool 2x2 -> 256ch, 52x52 -> 26x26"""
import sys
from pathlib import Path
import numpy as np
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))
from phase3_dpu_functional_model import maxpool_2x2

def main():
    for name in ("image_sim_out", "sim_out"):
        d = PROJECT_ROOT / name
        if d.is_dir() and (d / "layer16_output_ref.npy").exists():
            sim_out = d
            break
    else:
        print("Run 'run_image_to_detection.py --layers 18' first")
        return 1

    inp = np.load(sim_out / "layer16_output_ref.npy")
    expected = maxpool_2x2(inp, stride=2)

    print(f"Layer17 maxpool: {inp.shape} -> {expected.shape}")

    ref_path = sim_out / "layer17_output_ref.npy"
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
