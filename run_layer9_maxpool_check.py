#!/usr/bin/env python3
"""
Layer9 maxpool check: validates MaxPool 2x2 operation.
Layer 9: MaxPool 2x2, stride 2 -> (128, 104, 104) -> (128, 52, 52)

This is a simple comparison operation (no RTL simulation in this phase).

Usage:
  python run_layer9_maxpool_check.py
"""
import sys
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.resolve()
IMAGE_SIM_OUT = PROJECT_ROOT / "image_sim_out"
SIM_OUT = PROJECT_ROOT / "sim_out"


def need_layer9_data():
    for d in (IMAGE_SIM_OUT, SIM_OUT):
        if d.is_dir() and (d / "layer9_output_ref.npy").exists():
            return False
    return True


def main():
    print("=" * 60)
    print("Layer9 maxpool check (MaxPool 2x2: 104x104 -> 52x52)")
    print("=" * 60)

    if need_layer9_data():
        print("\n[1] Generating layers 0-9 (run_image_to_detection --synthetic --layers 10)")
        r = subprocess.run(
            [sys.executable, str(PROJECT_ROOT / "run_image_to_detection.py"), "--synthetic", "--layers", "10"],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
        )
        if r.returncode != 0:
            print(r.stderr or r.stdout or "run_image_to_detection failed")
            return 1
    else:
        print("\n[1] Layer9 data already present")

    print("\n[2] Python golden (layer9_maxpool_golden.py)")
    r = subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "tests" / "layer9_maxpool_golden.py")],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
    )
    print("  ", (r.stdout or "").replace("\n", "\n   "))
    if r.returncode != 0:
        print(r.stderr or "layer9_maxpool_golden.py failed")
        return 1

    print("\n[3] RTL: N/A (maxpool RTL validation in later phase)")

    print("\n" + "-" * 60)
    print("RESULT: PASS (Layer 9 maxpool validated)")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
