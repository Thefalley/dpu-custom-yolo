#!/usr/bin/env python3
"""
Layer3 route check: validates channel slicing (route split) operation.
Layer 3: route split (groups=2, group_id=1) - takes second half of 64 channels.

This is a data routing operation (no MACs, no RTL simulation needed).

Usage:
  python run_layer3_route_check.py
"""
import sys
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.resolve()
IMAGE_SIM_OUT = PROJECT_ROOT / "image_sim_out"
SIM_OUT = PROJECT_ROOT / "sim_out"


def need_layer3_data():
    for d in (IMAGE_SIM_OUT, SIM_OUT):
        if d.is_dir() and (d / "layer3_output_ref.npy").exists():
            return False
    return True


def main():
    print("=" * 60)
    print("Layer3 route check (channel split: 64 -> 32 channels)")
    print("=" * 60)

    if need_layer3_data():
        print("\n[1] Generating layers 0-3 (run_image_to_detection --synthetic --layers 4)")
        r = subprocess.run(
            [sys.executable, str(PROJECT_ROOT / "run_image_to_detection.py"), "--synthetic", "--layers", "4"],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
        )
        if r.returncode != 0:
            print(r.stderr or r.stdout or "run_image_to_detection failed")
            return 1
    else:
        print("\n[1] Layer3 data already present")

    print("\n[2] Python golden (layer3_route_golden.py)")
    r = subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "tests" / "layer3_route_golden.py")],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
    )
    print("  ", (r.stdout or "").replace("\n", "\n   "))
    if r.returncode != 0:
        print(r.stderr or "layer3_route_golden.py failed")
        return 1

    print("\n[3] RTL: N/A (route is data routing only, no compute)")

    print("\n" + "-" * 60)
    print("RESULT: PASS (Layer 3 route split validated)")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
