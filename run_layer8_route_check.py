#!/usr/bin/env python3
"""
Layer8 route check: validates channel concatenation (route concat) operation.
Layer 8: route concat (layer2 + layer7) -> 64+64 = 128 channels.

This is a data routing operation (no MACs, no RTL simulation needed).

Usage:
  python run_layer8_route_check.py
"""
import sys
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.resolve()
IMAGE_SIM_OUT = PROJECT_ROOT / "image_sim_out"
SIM_OUT = PROJECT_ROOT / "sim_out"


def need_layer8_data():
    for d in (IMAGE_SIM_OUT, SIM_OUT):
        if d.is_dir() and (d / "layer8_output_ref.npy").exists():
            return False
    return True


def main():
    print("=" * 60)
    print("Layer8 route check (channel concat: 64+64 -> 128 channels)")
    print("=" * 60)

    if need_layer8_data():
        print("\n[1] Generating layers 0-8 (run_image_to_detection --synthetic --layers 9)")
        r = subprocess.run(
            [sys.executable, str(PROJECT_ROOT / "run_image_to_detection.py"), "--synthetic", "--layers", "9"],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
        )
        if r.returncode != 0:
            print(r.stderr or r.stdout or "run_image_to_detection failed")
            return 1
    else:
        print("\n[1] Layer8 data already present")

    print("\n[2] Python golden (layer8_route_golden.py)")
    r = subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "tests" / "layer8_route_golden.py")],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
    )
    print("  ", (r.stdout or "").replace("\n", "\n   "))
    if r.returncode != 0:
        print(r.stderr or "layer8_route_golden.py failed")
        return 1

    print("\n[3] RTL: N/A (route is data routing only, no compute)")

    print("\n" + "-" * 60)
    print("RESULT: PASS (Layer 8 route concat validated)")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
