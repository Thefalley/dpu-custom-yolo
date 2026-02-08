#!/usr/bin/env python3
"""
Layer5 patch check: one pixel (0,0), 4 output channels, 288 MACs each.
1. Ensures layer0-5 data exist (run_image_to_detection.py --layers 6 if needed).
2. Runs Python golden (tests/layer5_patch_golden.py).
3. Runs RTL sim (layer5_patch_tb_iv) if iverilog available.

Usage:
  python run_layer5_patch_check.py
  python run_layer5_patch_check.py --python-only
"""
import sys
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.resolve()
IMAGE_SIM_OUT = PROJECT_ROOT / "image_sim_out"
SIM_OUT = PROJECT_ROOT / "sim_out"


def need_layer5_data():
    for d in (IMAGE_SIM_OUT, SIM_OUT):
        if d.is_dir() and (d / "layer5_weights.npy").exists():
            return False
    return True


def main():
    print("=" * 60)
    print("Layer5 patch check (one pixel, 4 channels, 288 MACs each)")
    print("=" * 60)

    if need_layer5_data():
        print("\n[1] Generating layers 0-5 (run_image_to_detection --synthetic --layers 6)")
        r = subprocess.run(
            [sys.executable, str(PROJECT_ROOT / "run_image_to_detection.py"), "--synthetic", "--layers", "6"],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
        )
        if r.returncode != 0:
            print(r.stderr or r.stdout or "run_image_to_detection failed")
            return 1
    else:
        print("\n[1] Layer5 data already present")

    print("\n[2] Python golden (layer5_patch_golden.py)")
    r = subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "tests" / "layer5_patch_golden.py")],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
    )
    print("  ", (r.stdout or "").replace("\n", "\n   "))
    if r.returncode != 0:
        print(r.stderr or "layer5_patch_golden.py failed")
        return 1

    if "--python-only" in sys.argv:
        print("\n[3] RTL: skipped (--python-only)")
        print("\n" + "-" * 60)
        print("RESULT: Python golden OK. RTL skipped.")
        print("=" * 60)
        return 0

    print("\n[3] RTL (layer5_patch_tb_iv)")
    sv_sim = PROJECT_ROOT / "verilog-sim-py" / "sv_simulator.py"
    if not sv_sim.exists():
        print("  [SKIP] verilog-sim-py/sv_simulator.py not found")
        return 0
    rtl_files = [
        PROJECT_ROOT / "rtl" / "dpu" / "primitives" / "mac_int8.sv",
        PROJECT_ROOT / "rtl" / "dpu" / "primitives" / "leaky_relu.sv",
        PROJECT_ROOT / "rtl" / "dpu" / "primitives" / "requantize.sv",
        PROJECT_ROOT / "rtl" / "tb" / "layer5_patch_tb_iv.sv",
    ]
    cmd = [
        sys.executable,
        str(sv_sim),
        "--no-wave",
        *[str(f) for f in rtl_files],
        "--top",
        "layer5_patch_tb",
    ]
    r = subprocess.run(cmd, cwd=str(PROJECT_ROOT), capture_output=True, text=True, timeout=600)
    # Show only essential output
    out = r.stdout or ""
    for line in out.split("\n"):
        if any(k in line for k in ["PASS", "FAIL", "RESULT", "===", "TOTAL"]):
            print(" ", line)
    if r.returncode != 0:
        print(r.stderr or "RTL sim failed")
        return 1

    print("\n" + "-" * 60)
    if "ALL PASS" in out:
        print("RESULT: ALL PASS")
    else:
        print("RESULT: SOME FAIL")
        return 1
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
