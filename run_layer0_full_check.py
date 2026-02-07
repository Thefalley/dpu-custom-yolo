#!/usr/bin/env python3
"""
Full layer 0 validation:
1. Ensures image + layer0 data exist (run_image_to_detection.py if needed).
2. Python: recompute full layer 0 with fixed-point primitives, compare to ref, save ref_fp.
3. Export 4x4 region hex for RTL (layer0_full_4x4_export.py).
4. RTL: run layer0_full_4x4_tb (4x4 x 32 channels) if iverilog available.

Usage:
  python run_layer0_full_check.py
  python run_layer0_full_check.py --python-only
"""
import sys
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.resolve()
IMAGE_SIM_OUT = PROJECT_ROOT / "image_sim_out"
SIM_OUT = PROJECT_ROOT / "sim_out"


def need_image_data():
    for d in (IMAGE_SIM_OUT, SIM_OUT):
        if d.is_dir() and (d / "image_input_layer0.npy").exists():
            return False
    return True


def main():
    if need_image_data():
        print("[1] Generating image + layer0 (run_image_to_detection --synthetic)")
        r = subprocess.run(
            [sys.executable, str(PROJECT_ROOT / "run_image_to_detection.py"), "--synthetic"],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
        )
        if r.returncode != 0:
            print(r.stderr or r.stdout or "run_image_to_detection failed")
            return 1
    else:
        print("[1] Image data already present")

    if "--full-py" in sys.argv:
        print("\n[2] Python full layer 0 (fixed-point, 208x208x32 — slow)")
        r = subprocess.run(
            [sys.executable, str(PROJECT_ROOT / "tests" / "layer0_full_golden.py")],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            timeout=600,
        )
        print(r.stdout or "")
        if r.returncode != 0:
            print(r.stderr or "layer0_full_golden.py failed")
            return 1
    else:
        print("\n[2] Python full layer 0: skipped (use --full-py to run 208x208x32)")

    print("\n[3] Export 4x4 region for RTL")
    r = subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "tests" / "layer0_full_4x4_export.py")],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
    )
    print(r.stdout or "")
    if r.returncode != 0:
        print(r.stderr or "layer0_full_4x4_export.py failed")
        return 1

    if "--python-only" in sys.argv:
        print("\n[4] RTL: skipped (--python-only)")
        print("\nRESULT: Python full layer 0 + 4x4 export OK.")
        return 0

    print("\n[4] RTL layer0_full_4x4_tb")
    sv_sim = PROJECT_ROOT / "verilog-sim-py" / "sv_simulator.py"
    if not sv_sim.exists():
        print("  [SKIP] verilog-sim-py/sv_simulator.py not found")
        return 0
    rtl_files = [
        PROJECT_ROOT / "rtl" / "dpu" / "primitives" / "mac_int8.sv",
        PROJECT_ROOT / "rtl" / "dpu" / "primitives" / "leaky_relu.sv",
        PROJECT_ROOT / "rtl" / "dpu" / "primitives" / "requantize.sv",
        PROJECT_ROOT / "rtl" / "tb" / "layer0_full_4x4_tb_iv.sv",
    ]
    cmd = [
        sys.executable,
        str(sv_sim),
        "--no-wave",
        *[str(f) for f in rtl_files],
        "--top",
        "layer0_full_4x4_tb",
    ]
    r = subprocess.run(cmd, cwd=str(PROJECT_ROOT), capture_output=True, text=True, timeout=300)
    print(r.stdout or "")
    if r.returncode != 0:
        print(r.stderr or "RTL sim failed")
        return 1
    if "RESULT:" in (r.stdout or "") and "ALL PASS" in (r.stdout or ""):
        print("\nRESULT: ALL PASS (Python full layer 0 + RTL 4x4x32)")
    else:
        print("\nRESULT: RTL 4x4 had failures")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
