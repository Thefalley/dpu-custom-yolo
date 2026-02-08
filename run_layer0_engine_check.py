#!/usr/bin/env python3
"""
Layer0 engine check: validates layer0_engine.sv (one channel, 27 MACs) vs golden.
1. Ensures layer0 patch data exist (run_layer0_patch_check.py or run_image_to_detection + layer0_patch_golden).
2. Runs RTL sim (layer0_engine_tb_iv) if iverilog available.

Usage:
  python run_layer0_engine_check.py
  python run_layer0_engine_check.py --python-only  # skip RTL
"""
import sys
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.resolve()


def need_data():
    for d in (PROJECT_ROOT / "image_sim_out", PROJECT_ROOT / "sim_out"):
        if d.is_dir() and (d / "layer0_patch_w0.hex").exists():
            return False
    return True


def main():
    print("=" * 60)
    print("Layer0 engine check (one channel, 27 MACs)")
    print("=" * 60)

    if need_data():
        print("\n[1] Generating layer0 patch data (run_layer0_patch_check.py)")
        r = subprocess.run(
            [sys.executable, str(PROJECT_ROOT / "run_layer0_patch_check.py")],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            timeout=120,
        )
        if r.returncode != 0:
            print(r.stderr or r.stdout or "run_layer0_patch_check failed")
            return 1
    else:
        print("\n[1] Layer0 patch data already present")

    if "--python-only" in sys.argv:
        print("\n[2] RTL: skipped (--python-only)")
        return 0

    print("\n[2] RTL (layer0_engine_tb)")
    sv_sim = PROJECT_ROOT / "verilog-sim-py" / "sv_simulator.py"
    if not sv_sim.exists():
        print("  [SKIP] verilog-sim-py/sv_simulator.py not found")
        return 0
    rtl_files = [
        PROJECT_ROOT / "rtl" / "dpu" / "primitives" / "mac_int8.sv",
        PROJECT_ROOT / "rtl" / "dpu" / "primitives" / "leaky_relu.sv",
        PROJECT_ROOT / "rtl" / "dpu" / "primitives" / "requantize.sv",
        PROJECT_ROOT / "rtl" / "dpu" / "layer0_engine.sv",
        PROJECT_ROOT / "rtl" / "tb" / "layer0_engine_tb_iv.sv",
    ]
    cmd = [
        sys.executable,
        str(sv_sim),
        "--no-wave",
        *[str(f) for f in rtl_files],
        "--top",
        "layer0_engine_tb",
    ]
    r = subprocess.run(cmd, cwd=str(PROJECT_ROOT), capture_output=True, text=True, timeout=60)
    print(r.stdout or "")
    if r.returncode != 0:
        print(r.stderr or "RTL sim failed")
        return 1
    if "ALL PASS" not in (r.stdout or ""):
        return 1
    print("\nRESULT: ALL PASS (layer0_engine matches golden)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
