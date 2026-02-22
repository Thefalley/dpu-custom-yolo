#!/usr/bin/env python3
"""
DPU System Top 36-layer AXI check: Python golden -> RTL system sim -> compare.

Tests the COMPLETE DPU through AXI interfaces only (AXI4-Lite + AXI-Stream DMA),
exactly as the CPU would control it on a real FPGA.

1. Runs tests/dpu_top_37layer_golden.py to generate hex files.
2. Compiles and runs the system-level RTL simulation (includes dpu_system_top,
   dpu_axi_dma, dpu_top, and all sub-modules).
3. Reports pass/fail.

Usage:
  python run_dpu_system_top_36layer_check.py
  python run_dpu_system_top_36layer_check.py --python-only
  python run_dpu_system_top_36layer_check.py --real-weights --input-image path.jpg
"""

import os
import sys
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.resolve()


def run_golden(real_weights=False, input_image=None):
    cmd = [sys.executable, str(PROJECT_ROOT / "tests" / "dpu_top_37layer_golden.py")]
    if real_weights:
        cmd.append("--real-weights")
    if input_image:
        cmd.extend(["--input-image", input_image])
    r = subprocess.run(
        cmd,
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        timeout=120,
    )
    out = (r.stdout or "") + (r.stderr or "")
    return r.returncode == 0 and "GOLDEN COMPLETE" in out, out


def run_rtl():
    sv_py = PROJECT_ROOT / "verilog-sim-py" / "sv_simulator.py"
    files = [
        # Primitives
        PROJECT_ROOT / "rtl" / "dpu" / "primitives" / "mac_int8.sv",
        PROJECT_ROOT / "rtl" / "dpu" / "primitives" / "leaky_relu.sv",
        PROJECT_ROOT / "rtl" / "dpu" / "primitives" / "requantize.sv",
        # Core modules
        PROJECT_ROOT / "rtl" / "dpu" / "mac_array_32x32.sv",
        PROJECT_ROOT / "rtl" / "dpu" / "post_process_array.sv",
        PROJECT_ROOT / "rtl" / "dpu" / "conv_engine_array.sv",
        PROJECT_ROOT / "rtl" / "dpu" / "maxpool_unit.sv",
        PROJECT_ROOT / "rtl" / "dpu" / "dpu_top.sv",
        # System wrapper + DMA
        PROJECT_ROOT / "rtl" / "dpu" / "dpu_axi_dma.sv",
        PROJECT_ROOT / "rtl" / "dpu" / "dpu_system_top.sv",
        # Testbench
        PROJECT_ROOT / "rtl" / "tb" / "dpu_system_top_36layer_tb.sv",
    ]
    cmd = ([sys.executable, str(sv_py), "--no-wave"] +
           [str(f) for f in files] +
           ["--top", "dpu_system_top_36layer_tb"])
    # Read OSS_CAD_PATH from .oss_cad_path file or env
    env = dict(os.environ)
    if "OSS_CAD_PATH" not in env:
        ocp = PROJECT_ROOT / ".oss_cad_path"
        if ocp.exists():
            env["OSS_CAD_PATH"] = ocp.read_text().strip()
    r = subprocess.run(cmd, cwd=PROJECT_ROOT, capture_output=True, text=True,
                       timeout=14400, env=env)  # 4hr timeout for AXI sim
    out = (r.stdout or "") + (r.stderr or "")
    return "ALL TESTS PASSED" in out, out


def main():
    import argparse
    ap = argparse.ArgumentParser(
        description="DPU System Top 36-layer AXI check")
    ap.add_argument("--python-only", action="store_true",
                    help="Skip RTL simulation")
    ap.add_argument("--real-weights", action="store_true",
                    help="Use real YOLOv4-tiny weights")
    ap.add_argument("--input-image", type=str, default=None,
                    help="Path to input image")
    args = ap.parse_args()

    mode = "REAL weights" if args.real_weights else "synthetic weights"
    if args.input_image:
        mode += f" + image: {Path(args.input_image).name}"
    print("=" * 60)
    print(f"DPU System Top 36-Layer AXI Check (H0=32, W0=32) [{mode}]")
    print("=" * 60)

    print("\n[1] Python golden model")
    ok, out = run_golden(real_weights=args.real_weights,
                         input_image=args.input_image)
    if not ok:
        print("  [FAIL] Golden model failed")
        for line in out.strip().splitlines():
            print("  ", line)
        return 1
    for line in out.strip().splitlines():
        print("  ", line)

    if args.python_only:
        print("\n[2] RTL: skipped (--python-only)")
        print("\nRESULT: Python golden OK")
        print("=" * 60)
        return 0

    print("\n[2] RTL system simulation (Icarus Verilog)")
    print("    Includes: dpu_system_top + dpu_axi_dma + dpu_top + primitives")
    print("    All data loaded via AXI interfaces (DMA + PIO)")
    rtl_ok, rtl_out = run_rtl()
    for line in rtl_out.strip().splitlines():
        print("  ", line)

    print("\n" + "-" * 60)
    if rtl_ok:
        print("RESULT: AXI SYSTEM TEST PASSED")
    else:
        print("RESULT: AXI SYSTEM TEST FAILED (see output above)")
    print("=" * 60)
    return 0 if rtl_ok else 1


if __name__ == "__main__":
    sys.exit(main())
