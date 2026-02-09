#!/usr/bin/env python3
"""
DPU Top 18-layer check: Python golden -> RTL sim -> compare.

1. Runs tests/dpu_top_18layer_golden.py to generate hex files.
2. Compiles and runs RTL simulation with Icarus Verilog.
3. Reports pass/fail.

Usage:
  python run_dpu_top_18layer_check.py
  python run_dpu_top_18layer_check.py --python-only
"""

import os
import sys
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.resolve()


def run_golden():
    r = subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "tests" / "dpu_top_18layer_golden.py")],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        timeout=60,
    )
    out = (r.stdout or "") + (r.stderr or "")
    return r.returncode == 0 and "GOLDEN COMPLETE" in out, out


def run_rtl():
    sv_py = PROJECT_ROOT / "verilog-sim-py" / "sv_simulator.py"
    files = [
        PROJECT_ROOT / "rtl" / "dpu" / "primitives" / "mac_int8.sv",
        PROJECT_ROOT / "rtl" / "dpu" / "primitives" / "leaky_relu.sv",
        PROJECT_ROOT / "rtl" / "dpu" / "primitives" / "requantize.sv",
        PROJECT_ROOT / "rtl" / "dpu" / "conv_engine.sv",
        PROJECT_ROOT / "rtl" / "dpu" / "maxpool_unit.sv",
        PROJECT_ROOT / "rtl" / "dpu" / "dpu_top.sv",
        PROJECT_ROOT / "rtl" / "tb" / "dpu_top_18layer_tb.sv",
    ]
    cmd = [sys.executable, str(sv_py), "--no-wave"] + [str(f) for f in files] + ["--top", "dpu_top_18layer_tb"]
    # Read OSS_CAD_PATH from .oss_cad_path file or env
    env = dict(os.environ)
    if "OSS_CAD_PATH" not in env:
        ocp = PROJECT_ROOT / ".oss_cad_path"
        if ocp.exists():
            env["OSS_CAD_PATH"] = ocp.read_text().strip()
    r = subprocess.run(cmd, cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=600, env=env)
    out = (r.stdout or "") + (r.stderr or "")
    return "ALL 18 LAYERS PASS" in out, out


def main():
    import argparse
    ap = argparse.ArgumentParser(description="DPU Top 18-layer check")
    ap.add_argument("--python-only", action="store_true", help="Skip RTL simulation")
    args = ap.parse_args()

    print("=" * 60)
    print("DPU Top 18-Layer Check (H0=16, W0=16)")
    print("=" * 60)

    print("\n[1] Python golden model")
    ok, out = run_golden()
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

    print("\n[2] RTL simulation (Icarus Verilog)")
    rtl_ok, rtl_out = run_rtl()
    for line in rtl_out.strip().splitlines()[-20:]:
        print("  ", line)

    print("\n" + "-" * 60)
    if rtl_ok:
        print("RESULT: ALL PASS")
    else:
        print("RESULT: RTL FAILED (see output above)")
    print("=" * 60)
    return 0 if rtl_ok else 1


if __name__ == "__main__":
    sys.exit(main())
