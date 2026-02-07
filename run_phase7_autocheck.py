#!/usr/bin/env python3
"""
Phase 7 — AutoCheck: verify DPU RTL against Python reference.

Runs:
  1. Python golden (tests/test_rtl_vectors.py) — same vectors as RTL TBs.
  2. RTL simulation for each primitive (mac_int8, leaky_relu, mult_shift_add)
     via verilog-sim-py + iverilog.

Parses outputs for PASS/FAIL and prints a single report.
Exit 0 only if Python golden and all RTL tests pass.

Usage (from project root):
  python run_phase7_autocheck.py
  python run_phase7_autocheck.py --python-only   # Only Python golden
  python run_phase7_autocheck.py --rtl-only      # Only RTL sims
"""

import subprocess
import sys
import os
import re
from pathlib import Path

# Project root: directory containing rtl/, tests/, verilog-sim-py/
SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR

RTL_TESTS = [
    {
        "name": "mac_int8",
        "rtl": "rtl/dpu/primitives/mac_int8.sv",
        "tb": "rtl/tb/mac_int8_tb_iv.sv",
        "top": "mac_int8_tb",
    },
    {
        "name": "leaky_relu",
        "rtl": "rtl/dpu/primitives/leaky_relu.sv",
        "tb": "rtl/tb/leaky_relu_tb_iv.sv",
        "top": "leaky_relu_tb",
    },
    {
        "name": "mult_shift_add",
        "rtl": "rtl/dpu/primitives/mult_shift_add.sv",
        "tb": "rtl/tb/mult_shift_add_tb_iv.sv",
        "top": "mult_shift_add_tb",
    },
]


def run_python_golden():
    """Run tests/test_rtl_vectors.py; return (passed: bool, output: str)."""
    script = PROJECT_ROOT / "tests" / "test_rtl_vectors.py"
    if not script.exists():
        return False, f"Golden script not found: {script}"
    result = subprocess.run(
        [sys.executable, str(script)],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        timeout=30,
    )
    out = (result.stdout or "") + (result.stderr or "")
    passed = result.returncode == 0
    return passed, out


def run_rtl_test(info):
    """Run one RTL test via verilog-sim-py; return (passed: bool, output: str)."""
    sv_py = PROJECT_ROOT / "verilog-sim-py" / "sv_simulator.py"
    rtl_path = PROJECT_ROOT / info["rtl"]
    tb_path = PROJECT_ROOT / info["tb"]
    if not sv_py.exists():
        return False, f"verilog-sim-py not found: {sv_py}"
    if not rtl_path.exists():
        return False, f"RTL not found: {rtl_path}"
    if not tb_path.exists():
        return False, f"TB not found: {tb_path}"

    cmd = [
        sys.executable,
        str(sv_py),
        "--no-wave",
        str(rtl_path),
        str(tb_path),
        "--top",
        info["top"],
    ]
    try:
        result = subprocess.run(
            cmd,
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=60,
        )
    except subprocess.TimeoutExpired:
        return False, "RTL simulation timed out"
    except FileNotFoundError:
        return False, "iverilog/vvp not found (add OSS CAD or Icarus to PATH)"

    out = (result.stdout or "") + (result.stderr or "")
    # Simulator may return 0 even when TB reports FAIL (TB uses $finish)
    if "RESULT: ALL PASS" in out:
        return True, out
    if "RESULT: SOME FAIL" in out:
        return False, out
    if result.returncode != 0:
        return False, out
    # No RESULT line: compile/sim error or iverilog not found
    if "Faltan herramientas" in out or "not found" in out.lower():
        out = out + "\n[Tip: add OSS CAD Suite or Icarus Verilog to PATH for RTL sims]"
    return False, out


def main():
    import argparse
    ap = argparse.ArgumentParser(description="Phase 7 AutoCheck: RTL vs Python")
    ap.add_argument("--python-only", action="store_true", help="Run only Python golden")
    ap.add_argument("--rtl-only", action="store_true", help="Run only RTL simulations")
    ap.add_argument("-q", "--quiet", action="store_true", help="Only print summary")
    args = ap.parse_args()

    run_py = not args.rtl_only
    run_rtl = not args.python_only

    print("=" * 60)
    print("PHASE 7 — AutoCheck: DPU RTL vs Python reference")
    print("=" * 60)

    results = []
    all_ok = True

    if run_py:
        print("\n[1] Python golden (tests/test_rtl_vectors.py)")
        py_ok, py_out = run_python_golden()
        results.append(("Python golden", py_ok))
        if not args.quiet:
            for line in py_out.strip().splitlines():
                print("  ", line)
        else:
            print("  ", "PASS" if py_ok else "FAIL")
        if not py_ok:
            all_ok = False

    if run_rtl:
        for i, info in enumerate(RTL_TESTS):
            label = f"[{i + 2}]" if run_py else f"[{i + 1}]"
            print(f"\n{label} RTL: {info['name']}")
            rtl_ok, rtl_out = run_rtl_test(info)
            results.append((info["name"], rtl_ok))
            if not args.quiet:
                for line in rtl_out.strip().splitlines():
                    if line.strip().startswith(("[", "  ", "===", "TOTAL", "RESULT")):
                        print("  ", line)
            else:
                print("  ", "PASS" if rtl_ok else "FAIL")
            if not rtl_ok:
                all_ok = False

    print("\n" + "-" * 60)
    print("Phase 7 report:")
    for name, ok in results:
        print(f"  {name}: {'PASS' if ok else 'FAIL'}")
    total = len(results)
    passed = sum(1 for _, ok in results if ok)
    print(f"TOTAL: {passed}/{total} passed")
    print("RESULT:", "ALL PASS" if all_ok else "SOME FAIL")
    print("=" * 60)
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
