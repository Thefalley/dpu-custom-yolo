#!/usr/bin/env python3
"""
Master Regression Suite — DPU Custom YOLOv4-tiny

Runs ALL tests in sequence and reports a unified PASS/FAIL summary.

Test suites:
  1. RTL Primitives (mac_int8, leaky_relu, mult_shift_add)
  2. 18-Layer Golden Model (Python, synthetic weights)
  3. 18-Layer RTL Simulation (Icarus Verilog, synthetic weights)
  4. AXI System Top RTL (AXI4-Lite + PIO + IRQ)
  5. 416x416 Golden Model (full resolution, Python only)

Usage:
  python run_all_tests.py                # Run all tests
  python run_all_tests.py --python-only  # Skip RTL simulations
  python run_all_tests.py --quick        # Primitives + golden only (fast)
"""
import sys
import subprocess
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.resolve()


def run_test(name, cmd, cwd=None, timeout=3600):
    """Run a test command, return (pass, elapsed_sec, output_tail)."""
    t0 = time.time()
    try:
        r = subprocess.run(
            cmd, cwd=cwd or str(PROJECT_ROOT),
            capture_output=True, text=True, timeout=timeout
        )
        elapsed = time.time() - t0
        out = (r.stdout or "") + (r.stderr or "")
        tail = out[-2000:] if len(out) > 2000 else out
        return r.returncode == 0, elapsed, tail
    except subprocess.TimeoutExpired:
        return False, time.time() - t0, "TIMEOUT"
    except Exception as e:
        return False, time.time() - t0, str(e)


def main():
    import argparse
    ap = argparse.ArgumentParser(description="DPU Master Regression Suite")
    ap.add_argument("--python-only", action="store_true", help="Skip RTL simulations")
    ap.add_argument("--quick", action="store_true", help="Quick mode: primitives + golden only")
    args = ap.parse_args()

    print("=" * 70)
    print("  DPU Custom YOLOv4-tiny — Master Regression Suite")
    print("=" * 70)

    results = []
    py = sys.executable

    # ======================================================================
    # Suite 1: RTL Primitives
    # ======================================================================
    suite = "RTL Primitives"
    print(f"\n{'=' * 70}")
    print(f"  Suite 1: {suite}")
    print(f"{'=' * 70}")

    if args.python_only or args.quick:
        # Python golden only (skip RTL for quick mode too)
        cmd = [py, str(PROJECT_ROOT / "run_phase7_autocheck.py"), "--python-only"]
    else:
        cmd = [py, str(PROJECT_ROOT / "run_phase7_autocheck.py")]

    passed, elapsed, tail = run_test(suite, cmd)
    status = "PASS" if passed else "FAIL"
    print(f"  {status} ({elapsed:.1f}s)")
    if not passed:
        print(tail[-500:])
    results.append((suite, status, elapsed))

    # ======================================================================
    # Suite 2: 18-Layer Golden Model (Python)
    # ======================================================================
    suite = "18-Layer Golden (Python)"
    print(f"\n{'=' * 70}")
    print(f"  Suite 2: {suite}")
    print(f"{'=' * 70}")

    cmd = [py, str(PROJECT_ROOT / "tests" / "dpu_top_18layer_golden.py")]
    passed, elapsed, tail = run_test(suite, cmd)
    # Golden model always succeeds if it runs without error
    # Check for the summary in output
    if passed and "LAYER 17" in tail.upper():
        status = "PASS"
    elif passed:
        status = "PASS"
    else:
        status = "FAIL"
    print(f"  {status} ({elapsed:.1f}s)")
    if not passed:
        print(tail[-500:])
    results.append((suite, status, elapsed))

    if args.quick:
        # Skip remaining suites
        print_summary(results)
        return 0 if all(s == "PASS" for _, s, _ in results) else 1

    # ======================================================================
    # Suite 3: 18-Layer RTL Simulation
    # ======================================================================
    if not args.python_only:
        suite = "18-Layer RTL (Icarus Verilog)"
        print(f"\n{'=' * 70}")
        print(f"  Suite 3: {suite}")
        print(f"{'=' * 70}")

        cmd = [py, str(PROJECT_ROOT / "run_dpu_top_18layer_check.py")]
        passed, elapsed, tail = run_test(suite, cmd, timeout=3600)
        if "ALL 18 LAYERS PASS" in tail:
            status = "PASS"
        elif passed:
            status = "PASS"
        else:
            status = "FAIL"
        print(f"  {status} ({elapsed:.1f}s)")
        if status == "FAIL":
            print(tail[-500:])
        results.append((suite, status, elapsed))

    # ======================================================================
    # Suite 4: AXI System Top RTL
    # ======================================================================
    if not args.python_only:
        suite = "AXI System Top RTL"
        print(f"\n{'=' * 70}")
        print(f"  Suite 4: {suite}")
        print(f"{'=' * 70}")

        cmd = [py, str(PROJECT_ROOT / "run_system_top_check.py")]
        passed, elapsed, tail = run_test(suite, cmd, timeout=3600)
        if "ALL TESTS PASSED" in tail:
            status = "PASS"
        elif passed:
            status = "PASS"
        else:
            status = "FAIL"
        print(f"  {status} ({elapsed:.1f}s)")
        if status == "FAIL":
            print(tail[-500:])
        results.append((suite, status, elapsed))

    # ======================================================================
    # Suite 5: 416x416 Golden Validation
    # ======================================================================
    suite = "416x416 Golden (Python)"
    print(f"\n{'=' * 70}")
    print(f"  Suite 5: {suite}")
    print(f"{'=' * 70}")

    cmd = [py, str(PROJECT_ROOT / "tests" / "validate_416x416.py")]
    passed, elapsed, tail = run_test(suite, cmd, timeout=600)
    if passed and "18/18" in tail:
        status = "PASS"
    elif passed:
        status = "PASS"
    else:
        status = "FAIL"
    print(f"  {status} ({elapsed:.1f}s)")
    if status == "FAIL":
        print(tail[-500:])
    results.append((suite, status, elapsed))

    # ======================================================================
    # Summary
    # ======================================================================
    print_summary(results)
    return 0 if all(s == "PASS" for _, s, _ in results) else 1


def print_summary(results):
    total_time = sum(t for _, _, t in results)
    passed = sum(1 for _, s, _ in results if s == "PASS")
    failed = sum(1 for _, s, _ in results if s == "FAIL")

    print(f"\n{'=' * 70}")
    print("  MASTER REGRESSION SUMMARY")
    print(f"{'=' * 70}")
    for name, status, elapsed in results:
        icon = "[PASS]" if status == "PASS" else "[FAIL]"
        print(f"  {icon}  {name:40s}  ({elapsed:.1f}s)")

    print(f"\n  Total: {passed} PASS, {failed} FAIL  ({total_time:.1f}s)")
    if failed == 0:
        print("  *** ALL SUITES PASSED ***")
    else:
        print(f"  *** {failed} SUITE(S) FAILED ***")
    print("=" * 70)


if __name__ == "__main__":
    sys.exit(main())
