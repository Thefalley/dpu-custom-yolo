#!/usr/bin/env python3
"""
Golden reference test: same vectors as RTL testbenches (mac_int8_tb, leaky_relu_tb, mult_shift_add_tb).
Run this to verify expected values before/after RTL changes. No YOLO, no slow sim.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

# Import Phase 3 primitives (same as RTL)
from phase3_dpu_functional_model import mac, leaky_relu_hardware

def test_mac():
    """Same cases as rtl/tb/mac_int8_tb.sv"""
    cases = [
        (10, 20, 0, 200),
        (-10, 20, 0, -200),
        (127, 127, 0, 16129),
        (50, 50, 1000, 3500),
    ]
    passed = 0
    for w, a, acc_in, expected in cases:
        r = int(mac(np.int8(w), np.int8(a), np.int32(acc_in)))
        ok = r == expected
        passed += 1 if ok else 0
        print(f"  MAC({w:4d}, {a:4d}, {acc_in:6d}) => {r:8d} (expected {expected:8d}) [{'PASS' if ok else 'FAIL'}]")
    return passed, len(cases)

def test_leaky_relu():
    """Same cases as rtl/tb/leaky_relu_tb.sv"""
    cases = [(-80, -10), (40, 40), (0, 0)]
    passed = 0
    for x_in, expected in cases:
        y = int(leaky_relu_hardware(np.array([x_in], dtype=np.int32))[0])
        ok = y == expected
        passed += 1 if ok else 0
        print(f"  LeakyReLU(x={x_in:4d}) => y={y:4d} (expected {expected:4d}) [{'PASS' if ok else 'FAIL'}]")
    return passed, len(cases)

def test_mult_shift_add():
    """Same cases as rtl/tb/mult_shift_add_tb.sv - INT8*INT8 product (no RTL import, compute here)."""
    cases = [(10, 20, 200), (127, 127, 16129), (-5, 4, -20)]
    passed = 0
    for a, b, expected in cases:
        # Python: int8 * int8 -> int16 range
        p = int(np.int8(a)) * int(np.int8(b))
        # Clamp to int16 for comparison with RTL 16-bit product
        if p > 32767:  p = 32767
        if p < -32768: p = -32768
        ok = p == expected
        passed += 1 if ok else 0
        print(f"  Mult({a:4d}, {b:4d}) => product={p:6d} (expected {expected:6d}) [{'PASS' if ok else 'FAIL'}]")
    return passed, len(cases)

def main():
    print("=" * 60)
    print("RTL GOLDEN VECTOR TEST (same as mac_int8_tb, leaky_relu_tb, mult_shift_add_tb)")
    print("=" * 60)

    total_pass, total_cases = 0, 0

    print("\n[1] MAC INT8 (mac_int8_tb vectors)")
    p, n = test_mac()
    total_pass += p; total_cases += n

    print("\n[2] LeakyReLU (leaky_relu_tb vectors)")
    p, n = test_leaky_relu()
    total_pass += p; total_cases += n

    print("\n[3] Mult shift-add (mult_shift_add_tb vectors)")
    p, n = test_mult_shift_add()
    total_pass += p; total_cases += n

    print("\n" + "-" * 60)
    print(f"TOTAL: {total_pass}/{total_cases} PASS, {total_cases - total_pass} FAIL")
    result = "ALL PASS" if total_pass == total_cases else "SOME FAIL"
    print(f"RESULT: {result}")
    print("=" * 60)
    return 0 if total_pass == total_cases else 1

if __name__ == "__main__":
    sys.exit(main())
