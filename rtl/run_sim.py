#!/usr/bin/env python3
"""
Run DPU RTL simulation with Icarus Verilog.
Usage: python run_sim.py [test_name]
  test_name: mac_int8 | leaky_relu | mult_shift_add (default: mac_int8)
"""
import subprocess
import sys
import os
from pathlib import Path

RTL_ROOT = Path(__file__).parent.resolve()
DPU_PRIM = RTL_ROOT / "dpu" / "primitives"
DPU = RTL_ROOT / "dpu"
TB = RTL_ROOT / "tb"

TESTS = {
    "mac_int8": {
        "files": [DPU_PRIM / "mac_int8.sv", TB / "mac_int8_tb.sv"],
        "top": "mac_int8_tb",
    },
    "leaky_relu": {
        "files": [DPU_PRIM / "leaky_relu.sv", TB / "leaky_relu_tb.sv"],
        "top": "leaky_relu_tb",
    },
    "mult_shift_add": {
        "files": [DPU_PRIM / "mult_shift_add.sv", TB / "mult_shift_add_tb.sv"],
        "top": "mult_shift_add_tb",
    },
}

def main():
    test_name = sys.argv[1] if len(sys.argv) > 1 else "mac_int8"
    if test_name not in TESTS:
        print(f"Unknown test: {test_name}")
        print(f"Available: {list(TESTS.keys())}")
        sys.exit(1)

    info = TESTS[test_name]
    files = [str(f) for f in info["files"]]
    top = info["top"]

    os.chdir(RTL_ROOT)
    cmd_compile = [
        "iverilog", "-g2012", "-o", "sim_out",
        "-s", top,
    ] + files
    print("[COMPILE]", " ".join(cmd_compile))
    r = subprocess.run(cmd_compile, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stderr or r.stdout)
        sys.exit(1)
    print("[RUN] vvp sim_out")
    r = subprocess.run(["vvp", "sim_out"], cwd=RTL_ROOT)
    sys.exit(r.returncode)

if __name__ == "__main__":
    main()
