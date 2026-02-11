#!/usr/bin/env python3
"""Run the debug TB simulation and capture output."""
import subprocess, os, sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.resolve()

env = dict(os.environ)
env["OSS_CAD_PATH"] = r"C:\iverilog"

# First compile
files = [
    PROJECT_ROOT / "rtl" / "dpu" / "primitives" / "mac_int8.sv",
    PROJECT_ROOT / "rtl" / "dpu" / "primitives" / "leaky_relu.sv",
    PROJECT_ROOT / "rtl" / "dpu" / "primitives" / "requantize.sv",
    PROJECT_ROOT / "rtl" / "dpu" / "mac_array_32x32.sv",
    PROJECT_ROOT / "rtl" / "dpu" / "post_process_array.sv",
    PROJECT_ROOT / "rtl" / "dpu" / "conv_engine_array.sv",
    PROJECT_ROOT / "rtl" / "dpu" / "maxpool_unit.sv",
    PROJECT_ROOT / "rtl" / "dpu" / "dpu_top.sv",
    PROJECT_ROOT / "rtl" / "tb" / "dpu_top_debug_tb.sv",
]

iverilog = r"C:\iverilog\bin\iverilog.exe"
vvp = r"C:\iverilog\bin\vvp.exe"

print("[COMPILE]")
cmd = [iverilog, "-g2012", "-o", "sim_debug.vvp", "-s", "dpu_top_debug_tb"] + [str(f) for f in files]
r = subprocess.run(cmd, cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=60, env=env)
if r.returncode != 0:
    print("COMPILE FAILED:")
    print(r.stdout)
    print(r.stderr)
    sys.exit(1)
if r.stderr:
    print("Warnings:", r.stderr.strip())
print("[OK] Compiled")

print("[SIMULATE]")
r = subprocess.run([vvp, "sim_debug.vvp"], cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=300, env=env)
out = r.stdout + r.stderr
lines = out.strip().splitlines()
print(f"Total output lines: {len(lines)}")

# Print first 100 and last 100 lines
if len(lines) <= 200:
    for line in lines:
        print(line)
else:
    for line in lines[:100]:
        print(line)
    print(f"... ({len(lines) - 200} lines omitted) ...")
    for line in lines[-100:]:
        print(line)

print(f"\nRC={r.returncode}")
