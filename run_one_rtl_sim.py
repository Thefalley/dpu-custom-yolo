#!/usr/bin/env python3
"""Run one RTL sim (mac_int8) and write output to sim_log.txt. Use OSS_CAD_PATH if set."""
import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.resolve()
os.chdir(PROJECT_ROOT)

env = os.environ.copy()
oss = os.environ.get("OSS_CAD_PATH")
if oss:
    p = Path(oss)
    if (p / "bin").exists():
        env["PATH"] = f"{p / 'bin'};{p / 'lib'};{env.get('PATH', '')}"

iverilog = "iverilog.exe" if os.name == "nt" else "iverilog"
vvp = "vvp.exe" if os.name == "nt" else "vvp"

# Compile
r1 = subprocess.run(
    [iverilog, "-g2012", "-o", "sim_mac", "-s", "mac_int8_tb",
     "rtl/dpu/primitives/mac_int8.sv", "rtl/tb/mac_int8_tb_iv.sv"],
    cwd=PROJECT_ROOT,
    env=env,
    capture_output=True,
    text=True,
    timeout=30,
)
log = f"=== COMPILE ===\nstdout: {r1.stdout}\nstderr: {r1.stderr}\nreturncode: {r1.returncode}\n"
if r1.returncode != 0:
    with open("sim_log.txt", "w") as f:
        f.write(log)
    print(log)
    sys.exit(1)

# Run vvp (may hang on Windows after $finish); redirect to file, keep files open until process ends
vvp_out = PROJECT_ROOT / "vvp_out.txt"
vvp_err = PROJECT_ROOT / "vvp_err.txt"
stdout = stderr = ""
passed = False
try:
    with open(vvp_out, "w") as fo, open(vvp_err, "w") as fe:
        proc = subprocess.Popen(
            [vvp, "sim_mac"],
            cwd=PROJECT_ROOT,
            env=env,
            stdout=fo,
            stderr=fe,
            text=True,
        )
        try:
            proc.wait(timeout=8)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
    stdout = vvp_out.read_text(encoding="utf-8", errors="replace") if vvp_out.exists() else ""
    stderr = vvp_err.read_text(encoding="utf-8", errors="replace") if vvp_err.exists() else ""
    log += f"\n=== RUN ===\nstdout: {stdout}\nstderr: {stderr}\nreturncode: {proc.returncode}\n"
    passed = proc.returncode == 0 and "ALL PASS" in stdout
except FileNotFoundError:
    log += "\n=== RUN === vvp not found\n"
except Exception as e:
    log += f"\n=== RUN === {e}\n"
    if vvp_out.exists():
        stdout = vvp_out.read_text(encoding="utf-8", errors="replace")
    if vvp_err.exists():
        stderr = vvp_err.read_text(encoding="utf-8", errors="replace")
    passed = "ALL PASS" in stdout

with open("sim_log.txt", "w", encoding="utf-8") as f:
    f.write(log)
try:
    print(stdout or stderr or log)
except OSError:
    pass
sys.exit(0 if passed else 1)
