#!/usr/bin/env python3
"""
Check if iverilog/vvp are available for RTL simulation.
Uses OSS_CAD_PATH from environment if set; otherwise looks in PATH.
Run from project root: python check_sim_tools.py
"""

import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.resolve()


def main():
    print("Checking simulation tools (iverilog, vvp)...")
    env = os.environ.copy()
    path_add = []
    oss = os.environ.get("OSS_CAD_PATH")
    if not oss:
        path_file = PROJECT_ROOT / ".oss_cad_path"
        if path_file.exists():
            for line in path_file.read_text().splitlines():
                line = line.strip()
                if line and not line.startswith("#"):
                    oss = line
                    break
    if oss:
        p = Path(oss)
        if (p / "bin").exists():
            path_add.append(str(p / "bin"))
            path_add.append(str(p / "lib"))
            print(f"  OSS_CAD_PATH: {oss}")
    if path_add:
        env["PATH"] = os.pathsep.join(path_add + [env.get("PATH", "")])

    for name in ("iverilog", "vvp"):
        exe_name = f"{name}.exe" if os.name == "nt" else name
        try:
            r = subprocess.run(
                [exe_name, "-V"] if name == "iverilog" else [exe_name, "-v"],
                env=env,
                capture_output=True,
                text=True,
                timeout=5,
            )
            out = (r.stdout or "") + (r.stderr or "")
            if r.returncode == 0 or "Icarus" in out or "icarus" in out.lower() or name in out:
                print(f"  [OK] {name} found")
            else:
                cmd = "where" if os.name == "nt" else "which"
                r2 = subprocess.run([cmd, name], env=env, capture_output=True, text=True, timeout=5)
                if r2.returncode == 0 and r2.stdout.strip():
                    print(f"  [OK] {name} found (PATH)")
                else:
                    print(f"  [--] {name} not found")
        except FileNotFoundError:
            print(f"  [--] {name} not found")
        except Exception as e:
            print(f"  [--] {name}: {e}")

    # Quick compile test via verilog-sim-py
    sv_py = PROJECT_ROOT / "verilog-sim-py" / "sv_simulator.py"
    mac_sv = PROJECT_ROOT / "rtl" / "dpu" / "primitives" / "mac_int8.sv"
    tb_sv = PROJECT_ROOT / "rtl" / "tb" / "mac_int8_tb_iv.sv"
    if sv_py.exists() and mac_sv.exists() and tb_sv.exists():
        print("\n  Trying compile (mac_int8 + tb)...")
        try:
            r = subprocess.run(
                [sys.executable, str(sv_py), "--no-wave", str(mac_sv), str(tb_sv), "--top", "mac_int8_tb"],
                cwd=PROJECT_ROOT,
                env=env,
                capture_output=True,
                text=True,
                timeout=30,
            )
            if r.returncode == 0:
                print("  [OK] Compile + sim succeeded")
            else:
                print("  [FAIL] Compile/sim failed:")
                print((r.stderr or r.stdout or "")[:500])
        except Exception as e:
            print(f"  [--] {e}")
    print("\nTo run all RTL sims: set OSS_CAD_PATH (or PATH), then .\\run_all_sims.ps1")
    return 0


if __name__ == "__main__":
    sys.exit(main())
