#!/usr/bin/env python3
"""
DPU Layer0 4x4 validation via command-interface TB (quick).
1. Export 4x4 hex (layer0_full_4x4_export.py).
2. Run RTL: dpu_layer0_cmd_tb_4x4 (LoadImage, LoadWeights, LoadBias, RunLayer0, CompareOutput).
3. Reports ALL PASS or FAIL (512 bytes vs expected).

Usage:
  python run_dpu_layer0_cmd_4x4_check.py
  python run_dpu_layer0_cmd_4x4_check.py --export-only
"""
import sys
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.resolve()


def need_image_data():
    for d in (PROJECT_ROOT / "image_sim_out", PROJECT_ROOT / "sim_out"):
        if d.is_dir() and (d / "image_input_layer0.npy").exists():
            return False
    return True


def main():
    if need_image_data():
        print("[1] Generating image + layer0 (run_image_to_detection --synthetic)")
        r = subprocess.run(
            [sys.executable, str(PROJECT_ROOT / "run_image_to_detection.py"), "--synthetic"],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            timeout=600,
        )
        if r.returncode != 0:
            print(r.stderr or r.stdout or "run_image_to_detection failed")
            return 1
    else:
        print("[1] Image data already present")

    print("\n[2] Export 4x4 region for RTL")
    r = subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "tests" / "layer0_full_4x4_export.py")],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
    )
    print(r.stdout or "")
    if r.returncode != 0:
        print(r.stderr or "layer0_full_4x4_export.py failed")
        return 1

    if "--export-only" in sys.argv:
        print("\n[3] RTL: skipped (--export-only)")
        return 0

    print("\n[3] RTL dpu_layer0_cmd_tb_4x4 (command interface 4x4)")
    sv_sim = PROJECT_ROOT / "verilog-sim-py" / "sv_simulator.py"
    if not sv_sim.exists():
        print("  [SKIP] verilog-sim-py/sv_simulator.py not found")
        return 0
    rtl_files = [
        PROJECT_ROOT / "rtl" / "dpu" / "primitives" / "mac_int8.sv",
        PROJECT_ROOT / "rtl" / "dpu" / "primitives" / "leaky_relu.sv",
        PROJECT_ROOT / "rtl" / "dpu" / "primitives" / "requantize.sv",
        PROJECT_ROOT / "rtl" / "dpu" / "layer0_engine.sv",
        PROJECT_ROOT / "rtl" / "dpu" / "dpu_layer0_top.sv",
        PROJECT_ROOT / "rtl" / "tb" / "dpu_layer0_cmd_tb_4x4.sv",
    ]
    cmd = [
        sys.executable,
        str(sv_sim),
        "--no-wave",
        *[str(f) for f in rtl_files],
        "--top",
        "dpu_layer0_cmd_tb_4x4",
    ]
    r = subprocess.run(cmd, cwd=str(PROJECT_ROOT), capture_output=True, text=True, timeout=300)
    print(r.stdout or "")
    if r.stderr:
        print(r.stderr)
    if r.returncode != 0:
        print("RTL sim failed")
        return 1
    if "ALL PASS" not in (r.stdout or ""):
        print("RTL did not report ALL PASS")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
