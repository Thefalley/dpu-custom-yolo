#!/usr/bin/env python3
"""
Full layer 0 (208x208x32) validation:
1. Ensures image + layer0 data exist.
2. Export padded input, weights, bias for RTL (layer0_full_208_export.py).
3. Run RTL TB layer0_full_208_tb (writes layer0_rtl_output.hex) — LONG (many minutes).
4. Compare RTL output with layer0_output_ref.npy (compare_layer0_rtl_ref.py).

Usage:
  python run_layer0_full208_check.py
  python run_layer0_full208_check.py --export-only   # only export hex, skip RTL
  python run_layer0_full208_check.py --compare-only  # only compare (after RTL run)
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
    if "--compare-only" in sys.argv:
        print("[Compare only] RTL output vs reference")
        r = subprocess.run(
            [sys.executable, str(PROJECT_ROOT / "tests" / "compare_layer0_rtl_ref.py")],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
        )
        print(r.stdout or "")
        if r.stderr:
            print(r.stderr)
        return r.returncode

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

    print("\n[2] Export full 208 padded + weights + bias")
    r = subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "tests" / "layer0_full_208_export.py")],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
    )
    print(r.stdout or "")
    if r.returncode != 0:
        print(r.stderr or "layer0_full_208_export failed")
        return 1

    if "--export-only" in sys.argv:
        print("\n[3] RTL: skipped (--export-only)")
        return 0

    print("\n[3] RTL layer0_full_208_tb (208x208x32 — LONG, may take 10–30+ min)")
    sv_sim = PROJECT_ROOT / "verilog-sim-py" / "sv_simulator.py"
    if not sv_sim.exists():
        print("  [SKIP] verilog-sim-py/sv_simulator.py not found")
        return 0
    rtl_files = [
        PROJECT_ROOT / "rtl" / "dpu" / "primitives" / "mac_int8.sv",
        PROJECT_ROOT / "rtl" / "dpu" / "primitives" / "leaky_relu.sv",
        PROJECT_ROOT / "rtl" / "dpu" / "primitives" / "requantize.sv",
        PROJECT_ROOT / "rtl" / "tb" / "layer0_full_208_tb_iv.sv",
    ]
    cmd = [
        sys.executable,
        str(sv_sim),
        "--no-wave",
        *[str(f) for f in rtl_files],
        "--top",
        "layer0_full_208_tb",
    ]
    r = subprocess.run(cmd, cwd=str(PROJECT_ROOT), capture_output=True, text=True, timeout=3600)
    print(r.stdout or "")
    if r.stderr:
        print(r.stderr)
    if r.returncode != 0:
        print("RTL sim failed")
        return 1

    print("\n[4] Compare RTL output vs reference")
    r = subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "tests" / "compare_layer0_rtl_ref.py")],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
    )
    print(r.stdout or "")
    if r.stderr:
        print(r.stderr)
    return r.returncode


if __name__ == "__main__":
    sys.exit(main())
