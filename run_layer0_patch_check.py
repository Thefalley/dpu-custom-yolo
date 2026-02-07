#!/usr/bin/env python3
"""
Layer0 patch check: one pixel of layer 0 (real image data) — Python golden + RTL.

1. Ensures image_sim_out/ has image + weights (runs run_image_to_detection --synthetic if needed).
2. Runs tests/layer0_patch_golden.py to generate patch vectors and hex files.
3. Runs RTL sim (layer0_patch_tb_iv) if iverilog is available.
4. Reports pass/fail.

Usage (from project root):
  python run_layer0_patch_check.py
  python run_layer0_patch_check.py --python-only
"""

import sys
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.resolve()
IMAGE_SIM_OUT = PROJECT_ROOT / "image_sim_out"
SIM_OUT = PROJECT_ROOT / "sim_out"


def need_image_data():
    inp = IMAGE_SIM_OUT / "image_input_layer0.npy"
    if inp.exists():
        return False
    if SIM_OUT.is_dir() and (SIM_OUT / "image_input_layer0.npy").exists():
        return False
    return True


def run_image_pipeline():
    r = subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "run_image_to_detection.py"), "--synthetic"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        timeout=60,
    )
    return r.returncode == 0


def run_golden():
    r = subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "tests" / "layer0_patch_golden.py")],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        timeout=30,
    )
    return r.returncode == 0, (r.stdout or "") + (r.stderr or "")


def run_rtl():
    sv_py = PROJECT_ROOT / "verilog-sim-py" / "sv_simulator.py"
    files = [
        PROJECT_ROOT / "rtl" / "dpu" / "primitives" / "mac_int8.sv",
        PROJECT_ROOT / "rtl" / "dpu" / "primitives" / "leaky_relu.sv",
        PROJECT_ROOT / "rtl" / "dpu" / "primitives" / "requantize.sv",
        PROJECT_ROOT / "rtl" / "tb" / "layer0_patch_tb_iv.sv",
    ]
    cmd = [sys.executable, str(sv_py), "--no-wave"] + [str(f) for f in files] + ["--top", "layer0_patch_tb"]
    r = subprocess.run(cmd, cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=90)
    out = (r.stdout or "") + (r.stderr or "")
    return "RESULT:" in out and "ALL PASS" in out and "SOME FAIL" not in out, out


def main():
    import argparse
    ap = argparse.ArgumentParser(description="Layer0 patch: Python golden + RTL (one pixel)")
    ap.add_argument("--python-only", action="store_true", help="Skip RTL simulation")
    args = ap.parse_args()

    print("=" * 60)
    print("Layer0 patch check (one pixel, real image data)")
    print("=" * 60)

    if need_image_data():
        print("\n[1] Generating image + layer0 weights (run_image_to_detection --synthetic)")
        if not run_image_pipeline():
            print("  [FAIL] run_image_to_detection failed")
            return 1
    else:
        print("\n[1] Image data already present (image_sim_out/ or sim_out/)")

    print("\n[2] Python golden (layer0_patch_golden.py)")
    ok, out = run_golden()
    if not ok:
        print("  [FAIL]", out.strip() or "layer0_patch_golden.py failed")
        return 1
    for line in out.strip().splitlines():
        print("  ", line)

    if args.python_only:
        print("\n[3] RTL: skipped (--python-only)")
        print("\nRESULT: Python golden OK. Run without --python-only when iverilog is in PATH.")
        print("=" * 60)
        return 0

    print("\n[3] RTL (layer0_patch_tb_iv)")
    rtl_ok, rtl_out = run_rtl()
    if rtl_ok:
        print("  PASS")
    else:
        print("  FAIL")
        if "Faltan herramientas" in rtl_out or "not found" in rtl_out.lower():
            print("  [Tip] Add OSS CAD Suite or Icarus Verilog to PATH for RTL sim.")

    print("\n" + "-" * 60)
    print("RESULT:", "ALL PASS" if rtl_ok else "RTL failed (Python golden OK)")
    print("=" * 60)
    return 0 if rtl_ok else 1


if __name__ == "__main__":
    sys.exit(main())
