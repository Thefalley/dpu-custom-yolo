#!/usr/bin/env python3
"""
Run DMA streaming testbench:
  1. Generate golden hex data (reuse dpu_top_18layer_golden)
  2. Compile and run RTL TB via Icarus Verilog
  3. Check PASS/FAIL
"""
import sys
import subprocess
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.resolve()


def main():
    print("=" * 60)
    print("  DPU DMA Streaming — AXI-Stream Integration Test")
    print("=" * 60)

    # Step 1: Generate golden data
    print("\n[1/3] Generating golden hex data...")
    r = subprocess.run([sys.executable, str(PROJECT_ROOT / "tests" / "dpu_top_18layer_golden.py")],
                       cwd=str(PROJECT_ROOT), capture_output=True, text=True)
    if r.returncode != 0:
        print(f"[FAIL] Golden model failed:\n{r.stderr}")
        return 1
    print("  Golden data generated.")

    # Step 2: Compile RTL
    print("\n[2/3] Compiling RTL (Icarus Verilog)...")
    env = os.environ.copy()
    oss_path = None
    oss_file = PROJECT_ROOT / ".oss_cad_path"
    if oss_file.exists():
        oss_path = oss_file.read_text().strip()
    elif 'OSS_CAD_PATH' in env:
        oss_path = env['OSS_CAD_PATH']

    iverilog_bin = "iverilog"
    vvp_bin = "vvp"
    if oss_path:
        bin_dir = os.path.join(oss_path, "bin")
        iverilog_bin = os.path.join(bin_dir, "iverilog")
        vvp_bin = os.path.join(bin_dir, "vvp")
        env['PATH'] = bin_dir + os.pathsep + env.get('PATH', '')

    rtl_dir = PROJECT_ROOT / "rtl" / "dpu"
    tb_dir = PROJECT_ROOT / "rtl" / "tb"

    srcs = [
        rtl_dir / "primitives" / "mac_int8.sv",
        rtl_dir / "primitives" / "leaky_relu.sv",
        rtl_dir / "primitives" / "requantize.sv",
        rtl_dir / "primitives" / "mult_shift_add.sv",
        rtl_dir / "mac_array_32x32.sv",
        rtl_dir / "post_process_array.sv",
        rtl_dir / "maxpool_unit.sv",
        rtl_dir / "conv_engine_array.sv",
        rtl_dir / "dpu_top.sv",
        rtl_dir / "dpu_axi_dma.sv",
        rtl_dir / "dpu_system_top.sv",
        tb_dir / "dpu_dma_tb.sv",
    ]

    vvp_file = PROJECT_ROOT / "sim_dma.vvp"
    compile_cmd = [iverilog_bin, "-g2012", "-o", str(vvp_file)]
    for s in srcs:
        compile_cmd.append(str(s))

    r = subprocess.run(compile_cmd, env=env, capture_output=True, text=True, cwd=str(PROJECT_ROOT))
    if r.returncode != 0:
        print(f"[FAIL] Compilation failed:\n{r.stderr}")
        return 1
    print("  Compilation OK.")

    # Step 3: Run simulation
    print("\n[3/3] Running simulation...")
    r = subprocess.run([vvp_bin, str(vvp_file)], env=env, capture_output=True, text=True,
                       cwd=str(PROJECT_ROOT), timeout=600)
    print(r.stdout[-3000:] if len(r.stdout) > 3000 else r.stdout)
    if r.stderr:
        print(r.stderr[-1000:])

    if "ALL TESTS PASSED" in r.stdout:
        print("\n*** DMA STREAMING TEST: PASS ***")
        return 0
    else:
        print("\n*** DMA STREAMING TEST: CHECK OUTPUT ***")
        return 1


if __name__ == "__main__":
    sys.exit(main())
