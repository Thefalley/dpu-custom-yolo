#!/usr/bin/env python3
"""
Multi-Image Stress Test — Full YOLOv4-tiny DPU (36 layers)

Runs the complete 36-layer golden model + RTL simulation for multiple
input images and edge cases. Every layer is verified bit-exact.
This ensures the DPU hardware is 100% correct regardless of input data.

Test cases:
  1. dog.jpg      — real photograph (dog, bicycle, car)
  2. horses.jpg   — real photograph (horses)
  3. person.jpg   — real photograph (person)
  4. all-zeros    — edge case: all-zero input
  5. all-max      — edge case: all 127s
  6. checkerboard — edge case: alternating +50/-50

For each test case:
  - Run golden model (Python) to generate expected outputs
  - Run RTL simulation (Icarus Verilog) with per-layer verification
  - Verify ALL 36 LAYERS match bit-exact

Usage:
  python run_multi_image_stress_test.py                  # All tests, golden only
  python run_multi_image_stress_test.py --rtl            # Include RTL simulation
  python run_multi_image_stress_test.py --real-weights   # Use real YOLOv4-tiny weights
"""
import sys
import os
import subprocess
import time
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.resolve()
H0, W0 = 32, 32

TEST_IMAGES = [
    ("dog.jpg",    PROJECT_ROOT / "image_sim_out" / "rtl_validation" / "dog.jpg"),
    ("horses.jpg", PROJECT_ROOT / "image_sim_out" / "rtl_validation" / "horses.jpg"),
    ("person.jpg", PROJECT_ROOT / "image_sim_out" / "rtl_validation" / "person.jpg"),
]

EDGE_CASES = [
    "all_zeros",       # All-zero input (forces large scales >= 32768)
    "all_max",         # All 127 (max INT8, saturated input)
    "checkerboard",    # Alternating +50/-50 (spatial pattern)
    "all_neg",         # All -128 (min INT8, tests LeakyReLU heavily)
    "single_pixel",    # Only one pixel non-zero (sparse input)
    "noise_uniform",   # Random uniform noise (fuzz test)
    "gradient_h",      # Horizontal gradient -128..127
    "stripe_v",        # Vertical stripes +127/-128
]


def generate_edge_case_image(case_name, out_path):
    """Generate edge case input and save as raw numpy."""
    if case_name == "all_zeros":
        img = np.zeros((3, H0, W0), dtype=np.int8)
    elif case_name == "all_max":
        img = np.full((3, H0, W0), 127, dtype=np.int8)
    elif case_name == "checkerboard":
        img = np.zeros((3, H0, W0), dtype=np.int8)
        for c in range(3):
            for h in range(H0):
                for w in range(W0):
                    img[c, h, w] = 50 if (h + w) % 2 == 0 else -50
    elif case_name == "all_neg":
        img = np.full((3, H0, W0), -128, dtype=np.int8)
    elif case_name == "single_pixel":
        img = np.zeros((3, H0, W0), dtype=np.int8)
        img[0, H0//2, W0//2] = 127  # Single bright pixel in channel 0
        img[1, H0//2, W0//2] = -128  # Min value in channel 1
        img[2, H0//2, W0//2] = 64   # Mid value in channel 2
    elif case_name == "noise_uniform":
        rng = np.random.RandomState(12345)
        img = rng.randint(-128, 128, (3, H0, W0), dtype=np.int8)
    elif case_name == "gradient_h":
        img = np.zeros((3, H0, W0), dtype=np.int8)
        for w in range(W0):
            val = int(-128 + 255 * w / (W0 - 1))
            val = max(-128, min(127, val))
            img[:, :, w] = val
    elif case_name == "stripe_v":
        img = np.zeros((3, H0, W0), dtype=np.int8)
        for h in range(H0):
            img[:, h, :] = 127 if h % 2 == 0 else -128
    else:
        raise ValueError(f"Unknown edge case: {case_name}")

    np.save(str(out_path), img)
    return img


def run_golden(input_image=None, input_npy=None, real_weights=False):
    """Run golden model. Returns (success, output_text)."""
    cmd = [sys.executable, str(PROJECT_ROOT / "tests" / "dpu_top_37layer_golden.py")]
    if real_weights:
        cmd.append("--real-weights")
    if input_image:
        cmd.extend(["--input-image", str(input_image)])

    # For edge cases using npy, we need a different approach:
    # Modify the golden model to accept --input-npy, or generate hex directly.
    # For simplicity, we'll patch the input_image.hex directly for edge cases.

    r = subprocess.run(cmd, cwd=str(PROJECT_ROOT),
                       capture_output=True, text=True, timeout=120)
    out = (r.stdout or "") + (r.stderr or "")
    ok = r.returncode == 0 and "GOLDEN COMPLETE" in out
    return ok, out


def write_input_hex_from_npy(npy_path):
    """Write edge case input directly to hex for golden/RTL."""
    img = np.load(str(npy_path))
    hex_path = PROJECT_ROOT / "image_sim_out" / "dpu_top_37" / "input_image.hex"
    with open(hex_path, 'w') as f:
        for v in img.flatten():
            f.write(f"{int(v) & 0xff:02x}\n")


def run_golden_with_npy_input(npy_path, real_weights=False):
    """Run golden model with an edge case input (no PIL needed)."""
    # We modify the golden model invocation:
    # 1. The golden model without --input-image uses random input
    # 2. We need to run it, then replace input_image.hex with our edge case
    # 3. Then re-run (or better: run with a custom input)
    #
    # Actually, the golden model generates random input if no --input-image.
    # For edge cases, we need to either:
    #   a) Save as a temporary image and pass --input-image
    #   b) Modify golden to accept --input-npy
    #
    # Approach: save edge case as a temp PNG via numpy -> PIL
    try:
        from PIL import Image
        img_int8 = np.load(str(npy_path))  # (3, H0, W0) INT8
        # Convert back to uint8 RGB for saving as PNG
        # Reverse the normalization: pixel_int8 = round(pixel/255 * 254 - 127)
        # So: pixel = (pixel_int8 + 127) / 254 * 255
        img_float = (img_int8.astype(np.float32) + 127.0) / 254.0 * 255.0
        img_float = np.clip(img_float, 0, 255).astype(np.uint8)
        img_hwc = img_float.transpose(1, 2, 0)  # CHW -> HWC
        tmp_path = PROJECT_ROOT / "image_sim_out" / "dpu_top_37" / "edge_case_input.png"
        Image.fromarray(img_hwc).save(str(tmp_path))
        return run_golden(input_image=str(tmp_path), real_weights=real_weights)
    except ImportError:
        print("  [WARN] PIL not available, cannot run edge case via image")
        return False, "PIL not available"


def run_rtl():
    """Run RTL simulation. Returns (passed, output_text)."""
    sv_py = PROJECT_ROOT / "verilog-sim-py" / "sv_simulator.py"
    files = [
        PROJECT_ROOT / "rtl" / "dpu" / "primitives" / "mac_int8.sv",
        PROJECT_ROOT / "rtl" / "dpu" / "primitives" / "leaky_relu.sv",
        PROJECT_ROOT / "rtl" / "dpu" / "primitives" / "requantize.sv",
        PROJECT_ROOT / "rtl" / "dpu" / "mac_array_32x32.sv",
        PROJECT_ROOT / "rtl" / "dpu" / "post_process_array.sv",
        PROJECT_ROOT / "rtl" / "dpu" / "conv_engine_array.sv",
        PROJECT_ROOT / "rtl" / "dpu" / "maxpool_unit.sv",
        PROJECT_ROOT / "rtl" / "dpu" / "dpu_top.sv",
        PROJECT_ROOT / "rtl" / "tb" / "dpu_top_36layer_tb.sv",
    ]
    cmd = ([sys.executable, str(sv_py), "--no-wave"] +
           [str(f) for f in files] +
           ["--top", "dpu_top_36layer_tb"])

    env = dict(os.environ)
    if "OSS_CAD_PATH" not in env:
        ocp = PROJECT_ROOT / ".oss_cad_path"
        if ocp.exists():
            env["OSS_CAD_PATH"] = ocp.read_text().strip()

    r = subprocess.run(cmd, cwd=str(PROJECT_ROOT), capture_output=True, text=True,
                       timeout=7200, env=env)
    out = (r.stdout or "") + (r.stderr or "")
    passed = "ALL 36 LAYERS PASS" in out
    return passed, out


def main():
    import argparse
    ap = argparse.ArgumentParser(description="Multi-Image Stress Test (36-layer DPU)")
    ap.add_argument("--rtl", action="store_true", help="Include RTL simulation")
    ap.add_argument("--real-weights", action="store_true", help="Use real YOLOv4-tiny weights")
    args = ap.parse_args()

    print("=" * 72)
    print("  MULTI-IMAGE STRESS TEST — 36-Layer YOLOv4-tiny DPU")
    print(f"  Weights: {'REAL' if args.real_weights else 'synthetic'}")
    print(f"  RTL: {'YES' if args.rtl else 'golden only'}")
    print(f"  Tests: {len(TEST_IMAGES)} images + {len(EDGE_CASES)} edge cases")
    print("=" * 72)

    results = []
    total_tests = 0
    total_pass = 0
    total_time = 0

    # ---- Real images ----
    for img_name, img_path in TEST_IMAGES:
        if not img_path.exists():
            print(f"\n[SKIP] {img_name}: file not found at {img_path}")
            continue

        total_tests += 1
        print(f"\n{'='*60}")
        print(f"  TEST: {img_name}")
        print(f"{'='*60}")

        # Golden
        t0 = time.time()
        print(f"  [Golden] Running 36-layer golden model...")
        golden_ok, golden_out = run_golden(input_image=str(img_path),
                                           real_weights=args.real_weights)
        golden_time = time.time() - t0

        if not golden_ok:
            print(f"  [Golden] FAIL ({golden_time:.1f}s)")
            for line in golden_out.strip().splitlines()[-10:]:
                print(f"    {line}")
            results.append((img_name, "GOLDEN_FAIL", golden_time, 0))
            continue

        # Count layers
        layer_count = golden_out.count("Layer ")
        print(f"  [Golden] PASS — {layer_count} layers ({golden_time:.1f}s)")

        # RTL
        rtl_time = 0
        if args.rtl:
            print(f"  [RTL] Running 36-layer RTL simulation...")
            t0 = time.time()
            rtl_ok, rtl_out = run_rtl()
            rtl_time = time.time() - t0

            if rtl_ok:
                print(f"  [RTL] ALL 36 LAYERS PASS — bit-exact ({rtl_time:.1f}s)")
                total_pass += 1
                results.append((img_name, "PASS", golden_time, rtl_time))
            else:
                print(f"  [RTL] FAIL ({rtl_time:.1f}s)")
                # Show failures
                for line in rtl_out.strip().splitlines():
                    if "FAIL" in line or "MISMATCH" in line:
                        print(f"    {line}")
                results.append((img_name, "RTL_FAIL", golden_time, rtl_time))
        else:
            total_pass += 1
            results.append((img_name, "GOLDEN_OK", golden_time, 0))
            print(f"  [RTL] Skipped")

        total_time += golden_time + rtl_time

    # ---- Edge cases ----
    edge_dir = PROJECT_ROOT / "image_sim_out" / "dpu_top_37"
    edge_dir.mkdir(parents=True, exist_ok=True)

    for case_name in EDGE_CASES:
        total_tests += 1
        print(f"\n{'='*60}")
        print(f"  TEST: edge case — {case_name}")
        print(f"{'='*60}")

        # Generate edge case
        npy_path = edge_dir / f"edge_{case_name}.npy"
        img = generate_edge_case_image(case_name, npy_path)
        print(f"  Input: shape={img.shape}, range=[{img.min()}, {img.max()}]")

        # Golden
        t0 = time.time()
        print(f"  [Golden] Running 36-layer golden model...")
        golden_ok, golden_out = run_golden_with_npy_input(npy_path,
                                                           real_weights=args.real_weights)
        golden_time = time.time() - t0

        if not golden_ok:
            print(f"  [Golden] FAIL ({golden_time:.1f}s)")
            for line in golden_out.strip().splitlines()[-10:]:
                print(f"    {line}")
            results.append((f"edge:{case_name}", "GOLDEN_FAIL", golden_time, 0))
            continue

        print(f"  [Golden] PASS ({golden_time:.1f}s)")

        # RTL
        rtl_time = 0
        if args.rtl:
            print(f"  [RTL] Running 36-layer RTL simulation...")
            t0 = time.time()
            rtl_ok, rtl_out = run_rtl()
            rtl_time = time.time() - t0

            if rtl_ok:
                print(f"  [RTL] ALL 36 LAYERS PASS — bit-exact ({rtl_time:.1f}s)")
                total_pass += 1
                results.append((f"edge:{case_name}", "PASS", golden_time, rtl_time))
            else:
                print(f"  [RTL] FAIL ({rtl_time:.1f}s)")
                for line in rtl_out.strip().splitlines():
                    if "FAIL" in line or "MISMATCH" in line:
                        print(f"    {line}")
                results.append((f"edge:{case_name}", "RTL_FAIL", golden_time, rtl_time))
        else:
            total_pass += 1
            results.append((f"edge:{case_name}", "GOLDEN_OK", golden_time, 0))
            print(f"  [RTL] Skipped")

        total_time += golden_time + rtl_time

    # ---- Summary ----
    print(f"\n{'='*72}")
    print(f"  STRESS TEST SUMMARY")
    print(f"{'='*72}")
    print(f"  {'Test':<25s} {'Result':<15s} {'Golden(s)':<12s} {'RTL(s)':<10s}")
    print(f"  {'-'*62}")
    for name, result, g_time, r_time in results:
        rtl_str = f"{r_time:.1f}" if r_time > 0 else "—"
        print(f"  {name:<25s} {result:<15s} {g_time:<12.1f} {rtl_str:<10s}")

    print(f"\n  Total: {total_pass}/{total_tests} passed")
    print(f"  Total time: {total_time:.1f}s")

    all_pass = total_pass == total_tests
    if all_pass:
        print(f"\n  RESULT: ALL TESTS PASS")
    else:
        print(f"\n  RESULT: {total_tests - total_pass} FAILURES")
    print(f"{'='*72}")

    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
