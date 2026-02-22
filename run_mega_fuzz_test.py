#!/usr/bin/env python3
"""
Mega Fuzz Test -- Full YOLOv4-tiny DPU (36 layers)

Generates a wide variety of adversarial, boundary, and structured input
patterns, then runs the 36-layer golden model (and optionally RTL) for
each one.  Every layer must match bit-exact.

Test categories:
  1. Random fuzz (20 seeds)  -- full INT8 range (-128..127) per seed
  2. Boundary value inputs   -- extreme/edge INT8 values
  3. Structured patterns     -- diagonal, ring, sparse, single-row/col

Usage:
  python run_mega_fuzz_test.py                     # golden only
  python run_mega_fuzz_test.py --rtl               # include RTL simulation
  python run_mega_fuzz_test.py --real-weights       # use real YOLOv4-tiny weights
  python run_mega_fuzz_test.py --rtl --real-weights # both
"""
import sys
import os
import subprocess
import time
import argparse
import numpy as np
from pathlib import Path

try:
    from PIL import Image
except ImportError:
    Image = None

PROJECT_ROOT = Path(__file__).parent.resolve()
H0, W0 = 32, 32

# ---------------------------------------------------------------------------
# Helper: INT8 CHW array  -->  temp PNG  (reverse the golden normalisation)
# golden: int8 = round(pixel/255 * 254 - 127)
# inverse: pixel = (int8 + 127) / 254 * 255
# ---------------------------------------------------------------------------

def int8_chw_to_png(arr_int8, out_path):
    """Save a (3, H0, W0) INT8 array as a PNG that the golden model will
    reload to (approximately) the same INT8 values."""
    assert Image is not None, "Pillow (PIL) is required"
    img_float = (arr_int8.astype(np.float32) + 127.0) / 254.0 * 255.0
    img_float = np.clip(img_float, 0, 255).astype(np.uint8)
    img_hwc = img_float.transpose(1, 2, 0)  # CHW -> HWC
    Image.fromarray(img_hwc).save(str(out_path))


# ===================================================================
#  INPUT GENERATORS
# ===================================================================

def gen_random_fuzz(seed):
    """Full INT8 random image for a given seed."""
    rng = np.random.RandomState(seed)
    return rng.randint(-128, 128, (3, H0, W0)).astype(np.int8)


# -- Boundary value inputs -------------------------------------------------

def gen_uniform(value):
    """All channels/pixels set to a single INT8 value."""
    return np.full((3, H0, W0), value, dtype=np.int8)


def gen_mixed_boundaries():
    """Channel 0 = -128, channel 1 = 0, channel 2 = 127."""
    img = np.zeros((3, H0, W0), dtype=np.int8)
    img[0, :, :] = -128
    img[1, :, :] = 0
    img[2, :, :] = 127
    return img


def gen_alternating_minmax_per_channel():
    """Per-channel alternation: for each pixel position (h,w) the three
    channels cycle through -128, 0, 127 shifted by channel index."""
    img = np.zeros((3, H0, W0), dtype=np.int8)
    vals = np.array([-128, 0, 127], dtype=np.int8)
    for c in range(3):
        for h in range(H0):
            for w in range(W0):
                img[c, h, w] = vals[(c + h * W0 + w) % 3]
    return img


# -- Structured patterns ---------------------------------------------------

def gen_diagonal_gradient(direction='tl_br'):
    """Diagonal gradient from -128 to 127.
    direction='tl_br' : top-left to bottom-right
    direction='br_tl' : bottom-right to top-left
    """
    img = np.zeros((3, H0, W0), dtype=np.int8)
    max_dist = (H0 - 1) + (W0 - 1)
    for h in range(H0):
        for w in range(W0):
            if direction == 'tl_br':
                d = h + w
            else:
                d = (H0 - 1 - h) + (W0 - 1 - w)
            val = int(-128 + 255 * d / max_dist)
            val = max(-128, min(127, val))
            img[:, h, w] = val
    return img


def gen_ring_pattern():
    """Concentric square rings of alternating +127 / -128."""
    img = np.zeros((3, H0, W0), dtype=np.int8)
    for h in range(H0):
        for w in range(W0):
            ring = min(h, w, H0 - 1 - h, W0 - 1 - w)
            img[:, h, w] = 127 if ring % 2 == 0 else -128
    return img


def gen_single_row_active():
    """Only the middle row is non-zero (all others zero)."""
    img = np.zeros((3, H0, W0), dtype=np.int8)
    mid = H0 // 2
    img[:, mid, :] = 127
    return img


def gen_single_col_active():
    """Only the middle column is non-zero (all others zero)."""
    img = np.zeros((3, H0, W0), dtype=np.int8)
    mid = W0 // 2
    img[:, :, mid] = 127
    return img


def gen_random_sparse(seed=99):
    """Random sparse: only ~10% of pixels are non-zero, rest zero."""
    rng = np.random.RandomState(seed)
    img = np.zeros((3, H0, W0), dtype=np.int8)
    mask = rng.rand(3, H0, W0) < 0.10
    vals = rng.randint(-128, 128, (3, H0, W0)).astype(np.int8)
    img[mask] = vals[mask]
    return img


# ===================================================================
#  Build the full test vector list
# ===================================================================

def build_test_vectors():
    """Return list of (name, int8_chw_array) test vectors."""
    vectors = []

    # 1. Random fuzz -- 20 seeds
    for seed in range(20):
        vectors.append((f"fuzz_seed{seed:02d}", gen_random_fuzz(seed)))

    # 2. Boundary value inputs
    for val in [-128, -127, -1, 0, 1, 126, 127]:
        vectors.append((f"uniform_{val}", gen_uniform(val)))
    vectors.append(("mixed_boundaries", gen_mixed_boundaries()))
    vectors.append(("alt_minmax_perchan", gen_alternating_minmax_per_channel()))

    # 3. Structured patterns
    vectors.append(("diag_gradient_tlbr", gen_diagonal_gradient('tl_br')))
    vectors.append(("diag_gradient_brtl", gen_diagonal_gradient('br_tl')))
    vectors.append(("ring_pattern", gen_ring_pattern()))
    vectors.append(("single_row_active", gen_single_row_active()))
    vectors.append(("single_col_active", gen_single_col_active()))
    vectors.append(("random_sparse_10pct", gen_random_sparse(seed=99)))

    return vectors


# ===================================================================
#  Golden + RTL runners  (same approach as run_multi_image_stress_test)
# ===================================================================

def run_golden(input_image, real_weights=False):
    """Run the 36-layer golden model. Returns (success, output_text)."""
    cmd = [sys.executable,
           str(PROJECT_ROOT / "tests" / "dpu_top_37layer_golden.py")]
    if real_weights:
        cmd.append("--real-weights")
    cmd.extend(["--input-image", str(input_image)])

    r = subprocess.run(cmd, cwd=str(PROJECT_ROOT),
                       capture_output=True, text=True, timeout=120)
    out = (r.stdout or "") + (r.stderr or "")
    ok = r.returncode == 0 and "GOLDEN COMPLETE" in out
    return ok, out


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

    r = subprocess.run(cmd, cwd=str(PROJECT_ROOT), capture_output=True,
                       text=True, timeout=7200, env=env)
    out = (r.stdout or "") + (r.stderr or "")
    passed = "ALL 36 LAYERS PASS" in out
    return passed, out


# ===================================================================
#  Main
# ===================================================================

def main():
    ap = argparse.ArgumentParser(
        description="Mega Fuzz Test -- 36-layer YOLOv4-tiny DPU")
    ap.add_argument("--rtl", action="store_true",
                    help="Include RTL simulation (default: golden only)")
    ap.add_argument("--real-weights", action="store_true",
                    help="Use real YOLOv4-tiny weights")
    args = ap.parse_args()

    if Image is None:
        print("[FATAL] Pillow (PIL) is required.  pip install Pillow")
        return 1

    vectors = build_test_vectors()

    print("=" * 72)
    print("  MEGA FUZZ TEST -- 36-Layer YOLOv4-tiny DPU")
    print(f"  Weights:  {'REAL' if args.real_weights else 'synthetic'}")
    print(f"  RTL:      {'YES' if args.rtl else 'golden only'}")
    print(f"  Vectors:  {len(vectors)}")
    print("=" * 72)

    tmp_dir = PROJECT_ROOT / "image_sim_out" / "dpu_top_37"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    results = []        # (name, status, golden_time, rtl_time)
    total_tests = 0
    total_pass = 0
    total_time = 0.0

    for name, arr in vectors:
        total_tests += 1
        print(f"\n{'=' * 60}")
        print(f"  TEST {total_tests}/{len(vectors)}: {name}")
        print(f"  Input: shape={arr.shape}, "
              f"range=[{arr.min()}, {arr.max()}], "
              f"nonzero={np.count_nonzero(arr)}/{arr.size}")
        print(f"{'=' * 60}")

        # Save as temp PNG
        png_path = tmp_dir / f"fuzz_{name}.png"
        int8_chw_to_png(arr, png_path)

        # -- Golden --
        t0 = time.time()
        print(f"  [Golden] Running 36-layer golden model ...")
        golden_ok, golden_out = run_golden(input_image=str(png_path),
                                           real_weights=args.real_weights)
        golden_time = time.time() - t0

        if not golden_ok:
            print(f"  [Golden] FAIL ({golden_time:.1f}s)")
            for line in golden_out.strip().splitlines()[-10:]:
                print(f"    {line}")
            results.append((name, "GOLDEN_FAIL", golden_time, 0.0))
            total_time += golden_time
            continue

        print(f"  [Golden] PASS ({golden_time:.1f}s)")

        # -- RTL (optional) --
        rtl_time = 0.0
        if args.rtl:
            print(f"  [RTL] Running 36-layer RTL simulation ...")
            t0 = time.time()
            rtl_ok, rtl_out = run_rtl()
            rtl_time = time.time() - t0

            if rtl_ok:
                print(f"  [RTL] ALL 36 LAYERS PASS -- bit-exact ({rtl_time:.1f}s)")
                total_pass += 1
                results.append((name, "PASS", golden_time, rtl_time))
            else:
                print(f"  [RTL] FAIL ({rtl_time:.1f}s)")
                for line in rtl_out.strip().splitlines():
                    if "FAIL" in line or "MISMATCH" in line:
                        print(f"    {line}")
                results.append((name, "RTL_FAIL", golden_time, rtl_time))
        else:
            total_pass += 1
            results.append((name, "GOLDEN_OK", golden_time, 0.0))

        total_time += golden_time + rtl_time

    # ----------------------------------------------------------------
    #  Summary table
    # ----------------------------------------------------------------
    print(f"\n{'=' * 78}")
    print(f"  MEGA FUZZ TEST SUMMARY")
    print(f"{'=' * 78}")
    print(f"  {'#':<4s} {'Test':<28s} {'Result':<14s} "
          f"{'Golden(s)':<12s} {'RTL(s)':<10s}")
    print(f"  {'-' * 72}")

    for idx, (name, status, g_time, r_time) in enumerate(results, 1):
        rtl_str = f"{r_time:.1f}" if r_time > 0 else "---"
        print(f"  {idx:<4d} {name:<28s} {status:<14s} "
              f"{g_time:<12.1f} {rtl_str:<10s}")

    total_fail = total_tests - total_pass
    print(f"\n  Total: {total_pass}/{total_tests} passed, "
          f"{total_fail} failed")
    print(f"  Total time: {total_time:.1f}s")

    if total_fail == 0:
        print(f"\n  RESULT: ALL {total_tests} TESTS PASS")
    else:
        print(f"\n  RESULT: {total_fail} FAILURE(S)")
    print(f"{'=' * 78}")

    return 0 if total_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
