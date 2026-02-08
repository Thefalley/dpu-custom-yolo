#!/usr/bin/env python3
"""Layer17 maxpool check: 2x2, 52x52 -> 26x26"""
import sys, subprocess
from pathlib import Path
PROJECT_ROOT = Path(__file__).parent.resolve()

def main():
    print("=" * 60)
    print("Layer17 maxpool check (2x2: 52x52 -> 26x26)")
    print("=" * 60)
    for d in (PROJECT_ROOT / "image_sim_out", PROJECT_ROOT / "sim_out"):
        if d.is_dir() and (d / "layer17_output_ref.npy").exists():
            break
    else:
        subprocess.run([sys.executable, str(PROJECT_ROOT / "run_image_to_detection.py"), "--synthetic", "--layers", "18"], cwd=str(PROJECT_ROOT), capture_output=True)
    r = subprocess.run([sys.executable, str(PROJECT_ROOT / "tests" / "layer17_maxpool_golden.py")], cwd=str(PROJECT_ROOT), capture_output=True, text=True)
    print("  ", (r.stdout or "").replace("\n", "\n   "))
    print("=" * 60)
    return r.returncode

if __name__ == "__main__":
    sys.exit(main())
