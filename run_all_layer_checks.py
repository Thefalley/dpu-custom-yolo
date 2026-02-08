#!/usr/bin/env python3
"""
Run all layer validation checks in sequence.
Usage:
  python run_all_layer_checks.py            # Run all (Python + RTL if available)
  python run_all_layer_checks.py --python-only  # Python golden only
"""
import sys
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.resolve()

CHECKS = [
    ("Layer 0 Patch (27 MACs)", "run_layer0_patch_check.py"),
    ("Layer 1 Patch (288 MACs)", "run_layer1_patch_check.py"),
    ("Layer 2 Patch (576 MACs)", "run_layer2_patch_check.py"),
    ("Layer 3 Route (channel split)", "run_layer3_route_check.py"),
    ("Layer 4 Patch (288 MACs)", "run_layer4_patch_check.py"),
    ("Layer 5 Patch (288 MACs)", "run_layer5_patch_check.py"),
    ("Layer 6 Route (channel concat)", "run_layer6_route_check.py"),
    ("Layer 7 Patch (64 MACs, 1x1)", "run_layer7_patch_check.py"),
    ("Layer 8 Route (channel concat)", "run_layer8_route_check.py"),
    ("Layer 9 MaxPool (2x2)", "run_layer9_maxpool_check.py"),
    ("Layer 10 Patch (1152 MACs)", "run_layer10_patch_check.py"),
    ("Layer 11 Route (split)", "run_layer11_route_check.py"),
    ("Layer 12 Patch (576 MACs)", "run_layer12_patch_check.py"),
    ("Layer 13 Patch (576 MACs)", "run_layer13_patch_check.py"),
    ("Layer 14 Route (concat)", "run_layer14_route_check.py"),
    ("Layer 15 Patch (128 MACs, 1x1)", "run_layer15_patch_check.py"),
    ("Layer 16 Route (concat)", "run_layer16_route_check.py"),
    ("Layer 17 MaxPool (2x2)", "run_layer17_maxpool_check.py"),
]


def main():
    args = sys.argv[1:]
    print("=" * 70)
    print("Running all layer validation checks")
    print("=" * 70)

    results = []
    for name, script in CHECKS:
        script_path = PROJECT_ROOT / script
        if not script_path.exists():
            print(f"\n[SKIP] {name}: {script} not found")
            results.append((name, "SKIP"))
            continue

        print(f"\n{'=' * 70}")
        print(f"[RUN] {name}")
        print("=" * 70)

        cmd = [sys.executable, str(script_path)] + args
        r = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
        if r.returncode == 0:
            results.append((name, "PASS"))
        else:
            results.append((name, "FAIL"))

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for name, status in results:
        icon = "[OK]" if status == "PASS" else ("[-]" if status == "SKIP" else "[X]")
        print(f"  {icon} {name}: {status}")

    passed = sum(1 for _, s in results if s == "PASS")
    failed = sum(1 for _, s in results if s == "FAIL")
    skipped = sum(1 for _, s in results if s == "SKIP")
    print(f"\nTotal: {passed} PASS, {failed} FAIL, {skipped} SKIP")

    if failed > 0:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
