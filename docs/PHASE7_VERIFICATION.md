# PHASE 7: Verification (AutoCheck Testbench)

## Overview

Phase 7 verifies the DPU RTL against the Python reference model. It provides a **single AutoCheck flow** that runs the Python golden vectors and the RTL simulations for each primitive, then reports pass/fail.

**Goal:** Ensure RTL outputs match the Python reference for the same deterministic inputs before moving to FPGA/Vivado/Vitis.

---

## 1. What Was Done

### 1.1 AutoCheck Mechanism

- **Python golden** (`tests/test_rtl_vectors.py`): Already present from Phase 3/6. Runs the same test vectors as the RTL testbenches using Phase 3 primitives (MAC, LeakyReLU, mult). Exit 0 = all pass.
- **RTL testbenches** (`rtl/tb/*_iv.sv`): Already compare RTL output to expected values (same numbers as Python) and print `RESULT: ALL PASS` or `RESULT: SOME FAIL`.
- **Phase 7 runner** (`run_phase7_autocheck.py`): Orchestrates both:
  1. Runs Python golden and captures result.
  2. For each primitive (mac_int8, leaky_relu, mult_shift_add), runs RTL simulation via `verilog-sim-py/sv_simulator.py` (iverilog + vvp).
  3. Parses RTL stdout for `RESULT: ALL PASS`.
  4. Prints a single report and exits 0 only if Python and all RTL tests pass.

### 1.2 What Is Verified

| Primitive       | Python reference        | RTL TB              | Vectors (aligned) |
|----------------|-------------------------|---------------------|--------------------|
| **mac_int8**   | `phase3_dpu_functional_model.mac` | mac_int8_tb_iv.sv  | 4 cases (W, A, acc_in → acc_out) |
| **leaky_relu** | `leaky_relu_hardware`   | leaky_relu_tb_iv.sv | 3 cases (x → y)   |
| **mult_shift_add** | int8×int8 product  | mult_shift_add_tb_iv.sv | 3 cases (a, b → product) |

The **same expected values** are used in Python and in the RTL TBs, so “RTL vs Python” is satisfied by construction when both pass.

### 1.3 Deliverables

- **Testbenches:** Existing `rtl/tb/mac_int8_tb_iv.sv`, `leaky_relu_tb_iv.sv`, `mult_shift_add_tb_iv.sv` with in-simulation PASS/FAIL.
- **Pass/fail reports:** From `run_phase7_autocheck.py` (summary) and from each RTL TB (per-case and total).
- **Verification documentation:** This file (PHASE7_VERIFICATION.md).

---

## 2. How to Run

### 2.1 Full AutoCheck (Python + RTL)

From project root:

```bash
python run_phase7_autocheck.py
```

**Requirements:**

- Python 3 with numpy and `phase3_dpu_functional_model` (project root in PYTHONPATH or run from root).
- For RTL: **Icarus Verilog** (or OSS CAD Suite) in PATH. If not installed, RTL steps will fail; use `--python-only` to still run the golden.

**Output:** Phase 7 report with one line per check (Python golden + mac_int8, leaky_relu, mult_shift_add) and final `RESULT: ALL PASS` or `SOME FAIL`. Exit code 0 only if all pass.

### 2.2 Python Golden Only

Useful when iverilog is not available or to quickly validate the reference:

```bash
python run_phase7_autocheck.py --python-only
```

### 2.3 RTL Only

Run only the three RTL simulations (no Python golden):

```bash
python run_phase7_autocheck.py --rtl-only
```

### 2.4 Quiet Mode

Only the summary lines (no per-case dump):

```bash
python run_phase7_autocheck.py -q
```

### 2.5 Individual RTL Tests (existing flow)

As in Phase 6 / CONTEXTO:

```powershell
.\run_dpu_sim.ps1 mac_int8
.\run_dpu_sim.ps1 leaky_relu
.\run_dpu_sim.ps1 mult_shift_add
```

Or with verilog-sim-py:

```bash
python verilog-sim-py/sv_simulator.py --no-wave rtl/dpu/primitives/mac_int8.sv rtl/tb/mac_int8_tb_iv.sv --top mac_int8_tb
```

---

## 3. Why This Design

- **Single entry point:** One command (`run_phase7_autocheck.py`) for “is RTL aligned with Python?”.
- **Deterministic inputs:** Same vectors in Python and RTL; no random stimulus in this phase.
- **In-TB check:** RTL TBs already compare to expected values; Phase 7 runner only needs to detect `RESULT: ALL PASS` in stdout.
- **No VCD parsing:** Pass/fail is derived from simulator stdout, keeping the flow simple and tool-agnostic.

---

## 4. What Worked

- Reusing `tests/test_rtl_vectors.py` as the single Python golden.
- Reusing existing `*_iv.sv` testbenches and their `RESULT: ALL PASS` / `SOME FAIL` output.
- One Python script that runs Python + all RTL tests and aggregates the report.

---

## 5. What Did Not / Limitations

- **iverilog required for RTL:** If iverilog/vvp are not in PATH, RTL tests are reported as FAIL (with a tip in the output). Use `--python-only` when RTL tools are unavailable.
- **mac_array_2x2 not in AutoCheck yet:** Phase 7 currently verifies only the three primitives. Adding a testbench and golden for `mac_array_2x2` would be a natural extension.
- **Full DPU:** The full DPU (dpu_top_stub) is not in the AutoCheck flow; it is a stub without a closed execution model yet.

---

## 6. Next Steps (Optional)

- Add `mac_array_2x2` testbench + golden vectors and include it in `run_phase7_autocheck.py`.
- When control and buffers exist, add a “full DPU” test (subset of layers or synthetic workload) to the verification flow.
- For FPGA: use this verified RTL as the baseline before Vivado/Vitis.

---

## 7. Summary

Phase 7 delivers **automated verification of DPU primitives (RTL vs Python)** via a single script, deterministic vectors, and clear pass/fail reports. The cycle “reference software ↔ RTL” is closed for the current primitives before growing the RTL further.
