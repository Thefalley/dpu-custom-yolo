# PHASE 6: RTL Implementation (Verilog)

## Overview

This phase implements the DPU primitives and a small subsystem in **SystemVerilog**, aligned with Phase 5 hardware architecture. Simulation uses **Icarus Verilog** (and optionally the `verilog-sim-py` repo).

**Repository for simulation:** [verilog-sim-py](https://github.com/Thefalley/verilog-sim-py) (cloned under `verilog-sim-py/`).

---

## 1. What Was Done

### 1.1 Repository Setup

- **Cloned** `verilog-sim-py` into the project for the simulation flow (Python + Icarus Verilog + GTKWave).
- **Created** `rtl/` under the project root for all DPU RTL (primitives, subsystem, top stub).

### 1.2 Primitives Implemented

| Module | File | Description |
|--------|------|-------------|
| **mac_int8** | `rtl/dpu/primitives/mac_int8.sv` | INT8×INT8 multiply + INT32 accumulate; 1-cycle latency |
| **leaky_relu** | `rtl/dpu/primitives/leaky_relu.sv` | x>0 ? x : (x>>>3); INT32→INT32 |
| **requantize** | `rtl/dpu/primitives/requantize.sv` | INT32×scale → round → clamp to INT8 |
| **mult_shift_add** | `rtl/dpu/primitives/mult_shift_add.sv` | Phase 4b LUT multiplier: INT8×INT8 via shift-and-add |

### 1.3 Subsystem

| Module | File | Description |
|--------|------|-------------|
| **mac_array_2x2** | `rtl/dpu/mac_array_2x2.sv` | 2×2 MAC array (2 output ch × 2 input ch); instantiates 4× mac_int8 |

### 1.4 Top-Level Stub

| Module | File | Description |
|--------|------|-------------|
| **dpu_top_stub** | `rtl/dpu/dpu_top_stub.sv` | Stub: mac_array_2x2 + inline bias/LeakyReLU/clamp; no buffers/control yet |

### 1.5 Testbenches

| Testbench | File | Tests |
|-----------|------|-------|
| **mac_int8_tb** | `rtl/tb/mac_int8_tb.sv` | mac(10,20,0)=200, (-10,20,0)=-200, (127,127,0)=16129, (50,50,1000)=3500 |
| **leaky_relu_tb** | `rtl/tb/leaky_relu_tb.sv` | x=-80→-10, x=40→40, x=0→0 |
| **mult_shift_add_tb** | `rtl/tb/mult_shift_add_tb.sv` | 10×20=200, 127×127=16129, -5×4=-20 |

### 1.6 Simulation Script

- **run_sim.py** | `rtl/run_sim.py` | Runs iverilog + vvp for a given test (`mac_int8`, `leaky_relu`, `mult_shift_add`).

---

## 2. Module Hierarchy

```
dpu_top_stub (stub)
  └── mac_array_2x2
        └── mac_int8 (x4)

Standalone primitives (with own testbenches):
  mac_int8, leaky_relu, requantize, mult_shift_add
```

---

## 3. How to Run Simulation

### 3.1 Prerequisites

- **Icarus Verilog:** [http://bleyer.org/icarus/](http://bleyer.org/icarus/) — add to PATH.
- Or **OSS CAD Suite** (portable): extract under `verilog-sim-py/oss-cad-suite/` and use its `bin/iverilog`, `bin/vvp`.

### 3.2 From Project Root

```bash
cd rtl
python run_sim.py mac_int8
python run_sim.py leaky_relu
python run_sim.py mult_shift_add
```

Each run compiles the design + testbench and runs `vvp sim_out`. Expected: PASS for all cases.

### 3.3 Using verilog-sim-py

From project root, passing RTL and TB files explicitly:

```bash
python verilog-sim-py/sv_simulator.py --top mac_int8_tb rtl/dpu/primitives/mac_int8.sv rtl/tb/mac_int8_tb.sv
```

(Requires iverilog/vvp in PATH or OSS CAD Suite.)

---

## 4. Mapping: Python (Phase 3) → RTL

| Python | RTL Module | Notes |
|--------|------------|-------|
| `mac(weight, activation, accumulator)` | `mac_int8` | INT8×INT8 → product; ACC += product |
| `leaky_relu_hardware(x)` | `leaky_relu` | x>0 ? x : (x>>>3) |
| `requantize(acc, scale, zero_point)` | `requantize` | acc*scale >> Q, clamp INT8 |
| Phase 4b shift-add multiply | `mult_shift_add` | Partial products = shifts; adder tree |

---

## 5. What Worked

- **Primitives** match Phase 3 Python behavior (same test vectors).
- **mac_array_2x2** builds on mac_int8; clear path to larger arrays (e.g. 32×32).
- **mult_shift_add** implements Phase 4b formula (no DSP); same interface as a behavioral 8×8 mult for substitution in a MAC.
- **Clean hierarchy:** primitives → subsystem → top stub.

---

## 6. What Did Not Work / Limitations

- **iverilog not in PATH** on the development machine: `run_sim.py` fails with `FileNotFoundError` until iverilog/vvp are installed and on PATH.
- **dpu_top_stub** does not drive weights/activations (tied to 0); full control and buffers are for a later step.
- **requantize** uses a 16-bit scale and fixed SCALE_Q; production would need per-channel scale/zero_point from calibration.

---

## 7. Files Delivered

| Path | Description |
|------|-------------|
| `rtl/dpu/primitives/mac_int8.sv` | MAC primitive |
| `rtl/dpu/primitives/leaky_relu.sv` | LeakyReLU primitive |
| `rtl/dpu/primitives/requantize.sv` | Requantize primitive |
| `rtl/dpu/primitives/mult_shift_add.sv` | LUT shift-and-add multiplier |
| `rtl/dpu/mac_array_2x2.sv` | 2×2 MAC array |
| `rtl/dpu/dpu_top_stub.sv` | Top stub |
| `rtl/tb/mac_int8_tb.sv` | MAC testbench |
| `rtl/tb/leaky_relu_tb.sv` | LeakyReLU testbench |
| `rtl/tb/mult_shift_add_tb.sv` | Mult shift-add testbench |
| `rtl/run_sim.py` | Simulation runner |
| `verilog-sim-py/` | Cloned simulation repo |

---

## 8. Next Steps (Phase 7 — Verification)

1. **AutoCheck testbench:** Drive RTL with the same vectors as Python (e.g. export from Phase 3 or hardcode).
2. **Compare RTL vs Python:** MAC, LeakyReLU, Requantize, mult_shift_add; report pass/fail.
3. **Extend to mac_array and top:** Compare small convolution outputs (RTL vs Python) once control and buffers are added.

**Phase 6 Status: COMPLETE**  
**RTL primitives and small MAC array implemented; simulation requires iverilog on PATH.**
