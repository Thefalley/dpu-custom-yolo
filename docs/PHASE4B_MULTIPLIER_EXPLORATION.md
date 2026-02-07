# PHASE 4B: Multiplier Architecture Exploration

## Overview

Phase 4 assumed **one DSP per MAC** (INT8×INT8). This phase explores **alternative multiplier implementations**: DSP-based vs **LUT-based (shift-and-add)** so that the DPU can:

- Run on FPGAs with few DSPs (e.g. ZedBoard / Zynq-7020).
- Be **ASIC-portable** (no vendor DSP blocks).
- Use **shifts and additions** instead of dedicated multipliers where desired.

**Target FPGA (exploration):** ZedBoard (Zynq-7020) – 220 DSP48E1, 53.2K LUTs.  
**Workload:** YOLOv4-tiny (~3.45 GMACs).

---

## 1. What We Did

1. **Defined multiplier implementation options**
   - **DSP:** 1× INT8×INT8 per DSP, or 2× packed per DSP.
   - **LUT (shift-and-add):** Serial (8 cycles), Booth radix-4 (4 cycles), parallel array, pipelined 2-stage.
   - **Hybrid:** Mix of DSP and LUT multipliers.

2. **Parameterized each option**
   - Resources per multiplier: DSPs, LUTs, FFs.
   - Latency, throughput (MACs/cycle), max frequency.

3. **Sized MAC arrays** for ZedBoard (resource limits) and **estimated YOLOv4-tiny FPS** for each configuration.

4. **Compared** DSP vs LUT vs hybrid and documented **ASIC considerations** for shift-and-add.

**Script:** `phase4b_multiplier_exploration.py`  
**Output:** `phase4b_multiplier_results.json` (after running the script).

---

## 2. Why We Did It

- **Phase 4** chose MEDIUM (32×32 MACs) assuming 1024 DSPs; Zynq-7020 has only 220 DSPs.
- To support **small FPGAs** and **ASIC** we need a path that does **not** rely on one DSP per MAC.
- **Shift-and-add multiplication** is a standard way to implement INT8×INT8 with LUTs only:
  - \( A \times B = \sum_{i=0}^{7} (A \cdot B[i]) \ll i \)
  - Multiplications by powers of 2 are **shifts**; then sum partial products (add tree or serial accumulator).

---

## 3. Multiplier Options Summary

| Multiplier | Type | DSP/MAC | LUTs/MAC | Throughput | Max Freq | Notes |
|------------|------|---------|----------|------------|----------|-------|
| DSP48E1 (Native) | DSP | 1.0 | ~20 | 1.0 | 200 MHz | Baseline |
| DSP48E1 (Packed 2×) | DSP | 0.5 | ~40 | 1.0 | 180 MHz | 2 INT8×INT8 per DSP |
| LUT Array (Parallel) | LUT | 0 | ~64 | 1.0 | 150 MHz | Full adder tree |
| LUT Serial (Shift-Add) | LUT | 0 | ~24 | 0.125 | 200 MHz | 8 cycles per multiply |
| LUT Booth Radix-4 | LUT | 0 | ~40 | 0.25 | 180 MHz | 4 cycles per multiply |
| LUT Pipelined (2-stage) | LUT | 0 | ~48 | 1.0 | 160 MHz | Recommended LUT option |
| Hybrid 50% DSP / 50% LUT | Hybrid | 0.5 | blended | 1.0 | blended | Trade-off |

---

## 4. Shift-and-Add Formulation (INT8 × INT8)

For \( A \times B \) (both INT8):

\[
A \times B = \sum_{i=0}^{7} (A \cdot B[i]) \cdot 2^i
\]

- \( B[i] \) = bit \( i \) of \( B \).
- \( A \cdot 2^i \) = **left shift** of \( A \) by \( i \) (no multiplier needed).
- Partial products are **shifts**; combine with **additions** (adder tree or serial accumulator).

**Hardware sketch:**

```
B[0] → A×1   ─┐
B[1] → A×2   ─┼→  ADD TREE  → Product (INT16)
...          ─┤
B[7] → A×128 ─┘
```

**Implementation variants:**

| Variant | Cycles/Mult | Area (LUTs) | Use case |
|---------|-------------|-------------|----------|
| **Parallel** | 1 (pipelined) | ~64 | Max throughput, more area |
| **Serial** | 8 | ~24 | Min area, 8× less throughput per unit |
| **Booth radix-4** | 4 | ~40 | Balanced |
| **Pipelined 2-stage** | 1 (after latency) | ~48 | **Recommended for ASIC / LUT-only** |

---

## 5. What Worked

- **Clear separation** of DSP vs LUT vs hybrid; easy to plug into Phase 4’s MAC array and cycle model.
- **FPS estimation** for ZedBoard: LUT-pipelined and LUT-parallel give viable FPS with **0 DSPs**; hybrid improves FPS when some DSPs are available.
- **ASIC rationale:** LUT-based (shift-and-add) is portable, no DSP blocks; pipelined parallel is a good compromise (throughput 1 MAC/cycle, regular structure).
- **Script** runs end-to-end and writes `phase4b_multiplier_results.json`.

---

## 6. What Did Not Work / Limitations

- **LUT/FF numbers** are **estimates** (not from synthesis); real FPGA/ASIC numbers may differ.
- **Frequency** is approximate; critical path depends on place-and-route (adder tree depth, pipeline stages).
- **Serial shift-add** has low throughput (0.125); need many parallel units to match one parallel multiplier, so total LUTs can be similar for same MAC/cycle.
- **Hybrid** model in the script is a simple linear blend; real design would allocate specific MACs to DSP vs LUT.

---

## 7. Recommendations

**For FPGA with enough DSPs (e.g. Zynq UltraScale+):**  
- Prefer **DSP-based** (or packed 2×) for best FPS and lower LUT usage.

**For FPGA with few DSPs (e.g. ZedBoard) or ASIC:**  
- Use **LUT-based multiplier** (shift-and-add).  
- **Recommended variant:** **LUT Pipelined (2-stage)** – 1 MAC/cycle after latency, ~48 LUTs per MAC, predictable timing.  
- Enables **ASIC portability** and scaling beyond DSP count.

**For Phase 5 (hardware architecture):**  
- Define **two multiplier back-ends**:  
  - **DSP back-end:** 1 INT8×INT8 per DSP (or packed 2×).  
  - **LUT back-end:** shift-and-add (pipelined parallel) for INT8×INT8.  
- Same MAC array and control; only the multiply unit and resource counts differ.

---

## 8. Deliverables

| Deliverable | Location |
|-------------|----------|
| Multiplier options and MAC array analysis | `phase4b_multiplier_exploration.py` |
| Numerical results (FPS, resources) | `phase4b_multiplier_results.json` (after run) |
| Phase 4b write-up | This document |

---

## 9. Next Steps (Phase 5)

1. **Hardware architecture definition:** datapath, control, memory blocks.
2. **Map Python primitives to hardware blocks**, including:
   - **MAC:** either DSP multiplier or **shift-and-add multiplier** (LUT) + accumulator.
3. **Interfaces:** internal (buffers, MAC array) and external (future AXI).
4. **Block diagram** showing optional multiplier back-ends (DSP vs LUT).

**Phase 4b status: COMPLETE**  
**Selected for shift-multiplier path:** LUT Pipelined (2-stage) – shifts + add tree, 1 MAC/cycle, ASIC-portable.
