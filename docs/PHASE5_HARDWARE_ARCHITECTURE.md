# PHASE 5: Hardware Architecture Definition

## Overview

This phase translates the **selected Python DPU model** (Phases 1–4 and 4b) into a **hardware architecture**: system-level blocks, datapath, control, memory, interfaces, and mapping from Python primitives to hardware.

**Reference configuration:** MEDIUM (32×32 MAC array) from Phase 4.  
**Multiplier option:** DSP-based or LUT-based (shift-and-add) from Phase 4b; same datapath, different multiply unit.

---

## 1. System-Level Definition

### 1.1 Top-Level Blocks

| Block | Role |
|-------|------|
| **Datapath** | MAC array, post-processing (LeakyReLU, Requant), bias add |
| **Control** | FSM: layer/tile sequencing, address generation, handshakes |
| **Memory** | Weight buffer, input buffer, line buffer, output buffer |

### 1.2 Datapath

- **Inputs:** INT8 weights (from Weight Buffer), INT8 activations (from Input/Line Buffer).
- **Compute:** Array of MACs (rows = output channels, cols = input channels); each MAC: `ACC += W * A` (INT8×INT8 → INT32).
- **Post-processing (per output channel):** Bias add (INT32) → LeakyReLU (INT32) → Requantize (INT32→INT8).
- **Output:** INT8 activations to Output Buffer.

**Multiplier back-end (selectable):**

- **DSP:** One INT8×INT8 multiplier per MAC (e.g. DSP48E1); 1 MAC per cycle per unit.
- **LUT (shift-and-add):** Partial products = shifts of A by bit i of B; adder tree; 2-stage pipeline (Phase 4b LUT Pipelined). Same functional behavior as DSP path.

### 1.3 Control

- **Layer FSM:** Load layer parameters (C_in, C_out, H, W, stride), then run tile loop.
- **Tile FSM:** For each output-channel tile (e.g. 32 channels), load weights; for each input-channel tile, load input slice; for each kernel position (9 for 3×3), run MAC array; then post-process and store output tile.
- **Address generators:** Weight address (per tile), input/line buffer read, output write.
- **Handshakes:** Start/done per layer; optional back-pressure for memory.

### 1.4 Memory Blocks

| Memory | Size (MEDIUM) | Role |
|--------|----------------|------|
| **Weight Buffer** | 128 KB | On-chip weights for current tile; double-buffer for prefetch |
| **Input Buffer** | 64 KB | Current input tile / lines; ping-pong with Line Buffer |
| **Line Buffer** | ~few lines | 3×3 sliding window (2–3 lines of width × C_in) |
| **Output Buffer** | 64 KB | Current output tile; double-buffer for write-back |

All buffers are **internal** to the DPU; they are filled/emptied via the **external memory interface** (future AXI).

---

## 2. Block Diagram

```
+================================================================================+
|                           DPU (Top Level)                                      |
+================================================================================+
|                                                                                |
|  +------------------+     +------------------+     +------------------+       |
|  |  WEIGHT BUFFER   |     |  INPUT BUFFER    |     |  LINE BUFFER      |       |
|  |  (128 KB)        |     |  (64 KB)         |     |  (2-3 lines)      |       |
|  |  double-buffer   |     |  ping-pong       |     |  sliding 3x3      |       |
|  +--------+--------+     +--------+--------+     +--------+--------+       |
|           |                         |                         |               |
|           v                         v                         v               |
|  +--------+--------+     +=========================================+       |
|  |  Weights (INT8)  |     |           MAC ARRAY (32 x 32)           |       |
|  |  to MAC array   |---->|  Row = output ch, Col = input ch       |       |
|  +-----------------+     |  Each cell: MULT (DSP or LUT) + ACC     |       |
|                          |  INT8 x INT8 -> INT32 accumulation      |       |
|                          +====================+====================+       |
|                                               |                             |
|                                               v                             |
|                          +------------------+------------------+           |
|                          |  ACC (INT32)     |  Bias Add (INT32) |           |
|                          +--------+---------+--------+----------+           |
|                                   v                   v                    |
|                          +--------+---------+  +------+--------+             |
|                          |  LeakyReLU       |  |  Requantize   |             |
|                          |  (x>0?x:x>>3)    |  |  INT32->INT8 |             |
|                          +--------+---------+  +------+--------+             |
|                                   v                   v                    |
|                          +=========================================+       |
|                          |       OUTPUT BUFFER (64 KB)              |       |
|                          |       double-buffer                     |       |
|                          +=========================================+       |
|                                                                                |
|  +------------------+                                                          |
|  |  CONTROL FSM     |  Layer / tile / address gen / handshakes                 |
|  +------------------+                                                          |
|                                                                                |
+================================================================================+
|                     EXTERNAL INTERFACE (future AXI)                             |
|  - Read:  weights, input feature maps                                          |
|  - Write: output feature maps                                                   |
|  - Control: start, layer params, done, error                                  |
+================================================================================+
```

### 2.1 Multiplier Block (DSP vs LUT)

Each MAC cell contains one multiplier. Two implementations:

```
OPTION A: DSP                          OPTION B: LUT (shift-and-add)
+-------------------+                   +-------------------+
|  A (INT8)  B(INT8)|                   |  A (INT8)  B(INT8)|
|       v     v     |                   |       v     v     |
|  +----+-----+----+                   |  Partial products  |
|  |  DSP48   |    |  --> P (INT16)    |  (A<<0..A<<7)      |
|  | 8x8 mult |    |                   |  + Adder tree      |
|  +----------+----+                   |  (2-stage pipe)     |
|       P + ACC -> ACC                 |  --> P (INT16)     |
+-------------------+                   |  P + ACC -> ACC    |
                                        +-------------------+
```

Same interface: (A, B, ACC) → (ACC_new). Only the internal implementation of the multiply differs.

---

## 3. Internal Interfaces

### 3.1 Weight Buffer → MAC Array

- **Data:** INT8 weights; width = MAC_cols (e.g. 32) × 1 byte; one weight per MAC column per cycle (broadcast along row if needed).
- **Addressing:** Linear by (output_channel_tile, input_channel_tile, ky, kx).
- **Control:** Valid signal; address from Control.

### 3.2 Input / Line Buffer → MAC Array

- **Data:** INT8 activations; width = MAC_cols × 1 byte (one activation per column).
- **Line buffer:** Delivers 3×3 window per position; 9 values × MAC_cols (or time-multiplexed).
- **Control:** Read enable, address from Control.

### 3.3 MAC Array → Accumulator → Post-Processing

- **Data:** INT32 partial sum per row (output channel); one value per MAC row per cycle (after accumulation over kernel and input channels).
- **Pipeline:** ACC → Bias Add → LeakyReLU → Requantize → INT8.
- **Control:** Valid per stage; post-processing is fixed function (no config per layer beyond scale/zero_point).

### 3.4 Post-Processing → Output Buffer

- **Data:** INT8; width = MAC_rows × 1 byte.
- **Addressing:** Linear by (output_channel_tile, H_out, W_out).
- **Control:** Write enable, address from Control.

### 3.5 Control → All Blocks

- **Signals:** Start layer, start tile, kernel index, addresses, read/write enables, done.
- **No data path:** Control is FSM + address generators only.

---

## 4. External Interface (Future AXI)

Described at a **definition level** only (no RTL in this phase).

### 4.1 AXI4-Stream or AXI4-MM (to be decided in implementation)

- **Read:**  
  - Weights (per layer or per tile).  
  - Input feature map (per tile or full layer).
- **Write:**  
  - Output feature map (per tile or full layer).
- **Control/status:**  
  - Registers or lightweight AXI-Lite: start, layer id, dimensions (C_in, C_out, H, W, stride), done, error.

### 4.2 Requirements

- **Bandwidth:** Sufficient to feed Weight + Input buffers and drain Output buffer without stalling the DPU (Phase 4 assumed overlap of compute and memory).
- **Burst:** Prefer burst reads/writes for efficiency.

---

## 5. Mapping: Python Primitives → Hardware Blocks

| Python Primitive | Hardware Block | Notes |
|------------------|----------------|------|
| `mac(weight, activation, accumulator)` | One MAC cell: **Multiplier (DSP or LUT)** + 32-bit adder + ACC register | Phase 4b: multiplier = DSP or shift-and-add |
| `conv2d_3x3(...)` | **MAC array** + **Line Buffer** (3×3 window) + **Weight Buffer** + Control (tile loop, kernel 9 positions) | Same array as 1×1; line buffer feeds 9 inputs per position |
| `conv2d_1x1(...)` | **MAC array** + **Weight Buffer** + **Input Buffer** (no line buffer) | Subset of 3×3 flow; K=1, no sliding window |
| `leaky_relu_hardware(x)` | **LeakyReLU block:** comparator (x>0), shifter (x>>3), MUX | One per output channel; INT32→INT32 |
| `requantize(acc, scale, zero_point)` | **Requantize block:** multiply-add (scale, zero_point), round, clamp to [-128,127] | One per output channel; INT32→INT8 |
| Bias add (fused BN) | **Bias adder:** ACC + bias[channel] (INT32) | In pipeline between MAC and LeakyReLU |
| MaxPool 2×2 | Comparator tree (max of 4); or deferred to software / later phase | Not in first RTL slice if scope reduced |
| Route/Concat | Address generator + buffer indexing | No extra compute block |

### 5.1 Data Types (Aligned with Phase 2)

| Signal | Type | Width |
|--------|------|-------|
| Weight | INT8 | 8 |
| Activation (input) | INT8 | 8 |
| Accumulator | INT32 | 32 |
| Bias | INT32 | 32 |
| Post-LeakyReLU | INT32 | 32 |
| Activation (output) | INT8 | 8 |

---

## 6. Configuration Summary (MEDIUM + Phase 4b Option)

| Parameter | Value |
|-----------|-------|
| MAC array | 32 × 32 |
| Weight Buffer | 128 KB |
| Input Buffer | 64 KB |
| Output Buffer | 64 KB |
| Multiplier | DSP (1 per MAC) **or** LUT Pipelined (shift-and-add) |
| Post-processing | Bias add, LeakyReLU, Requantize (per channel) |
| Dataflow | Output-stationary |
| Target | YOLOv4-tiny, 53.7 FPS (DSP) / 24.26 FPS (LUT on ZedBoard) |

---

## 7. Deliverables

| Deliverable | Location |
|-------------|----------|
| Hardware architecture description | This document (Sections 1–2) |
| Block diagram | This document (Section 2) |
| Internal interfaces | This document (Section 3) |
| External interface (future AXI) | This document (Section 4) |
| Mapping Python primitives → HW | This document (Section 5) |

---

## 8. Next Steps (Phase 6 — RTL Implementation)

1. Clone/inspect Verilog simulation repo (e.g. `verilog-sim-py`).
2. Implement **primitives** in Verilog: MAC (with DSP or LUT multiplier), LeakyReLU, Requantize.
3. Implement **subsystems:** MAC array, buffers (or wrappers), control FSM.
4. Integrate into **full DPU** with internal interfaces as above.
5. Clean module hierarchy and naming for Phase 7 verification.

**Phase 5 Status: COMPLETE**
