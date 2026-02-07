# PHASE 2: DPU Primitives Identification

## Overview

This document defines the minimal set of hardware primitives required to accelerate YOLOv3-tiny and YOLOv4-tiny on a custom DPU.

**Models Analyzed:** YOLOv3-tiny, YOLOv4-tiny
**Common Activation:** LeakyReLU (alpha=0.1) - Hardware Friendly

---

## 1. Model Comparison

| Metric | YOLOv3-tiny | YOLOv4-tiny |
|--------|-------------|-------------|
| Architecture | Linear | CSP (skip paths) |
| Conv layers | 13 | 21 |
| MaxPool layers | 6 | 3 |
| Route layers | 2 | 11 |
| 3x3 kernels | 10 | 14 |
| 1x1 kernels | 3 | 7 |
| Max channels | 1024 | 512 |
| Parameters | 8.7M | 6.06M |
| FLOPs | 5.6G | 6.9G |
| mAP@0.5 | 33.1% | 40.2% |

---

## 2. DPU Primitives

### 2.1 CRITICAL: MAC (Multiply-Accumulate)

The fundamental compute unit for all convolutions.

```
+-------+     +-------+
|  A    |---->|       |
| INT8  |     |  MAC  |----> P (INT32)
+-------+     |       |
              | P=P+AxB|
+-------+     |       |
|  B    |---->|       |
| INT8  |     +-------+
+-------+
```

| Specification | Value |
|--------------|-------|
| Input A | INT8 (weight) |
| Input B | INT8 (activation) |
| Output P | INT32 (accumulated) |
| Operation | P_new = P_old + A * B |
| Hardware | 8x8 multiplier + 32-bit adder |
| Latency | 1 cycle (pipelined) |

---

### 2.2 HIGH: Conv 3x3

Main feature extraction operation.

| Specification | Value |
|--------------|-------|
| Kernel size | 3x3 |
| Stride | 1 or 2 |
| Padding | 1 (same) |
| Max C_in | 512 (v4) / 1024 (v3) |
| Max C_out | 512 |
| MACs per pixel | 9 x C_in x C_out |

**Hardware Requirements:**
- MAC array (systolic or output-stationary)
- Line buffer (2 lines for 3x3 window)
- Weight buffer

---

### 2.3 HIGH: Conv 1x1 (Pointwise)

Channel mixing and dimension reduction.

| Specification | Value |
|--------------|-------|
| Kernel size | 1x1 |
| Stride | 1 |
| Padding | 0 |
| Max C_in | 1024 |
| Max C_out | 512 |
| MACs per pixel | C_in x C_out |

**Hardware Requirements:**
- Same MAC array as 3x3
- No line buffer needed

---

### 2.4 HIGH: LeakyReLU

Non-linear activation function.

```
y = x       if x > 0
y = 0.1*x   if x <= 0
```

| Specification | Value |
|--------------|-------|
| Alpha | 0.1 |
| Approximation | 1/8 = 0.125 (shift right 3) |
| Hardware | 1 comparator + 1 shifter + 1 MUX |
| Latency | 1 cycle |

**Hardware Implementation:**
```verilog
// LeakyReLU approximation (alpha = 1/8)
assign y = (x > 0) ? x : (x >>> 3);
```

**Comparison with SiLU (NOT USED):**
| Aspect | LeakyReLU | SiLU |
|--------|-----------|------|
| Formula | max(0.1x, x) | x * sigmoid(x) |
| Hardware | Compare + MUX | Sigmoid LUT + Mult |
| Latency | 1 cycle | 10+ cycles |
| Area | Minimal | Large |

---

### 2.5 HIGH: Requantization

Convert INT32 accumulator back to INT8.

```
y = clamp(round(acc * scale) + zero_point, -128, 127)
```

| Specification | Value |
|--------------|-------|
| Input | INT32 accumulator |
| Output | INT8 activation |
| Scale | Per-channel (INT32 or fixed-point) |
| Hardware | Multiplier + Adder + Saturator |
| Latency | 1-2 cycles |

---

### 2.6 MEDIUM: MaxPool 2x2

Spatial downsampling.

```
y = max(x[0,0], x[0,1], x[1,0], x[1,1])
```

| Specification | Value |
|--------------|-------|
| Kernel | 2x2 |
| Stride | 2 |
| Hardware | 3 comparators (tree) |
| Latency | 1 cycle |

---

### 2.7 MEDIUM: Route/Concat

Feature map manipulation.

| Operation | Description |
|-----------|-------------|
| Concat | Combine feature maps along channel dim |
| Split | Take subset of channels |

| Specification | Value |
|--------------|-------|
| Compute | NONE (memory only) |
| Hardware | Address generator |
| YOLOv3-tiny | 2 route operations |
| YOLOv4-tiny | 11 route operations |

---

### 2.8 LOW: Upsample 2x

Nearest neighbor upscaling.

| Specification | Value |
|--------------|-------|
| Scale | 2x |
| Method | Nearest neighbor |
| Compute | NONE (pixel duplication) |
| Hardware | Address generator |

---

### 2.9 HIGH: Bias Add (Fused BatchNorm)

Per-channel bias addition.

| Specification | Value |
|--------------|-------|
| Operation | y = acc + bias[channel] |
| Hardware | 32-bit adder |
| Note | Fused with MAC pipeline |

**BatchNorm Folding (at compile time):**
```
W_folded = W * gamma / sqrt(var + eps)
b_folded = beta - mean * gamma / sqrt(var + eps)
```

---

## 3. Data Types

| Type | Bits | Signed | Usage |
|------|------|--------|-------|
| Weights | 8 | Yes | Conv kernels |
| Activations | 8 | Yes | Feature maps |
| Accumulator | 32 | Yes | Partial sums |
| Bias | 32 | Yes | Fused BN bias |
| Scale | 32 | - | Requant scale |

**Overflow Analysis:**
- Max MACs per output = 1024 x 3 x 3 = 9216
- Max product = 127 x 127 = 16,129
- Max accumulation = 16,129 x 9,216 = 148,660,224
- INT32 max = 2,147,483,647 (safe margin)

---

## 4. What is NOT Needed

| Item | Reason |
|------|--------|
| SiLU / Swish | Requires sigmoid (exponential) |
| Mish | Requires softplus and tanh |
| Sigmoid | Not in inference path |
| Softmax | Post-processing only (CPU) |
| Depthwise Conv | Not used in v3/v4-tiny |
| Dilated Conv | Not used |
| Transposed Conv | Not used |
| 5x5, 7x7 Conv | Not used |
| Floating Point | INT8 quantized inference |

---

## 5. Execution Model

### Convolution Pipeline

```
+----------+   +-------+   +--------+   +---------+   +---------+
| Weight   |-->|  MAC  |-->| Bias   |-->| Leaky   |-->| Requant |-->OUT
| Buffer   |   | Array |   | Add    |   | ReLU    |   |         |
+----------+   +-------+   +--------+   +---------+   +---------+
                  ^
                  |
            +----------+
            | Line     |
            | Buffer   |
            +----------+
                  ^
                  |
            +----------+
            | Input    |
            | SRAM     |
            +----------+
```

### Per-Output Computation

```python
for c_out in range(C_out):
    acc = 0
    for c_in in range(C_in):
        for kh in range(3):
            for kw in range(3):
                acc += weight[c_out][c_in][kh][kw] * input[c_in][h+kh][w+kw]
    acc += bias[c_out]
    out = leaky_relu(acc)
    output[c_out][h][w] = requantize(out)
```

---

## 6. Primitives Summary

| # | Primitive | Priority | Hardware | Compute |
|---|-----------|----------|----------|---------|
| 1 | MAC | CRITICAL | 8x8 mult + 32b add | 1 MAC/cyc |
| 2 | Conv 3x3 | HIGH | MAC array + buffers | 9xCin MACs |
| 3 | Conv 1x1 | HIGH | MAC array | Cin MACs |
| 4 | LeakyReLU | HIGH | cmp + shift + mux | 1 cycle |
| 5 | Requant | HIGH | mult + add + clamp | 1-2 cyc |
| 6 | MaxPool | MEDIUM | 3 comparators | 1 cycle |
| 7 | Route | MEDIUM | Memory only | 0 MACs |
| 8 | Upsample | LOW | Memory only | 0 MACs |
| 9 | Bias Add | HIGH | 32b adder | Fused |

---

## 7. Model Recommendation

### For Simplest DPU: YOLOv3-tiny
- Linear dataflow (no complex routing)
- Only 2 route operations
- Simpler control logic
- Good for first implementation

### For Best Accuracy: YOLOv4-tiny
- +7% mAP improvement
- Complex routing (11 routes)
- CSP architecture with skip paths
- Requires flexible memory addressing

### Common Requirements
- Conv 3x3 and 1x1
- LeakyReLU (alpha=0.1)
- MaxPool 2x2
- INT8 weights and activations
- INT32 accumulation

---

## Files Generated

| File | Description |
|------|-------------|
| `phase2_dpu_primitives.py` | Analysis script |
| `phase2_primitives_results.json` | Machine-readable results |
| `docs/PHASE2_DPU_PRIMITIVES.md` | This document |
| `yolov3-tiny.cfg` | YOLOv3-tiny config |

---

**Phase 2 Status: COMPLETE**
**Date:** 2026-02-07
**Ready for Phase 3: Python Functional DPU Model**
