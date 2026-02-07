# PHASE 3: Python Functional DPU Model

## Overview

This phase implements each DPU primitive as a Python function, creating a **golden reference model** that will be used to verify the RTL implementation.

All operations use **INT8/INT32 arithmetic** to match hardware behavior exactly.

---

## 1. Primitives Implemented

### 1.1 MAC (Multiply-Accumulate)

```python
def mac(weight: np.int8, activation: np.int8, accumulator: np.int32) -> np.int32:
    product = np.int32(weight) * np.int32(activation)  # 8x8 -> 16 bits
    result = accumulator + product                      # 32-bit accumulation
    return np.int32(result)
```

**Test Results:**
```
MAC(  10,   20,      0) =      200  [PASS]
MAC( -10,   20,      0) =     -200  [PASS]
MAC( 127,  127,      0) =    16129  [PASS]
MAC(-128,  127,      0) =   -16256  [PASS]
MAC(  50,   50,   1000) =     3500  [PASS]
```

---

### 1.2 Convolution 3x3

```python
def conv2d_3x3(input_fm, weights, bias, stride=1, padding=1):
    # For each output position
    for c_out in range(C_out):
        for h_out in range(H_out):
            for w_out in range(W_out):
                # Extract 3x3 window
                window = input_padded[:, h_in:h_in+3, w_in:w_in+3]
                # MAC array: dot product
                acc = mac_array(weights[c_out], window)
                # Add bias (fused BatchNorm)
                output[c_out, h_out, w_out] = acc + bias[c_out]
    return output  # INT32
```

**Key Features:**
- Supports stride 1 and 2
- Padding = 1 for same output size
- INT8 inputs, INT32 output

---

### 1.3 Convolution 1x1

```python
def conv2d_1x1(input_fm, weights, bias):
    # Matrix multiplication (more efficient than 3x3 loop)
    w = weights.reshape(C_out, C_in).astype(np.int32)
    x = input_fm.reshape(C_in, H * W).astype(np.int32)
    output = np.dot(w, x) + bias.reshape(C_out, 1)
    return output  # INT32
```

---

### 1.4 LeakyReLU (Hardware Version)

```python
def leaky_relu_hardware(x: np.ndarray) -> np.ndarray:
    # Arithmetic right shift by 3 (equivalent to divide by 8)
    negative_scaled = np.right_shift(x.astype(np.int32), 3)
    # MUX: select based on sign
    return np.where(x > 0, x, negative_scaled)
```

**Test Results:**
```
Input:    [-80 -40  -8   0   8  40  80]
Output:   [-10  -5  -1   0   8  40  80]
Expected: [-10  -5  -1   0   8  40  80]
Status:   PASS
```

**Hardware Mapping:**
- Alpha = 0.1 approximated as 1/8 = 0.125
- Implementation: `x >> 3` (right shift)
- Error: 25% on negative values (acceptable for inference)

---

### 1.5 Requantization

```python
def requantize(acc, scale, zero_point=0):
    scaled = np.round(acc.astype(np.float64) * scale + zero_point)
    output = np.clip(scaled, -128, 127)
    return output.astype(np.int8)
```

**Fixed-Point Version (for hardware):**
```python
def requantize_fixed_point(acc, multiplier, shift):
    product = acc.astype(np.int64) * np.int64(multiplier)
    shifted = product >> shift
    return np.clip(shifted, -128, 127).astype(np.int8)
```

---

### 1.6 MaxPool 2x2

```python
def maxpool_2x2(input_fm, stride=2):
    for c in range(C):
        for h in range(H_out):
            for w in range(W_out):
                window = input_fm[c, h*2:h*2+2, w*2:w*2+2]
                output[c, h, w] = np.max(window)
    return output
```

**Test Results:**
```
Input:
[[ 1  2  3  4]
 [ 5  6  7  8]
 [ 9 10 11 12]
 [13 14 15 16]]

Output:
[[ 6  8]
 [14 16]]

Status: PASS
```

---

### 1.7 Route/Concat

```python
def route_concat(fm_list):
    return np.concatenate(fm_list, axis=0)  # Along channel dimension

def route_split(input_fm, groups=2, group_id=0):
    channels_per_group = C // groups
    start = group_id * channels_per_group
    return input_fm[start:start+channels_per_group]
```

---

### 1.8 Upsample 2x

```python
def upsample_2x(input_fm):
    # Duplicate each pixel to 2x2 block
    for c in range(C):
        for h in range(H):
            for w in range(W):
                output[c, h*2:h*2+2, w*2:w*2+2] = input_fm[c, h, w]
    return output
```

---

## 2. Complete Layer Pipeline

The most common operation in YOLO: **Conv + BN + LeakyReLU + Requant**

```python
def conv_bn_leaky(input_fm, weights, bias, scale, stride=1, kernel_size=3):
    # Step 1: Convolution (INT8 -> INT32)
    if kernel_size == 3:
        conv_out = conv2d_3x3(input_fm, weights, bias, stride=stride)
    else:
        conv_out = conv2d_1x1(input_fm, weights, bias)

    # Step 2: LeakyReLU (INT32 -> INT32)
    relu_out = leaky_relu_hardware(conv_out)

    # Step 3: Requantize (INT32 -> INT8)
    output = requantize(relu_out, scale)

    return output
```

**Data Flow:**
```
INT8 input -> Conv -> INT32 -> LeakyReLU -> INT32 -> Requant -> INT8 output
```

---

## 3. YOLOv4-tiny Simulation

Simulated the first 10 layers of YOLOv4-tiny:

| Layer | Operation | Input Shape | Output Shape | Status |
|-------|-----------|-------------|--------------|--------|
| 0 | Conv 3x3 s2 | (3, 416, 416) | (32, 208, 208) | PASS |
| 1 | Conv 3x3 s2 | (32, 208, 208) | (64, 104, 104) | PASS |
| 2 | Conv 3x3 | (64, 104, 104) | (64, 104, 104) | PASS |
| 3 | Route split | (64, 104, 104) | (32, 104, 104) | PASS |
| 4 | Conv 3x3 | (32, 104, 104) | (32, 104, 104) | PASS |
| 5 | Conv 3x3 | (32, 104, 104) | (32, 104, 104) | PASS |
| 6 | Route concat | (32+32, 104, 104) | (64, 104, 104) | PASS |
| 7 | Conv 1x1 | (64, 104, 104) | (64, 104, 104) | PASS |
| 8 | Route concat | (64+64, 104, 104) | (128, 104, 104) | PASS |
| 9 | MaxPool 2x2 | (128, 104, 104) | (128, 52, 52) | PASS |

**All layers execute correctly with INT8/INT32 arithmetic.**

---

## 4. Validation Summary

| Primitive | Status |
|-----------|--------|
| MAC | PASS |
| LeakyReLU | PASS |
| Conv3x3 | PASS |
| Conv1x1 | PASS |
| MaxPool | PASS |
| Requantize | PASS |
| Full Layer | PASS |
| YOLO Simulation | PASS |

---

## 5. Key Implementation Details

### 5.1 INT8 Range
```python
INT8_MIN = -128
INT8_MAX = 127
```

### 5.2 Overflow Prevention
```
Max MACs per output = C_in × K × K = 1024 × 3 × 3 = 9,216
Max product = 127 × 127 = 16,129
Max accumulation = 16,129 × 9,216 = 148,660,224
INT32 max = 2,147,483,647 → Safe!
```

### 5.3 LeakyReLU Approximation
```
Exact:   alpha = 0.1
Approx:  alpha = 1/8 = 0.125 (shift right 3)
Error:   25% on negative values (acceptable)
```

---

## 6. Files Generated

| File | Description |
|------|-------------|
| `phase3_dpu_functional_model.py` | Python functional model |
| `phase3_validation_results.json` | Test results |
| `docs/PHASE3_FUNCTIONAL_MODEL.md` | This document |

---

## 7. Usage as Golden Reference

This model will be used in Phase 7 for RTL verification:

```python
# Generate test vectors
input_data = generate_random_input()
expected_output = conv_bn_leaky(input_data, weights, bias, scale)

# Compare with RTL simulation
rtl_output = run_rtl_simulation(input_data)
assert np.array_equal(expected_output, rtl_output)
```

---

**Phase 3 Status: COMPLETE**
**All primitives validated with INT8/INT32 arithmetic**
**Ready for Phase 4: Architecture Exploration**
