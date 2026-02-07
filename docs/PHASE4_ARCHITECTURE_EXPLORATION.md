# PHASE 4: Software Benchmarking & Architecture Exploration

## Overview

This phase explores different DPU architectures to find the optimal configuration for YOLOv4-tiny, balancing performance, resources, and efficiency.

---

## 1. Network Requirements

### YOLOv4-tiny Analysis

| Metric | Value |
|--------|-------|
| Conv Layers | 21 |
| Total MACs | 3.45 GMACs |
| Total Weights | 5.9 MB |
| Max Channels | 512 |
| Input Size | 416 x 416 |

---

## 2. Configurations Explored

### 2.1 TINY (8x8)

| Parameter | Value |
|-----------|-------|
| MAC Array | 8 x 8 = 64 MACs/cycle |
| Frequency | 100 MHz |
| Peak TOPS | 0.006 |
| Weight Buffer | 32 KB |
| DSPs | 64 |

**Performance:** 1.8 FPS (too slow)

---

### 2.2 SMALL (16x16)

| Parameter | Value |
|-----------|-------|
| MAC Array | 16 x 16 = 256 MACs/cycle |
| Frequency | 150 MHz |
| Peak TOPS | 0.038 |
| Weight Buffer | 64 KB |
| DSPs | 256 |

**Performance:** 10.6 FPS, 96% utilization

**Best for:** First implementation, debugging, small FPGAs

---

### 2.3 MEDIUM (32x32) - RECOMMENDED

| Parameter | Value |
|-----------|-------|
| MAC Array | 32 x 32 = 1024 MACs/cycle |
| Frequency | 200 MHz |
| Peak TOPS | 0.205 |
| Weight Buffer | 128 KB |
| DSPs | 1024 |

**Performance:** 53.7 FPS, 95.6% utilization

**Best for:** Real-time inference, mid-range FPGAs

---

### 2.4 LARGE (64x64)

| Parameter | Value |
|-----------|-------|
| MAC Array | 64 x 64 = 4096 MACs/cycle |
| Frequency | 250 MHz |
| Peak TOPS | 1.024 |
| Weight Buffer | 256 KB |
| DSPs | 4096 |

**Performance:** 173.6 FPS, 81.6% utilization

**Best for:** High-performance applications

---

### 2.5 XLARGE (128x64)

| Parameter | Value |
|-----------|-------|
| MAC Array | 128 x 64 = 8192 MACs/cycle |
| Frequency | 300 MHz |
| Peak TOPS | 2.458 |
| Weight Buffer | 512 KB |
| DSPs | 8192 |

**Performance:** 260 FPS, 65.9% utilization

**Note:** Lower utilization due to tiling overhead

---

## 3. Comparison Table

| Config | MAC Array | Freq | Peak TOPS | FPS | Util | DSPs | BRAM |
|--------|-----------|------|-----------|-----|------|------|------|
| TINY | 8x8 | 100 | 0.006 | 1.8 | 97% | 64 | 64KB |
| SMALL | 16x16 | 150 | 0.038 | 10.6 | 96% | 256 | 128KB |
| **MEDIUM** | **32x32** | **200** | **0.205** | **53.7** | **96%** | **1024** | **256KB** |
| LARGE | 64x64 | 250 | 1.024 | 173.6 | 82% | 4096 | 512KB |
| XLARGE | 128x64 | 300 | 2.458 | 260.0 | 66% | 8192 | 1024KB |

---

## 4. Efficiency Analysis

| Config | FPS/DSP | FPS/TOPS | Efficiency Score |
|--------|---------|----------|------------------|
| TINY | 0.028 | 284.4 | Low |
| SMALL | 0.041 | 276.5 | Good |
| **MEDIUM** | **0.052** | **262.1** | **Best** |
| LARGE | 0.042 | 169.6 | Good |
| XLARGE | 0.032 | 105.8 | Moderate |

**Observation:** MEDIUM has best FPS per DSP ratio.

---

## 5. Layer-by-Layer Analysis (MEDIUM)

| Layer | Shape | MACs | Cycles | Utilization |
|-------|-------|------|--------|-------------|
| conv0 | 3->32 3x3 | 37M | 389K | 95% |
| conv1 | 32->64 3x3 | 199M | 389K | 100% |
| conv2 | 64->64 3x3 | 398M | 778K | 100% |
| conv10 | 128->128 3x3 | 398M | 778K | 100% |
| conv18 | 256->256 3x3 | 398M | 778K | 100% |
| conv26 | 512->512 3x3 | 398M | 778K | 100% |
| conv35 | 384->256 3x3 | 598M | 1.17M | 100% |

**Total:** 3.45 GMACs, 3.7M cycles, 18.6ms at 200MHz = **53.7 FPS**

---

## 6. Selected Architecture: MEDIUM

### Specifications

```
+--------------------------------------------------+
|              DPU ARCHITECTURE (MEDIUM)           |
+--------------------------------------------------+
|  MAC Array:        32 x 32 = 1024 MACs/cycle     |
|  Frequency:        200 MHz                        |
|  Peak Performance: 0.205 TOPS                     |
+--------------------------------------------------+
|  BUFFERS:                                         |
|    Weight Buffer:  128 KB                         |
|    Input Buffer:   64 KB                          |
|    Output Buffer:  64 KB                          |
+--------------------------------------------------+
|  RESOURCES (estimated):                           |
|    DSPs:           1024                           |
|    LUTs:           ~125K                          |
|    FFs:            ~125K                          |
|    BRAM:           256 KB                         |
+--------------------------------------------------+
|  PERFORMANCE:                                     |
|    YOLOv4-tiny:    53.7 FPS                       |
|    Utilization:    95.6%                          |
|    Inference time: 18.6 ms                        |
+--------------------------------------------------+
```

### Why MEDIUM?

1. **Real-time capable:** >30 FPS achieved
2. **High utilization:** 95.6% of peak performance
3. **Resource efficient:** Best FPS/DSP ratio
4. **Fits mid-range FPGAs:** ZU3EG, ZU4EV, etc.
5. **Power efficient:** ~1024 DSPs at 200 MHz

---

## 7. Architecture Recommendations

### MAC Array Design

```
                     32 Output Channels
              +--+--+--+--+--+--+--+--+
              |  |  |  |  |  |  |  |  |
          +---+--+--+--+--+--+--+--+--+---+
          |   MAC MAC MAC MAC MAC MAC MAC |
    32    |   MAC MAC MAC MAC MAC MAC MAC |
  Input   |   MAC MAC MAC MAC MAC MAC MAC |
 Channels |   MAC MAC MAC MAC MAC MAC MAC |
          |   ...                     ... |
          |   MAC MAC MAC MAC MAC MAC MAC |
          +-------------------------------+
                         |
                         v
                   32 Accumulators (INT32)
                         |
                         v
                   32 LeakyReLU units
                         |
                         v
                   32 Requantize units
                         |
                         v
                   32 INT8 outputs
```

### Dataflow: Output-Stationary

```
For each output channel tile (32 at a time):
    Load weights for 32 output channels
    For each input channel tile (32 at a time):
        For each kernel position (9 for 3x3):
            For each output position:
                Accumulate: ACC += W * A
    Apply LeakyReLU
    Requantize to INT8
    Store output
```

### Buffer Strategy

```
+-------------------+
|   Weight Buffer   |  128 KB
|   (double buffer) |  Prefetch next layer while computing
+-------------------+
         |
         v
+-------------------+     +-------------------+
|   Input Buffer    | <-- |   Line Buffer     |
|   (ping-pong)     |     |   (2-3 lines)     |
+-------------------+     +-------------------+
         |
         v
+-------------------+
|   Output Buffer   |  64 KB
|   (double buffer) |  Write while computing next tile
+-------------------+
```

---

## 8. Implementation Roadmap

### Phase 1: SMALL (16x16) - Prototype

- 256 MACs/cycle
- ~10 FPS on YOLOv4-tiny
- Easier to debug and verify
- ~256 DSPs

### Phase 2: MEDIUM (32x32) - Production

- 1024 MACs/cycle
- ~54 FPS on YOLOv4-tiny
- Full real-time capability
- ~1024 DSPs

### Phase 3: LARGE (64x64) - High Performance

- 4096 MACs/cycle
- ~174 FPS on YOLOv4-tiny
- For demanding applications
- ~4096 DSPs

---

## 9. Resource Mapping

### Target: Zynq UltraScale+ ZU3EG

| Resource | Available | Used (MEDIUM) | Utilization |
|----------|-----------|---------------|-------------|
| DSP48E2 | 360 | 256 (SMALL) | 71% |
| LUTs | 71,280 | ~32K (SMALL) | 45% |
| BRAM | 432 KB | 128 KB (SMALL) | 30% |

**Note:** ZU3EG fits SMALL config. For MEDIUM, use ZU7EV or larger.

### Target: Zynq UltraScale+ ZU7EV

| Resource | Available | Used (MEDIUM) | Utilization |
|----------|-----------|---------------|-------------|
| DSP48E2 | 1728 | 1024 | 59% |
| LUTs | 504,000 | ~125K | 25% |
| BRAM | 1.5 MB | 256 KB | 17% |

**MEDIUM configuration fits well.**

---

## 10. Files Generated

| File | Description |
|------|-------------|
| `phase4_architecture_exploration.py` | Exploration script |
| `phase4_architecture_results.json` | Results data |
| `docs/PHASE4_ARCHITECTURE_EXPLORATION.md` | This document |

---

## 11. Next Steps (Phase 5)

1. Define detailed hardware block diagram
2. Specify internal interfaces
3. Define external AXI interface
4. Create register map
5. Design state machine for control

---

**Phase 4 Status: COMPLETE**
**Selected: MEDIUM (32x32 MAC Array)**
**Target: 53.7 FPS on YOLOv4-tiny**
