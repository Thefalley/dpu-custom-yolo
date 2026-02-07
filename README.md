# DPU Custom YOLO

Custom DPU (Deep Learning Processing Unit) architecture designed and validated for a **hardware-friendly YOLO** model, using a strict bottom-up methodology.

---

## Project specification (7 phases)

| Phase | Goal | Status |
|-------|------|--------|
| **1** | YOLO model analysis (software) | **DONE** ? YOLOv4-tiny |
| **2** | Identification of DPU primitives | **DONE** |
| **3** | Python functional DPU model | **DONE** |
| **4** | Software benchmarking & architecture exploration | **DONE** ? MEDIUM 32?32 selected |
| **4b** | Multiplier exploration (DSP vs shift-and-add) | **DONE** ? documented |
| **5** | Hardware architecture definition | **DONE** |
| **6** | RTL implementation (Verilog) | Pending |
| **7** | Verification (AutoCheck testbench) | Pending |

**Model choice:** YOLOv4-tiny (LeakyReLU, hardware-friendly; see `docs/PHASE1_MODEL_SELECTION.md`).  
**Phase 4b:** Explores using **shift-and-add multipliers** instead of DSPs for ASIC portability and FPGA resource flexibility.

---

## Repository layout

```
docs/                    # Phase write-ups and analysis
  PHASE1_ANALYSIS.md
  PHASE1_MODEL_SELECTION.md   # Why YOLOv4-tiny (LeakyReLU)
  PHASE2_DPU_PRIMITIVES.md
  PHASE3_FUNCTIONAL_MODEL.md
  PHASE4_ARCHITECTURE_EXPLORATION.md
  PHASE4B_MULTIPLIER_EXPLORATION.md   # DSP vs LUT (shift-add)
  PHASE5_HARDWARE_ARCHITECTURE.md     # Datapath, control, memory, interfaces
phase1_yolov4tiny_analysis.py         # YOLOv4-tiny layer analysis
phase2_dpu_primitives.py
phase3_dpu_functional_model.py
phase4_architecture_exploration.py
phase4b_multiplier_exploration.py    # Multiplier options & FPS
yolov4-tiny.cfg
```

---

## Quick run

- **Phase 1 (YOLOv4-tiny):** `python phase1_yolov4tiny_analysis.py`
- **Phase 4 (architecture):** `python phase4_architecture_exploration.py`
- **Phase 4b (multipliers):** `python phase4b_multiplier_exploration.py`

FPGA/Vivado/Vitis are out of scope for the current phases.

---

## Git (after meaningful steps)

Suggested commit after Phase 4b:

```bash
git add README.md docs/PHASE4B_MULTIPLIER_EXPLORATION.md phase4b_multiplier_exploration.py phase4b_multiplier_results.json
git commit -m "Phase 4b: multiplier exploration (DSP vs shift-and-add), doc and results"
```
