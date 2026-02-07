# PHASE 1 (REVISED): Hardware-Friendly YOLO Model Selection

## Problem Statement

The initial selection of YOLOv8n was rejected because it uses **SiLU (Swish)** activation:
```
SiLU(x) = x × sigmoid(x)
```

**Why SiLU is problematic for hardware:**
- Requires sigmoid computation (exponential function)
- Needs LUT with many entries or iterative approximation
- High latency and area cost
- Not natively supported by most DPU architectures

**Preferred activations for hardware:**
- **ReLU:** `max(0, x)` - single comparison
- **LeakyReLU:** `x if x > 0 else α×x` - comparison + multiply
- Both map to simple hardware primitives

---

## Activation Functions by YOLO Version

| YOLO Version | Activation | Hardware Friendly? |
|--------------|------------|-------------------|
| YOLOv1-v2 | LeakyReLU | YES |
| YOLOv3 | LeakyReLU | YES |
| YOLOv3-tiny | LeakyReLU | YES |
| YOLOv4 (backbone) | Mish | NO (but replaceable) |
| YOLOv4-tiny | LeakyReLU | YES |
| YOLOv5 | SiLU/Swish | NO |
| YOLOv6 | SiLU/ReLU | PARTIAL |
| YOLOv7 | SiLU | NO |
| YOLOv8 | SiLU | NO |

**Source:** [Vitis AI Activation Analysis](https://www.hackster.io/LogicTronix/activation-function-its-effect-analysis-with-amd-vitis-ai-c6d1b0)

---

## Candidate Models Analysis

### 1. YOLOv3-tiny

| Specification | Value |
|--------------|-------|
| Input Size | 416×416 |
| Parameters | ~8.7M |
| FLOPs | 5.6 BFLOPs |
| Model Size | 33.7 MB |
| Activation | **LeakyReLU (α=0.1)** |
| mAP@0.5 (COCO) | 33.1% |
| FPS (GPU) | 345 |

**Architecture:**
- 7 convolutional layers (3×3)
- 1 convolutional layer (1×1)
- 6 max pooling layers
- 2 detection heads

**Pros:**
- Native LeakyReLU throughout
- Proven FPGA implementations
- Simple architecture
- Well documented

**Cons:**
- Lower accuracy than v4-tiny
- Older architecture

**Sources:**
- [Darknet YOLOv3-tiny config](https://github.com/pjreddie/darknet/blob/master/cfg/yolov3-tiny.cfg)
- [FPGA Implementation](https://ieeexplore.ieee.org/document/9576092/)

---

### 2. YOLOv4-tiny (RECOMMENDED)

| Specification | Value |
|--------------|-------|
| Input Size | 416×416 |
| Parameters | ~6.06M |
| FLOPs | 6.9 BFLOPs |
| Model Size | 23.1 MB |
| Activation | **LeakyReLU (α=0.1)** |
| mAP@0.5 (COCO) | 40.2% |
| FPS (GPU) | 371 |

**Architecture:**
- CSPDarknet-tiny backbone
- PANet neck (simplified)
- 2 detection heads (13×13, 26×26)
- Uses LeakyReLU (NOT Mish like full YOLOv4)

**Pros:**
- Native LeakyReLU throughout
- Better accuracy than v3-tiny (+7%)
- Smaller model size
- Extensive FPGA implementations available
- Proven INT8 quantization (3% accuracy drop)

**Cons:**
- Slightly more compute than v3-tiny

**Sources:**
- [Roboflow YOLOv4-tiny Guide](https://roboflow.com/model/yolov4-tiny)
- [FPGA Implementation on ZYNQ](https://pmc.ncbi.nlm.nih.gov/articles/PMC9697515/)
- [MATLAB FPGA Deployment](https://www.mathworks.com/help/deep-learning-hdl/ug/yolov4-tiny-object-detection-using-fpga.html)

---

### 3. OFA-YOLO (Vitis AI Optimized)

| Specification | Value |
|--------------|-------|
| Input Size | 640×640 |
| GOPs | 24.62-48.88 (pruned variants) |
| Activation | LeakyReLU (Vitis compatible) |
| mAP (COCO) | 37.8-43.6% |
| Framework | PyTorch |

**Variants in Vitis AI Model Zoo:**
- `pt_OFA-yolo_coco_640_640_48.88G` - Full
- `pt_OFA-yolo_coco_640_640_0.3_34.72G` - 30% pruned
- `pt_OFA-yolo_coco_640_640_0.5_24.62G` - 50% pruned

**Pros:**
- Officially optimized for Vitis AI DPU
- Pre-quantized models available
- Pruned variants for different resource budgets

**Cons:**
- Higher compute than tiny variants
- Larger input resolution required
- Less documentation on custom DPU

**Source:** [Vitis AI Model Zoo](https://github.com/Xilinx/Vitis-AI/tree/v2.5/model_zoo)

---

### 4. Vitis AI Standard YOLO Models

Available in Vitis AI 2.5/3.0 Model Zoo:

| Model | Framework | Input | GOPs | Accuracy |
|-------|-----------|-------|------|----------|
| tf_yolov3_voc_416_416 | TF1 | 416×416 | 65.63 | 78.5% |
| tf2_yolov3_coco_416_416 | TF2 | 416×416 | 65.9 | 33.1% |
| tf_yolov4_coco_416_416 | TF1 | 416×416 | 60.3 | 39.3% |
| tf_yolov4_coco_512_512 | TF1 | 512×512 | 91.2 | 41.2% |

**Note:** These are full-size models, NOT tiny variants. Much higher compute.

---

## Comparison Table

| Model | Activation | GOPs | Params | mAP@0.5 | HW Friendly |
|-------|------------|------|--------|---------|-------------|
| YOLOv3-tiny | LeakyReLU | 5.6 | 8.7M | 33.1% | **YES** |
| **YOLOv4-tiny** | **LeakyReLU** | **6.9** | **6.1M** | **40.2%** | **YES** |
| YOLOv5n | SiLU | 4.5 | 1.9M | 28.0% | NO |
| YOLOv8n | SiLU | 8.7 | 3.2M | 37.3% | NO |
| OFA-YOLO-50% | LeakyReLU | 24.6 | - | 37.8% | YES |

---

## Vitis AI DPU Supported Activations

According to [Vitis AI documentation](https://www.hackster.io/LogicTronix/activation-function-its-effect-analysis-with-amd-vitis-ai-c6d1b0):

| Activation | Supported | Recommended |
|------------|-----------|-------------|
| ReLU | YES | YES |
| LeakyReLU | YES | **BEST FOR YOLO** |
| ReLU6 | YES | Converted to ReLU |
| Hard Tanh | YES | Poor results |
| Hard Sigmoid | YES | Poor results |
| Hard Swish | PARTIAL | Compilation issues |
| Mish | NO | Use LeakyReLU instead |
| SiLU/Swish | NO | Not supported |

---

## RECOMMENDATION

### Primary Choice: YOLOv4-tiny

**Rationale:**

1. **Native LeakyReLU** - No activation function modification needed
2. **Proven FPGA implementations** - Multiple academic papers with working designs
3. **Good accuracy/compute tradeoff** - 40.2% mAP with only 6.9 GFLOPs
4. **Small model size** - 23.1 MB fits in on-chip memory
5. **INT8 quantization validated** - Only 3% accuracy drop
6. **Darknet format available** - Easy to parse and analyze

### Where to obtain:

1. **Original Darknet weights:**
   - https://github.com/AlexeyAB/darknet (yolov4-tiny.weights)

2. **Pre-trained models:**
   - COCO dataset: yolov4-tiny.weights (22.4 MB)
   - Config: yolov4-tiny.cfg

3. **Framework conversions:**
   - TensorFlow: tensorflow-yolov4-tflite
   - PyTorch: pytorch-yolov4
   - ONNX: Available

### Alternative: YOLOv3-tiny

If even simpler architecture is desired:
- 5.6 GFLOPs (lower than v4-tiny)
- Same LeakyReLU activation
- More basic architecture (easier to implement)

---

## What to AVOID

| Model | Reason |
|-------|--------|
| YOLOv5 (all variants) | Uses SiLU activation |
| YOLOv6 | Uses SiLU in most layers |
| YOLOv7 | Uses SiLU activation |
| YOLOv8 (all variants) | Uses SiLU activation |
| YOLOv4 (full) | Uses Mish in backbone |

---

## Next Steps

1. Download YOLOv4-tiny weights and config
2. Parse the architecture for exact layer specifications
3. Re-run Phase 1 analysis with YOLOv4-tiny
4. Proceed to Phase 2 with LeakyReLU-based primitives

---

## References

1. [Vitis AI Model Zoo v2.5](https://github.com/Xilinx/Vitis-AI/tree/v2.5/model_zoo)
2. [Vitis AI Activation Function Analysis](https://www.hackster.io/LogicTronix/activation-function-its-effect-analysis-with-amd-vitis-ai-c6d1b0)
3. [YOLOv4-tiny FPGA Implementation](https://pmc.ncbi.nlm.nih.gov/articles/PMC9697515/)
4. [AlexeyAB Darknet Repository](https://github.com/AlexeyAB/darknet)
5. [YOLOv3-tiny FPGA Accelerator](https://ieeexplore.ieee.org/document/9576092/)
6. [MATLAB YOLOv4-tiny FPGA](https://www.mathworks.com/help/deep-learning-hdl/ug/yolov4-tiny-object-detection-using-fpga.html)

---

**Document Status:** Model Selection Complete
**Selected Model:** YOLOv4-tiny
**Activation:** LeakyReLU (α=0.1)
**Ready for:** Detailed Architecture Analysis
