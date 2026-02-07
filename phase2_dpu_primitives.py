"""
PHASE 2: DPU Primitives Identification
======================================
Define the minimal set of hardware primitives needed to accelerate
YOLOv3-tiny and YOLOv4-tiny on a custom DPU.
"""

import json

def analyze_both_models():
    """Compare YOLOv3-tiny and YOLOv4-tiny architectures"""

    print("=" * 70)
    print("PHASE 2: DPU PRIMITIVES IDENTIFICATION")
    print("=" * 70)

    # YOLOv3-tiny architecture (from cfg analysis)
    yolov3_tiny = {
        'name': 'YOLOv3-tiny',
        'input': (416, 416, 3),
        'conv_layers': 13,
        'maxpool_layers': 6,
        'route_layers': 2,
        'upsample_layers': 1,
        'yolo_heads': 2,
        'kernel_sizes': {'3x3': 10, '1x1': 3},
        'strides': {'1': 13, '2': 0},  # All stride 1, maxpool does downsampling
        'max_channels': 1024,
        'activations': {'leaky': 11, 'linear': 2},
        'params': 8.7e6,
        'flops': 5.6e9,
    }

    # YOLOv4-tiny architecture (from our analysis)
    yolov4_tiny = {
        'name': 'YOLOv4-tiny',
        'input': (416, 416, 3),
        'conv_layers': 21,
        'maxpool_layers': 3,
        'route_layers': 11,
        'upsample_layers': 1,
        'yolo_heads': 2,
        'kernel_sizes': {'3x3': 14, '1x1': 7},
        'strides': {'1': 19, '2': 2},  # Some conv with stride 2
        'max_channels': 512,
        'activations': {'leaky': 19, 'linear': 2},
        'params': 6.06e6,
        'flops': 6.9e9,
    }

    print("\n[1] MODEL COMPARISON")
    print("-" * 70)
    print(f"{'Metric':<25} {'YOLOv3-tiny':<20} {'YOLOv4-tiny':<20}")
    print("-" * 70)
    print(f"{'Conv layers':<25} {yolov3_tiny['conv_layers']:<20} {yolov4_tiny['conv_layers']:<20}")
    print(f"{'MaxPool layers':<25} {yolov3_tiny['maxpool_layers']:<20} {yolov4_tiny['maxpool_layers']:<20}")
    print(f"{'Route layers':<25} {yolov3_tiny['route_layers']:<20} {yolov4_tiny['route_layers']:<20}")
    print(f"{'3x3 kernels':<25} {yolov3_tiny['kernel_sizes']['3x3']:<20} {yolov4_tiny['kernel_sizes']['3x3']:<20}")
    print(f"{'1x1 kernels':<25} {yolov3_tiny['kernel_sizes']['1x1']:<20} {yolov4_tiny['kernel_sizes']['1x1']:<20}")
    print(f"{'Max channels':<25} {yolov3_tiny['max_channels']:<20} {yolov4_tiny['max_channels']:<20}")
    print(f"{'Parameters':<25} {yolov3_tiny['params']/1e6:.2f}M{'':<14} {yolov4_tiny['params']/1e6:.2f}M")
    print(f"{'FLOPs':<25} {yolov3_tiny['flops']/1e9:.1f}G{'':<15} {yolov4_tiny['flops']/1e9:.1f}G")
    print(f"{'Activation':<25} {'LeakyReLU':<20} {'LeakyReLU':<20}")

    return yolov3_tiny, yolov4_tiny


def define_primitives():
    """Define all DPU primitives with detailed specifications"""

    print("\n" + "=" * 70)
    print("[2] DPU PRIMITIVES DEFINITION")
    print("=" * 70)

    primitives = {}

    # =========================================================================
    # PRIMITIVE 1: MAC (Multiply-Accumulate)
    # =========================================================================
    primitives['MAC'] = {
        'name': 'Multiply-Accumulate Unit',
        'priority': 'CRITICAL',
        'description': 'Core compute element for convolutions',
        'inputs': {
            'A': 'INT8 (signed) - Weight',
            'B': 'INT8 (signed/unsigned) - Activation',
        },
        'output': {
            'P': 'INT32 (signed) - Partial sum accumulation',
        },
        'operation': 'P_new = P_old + (A * B)',
        'hardware': {
            'multiplier': '8x8 -> 16-bit product',
            'accumulator': '32-bit adder',
            'latency': '1 cycle (pipelined)',
        },
        'notes': 'INT32 accumulator prevents overflow for up to 65536 MAC operations',
    }

    print("\n--- PRIMITIVE 1: MAC (Multiply-Accumulate) ---")
    print(f"  Purpose:    Core compute for convolutions")
    print(f"  Inputs:     A (INT8 weight), B (INT8 activation)")
    print(f"  Output:     P (INT32 accumulated sum)")
    print(f"  Operation:  P = P + A * B")
    print(f"  Latency:    1 cycle (pipelined)")

    # =========================================================================
    # PRIMITIVE 2: Conv 3x3
    # =========================================================================
    primitives['CONV3x3'] = {
        'name': '3x3 Convolution',
        'priority': 'HIGH',
        'description': 'Main feature extraction operation',
        'inputs': {
            'input_fm': 'INT8[C_in][H][W] - Input feature map',
            'weights': 'INT8[C_out][C_in][3][3] - Kernel weights',
            'bias': 'INT32[C_out] - Bias (optional, fused BN)',
        },
        'output': {
            'output_fm': 'INT32[C_out][H_out][W_out] - Before requantization',
        },
        'parameters': {
            'stride': [1, 2],
            'padding': 1,
            'C_in_max': 512,
            'C_out_max': 512,
        },
        'macs_per_output': '9 * C_in',
        'hardware': {
            'mac_array': 'Systolic or Output-stationary',
            'line_buffer': '2 lines for 3x3 window',
            'weight_buffer': 'C_out * C_in * 9 bytes',
        },
    }

    print("\n--- PRIMITIVE 2: Conv 3x3 ---")
    print(f"  Purpose:    Feature extraction (dominant operation)")
    print(f"  Kernel:     3x3")
    print(f"  Stride:     1 or 2")
    print(f"  Padding:    1 (same padding)")
    print(f"  MACs/pixel: 9 * C_in * C_out")
    print(f"  Max C_in:   512")
    print(f"  Max C_out:  512")

    # =========================================================================
    # PRIMITIVE 3: Conv 1x1
    # =========================================================================
    primitives['CONV1x1'] = {
        'name': '1x1 Convolution (Pointwise)',
        'priority': 'HIGH',
        'description': 'Channel mixing and dimension reduction',
        'inputs': {
            'input_fm': 'INT8[C_in][H][W]',
            'weights': 'INT8[C_out][C_in][1][1]',
            'bias': 'INT32[C_out]',
        },
        'output': {
            'output_fm': 'INT32[C_out][H][W]',
        },
        'parameters': {
            'stride': 1,
            'padding': 0,
            'C_in_max': 1024,
            'C_out_max': 512,
        },
        'macs_per_output': 'C_in',
        'hardware': {
            'mac_array': 'Same as 3x3, but no line buffer needed',
            'weight_buffer': 'C_out * C_in bytes',
        },
    }

    print("\n--- PRIMITIVE 3: Conv 1x1 (Pointwise) ---")
    print(f"  Purpose:    Channel mixing, dimension reduction")
    print(f"  Kernel:     1x1")
    print(f"  Stride:     1")
    print(f"  Padding:    0")
    print(f"  MACs/pixel: C_in * C_out")
    print(f"  Max C_in:   1024 (YOLOv3-tiny)")
    print(f"  Max C_out:  512")

    # =========================================================================
    # PRIMITIVE 4: LeakyReLU
    # =========================================================================
    primitives['LEAKY_RELU'] = {
        'name': 'Leaky ReLU Activation',
        'priority': 'HIGH',
        'description': 'Non-linear activation function',
        'inputs': {
            'x': 'INT32 or INT8 - Input value',
        },
        'output': {
            'y': 'INT8 - Activated value',
        },
        'parameters': {
            'alpha': 0.1,  # Negative slope
        },
        'operation': '''
            if x > 0:
                y = x
            else:
                y = x * 0.1  (or x >> 3 for approximation)
        ''',
        'hardware': {
            'comparator': '1x signed comparator',
            'shifter': 'Right shift by 3 (approximates 0.1 = 1/8)',
            'mux': '2:1 multiplexer',
            'latency': '1 cycle',
        },
        'approximation': 'alpha=0.1 approx= 1/8 = 0.125 (shift right 3)',
    }

    print("\n--- PRIMITIVE 4: LeakyReLU ---")
    print(f"  Purpose:    Non-linear activation")
    print(f"  Operation:  y = x if x > 0 else alpha * x")
    print(f"  Alpha:      0.1 (can approximate as 1/8 = 0.125)")
    print(f"  Hardware:   1 comparator + 1 shifter + 1 mux")
    print(f"  Latency:    1 cycle")
    print(f"  Note:       MUCH simpler than SiLU!")

    # =========================================================================
    # PRIMITIVE 5: Requantization
    # =========================================================================
    primitives['REQUANT'] = {
        'name': 'Requantization',
        'priority': 'HIGH',
        'description': 'Scale INT32 accumulator back to INT8',
        'inputs': {
            'acc': 'INT32 - Accumulated value',
            'scale': 'INT32 or fixed-point scale factor',
            'zero_point': 'INT8 - Zero point offset',
        },
        'output': {
            'y': 'INT8 - Quantized output',
        },
        'operation': '''
            y = clamp(round(acc * scale) + zero_point, -128, 127)
        ''',
        'hardware': {
            'multiplier': '32x32 -> 64 bit (or shift approximation)',
            'adder': '32-bit',
            'saturator': 'Clamp to INT8 range',
            'latency': '1-2 cycles',
        },
    }

    print("\n--- PRIMITIVE 5: Requantization ---")
    print(f"  Purpose:    Convert INT32 accumulator to INT8")
    print(f"  Operation:  y = clamp(round(acc * scale) + zp)")
    print(f"  Hardware:   Multiplier + Adder + Saturator")
    print(f"  Latency:    1-2 cycles")

    # =========================================================================
    # PRIMITIVE 6: MaxPool 2x2
    # =========================================================================
    primitives['MAXPOOL'] = {
        'name': 'Max Pooling 2x2',
        'priority': 'MEDIUM',
        'description': 'Spatial downsampling via max operation',
        'inputs': {
            'x': 'INT8[C][H][W] - Input feature map',
        },
        'output': {
            'y': 'INT8[C][H/2][W/2] - Downsampled output',
        },
        'parameters': {
            'kernel': '2x2',
            'stride': 2,
        },
        'operation': 'y = max(x[0,0], x[0,1], x[1,0], x[1,1])',
        'hardware': {
            'comparators': '3x signed comparators (tree)',
            'line_buffer': '1 line',
            'latency': '1 cycle',
        },
    }

    print("\n--- PRIMITIVE 6: MaxPool 2x2 ---")
    print(f"  Purpose:    Spatial downsampling")
    print(f"  Kernel:     2x2")
    print(f"  Stride:     2")
    print(f"  Hardware:   3 comparators (tree structure)")
    print(f"  Latency:    1 cycle")

    # =========================================================================
    # PRIMITIVE 7: Route/Concat
    # =========================================================================
    primitives['ROUTE'] = {
        'name': 'Route / Concatenation',
        'priority': 'MEDIUM',
        'description': 'Concatenate feature maps or split channels',
        'inputs': {
            'fm1': 'INT8[C1][H][W] - Feature map 1',
            'fm2': 'INT8[C2][H][W] - Feature map 2 (optional)',
        },
        'output': {
            'y': 'INT8[C1+C2][H][W] - Concatenated output',
        },
        'operations': {
            'concat': 'Combine along channel dimension',
            'split': 'Take subset of channels (groups)',
        },
        'hardware': {
            'type': 'Memory operation ONLY',
            'compute': 'None (just address calculation)',
            'latency': 'Memory bandwidth limited',
        },
    }

    print("\n--- PRIMITIVE 7: Route/Concat ---")
    print(f"  Purpose:    Feature map concatenation/split")
    print(f"  Operation:  Memory addressing only")
    print(f"  Compute:    NONE (zero MACs)")
    print(f"  Hardware:   Address generator + memory controller")

    # =========================================================================
    # PRIMITIVE 8: Upsample 2x
    # =========================================================================
    primitives['UPSAMPLE'] = {
        'name': 'Nearest Neighbor Upsample 2x',
        'priority': 'LOW',
        'description': 'Spatial upscaling for FPN',
        'inputs': {
            'x': 'INT8[C][H][W]',
        },
        'output': {
            'y': 'INT8[C][2H][2W]',
        },
        'operation': 'Duplicate each pixel to 2x2 block',
        'hardware': {
            'type': 'Memory operation',
            'compute': 'None (pixel duplication)',
            'latency': 'Memory bandwidth limited',
        },
    }

    print("\n--- PRIMITIVE 8: Upsample 2x ---")
    print(f"  Purpose:    Spatial upscaling for FPN")
    print(f"  Operation:  Nearest neighbor (pixel duplication)")
    print(f"  Compute:    NONE")
    print(f"  Hardware:   Address generator only")

    # =========================================================================
    # PRIMITIVE 9: Bias Add (fused BatchNorm)
    # =========================================================================
    primitives['BIAS_ADD'] = {
        'name': 'Bias Addition (Fused BatchNorm)',
        'priority': 'HIGH',
        'description': 'Add per-channel bias after convolution',
        'inputs': {
            'acc': 'INT32[C][H][W] - Accumulated conv output',
            'bias': 'INT32[C] - Per-channel bias',
        },
        'output': {
            'y': 'INT32[C][H][W]',
        },
        'operation': 'y[c] = acc[c] + bias[c]',
        'hardware': {
            'adder': '32-bit',
            'latency': '1 cycle (fused with MAC)',
        },
        'note': 'BatchNorm is folded into weights and bias at compile time',
    }

    print("\n--- PRIMITIVE 9: Bias Add (Fused BatchNorm) ---")
    print(f"  Purpose:    Add per-channel bias")
    print(f"  Operation:  y = acc + bias")
    print(f"  Note:       BatchNorm folded at compile time:")
    print(f"              W' = W * gamma / sqrt(var + eps)")
    print(f"              b' = beta - mean * gamma / sqrt(var + eps)")

    return primitives


def define_data_types():
    """Define data types for the DPU"""

    print("\n" + "=" * 70)
    print("[3] DATA TYPES")
    print("=" * 70)

    data_types = {
        'weights': {
            'type': 'INT8',
            'bits': 8,
            'signed': True,
            'range': '[-128, 127]',
            'quantization': 'Per-channel symmetric',
        },
        'activations': {
            'type': 'INT8',
            'bits': 8,
            'signed': True,  # After LeakyReLU can be negative
            'range': '[-128, 127]',
            'quantization': 'Per-tensor asymmetric',
        },
        'accumulator': {
            'type': 'INT32',
            'bits': 32,
            'signed': True,
            'range': '[-2^31, 2^31-1]',
            'note': 'Prevents overflow: 8+8=16 bit product, up to 2^16 accumulations',
        },
        'bias': {
            'type': 'INT32',
            'bits': 32,
            'signed': True,
            'note': 'Fused BatchNorm bias, per-channel',
        },
        'scale': {
            'type': 'INT32 or Fixed-Point',
            'bits': 32,
            'note': 'Requantization scale factor',
        },
    }

    print(f"\n{'Data Type':<20} {'Bits':<8} {'Signed':<8} {'Usage'}")
    print("-" * 70)
    print(f"{'Weights':<20} {'8':<8} {'Yes':<8} Convolution kernels")
    print(f"{'Activations':<20} {'8':<8} {'Yes':<8} Feature maps (after LeakyReLU)")
    print(f"{'Accumulator':<20} {'32':<8} {'Yes':<8} Partial sums during conv")
    print(f"{'Bias':<20} {'32':<8} {'Yes':<8} Fused BatchNorm bias")
    print(f"{'Scale':<20} {'32':<8} {'N/A':<8} Requantization scale")

    print("\n  Overflow Analysis:")
    print("  - Max MACs per output = C_in * K * K = 1024 * 3 * 3 = 9216")
    print("  - Max product = 127 * 127 = 16129 (fits in 16 bits)")
    print("  - Max accumulation = 16129 * 9216 = 148,660,224")
    print("  - Fits in INT32 (max 2,147,483,647) with margin")

    return data_types


def what_is_not_needed():
    """Explicitly list what is NOT needed"""

    print("\n" + "=" * 70)
    print("[4] WHAT IS NOT NEEDED")
    print("=" * 70)

    not_needed = {
        'SiLU / Swish': 'Complex: requires sigmoid (exponential)',
        'Mish': 'Complex: requires softplus and tanh',
        'Sigmoid': 'Not used in inference path',
        'Tanh': 'Not used',
        'Softmax': 'Only in post-processing (CPU)',
        'Depthwise Conv': 'Not used in YOLOv3/v4-tiny',
        'Group Conv': 'Only groups=1 used',
        'Dilated Conv': 'Not used',
        'Transposed Conv': 'Not used (upsample is nearest neighbor)',
        'Large Kernels': 'Only 3x3 and 1x1',
        '5x5, 7x7 Conv': 'Not used',
        'Batch Norm (runtime)': 'Folded into weights at compile time',
        'Dropout': 'Only used during training',
        'Floating Point': 'INT8 quantized inference',
    }

    for item, reason in not_needed.items():
        print(f"  [X] {item:<25} -> {reason}")

    return not_needed


def execution_model():
    """Define simplified execution model"""

    print("\n" + "=" * 70)
    print("[5] SIMPLIFIED EXECUTION MODEL")
    print("=" * 70)

    print("""
  CONVOLUTION EXECUTION FLOW:
  ===========================

  For each output pixel (h, w) and output channel (c_out):

    1. LOAD weights[c_out][0:C_in][0:3][0:3]  (from weight buffer)

    2. LOAD input window[0:C_in][h:h+3][w:w+3]  (from line buffer)

    3. COMPUTE (MAC array):
       acc = 0
       for c_in in range(C_in):
           for kh in range(3):
               for kw in range(3):
                   acc += weight[c_out][c_in][kh][kw] * input[c_in][h+kh][w+kw]

    4. BIAS ADD:
       acc = acc + bias[c_out]

    5. ACTIVATION (LeakyReLU):
       if acc > 0:
           out = acc
       else:
           out = acc >> 3  (approximate alpha=0.1)

    6. REQUANTIZE:
       out_int8 = clamp(round(out * scale), -128, 127)

    7. STORE to output buffer


  PIPELINE:
  =========

  +---------+    +--------+    +----------+    +---------+    +-------+
  | Weight  | -> |  MAC   | -> | Bias Add | -> | LeakyRU | -> | ReQuant|
  | Buffer  |    | Array  |    |          |    |         |    |        |
  +---------+    +--------+    +----------+    +---------+    +-------+
       ^              ^
       |              |
  +---------+    +---------+
  | Weight  |    |  Line   |
  |  SRAM   |    | Buffer  |
  +---------+    +---------+
                      ^
                      |
               +-------------+
               | Input SRAM  |
               +-------------+
""")


def generate_primitive_summary():
    """Generate summary table"""

    print("\n" + "=" * 70)
    print("[6] PRIMITIVES SUMMARY TABLE")
    print("=" * 70)

    print(f"\n{'#':<4} {'Primitive':<20} {'Priority':<10} {'HW Complexity':<20} {'Compute'}")
    print("-" * 80)
    print(f"{'1':<4} {'MAC Unit':<20} {'CRITICAL':<10} {'8x8 mult + 32b add':<20} {'1 MAC/cycle'}")
    print(f"{'2':<4} {'Conv 3x3':<20} {'HIGH':<10} {'MAC array + buffers':<20} {'9*Cin MACs/px'}")
    print(f"{'3':<4} {'Conv 1x1':<20} {'HIGH':<10} {'MAC array':<20} {'Cin MACs/px'}")
    print(f"{'4':<4} {'LeakyReLU':<20} {'HIGH':<10} {'1 cmp + 1 shift + mux':<20} {'1 cycle'}")
    print(f"{'5':<4} {'Requantize':<20} {'HIGH':<10} {'mult + add + clamp':<20} {'1-2 cycles'}")
    print(f"{'6':<4} {'MaxPool 2x2':<20} {'MEDIUM':<10} {'3 comparators':<20} {'1 cycle'}")
    print(f"{'7':<4} {'Route/Concat':<20} {'MEDIUM':<10} {'Memory only':<20} {'0 MACs'}")
    print(f"{'8':<4} {'Upsample 2x':<20} {'LOW':<10} {'Memory only':<20} {'0 MACs'}")
    print(f"{'9':<4} {'Bias Add':<20} {'HIGH':<10} {'32b adder':<20} {'Fused w/MAC'}")


def model_comparison_for_dpu():
    """Compare models from DPU perspective"""

    print("\n" + "=" * 70)
    print("[7] MODEL COMPARISON FOR DPU DESIGN")
    print("=" * 70)

    print("""
  +------------------+-------------------+-------------------+
  | Aspect           | YOLOv3-tiny       | YOLOv4-tiny       |
  +------------------+-------------------+-------------------+
  | Architecture     | Linear (simple)   | CSP (skip paths)  |
  | Conv layers      | 13                | 21                |
  | Max channels     | 1024              | 512               |
  | Route complexity | Simple (2 routes) | Complex (11 routes)|
  | Downsampling     | MaxPool only      | MaxPool + Stride2 |
  | Parameters       | 8.7M              | 6.06M             |
  | FLOPs            | 5.6G              | 6.9G              |
  | Accuracy (mAP)   | 33.1%             | 40.2%             |
  +------------------+-------------------+-------------------+

  RECOMMENDATION:
  ===============

  For SIMPLEST DPU design:     YOLOv3-tiny
    - Linear dataflow
    - Fewer route operations
    - Simpler control logic
    - Good for first implementation

  For BEST accuracy/compute:   YOLOv4-tiny
    - Better accuracy (+7%)
    - More complex routing
    - Requires flexible memory addressing
    - Better for optimized DPU

  COMMON REQUIREMENTS (both models):
    - Conv 3x3 and 1x1
    - LeakyReLU (alpha=0.1)
    - MaxPool 2x2
    - INT8 weights and activations
    - INT32 accumulation
""")


def save_results():
    """Save Phase 2 results"""

    results = {
        'phase': 2,
        'title': 'DPU Primitives Identification',
        'models_analyzed': ['YOLOv3-tiny', 'YOLOv4-tiny'],
        'primitives': {
            'critical': ['MAC'],
            'high': ['Conv3x3', 'Conv1x1', 'LeakyReLU', 'Requantize', 'BiasAdd'],
            'medium': ['MaxPool2x2', 'Route'],
            'low': ['Upsample2x'],
        },
        'data_types': {
            'weights': 'INT8',
            'activations': 'INT8',
            'accumulator': 'INT32',
            'bias': 'INT32',
        },
        'not_needed': [
            'SiLU', 'Mish', 'Sigmoid', 'Softmax',
            'Depthwise Conv', 'Dilated Conv', 'Transposed Conv',
            'Large kernels (5x5, 7x7)', 'Floating point'
        ],
        'leaky_relu': {
            'alpha': 0.1,
            'approximation': '1/8 (shift right 3)',
            'hardware': 'comparator + shifter + mux',
        },
        'max_channels': {
            'YOLOv3-tiny': 1024,
            'YOLOv4-tiny': 512,
        },
        'recommendation': {
            'simple_dpu': 'YOLOv3-tiny',
            'optimized_dpu': 'YOLOv4-tiny',
        }
    }

    with open('phase2_primitives_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 70)
    print("PHASE 2 COMPLETE")
    print("=" * 70)
    print("Results saved to: phase2_primitives_results.json")


def main():
    """Main function"""

    # Compare both models
    yolov3, yolov4 = analyze_both_models()

    # Define primitives
    primitives = define_primitives()

    # Define data types
    data_types = define_data_types()

    # What is not needed
    not_needed = what_is_not_needed()

    # Execution model
    execution_model()

    # Summary
    generate_primitive_summary()

    # Model comparison
    model_comparison_for_dpu()

    # Save results
    save_results()

    print("\nReady for PHASE 3: Python Functional DPU Model")


if __name__ == "__main__":
    main()
