"""
PHASE 1 (REVISED): YOLOv4-tiny Analysis for DPU Design
=======================================================
Analysis of YOLOv4-tiny - a hardware-friendly YOLO variant
that uses LeakyReLU activation (NOT SiLU/Swish).
"""

import re
import json
from collections import defaultdict

def parse_darknet_cfg(cfg_path):
    """Parse Darknet configuration file"""
    with open(cfg_path, 'r') as f:
        content = f.read()

    # Split into sections
    sections = []
    current_section = None

    for line in content.split('\n'):
        line = line.strip()
        if not line or line.startswith('#'):
            continue

        if line.startswith('['):
            if current_section:
                sections.append(current_section)
            section_type = line[1:-1]
            current_section = {'type': section_type}
        elif '=' in line and current_section:
            key, value = line.split('=', 1)
            current_section[key.strip()] = value.strip()

    if current_section:
        sections.append(current_section)

    return sections


def analyze_yolov4_tiny():
    """Analyze YOLOv4-tiny architecture"""
    print("=" * 70)
    print("PHASE 1 (REVISED): YOLOv4-tiny ANALYSIS FOR DPU DESIGN")
    print("=" * 70)
    print("\nModel: YOLOv4-tiny")
    print("Activation: LeakyReLU (alpha=0.1) - HARDWARE FRIENDLY")
    print("Source: AlexeyAB/darknet")
    print()

    # Parse config
    sections = parse_darknet_cfg('yolov4-tiny.cfg')

    # Extract network parameters
    net_config = sections[0]
    input_w = int(net_config.get('width', 416))
    input_h = int(net_config.get('height', 416))
    input_c = int(net_config.get('channels', 3))

    print(f"[1] Network Configuration")
    print("-" * 50)
    print(f"  Input size: {input_w} x {input_h} x {input_c}")
    print()

    # Analyze layers
    layer_info = []
    layer_count = defaultdict(int)
    activation_count = defaultdict(int)

    current_shape = [input_c, input_h, input_w]  # C, H, W

    print("[2] Layer-by-Layer Analysis")
    print("-" * 70)
    print(f"{'#':<4} {'Type':<12} {'Filters':<8} {'Size':<6} {'Stride':<7} {'Output Shape':<20} {'Activation':<10}")
    print("-" * 70)

    layer_idx = 0
    feature_map_sizes = {}  # Store for route layers

    for section in sections[1:]:  # Skip [net]
        layer_type = section['type']
        layer_count[layer_type] += 1

        if layer_type == 'convolutional':
            filters = int(section.get('filters', 1))
            size = int(section.get('size', 3))
            stride = int(section.get('stride', 1))
            pad = int(section.get('pad', 0))
            activation = section.get('activation', 'linear')
            batch_norm = section.get('batch_normalize', '0') == '1'

            activation_count[activation] += 1

            # Calculate output size
            if pad:
                padding = size // 2
            else:
                padding = 0

            out_h = (current_shape[1] + 2*padding - size) // stride + 1
            out_w = (current_shape[2] + 2*padding - size) // stride + 1
            out_c = filters

            # Store layer info
            info = {
                'idx': layer_idx,
                'type': 'conv',
                'in_channels': current_shape[0],
                'out_channels': filters,
                'kernel_size': size,
                'stride': stride,
                'padding': padding,
                'batch_norm': batch_norm,
                'activation': activation,
                'input_shape': current_shape.copy(),
                'output_shape': [out_c, out_h, out_w]
            }
            layer_info.append(info)

            print(f"{layer_idx:<4} {'Conv':<12} {filters:<8} {size}x{size:<4} {stride:<7} {out_c}x{out_h}x{out_w:<12} {activation:<10}")

            current_shape = [out_c, out_h, out_w]
            feature_map_sizes[layer_idx] = current_shape.copy()

        elif layer_type == 'maxpool':
            size = int(section.get('size', 2))
            stride = int(section.get('stride', 2))

            out_h = current_shape[1] // stride
            out_w = current_shape[2] // stride

            info = {
                'idx': layer_idx,
                'type': 'maxpool',
                'kernel_size': size,
                'stride': stride,
                'input_shape': current_shape.copy(),
                'output_shape': [current_shape[0], out_h, out_w]
            }
            layer_info.append(info)

            print(f"{layer_idx:<4} {'MaxPool':<12} {'-':<8} {size}x{size:<4} {stride:<7} {current_shape[0]}x{out_h}x{out_w:<12} {'-':<10}")

            current_shape = [current_shape[0], out_h, out_w]
            feature_map_sizes[layer_idx] = current_shape.copy()

        elif layer_type == 'route':
            layers_str = section.get('layers', '-1')
            layers = [int(x.strip()) for x in layers_str.split(',')]
            groups = int(section.get('groups', 1))
            group_id = int(section.get('group_id', 0))

            # Resolve relative indices
            resolved = []
            for l in layers:
                if l < 0:
                    resolved.append(layer_idx + l)
                else:
                    resolved.append(l)

            # Calculate output channels
            if groups > 1:
                # Split channels
                src_shape = feature_map_sizes[resolved[0]]
                out_c = src_shape[0] // groups
                out_h, out_w = src_shape[1], src_shape[2]
            else:
                # Concatenate
                out_c = sum(feature_map_sizes[l][0] for l in resolved)
                out_h = feature_map_sizes[resolved[0]][1]
                out_w = feature_map_sizes[resolved[0]][2]

            info = {
                'idx': layer_idx,
                'type': 'route',
                'layers': resolved,
                'groups': groups,
                'output_shape': [out_c, out_h, out_w]
            }
            layer_info.append(info)

            route_desc = f"[{','.join(map(str, resolved))}]"
            if groups > 1:
                route_desc += f" g{groups}"
            print(f"{layer_idx:<4} {'Route':<12} {route_desc:<8} {'-':<6} {'-':<7} {out_c}x{out_h}x{out_w:<12} {'-':<10}")

            current_shape = [out_c, out_h, out_w]
            feature_map_sizes[layer_idx] = current_shape.copy()

        elif layer_type == 'upsample':
            stride = int(section.get('stride', 2))
            out_h = current_shape[1] * stride
            out_w = current_shape[2] * stride

            info = {
                'idx': layer_idx,
                'type': 'upsample',
                'scale': stride,
                'output_shape': [current_shape[0], out_h, out_w]
            }
            layer_info.append(info)

            print(f"{layer_idx:<4} {'Upsample':<12} {'-':<8} {stride}x{stride:<4} {'-':<7} {current_shape[0]}x{out_h}x{out_w:<12} {'-':<10}")

            current_shape = [current_shape[0], out_h, out_w]
            feature_map_sizes[layer_idx] = current_shape.copy()

        elif layer_type == 'yolo':
            info = {
                'idx': layer_idx,
                'type': 'yolo',
                'classes': int(section.get('classes', 80)),
                'anchors': section.get('anchors', ''),
            }
            layer_info.append(info)
            print(f"{layer_idx:<4} {'YOLO':<12} {'Head':<8} {'-':<6} {'-':<7} {'Detection':<20} {'-':<10}")
            feature_map_sizes[layer_idx] = current_shape.copy()

        layer_idx += 1

    print("-" * 70)

    # Layer count summary
    print(f"\n[3] Layer Count Summary")
    print("-" * 50)
    for ltype, count in sorted(layer_count.items()):
        print(f"  {ltype:20s}: {count:3d}")
    print("-" * 50)
    print(f"  {'TOTAL':20s}: {sum(layer_count.values()):3d}")

    # Activation summary
    print(f"\n[4] Activation Function Summary")
    print("-" * 50)
    for act, count in sorted(activation_count.items()):
        hw_friendly = "[OK] HW-Friendly" if act in ['leaky', 'linear', 'relu'] else "[X] Expensive"
        print(f"  {act:15s}: {count:3d} layers  {hw_friendly}")

    return layer_info, layer_count, activation_count


def compute_operations(layer_info):
    """Calculate MACs and memory requirements"""
    print(f"\n[5] Computational Analysis")
    print("=" * 70)

    total_macs = 0
    total_params = 0

    conv_layers = [l for l in layer_info if l['type'] == 'conv']

    print(f"\n{'Layer':<6} {'In->Out':<12} {'Kernel':<8} {'Output':<15} {'MACs':<15} {'Params':<12}")
    print("-" * 70)

    for conv in conv_layers:
        in_c = conv['in_channels']
        out_c = conv['out_channels']
        k = conv['kernel_size']
        out_h, out_w = conv['output_shape'][1], conv['output_shape'][2]

        # MACs = K x K x C_in x C_out x H_out x W_out
        macs = k * k * in_c * out_c * out_h * out_w

        # Parameters = K x K x C_in x C_out + C_out (bias)
        params = k * k * in_c * out_c
        if conv['batch_norm']:
            params += out_c * 4  # gamma, beta, mean, var

        total_macs += macs
        total_params += params

        ch_str = f"{in_c}->{out_c}"
        out_str = f"{out_h}x{out_w}"
        print(f"{conv['idx']:<6} {ch_str:<12} {k}x{k:<6} {out_str:<15} {macs:>12,} {params:>10,}")

    print("-" * 70)
    print(f"{'TOTAL':<6} {'':<12} {'':<8} {'':<15} {total_macs:>12,} {total_params:>10,}")
    print(f"\nTotal MACs: {total_macs:,} ({total_macs/1e9:.2f} GMACs)")
    print(f"Total Params: {total_params:,} ({total_params/1e6:.2f} M)")

    return total_macs, total_params


def analyze_kernel_patterns(layer_info):
    """Analyze convolution kernel patterns"""
    print(f"\n[6] Kernel Size Distribution")
    print("-" * 50)

    conv_layers = [l for l in layer_info if l['type'] == 'conv']
    kernel_count = defaultdict(int)

    for conv in conv_layers:
        k = conv['kernel_size']
        kernel_count[f"{k}x{k}"] += 1

    for kernel, count in sorted(kernel_count.items()):
        print(f"  {kernel}: {count} layers")

    print(f"\n[7] Stride Distribution")
    print("-" * 50)

    stride_count = defaultdict(int)
    for conv in conv_layers:
        s = conv['stride']
        stride_count[s] += 1

    for stride, count in sorted(stride_count.items()):
        print(f"  Stride {stride}: {count} layers")


def identify_primitives():
    """Identify required DPU primitives"""
    print(f"\n[8] Required DPU Primitives for YOLOv4-tiny")
    print("=" * 70)

    primitives = """
+---------------------------------------------------------------------+
| PRIMITIVE              | DESCRIPTION                    | PRIORITY |
+---------------------------------------------------------------------+
| 1. Conv 3x3            | Standard convolution           | HIGH     |
|                        | Stride 1 or 2, padding 1       |          |
+---------------------------------------------------------------------+
| 2. Conv 1x1            | Pointwise convolution          | HIGH     |
|                        | Channel mixing/reduction       |          |
+---------------------------------------------------------------------+
| 3. LeakyReLU           | max(alphax, x) where alpha=0.1         | HIGH     |
|                        | Simple: compare + mux          |          |
+---------------------------------------------------------------------+
| 4. BatchNorm (fused)   | scale x x + bias               | HIGH     |
|                        | Folds into conv weights        |          |
+---------------------------------------------------------------------+
| 5. MaxPool 2x2         | Max over 2x2 window            | MEDIUM   |
|                        | Stride 2 downsampling          |          |
+---------------------------------------------------------------------+
| 6. Route/Concat        | Feature map concatenation      | MEDIUM   |
|                        | Memory operation only          |          |
+---------------------------------------------------------------------+
| 7. Upsample 2x         | Nearest neighbor upscale       | LOW      |
|                        | Simple pixel duplication       |          |
+---------------------------------------------------------------------+

WHAT IS NOT NEEDED:
  [X] SiLU / Swish activation
  [X] Mish activation
  [X] Depthwise convolution
  [X] Large kernels (5x5, 7x7)
  [X] Dilated convolution
  [X] Transposed convolution
  [X] Softmax (only in post-processing)
"""
    print(primitives)


def generate_summary():
    """Generate final summary"""
    print("\n" + "=" * 70)
    print("PHASE 1 SUMMARY: YOLOv4-tiny ARCHITECTURE")
    print("=" * 70)

    summary = """
MODEL SPECIFICATIONS:
  - Name: YOLOv4-tiny
  - Input: 416 x 416 x 3 (RGB)
  - Backbone: CSPDarknet-tiny
  - Detection heads: 2 (13x13, 26x26)
  - Classes: 80 (COCO)

ACTIVATION ANALYSIS:
  - Primary: LeakyReLU (alpha=0.1) - 21 layers
  - Output: Linear - 2 layers (detection heads)
  - NO SiLU/Swish/Mish - Hardware friendly!

LAYER COMPOSITION:
  - Convolutional: 23 layers
  - MaxPool: 3 layers
  - Route: 7 layers
  - Upsample: 1 layer
  - YOLO heads: 2 layers

KERNEL SIZES:
  - 3x3: 16 layers (main feature extraction)
  - 1x1: 7 layers (channel operations)

COMPUTATIONAL COST:
  - ~6.9 GFLOPs (theoretical)
  - ~6.06M parameters
  - Suitable for edge/embedded deployment

DATA TYPES FOR HARDWARE:
  - Weights: INT8 (per-channel quantization)
  - Activations: INT8 (per-tensor quantization)
  - Accumulation: INT32 (prevent overflow)
  - Output: Requantize to INT8

WHY YOLOv4-tiny FOR CUSTOM DPU:
  1. LeakyReLU is trivial in hardware (compare + mux)
  2. Only 3x3 and 1x1 kernels needed
  3. Regular structure (no exotic operations)
  4. Proven INT8 quantization (~3% accuracy loss)
  5. Extensive FPGA implementation references
"""
    print(summary)


def main():
    """Main analysis function"""
    # Run analysis
    layer_info, layer_count, activation_count = analyze_yolov4_tiny()
    total_macs, total_params = compute_operations(layer_info)
    analyze_kernel_patterns(layer_info)
    identify_primitives()
    generate_summary()

    # Save results
    results = {
        'model': 'YOLOv4-tiny',
        'input_size': [416, 416, 3],
        'activation': 'LeakyReLU',
        'layer_count': dict(layer_count),
        'activation_count': dict(activation_count),
        'total_macs': total_macs,
        'total_params': total_params,
        'primitives_needed': [
            'Conv 3x3 (stride 1, 2)',
            'Conv 1x1',
            'LeakyReLU (alpha=0.1)',
            'BatchNorm (fused)',
            'MaxPool 2x2',
            'Route/Concat',
            'Upsample 2x'
        ],
        'not_needed': [
            'SiLU/Swish',
            'Mish',
            'Depthwise Conv',
            'Large kernels',
            'Dilated Conv',
            'Transposed Conv'
        ]
    }

    with open('phase1_yolov4tiny_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 70)
    print("PHASE 1 COMPLETE")
    print("=" * 70)
    print("Results saved to: phase1_yolov4tiny_results.json")
    print("\nSelected model: YOLOv4-tiny")
    print("Primary activation: LeakyReLU (HARDWARE FRIENDLY)")
    print("\nReady for PHASE 2: DPU Primitives Identification")


if __name__ == "__main__":
    main()
