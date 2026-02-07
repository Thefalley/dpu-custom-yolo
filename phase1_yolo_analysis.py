"""
PHASE 1: YOLO Model Analysis for DPU Design
============================================
This script analyzes YOLOv5n to understand the computational requirements
for designing a custom DPU architecture.
"""

import torch
import torch.nn as nn
from ultralytics import YOLO
import numpy as np
import json
from collections import defaultdict

def analyze_yolov5n():
    """
    Analyze YOLOv5n architecture for DPU design.
    We use YOLOv8n from ultralytics as it's the modern equivalent
    and shares similar architecture with YOLOv5n.
    """
    print("=" * 60)
    print("PHASE 1: YOLO MODEL ANALYSIS FOR DPU DESIGN")
    print("=" * 60)

    # Load YOLOv8n (nano version - hardware friendly)
    print("\n[1] Loading YOLOv8n model (nano version)...")
    model = YOLO('yolov8n.pt')

    # Get the PyTorch model
    pt_model = model.model

    print(f"Model loaded successfully")
    print(f"Model type: {type(pt_model)}")

    # Analyze model structure
    print("\n[2] Analyzing model structure...")

    layer_info = []
    layer_count = defaultdict(int)

    def get_layer_info(module, name, input_shape=None):
        """Extract layer information"""
        info = {
            'name': name,
            'type': module.__class__.__name__,
            'params': sum(p.numel() for p in module.parameters()),
        }

        # Convolution layers
        if isinstance(module, nn.Conv2d):
            info['kernel_size'] = module.kernel_size
            info['stride'] = module.stride
            info['padding'] = module.padding
            info['in_channels'] = module.in_channels
            info['out_channels'] = module.out_channels
            info['groups'] = module.groups
            info['bias'] = module.bias is not None
            info['dilation'] = module.dilation
            layer_count['Conv2d'] += 1

        # Batch Normalization
        elif isinstance(module, nn.BatchNorm2d):
            info['num_features'] = module.num_features
            layer_count['BatchNorm2d'] += 1

        # Activation functions
        elif isinstance(module, (nn.SiLU, nn.ReLU, nn.LeakyReLU, nn.Sigmoid)):
            layer_count[module.__class__.__name__] += 1

        # Pooling
        elif isinstance(module, (nn.MaxPool2d, nn.AvgPool2d)):
            info['kernel_size'] = module.kernel_size
            info['stride'] = module.stride
            layer_count[module.__class__.__name__] += 1

        # Upsample
        elif isinstance(module, nn.Upsample):
            info['scale_factor'] = module.scale_factor
            info['mode'] = module.mode
            layer_count['Upsample'] += 1

        return info

    # Recursive function to traverse all modules
    def traverse_model(module, prefix=''):
        for name, child in module.named_children():
            full_name = f"{prefix}.{name}" if prefix else name
            info = get_layer_info(child, full_name)

            # Only add leaf modules or important containers
            if len(list(child.children())) == 0:
                layer_info.append(info)

            # Recurse
            traverse_model(child, full_name)

    traverse_model(pt_model)

    # Print layer count summary
    print("\n[3] Layer Type Summary:")
    print("-" * 40)
    for layer_type, count in sorted(layer_count.items()):
        print(f"  {layer_type:20s}: {count:4d}")
    print("-" * 40)
    print(f"  {'TOTAL':20s}: {sum(layer_count.values()):4d}")

    return pt_model, layer_info, layer_count


def detailed_conv_analysis(model):
    """
    Detailed analysis of convolution layers - the dominant operation.
    """
    print("\n" + "=" * 60)
    print("DETAILED CONVOLUTION ANALYSIS")
    print("=" * 60)

    conv_layers = []

    def find_convs(module, prefix=''):
        for name, child in module.named_children():
            full_name = f"{prefix}.{name}" if prefix else name
            if isinstance(child, nn.Conv2d):
                conv_layers.append({
                    'name': full_name,
                    'in_ch': child.in_channels,
                    'out_ch': child.out_channels,
                    'kernel': child.kernel_size,
                    'stride': child.stride,
                    'padding': child.padding,
                    'groups': child.groups,
                    'params': child.weight.numel() + (child.bias.numel() if child.bias is not None else 0),
                    'is_depthwise': child.groups == child.in_channels and child.groups > 1,
                    'is_pointwise': child.kernel_size == (1, 1),
                })
            find_convs(child, full_name)

    find_convs(model)

    # Categorize convolutions
    conv_types = defaultdict(list)
    for conv in conv_layers:
        if conv['is_depthwise']:
            conv_types['Depthwise'].append(conv)
        elif conv['is_pointwise']:
            conv_types['Pointwise (1x1)'].append(conv)
        else:
            k = conv['kernel']
            conv_types[f'Standard {k[0]}x{k[1]}'].append(conv)

    print("\n[4] Convolution Types Distribution:")
    print("-" * 50)
    for conv_type, convs in sorted(conv_types.items()):
        total_params = sum(c['params'] for c in convs)
        print(f"  {conv_type:25s}: {len(convs):3d} layers, {total_params:,} params")

    # Kernel size analysis
    print("\n[5] Kernel Size Analysis:")
    print("-" * 50)
    kernel_sizes = defaultdict(int)
    for conv in conv_layers:
        kernel_sizes[conv['kernel']] += 1
    for k, count in sorted(kernel_sizes.items()):
        print(f"  Kernel {k[0]}x{k[1]}: {count} layers")

    # Stride analysis
    print("\n[6] Stride Analysis:")
    print("-" * 50)
    strides = defaultdict(int)
    for conv in conv_layers:
        strides[conv['stride']] += 1
    for s, count in sorted(strides.items()):
        print(f"  Stride {s}: {count} layers")

    return conv_layers


def trace_model_shapes(model):
    """
    Trace the model with a sample input to get actual tensor shapes.
    """
    print("\n" + "=" * 60)
    print("LAYER-BY-LAYER SHAPE ANALYSIS")
    print("=" * 60)

    # Create sample input (640x640 is standard YOLO input)
    sample_input = torch.randn(1, 3, 640, 640)

    layer_shapes = []
    hooks = []

    def hook_fn(name):
        def hook(module, input, output):
            input_shape = input[0].shape if isinstance(input, tuple) and len(input) > 0 else "N/A"
            output_shape = output.shape if isinstance(output, torch.Tensor) else "Complex"

            layer_shapes.append({
                'name': name,
                'type': module.__class__.__name__,
                'input_shape': str(input_shape),
                'output_shape': str(output_shape),
            })
        return hook

    # Register hooks on important layers
    def register_hooks(module, prefix=''):
        for name, child in module.named_children():
            full_name = f"{prefix}.{name}" if prefix else name
            if isinstance(child, (nn.Conv2d, nn.BatchNorm2d, nn.MaxPool2d,
                                  nn.Upsample, nn.SiLU)):
                hooks.append(child.register_forward_hook(hook_fn(full_name)))
            register_hooks(child, full_name)

    register_hooks(model)

    # Run forward pass
    with torch.no_grad():
        try:
            _ = model(sample_input)
        except:
            print("Note: Forward pass may have issues, using partial trace")

    # Remove hooks
    for h in hooks:
        h.remove()

    return layer_shapes


def compute_operations_analysis(conv_layers, input_size=640):
    """
    Analyze computational requirements (MACs/FLOPs).
    """
    print("\n" + "=" * 60)
    print("COMPUTATIONAL REQUIREMENTS ANALYSIS")
    print("=" * 60)

    total_macs = 0
    layer_macs = []

    # Simplified MAC calculation for each conv layer
    # MACs = Output_H * Output_W * Kernel_H * Kernel_W * In_Ch * Out_Ch / Groups

    current_h, current_w = input_size, input_size

    print("\n[7] Top 10 Computationally Intensive Layers:")
    print("-" * 70)
    print(f"{'Layer':<35} {'Shape':<15} {'MACs':>15}")
    print("-" * 70)

    for conv in conv_layers:
        k_h, k_w = conv['kernel']
        s_h, s_w = conv['stride']
        in_ch = conv['in_ch']
        out_ch = conv['out_ch']
        groups = conv['groups']

        # Calculate output size (simplified)
        out_h = current_h // s_h
        out_w = current_w // s_w

        # MACs for this layer
        macs = out_h * out_w * k_h * k_w * in_ch * out_ch // groups
        total_macs += macs

        layer_macs.append({
            'name': conv['name'],
            'macs': macs,
            'shape': f"{in_ch}->{out_ch}",
        })

        # Update current size for next layer (simplified tracking)
        if s_h > 1:
            current_h = out_h
            current_w = out_w

    # Sort by MACs and show top 10
    layer_macs.sort(key=lambda x: x['macs'], reverse=True)
    for layer in layer_macs[:10]:
        print(f"{layer['name']:<35} {layer['shape']:<15} {layer['macs']:>15,}")

    print("-" * 70)
    print(f"{'TOTAL MACs':<35} {'':<15} {total_macs:>15,}")
    print(f"{'TOTAL GMACs':<35} {'':<15} {total_macs/1e9:>15.2f}")

    return total_macs, layer_macs


def data_reuse_analysis():
    """
    Analyze data reuse patterns for DPU buffer design.
    """
    print("\n" + "=" * 60)
    print("DATA REUSE PATTERN ANALYSIS")
    print("=" * 60)

    print("""
[8] Data Reuse Patterns in YOLO Convolutions:

1. WEIGHT REUSE (across spatial positions):
   - Same kernel weights applied to all positions in feature map
   - Reuse factor = H_out × W_out
   - Critical for: Weight buffer design, broadcast architecture

2. INPUT REUSE (across output channels):
   - Same input pixel used for all output channel computations
   - Reuse factor = C_out
   - Critical for: Input buffer design, parallelism strategy

3. OUTPUT REUSE (accumulation):
   - Partial sums accumulated across input channels and kernel positions
   - Reuse factor = C_in × K_h × K_w
   - Critical for: Accumulator design, INT32 intermediate storage

4. SLIDING WINDOW REUSE:
   - Adjacent output positions share overlapping input regions
   - Reuse factor = K_h × K_w - stride (approximate)
   - Critical for: Line buffer design, streaming architecture

YOLO-Specific Observations:
- Dominant 3×3 convolutions have 9× weight reuse per position
- 1×1 pointwise convolutions have maximum weight reuse (no overlap waste)
- Stride-2 layers reduce spatial dimensions, changing reuse patterns
- Skip connections require buffering multiple resolution feature maps
""")


def generate_summary_table(conv_layers):
    """
    Generate the deliverable summary table.
    """
    print("\n" + "=" * 60)
    print("DELIVERABLE: LAYER-BY-LAYER SUMMARY TABLE")
    print("=" * 60)

    print(f"\n{'#':<4} {'Type':<12} {'In→Out Ch':<12} {'Kernel':<8} {'Stride':<8} {'Groups':<8} {'Params':<12}")
    print("-" * 72)

    for i, conv in enumerate(conv_layers[:30]):  # First 30 layers
        kernel_str = f"{conv['kernel'][0]}x{conv['kernel'][1]}"
        stride_str = f"{conv['stride'][0]}x{conv['stride'][1]}"
        ch_str = f"{conv['in_ch']:>3}→{conv['out_ch']:<3}"

        conv_type = "DW" if conv['is_depthwise'] else ("PW" if conv['is_pointwise'] else "STD")

        print(f"{i:<4} {conv_type:<12} {ch_str:<12} {kernel_str:<8} {stride_str:<8} {conv['groups']:<8} {conv['params']:<12,}")

    if len(conv_layers) > 30:
        print(f"... and {len(conv_layers) - 30} more layers")


def identify_dominant_operations(layer_count, conv_layers):
    """
    Identify dominant operations for DPU optimization.
    """
    print("\n" + "=" * 60)
    print("DOMINANT OPERATIONS IDENTIFICATION")
    print("=" * 60)

    print("""
[9] Dominant Operations in YOLOv8n (Priority Order):

1. CONVOLUTION (Highest Priority)
   - 3×3 standard convolutions: Most common, highest compute
   - 1×1 pointwise convolutions: Channel mixing, frequent
   - Depthwise convolutions: Present but less dominant in YOLOv8n

2. BATCH NORMALIZATION
   - Fused with Conv during inference
   - Per-channel scale and bias
   - Can be folded into convolution weights

3. ACTIVATION (SiLU/Swish)
   - SiLU(x) = x × sigmoid(x)
   - Applied after every Conv+BN
   - Can be approximated with LUT for hardware

4. ELEMENT-WISE OPERATIONS
   - Add (for skip connections)
   - Concatenation (for feature fusion)

5. UPSAMPLING
   - Nearest-neighbor (simple)
   - 2× scale factor typical

6. MAX POOLING
   - Spatial pyramid pooling
   - Various kernel sizes
""")

    # Statistics
    total_conv_params = sum(c['params'] for c in conv_layers)
    print(f"\nKey Statistics:")
    print(f"  - Total convolution parameters: {total_conv_params:,}")
    print(f"  - Number of Conv2d layers: {len(conv_layers)}")
    print(f"  - Average params per conv: {total_conv_params // len(conv_layers):,}")


def main():
    """Main analysis function"""

    # Run all analyses
    model, layer_info, layer_count = analyze_yolov5n()
    conv_layers = detailed_conv_analysis(model)
    total_macs, layer_macs = compute_operations_analysis(conv_layers)
    data_reuse_analysis()
    generate_summary_table(conv_layers)
    identify_dominant_operations(layer_count, conv_layers)

    # Save results to JSON for later phases
    results = {
        'model': 'YOLOv8n',
        'total_layers': sum(layer_count.values()),
        'layer_count': dict(layer_count),
        'total_macs': total_macs,
        'conv_layers': conv_layers,
        'dominant_operations': [
            'Conv2d (3x3)',
            'Conv2d (1x1)',
            'BatchNorm2d (fused)',
            'SiLU activation',
            'Element-wise Add',
            'Concatenation',
            'Upsample',
            'MaxPool2d'
        ]
    }

    with open('phase1_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print("\n" + "=" * 60)
    print("PHASE 1 COMPLETE")
    print("=" * 60)
    print("Results saved to: phase1_results.json")
    print("\nReady to proceed to PHASE 2: DPU Primitives Identification")

    return results


if __name__ == "__main__":
    main()
