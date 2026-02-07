"""
PHASE 3: Python Functional DPU Model
=====================================
Functional implementation of DPU primitives in Python.
This serves as the golden reference for RTL verification.

All operations use INT8/INT32 arithmetic to match hardware behavior.
"""

import numpy as np
from typing import Tuple, Optional
import json

# =============================================================================
# DATA TYPE DEFINITIONS
# =============================================================================

# Quantization parameters
INT8_MIN = -128
INT8_MAX = 127
INT32_MIN = -2147483648
INT32_MAX = 2147483647

def clip_int8(x: np.ndarray) -> np.ndarray:
    """Clip values to INT8 range"""
    return np.clip(x, INT8_MIN, INT8_MAX).astype(np.int8)

def clip_int32(x: np.ndarray) -> np.ndarray:
    """Clip values to INT32 range"""
    return np.clip(x, INT32_MIN, INT32_MAX).astype(np.int32)


# =============================================================================
# PRIMITIVE 1: MAC (Multiply-Accumulate)
# =============================================================================

def mac(weight: np.int8, activation: np.int8, accumulator: np.int32) -> np.int32:
    """
    Single MAC operation: acc = acc + weight * activation

    Args:
        weight: INT8 weight value
        activation: INT8 activation value
        accumulator: INT32 accumulated sum

    Returns:
        Updated INT32 accumulator
    """
    # Multiply: INT8 x INT8 -> INT16 (but we use INT32 for safety)
    product = np.int32(weight) * np.int32(activation)
    # Accumulate: INT32 + INT32 -> INT32
    result = accumulator + product
    return np.int32(result)


def mac_array(weights: np.ndarray, activations: np.ndarray) -> np.int32:
    """
    Perform multiple MAC operations (dot product)

    Args:
        weights: INT8 array of weights
        activations: INT8 array of activations

    Returns:
        INT32 accumulated sum
    """
    # Convert to INT32 for computation
    w = weights.astype(np.int32)
    a = activations.astype(np.int32)
    # Dot product with accumulation
    result = np.sum(w * a)
    return np.int32(result)


# =============================================================================
# PRIMITIVE 2: Convolution 3x3
# =============================================================================

def conv2d_3x3(
    input_fm: np.ndarray,      # INT8 [C_in, H, W]
    weights: np.ndarray,        # INT8 [C_out, C_in, 3, 3]
    bias: np.ndarray,           # INT32 [C_out]
    stride: int = 1,
    padding: int = 1
) -> np.ndarray:                # INT32 [C_out, H_out, W_out]
    """
    3x3 Convolution with INT8 inputs and INT32 accumulation.

    Args:
        input_fm: Input feature map [C_in, H, W] as INT8
        weights: Convolution weights [C_out, C_in, 3, 3] as INT8
        bias: Bias values [C_out] as INT32 (fused BatchNorm)
        stride: Stride (1 or 2)
        padding: Padding (typically 1 for 3x3)

    Returns:
        Output feature map [C_out, H_out, W_out] as INT32
    """
    C_in, H, W = input_fm.shape
    C_out = weights.shape[0]

    # Pad input
    if padding > 0:
        input_padded = np.pad(
            input_fm,
            ((0, 0), (padding, padding), (padding, padding)),
            mode='constant',
            constant_values=0
        )
    else:
        input_padded = input_fm

    # Output dimensions
    H_out = (H + 2*padding - 3) // stride + 1
    W_out = (W + 2*padding - 3) // stride + 1

    # Allocate output (INT32 for accumulation)
    output = np.zeros((C_out, H_out, W_out), dtype=np.int32)

    # Convolution loop (functional reference, not optimized)
    for c_out in range(C_out):
        for h_out in range(H_out):
            for w_out in range(W_out):
                # Input window position
                h_in = h_out * stride
                w_in = w_out * stride

                # Extract 3x3 window for all input channels
                window = input_padded[:, h_in:h_in+3, w_in:w_in+3]

                # MAC operation: dot product
                acc = mac_array(
                    weights[c_out].flatten(),
                    window.flatten()
                )

                # Add bias
                acc = acc + bias[c_out]

                output[c_out, h_out, w_out] = acc

    return output


# =============================================================================
# PRIMITIVE 3: Convolution 1x1
# =============================================================================

def conv2d_1x1(
    input_fm: np.ndarray,      # INT8 [C_in, H, W]
    weights: np.ndarray,        # INT8 [C_out, C_in, 1, 1]
    bias: np.ndarray            # INT32 [C_out]
) -> np.ndarray:                # INT32 [C_out, H, W]
    """
    1x1 Convolution (pointwise) with INT8 inputs and INT32 accumulation.

    Args:
        input_fm: Input feature map [C_in, H, W] as INT8
        weights: Convolution weights [C_out, C_in, 1, 1] as INT8
        bias: Bias values [C_out] as INT32

    Returns:
        Output feature map [C_out, H, W] as INT32
    """
    C_in, H, W = input_fm.shape
    C_out = weights.shape[0]

    # Reshape weights for matrix multiplication
    w = weights.reshape(C_out, C_in).astype(np.int32)

    # Reshape input for matrix multiplication
    x = input_fm.reshape(C_in, H * W).astype(np.int32)

    # Matrix multiplication
    output = np.dot(w, x)  # [C_out, H*W]

    # Add bias (broadcast)
    output = output + bias.reshape(C_out, 1)

    # Reshape back to feature map
    output = output.reshape(C_out, H, W)

    return output.astype(np.int32)


# =============================================================================
# PRIMITIVE 4: LeakyReLU
# =============================================================================

def leaky_relu(
    x: np.ndarray,              # INT32 input
    alpha_shift: int = 3,       # Right shift for alpha approximation (1/8)
    use_exact: bool = False,    # Use exact 0.1 or shift approximation
    alpha: float = 0.1
) -> np.ndarray:                # INT32 output
    """
    LeakyReLU activation: y = x if x > 0 else alpha * x

    For hardware, we approximate alpha=0.1 as 1/8 using right shift by 3.

    Args:
        x: Input values (INT32)
        alpha_shift: Right shift amount for approximation (3 -> 1/8)
        use_exact: If True, use exact alpha multiplication
        alpha: Exact alpha value (only used if use_exact=True)

    Returns:
        Activated values (INT32)
    """
    if use_exact:
        # Exact computation (for reference)
        negative_part = (x * alpha).astype(np.int32)
    else:
        # Hardware approximation: x >> 3 (divide by 8)
        # Arithmetic right shift preserves sign
        negative_part = x >> alpha_shift

    # Select: x if x > 0 else negative_part
    output = np.where(x > 0, x, negative_part)

    return output.astype(np.int32)


def leaky_relu_hardware(x: np.ndarray) -> np.ndarray:
    """
    Hardware-accurate LeakyReLU using shift instead of multiply.
    This exactly matches what the RTL will do.
    """
    # Arithmetic right shift by 3 (equivalent to divide by 8, preserving sign)
    negative_scaled = np.right_shift(x.astype(np.int32), 3)
    # MUX: select based on sign
    return np.where(x > 0, x, negative_scaled).astype(np.int32)


# =============================================================================
# PRIMITIVE 5: Requantization
# =============================================================================

def requantize(
    acc: np.ndarray,            # INT32 accumulator values
    scale: float,               # Quantization scale
    zero_point: int = 0         # Zero point offset
) -> np.ndarray:                # INT8 output
    """
    Requantize INT32 accumulator to INT8.

    Args:
        acc: Accumulated values (INT32)
        scale: Scale factor (typically < 1)
        zero_point: Zero point offset

    Returns:
        Quantized values (INT8)
    """
    # Scale and add zero point
    scaled = np.round(acc.astype(np.float64) * scale + zero_point)
    # Clip to INT8 range
    output = np.clip(scaled, INT8_MIN, INT8_MAX)
    return output.astype(np.int8)


def requantize_fixed_point(
    acc: np.ndarray,            # INT32 accumulator values
    multiplier: np.int32,       # Fixed-point multiplier
    shift: int                  # Right shift amount
) -> np.ndarray:                # INT8 output
    """
    Hardware-friendly requantization using fixed-point arithmetic.

    output = (acc * multiplier) >> shift

    Args:
        acc: Accumulated values (INT32)
        multiplier: Fixed-point scale multiplier
        shift: Right shift amount

    Returns:
        Quantized values (INT8)
    """
    # Multiply (produces 64-bit result)
    product = acc.astype(np.int64) * np.int64(multiplier)
    # Right shift
    shifted = product >> shift
    # Clip to INT8
    output = np.clip(shifted, INT8_MIN, INT8_MAX)
    return output.astype(np.int8)


# =============================================================================
# PRIMITIVE 6: MaxPool 2x2
# =============================================================================

def maxpool_2x2(
    input_fm: np.ndarray,       # INT8 [C, H, W]
    stride: int = 2
) -> np.ndarray:                # INT8 [C, H/2, W/2]
    """
    2x2 Max Pooling with stride 2.

    Args:
        input_fm: Input feature map [C, H, W] as INT8
        stride: Stride (typically 2)

    Returns:
        Pooled feature map [C, H/2, W/2] as INT8
    """
    C, H, W = input_fm.shape
    H_out = H // stride
    W_out = W // stride

    output = np.zeros((C, H_out, W_out), dtype=np.int8)

    for c in range(C):
        for h in range(H_out):
            for w in range(W_out):
                # Extract 2x2 window
                h_in = h * stride
                w_in = w * stride
                window = input_fm[c, h_in:h_in+2, w_in:w_in+2]
                # Max operation
                output[c, h, w] = np.max(window)

    return output


# =============================================================================
# PRIMITIVE 7: Route / Concatenation
# =============================================================================

def route_concat(
    fm_list: list               # List of feature maps [C_i, H, W]
) -> np.ndarray:                # Concatenated [sum(C_i), H, W]
    """
    Concatenate feature maps along channel dimension.

    Args:
        fm_list: List of feature maps to concatenate

    Returns:
        Concatenated feature map
    """
    return np.concatenate(fm_list, axis=0)


def route_split(
    input_fm: np.ndarray,       # [C, H, W]
    groups: int = 2,
    group_id: int = 0
) -> np.ndarray:                # [C/groups, H, W]
    """
    Split feature map and take one group.

    Args:
        input_fm: Input feature map
        groups: Number of groups to split into
        group_id: Which group to take (0-indexed)

    Returns:
        Selected group of channels
    """
    C = input_fm.shape[0]
    channels_per_group = C // groups
    start = group_id * channels_per_group
    end = start + channels_per_group
    return input_fm[start:end]


# =============================================================================
# PRIMITIVE 8: Upsample 2x (Nearest Neighbor)
# =============================================================================

def upsample_2x(
    input_fm: np.ndarray        # [C, H, W]
) -> np.ndarray:                # [C, 2H, 2W]
    """
    2x nearest neighbor upsampling.

    Args:
        input_fm: Input feature map

    Returns:
        Upsampled feature map (2x in each spatial dimension)
    """
    C, H, W = input_fm.shape
    output = np.zeros((C, H*2, W*2), dtype=input_fm.dtype)

    # Duplicate each pixel to 2x2 block
    for c in range(C):
        for h in range(H):
            for w in range(W):
                output[c, h*2:h*2+2, w*2:w*2+2] = input_fm[c, h, w]

    return output


# =============================================================================
# COMPOSITE: Conv + BN + LeakyReLU + Requant
# =============================================================================

def conv_bn_leaky(
    input_fm: np.ndarray,       # INT8 [C_in, H, W]
    weights: np.ndarray,        # INT8 [C_out, C_in, K, K]
    bias: np.ndarray,           # INT32 [C_out] (fused BN)
    scale: float,               # Requantization scale
    stride: int = 1,
    kernel_size: int = 3
) -> np.ndarray:                # INT8 [C_out, H_out, W_out]
    """
    Complete convolution block: Conv + (fused BN) + LeakyReLU + Requantize

    This is the most common operation in YOLO.
    """
    # Step 1: Convolution (produces INT32)
    if kernel_size == 3:
        conv_out = conv2d_3x3(input_fm, weights, bias, stride=stride)
    else:  # kernel_size == 1
        conv_out = conv2d_1x1(input_fm, weights, bias)

    # Step 2: LeakyReLU (INT32 -> INT32)
    relu_out = leaky_relu_hardware(conv_out)

    # Step 3: Requantize (INT32 -> INT8)
    output = requantize(relu_out, scale)

    return output


# =============================================================================
# VALIDATION FUNCTIONS
# =============================================================================

def validate_mac():
    """Validate MAC operation"""
    print("\n[TEST] MAC Operation")
    print("-" * 40)

    # Test cases
    test_cases = [
        (10, 20, 0),       # Simple positive
        (-10, 20, 0),      # Negative weight
        (127, 127, 0),     # Max positive
        (-128, 127, 0),    # Max negative
        (50, 50, 1000),    # With accumulator
    ]

    all_pass = True
    for w, a, acc in test_cases:
        result = mac(np.int8(w), np.int8(a), np.int32(acc))
        expected = acc + w * a
        passed = result == expected
        all_pass = all_pass and passed
        status = "PASS" if passed else "FAIL"
        print(f"  MAC({w:4d}, {a:4d}, {acc:6d}) = {result:8d}  [{status}]")

    return all_pass


def validate_leaky_relu():
    """Validate LeakyReLU operation"""
    print("\n[TEST] LeakyReLU Operation")
    print("-" * 40)

    # Test values
    x = np.array([-80, -40, -8, 0, 8, 40, 80], dtype=np.int32)

    # Hardware version (shift by 3)
    y_hw = leaky_relu_hardware(x)

    # Expected: positive values unchanged, negative divided by 8
    expected = np.array([-10, -5, -1, 0, 8, 40, 80], dtype=np.int32)

    print(f"  Input:    {x}")
    print(f"  Output:   {y_hw}")
    print(f"  Expected: {expected}")

    passed = np.array_equal(y_hw, expected)
    print(f"  Status:   {'PASS' if passed else 'FAIL'}")

    return passed


def validate_conv3x3():
    """Validate 3x3 convolution"""
    print("\n[TEST] Conv 3x3 Operation")
    print("-" * 40)

    # Simple test: 1 input channel, 1 output channel, 4x4 input
    C_in, H, W = 1, 4, 4
    C_out = 1

    # All-ones input
    input_fm = np.ones((C_in, H, W), dtype=np.int8) * 10

    # All-ones weights
    weights = np.ones((C_out, C_in, 3, 3), dtype=np.int8)

    # Zero bias
    bias = np.zeros(C_out, dtype=np.int32)

    # Compute
    output = conv2d_3x3(input_fm, weights, bias, stride=1, padding=1)

    # Expected: center pixels should be 10*9=90 (full overlap)
    print(f"  Input shape:  {input_fm.shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  Center value: {output[0, 1, 1]} (expected: 90)")

    passed = output[0, 1, 1] == 90
    print(f"  Status:       {'PASS' if passed else 'FAIL'}")

    return passed


def validate_maxpool():
    """Validate MaxPool 2x2"""
    print("\n[TEST] MaxPool 2x2 Operation")
    print("-" * 40)

    # 1 channel, 4x4 input
    input_fm = np.array([
        [[1, 2, 3, 4],
         [5, 6, 7, 8],
         [9, 10, 11, 12],
         [13, 14, 15, 16]]
    ], dtype=np.int8)

    output = maxpool_2x2(input_fm)

    expected = np.array([
        [[6, 8],
         [14, 16]]
    ], dtype=np.int8)

    print(f"  Input:\n{input_fm[0]}")
    print(f"  Output:\n{output[0]}")
    print(f"  Expected:\n{expected[0]}")

    passed = np.array_equal(output, expected)
    print(f"  Status: {'PASS' if passed else 'FAIL'}")

    return passed


def validate_requantize():
    """Validate requantization"""
    print("\n[TEST] Requantization Operation")
    print("-" * 40)

    # INT32 accumulator values
    acc = np.array([-1000, -100, 0, 100, 1000], dtype=np.int32)

    # Scale factor
    scale = 0.1

    output = requantize(acc, scale)
    expected = np.array([-100, -10, 0, 10, 100], dtype=np.int8)

    print(f"  Input (INT32): {acc}")
    print(f"  Scale:         {scale}")
    print(f"  Output (INT8): {output}")
    print(f"  Expected:      {expected}")

    passed = np.array_equal(output, expected)
    print(f"  Status:        {'PASS' if passed else 'FAIL'}")

    return passed


def validate_full_layer():
    """Validate complete layer: Conv + BN + LeakyReLU + Requant"""
    print("\n[TEST] Full Layer (Conv+BN+LeakyReLU+Requant)")
    print("-" * 40)

    # Create test input
    C_in, H, W = 3, 8, 8
    C_out = 16

    np.random.seed(42)
    input_fm = np.random.randint(-50, 50, (C_in, H, W), dtype=np.int8)
    weights = np.random.randint(-20, 20, (C_out, C_in, 3, 3), dtype=np.int8)
    bias = np.random.randint(-100, 100, C_out, dtype=np.int32)
    scale = 0.01

    # Run full layer
    output = conv_bn_leaky(input_fm, weights, bias, scale, stride=1, kernel_size=3)

    print(f"  Input shape:  {input_fm.shape}")
    print(f"  Weight shape: {weights.shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  Output dtype: {output.dtype}")
    print(f"  Output range: [{output.min()}, {output.max()}]")

    # Check output is valid INT8
    valid_range = (output.min() >= INT8_MIN) and (output.max() <= INT8_MAX)
    valid_shape = output.shape == (C_out, H, W)

    passed = valid_range and valid_shape
    print(f"  Status:       {'PASS' if passed else 'FAIL'}")

    return passed


# =============================================================================
# YOLO LAYER SIMULATION
# =============================================================================

def simulate_yolov4_tiny_first_layers():
    """
    Simulate the first few layers of YOLOv4-tiny to validate the DPU model.
    """
    print("\n" + "=" * 70)
    print("YOLOV4-TINY LAYER SIMULATION")
    print("=" * 70)

    np.random.seed(42)

    # Layer 0: Conv 3x3, 3->32, stride 2
    print("\n[Layer 0] Conv 3x3, 3->32, stride=2")
    input_img = np.random.randint(-128, 127, (3, 416, 416), dtype=np.int8)
    weights_0 = np.random.randint(-30, 30, (32, 3, 3, 3), dtype=np.int8)
    bias_0 = np.random.randint(-500, 500, 32, dtype=np.int32)

    layer0_out = conv_bn_leaky(input_img, weights_0, bias_0, scale=0.01, stride=2)
    print(f"  Input:  {input_img.shape} -> Output: {layer0_out.shape}")
    assert layer0_out.shape == (32, 208, 208), "Layer 0 shape mismatch"
    print("  [PASS]")

    # Layer 1: Conv 3x3, 32->64, stride 2
    print("\n[Layer 1] Conv 3x3, 32->64, stride=2")
    weights_1 = np.random.randint(-30, 30, (64, 32, 3, 3), dtype=np.int8)
    bias_1 = np.random.randint(-500, 500, 64, dtype=np.int32)

    layer1_out = conv_bn_leaky(layer0_out, weights_1, bias_1, scale=0.01, stride=2)
    print(f"  Input:  {layer0_out.shape} -> Output: {layer1_out.shape}")
    assert layer1_out.shape == (64, 104, 104), "Layer 1 shape mismatch"
    print("  [PASS]")

    # Layer 2: Conv 3x3, 64->64, stride 1
    print("\n[Layer 2] Conv 3x3, 64->64, stride=1")
    weights_2 = np.random.randint(-30, 30, (64, 64, 3, 3), dtype=np.int8)
    bias_2 = np.random.randint(-500, 500, 64, dtype=np.int32)

    layer2_out = conv_bn_leaky(layer1_out, weights_2, bias_2, scale=0.01, stride=1)
    print(f"  Input:  {layer1_out.shape} -> Output: {layer2_out.shape}")
    assert layer2_out.shape == (64, 104, 104), "Layer 2 shape mismatch"
    print("  [PASS]")

    # Layer 3: Route split (groups=2, group_id=1)
    print("\n[Layer 3] Route split (groups=2, take second half)")
    layer3_out = route_split(layer2_out, groups=2, group_id=1)
    print(f"  Input:  {layer2_out.shape} -> Output: {layer3_out.shape}")
    assert layer3_out.shape == (32, 104, 104), "Layer 3 shape mismatch"
    print("  [PASS]")

    # Layer 4: Conv 3x3, 32->32, stride 1
    print("\n[Layer 4] Conv 3x3, 32->32, stride=1")
    weights_4 = np.random.randint(-30, 30, (32, 32, 3, 3), dtype=np.int8)
    bias_4 = np.random.randint(-500, 500, 32, dtype=np.int32)

    layer4_out = conv_bn_leaky(layer3_out, weights_4, bias_4, scale=0.01, stride=1)
    print(f"  Input:  {layer3_out.shape} -> Output: {layer4_out.shape}")
    assert layer4_out.shape == (32, 104, 104), "Layer 4 shape mismatch"
    print("  [PASS]")

    # Layer 5: Conv 3x3, 32->32, stride 1
    print("\n[Layer 5] Conv 3x3, 32->32, stride=1")
    weights_5 = np.random.randint(-30, 30, (32, 32, 3, 3), dtype=np.int8)
    bias_5 = np.random.randint(-500, 500, 32, dtype=np.int32)

    layer5_out = conv_bn_leaky(layer4_out, weights_5, bias_5, scale=0.01, stride=1)
    print(f"  Input:  {layer4_out.shape} -> Output: {layer5_out.shape}")
    assert layer5_out.shape == (32, 104, 104), "Layer 5 shape mismatch"
    print("  [PASS]")

    # Layer 6: Route concat (layer5 + layer4)
    print("\n[Layer 6] Route concat (layer5 + layer4)")
    layer6_out = route_concat([layer5_out, layer4_out])
    print(f"  Input:  {layer5_out.shape} + {layer4_out.shape} -> Output: {layer6_out.shape}")
    assert layer6_out.shape == (64, 104, 104), "Layer 6 shape mismatch"
    print("  [PASS]")

    # Layer 7: Conv 1x1, 64->64
    print("\n[Layer 7] Conv 1x1, 64->64")
    weights_7 = np.random.randint(-30, 30, (64, 64, 1, 1), dtype=np.int8)
    bias_7 = np.random.randint(-500, 500, 64, dtype=np.int32)

    layer7_out = conv_bn_leaky(layer6_out, weights_7, bias_7, scale=0.01, stride=1, kernel_size=1)
    print(f"  Input:  {layer6_out.shape} -> Output: {layer7_out.shape}")
    assert layer7_out.shape == (64, 104, 104), "Layer 7 shape mismatch"
    print("  [PASS]")

    # Layer 8: Route concat (layer2 + layer7)
    print("\n[Layer 8] Route concat (layer2 + layer7)")
    layer8_out = route_concat([layer2_out, layer7_out])
    print(f"  Input:  {layer2_out.shape} + {layer7_out.shape} -> Output: {layer8_out.shape}")
    assert layer8_out.shape == (128, 104, 104), "Layer 8 shape mismatch"
    print("  [PASS]")

    # Layer 9: MaxPool 2x2
    print("\n[Layer 9] MaxPool 2x2")
    layer9_out = maxpool_2x2(layer8_out)
    print(f"  Input:  {layer8_out.shape} -> Output: {layer9_out.shape}")
    assert layer9_out.shape == (128, 52, 52), "Layer 9 shape mismatch"
    print("  [PASS]")

    print("\n" + "=" * 70)
    print("FIRST 10 LAYERS OF YOLOV4-TINY: ALL PASS")
    print("=" * 70)

    return True


# =============================================================================
# MAIN
# =============================================================================

def run_all_validations():
    """Run all validation tests"""
    print("=" * 70)
    print("PHASE 3: DPU FUNCTIONAL MODEL VALIDATION")
    print("=" * 70)

    results = {
        'MAC': validate_mac(),
        'LeakyReLU': validate_leaky_relu(),
        'Conv3x3': validate_conv3x3(),
        'MaxPool': validate_maxpool(),
        'Requantize': validate_requantize(),
        'FullLayer': validate_full_layer(),
    }

    print("\n" + "=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)

    all_pass = True
    for name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {name:<20}: [{status}]")
        all_pass = all_pass and passed

    print("-" * 70)
    print(f"  {'OVERALL':<20}: [{'PASS' if all_pass else 'FAIL'}]")

    # Run YOLO simulation
    yolo_pass = simulate_yolov4_tiny_first_layers()

    # Save results
    results['YOLO_Simulation'] = bool(yolo_pass)
    results['all_pass'] = bool(all_pass and yolo_pass)

    # Convert all values to Python bool for JSON serialization
    results = {k: bool(v) for k, v in results.items()}

    with open('phase3_validation_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 70)
    print("PHASE 3 COMPLETE")
    print("=" * 70)
    print("Results saved to: phase3_validation_results.json")
    print("\nReady for PHASE 4: Architecture Exploration")

    return all_pass and yolo_pass


if __name__ == "__main__":
    run_all_validations()
