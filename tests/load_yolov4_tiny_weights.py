#!/usr/bin/env python3
"""
Load real YOLOv4-tiny darknet weights, fold batch normalization,
quantize to INT8/INT32, and export for the DPU golden model.

Downloads yolov4-tiny.weights if not present, parses the darknet binary
format, folds BN parameters into conv weights, and quantizes.

Output: image_sim_out/dpu_top_real/ directory with hex files.

Usage: python tests/load_yolov4_tiny_weights.py
"""

import sys
import struct
from pathlib import Path
import numpy as np

SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

WEIGHTS_URL = "https://github.com/AlexeyAB/darknet/releases/download/yolov4/yolov4-tiny.weights"
WEIGHTS_FILE = PROJECT_ROOT / "yolov4-tiny.weights"

# DPU conv layer definitions (index in our 18-layer sequence)
# Each: (dpu_layer, c_in, c_out, kernel, stride, has_bn)
# These are the conv layers in YOLOv4-tiny backbone (first 18 darknet layers)
CONV_LAYERS = [
    (0,   3,  32, 3, 2, True),
    (1,  32,  64, 3, 2, True),
    (2,  64,  64, 3, 1, True),
    # layer 3 = route split (no weights)
    (4,  32,  32, 3, 1, True),
    (5,  32,  32, 3, 1, True),
    # layer 6 = route concat (no weights)
    (7,  64,  64, 1, 1, True),
    # layer 8 = route concat (no weights)
    # layer 9 = maxpool (no weights)
    (10, 128, 128, 3, 1, True),
    # layer 11 = route split (no weights)
    (12,  64,  64, 3, 1, True),
    (13,  64,  64, 3, 1, True),
    # layer 14 = route concat (no weights)
    (15, 128, 128, 1, 1, True),
    # layer 16 = route concat (no weights)
    # layer 17 = maxpool (no weights)
]


def download_weights():
    """Download YOLOv4-tiny weights if not present."""
    if WEIGHTS_FILE.exists():
        print(f"[OK] Weights file exists: {WEIGHTS_FILE}")
        return True

    print(f"[DOWNLOAD] Downloading YOLOv4-tiny weights...")
    try:
        import urllib.request
        urllib.request.urlretrieve(WEIGHTS_URL, str(WEIGHTS_FILE))
        print(f"[OK] Downloaded to {WEIGHTS_FILE} ({WEIGHTS_FILE.stat().st_size} bytes)")
        return True
    except Exception as e:
        print(f"[ERROR] Failed to download: {e}")
        print(f"Please manually download from: {WEIGHTS_URL}")
        print(f"And place at: {WEIGHTS_FILE}")
        return False


def parse_darknet_weights(filepath):
    """
    Parse darknet .weights binary file.
    Returns list of layer weight dicts in order.
    """
    with open(filepath, 'rb') as f:
        # Header: major, minor, revision (3 x int32) + images_seen (int64)
        major, minor, revision = struct.unpack('3i', f.read(12))
        if major * 10 + minor >= 2:
            seen = struct.unpack('Q', f.read(8))[0]
        else:
            seen = struct.unpack('I', f.read(4))[0]

        print(f"[PARSE] Darknet weights v{major}.{minor}.{revision}, seen={seen}")

        layers = []
        for dpu_idx, c_in, c_out, kernel, stride, has_bn in CONV_LAYERS:
            n_weights = c_out * c_in * kernel * kernel

            if has_bn:
                # Order: biases, bn_scales, bn_running_mean, bn_running_var, weights
                biases = np.frombuffer(f.read(c_out * 4), dtype=np.float32).copy()
                bn_scales = np.frombuffer(f.read(c_out * 4), dtype=np.float32).copy()
                bn_means = np.frombuffer(f.read(c_out * 4), dtype=np.float32).copy()
                bn_vars = np.frombuffer(f.read(c_out * 4), dtype=np.float32).copy()
                weights = np.frombuffer(f.read(n_weights * 4), dtype=np.float32).copy()
            else:
                biases = np.frombuffer(f.read(c_out * 4), dtype=np.float32).copy()
                bn_scales = None
                bn_means = None
                bn_vars = None
                weights = np.frombuffer(f.read(n_weights * 4), dtype=np.float32).copy()

            weights = weights.reshape(c_out, c_in, kernel, kernel)

            layers.append({
                'dpu_idx': dpu_idx,
                'c_in': c_in, 'c_out': c_out,
                'kernel': kernel, 'stride': stride,
                'weights': weights, 'biases': biases,
                'bn_scales': bn_scales, 'bn_means': bn_means, 'bn_vars': bn_vars,
            })

        remaining = len(f.read())
        print(f"[PARSE] Loaded {len(layers)} conv layers, {remaining} bytes remaining (detection head)")

    return layers


def fold_bn(layer, eps=1e-5):
    """Fold batch normalization into conv weights and biases."""
    w = layer['weights']  # [c_out, c_in, k, k]
    b = layer['biases']   # [c_out]

    if layer['bn_scales'] is None:
        return w, b

    gamma = layer['bn_scales']  # [c_out]
    mean = layer['bn_means']    # [c_out]
    var = layer['bn_vars']      # [c_out]

    # Fold: w_new[co] = w[co] * gamma[co] / sqrt(var[co] + eps)
    #        b_new[co] = gamma[co] * (b[co] - mean[co]) / sqrt(var[co] + eps) + 0
    # (no beta in darknet BN, beta is absorbed into biases)
    std = np.sqrt(var + eps)
    scale = gamma / std

    w_folded = w * scale.reshape(-1, 1, 1, 1)
    b_folded = (b - mean) * scale

    return w_folded, b_folded


def quantize_symmetric_int8(tensor, clip_percentile=None):
    """
    Symmetric INT8 quantization.
    Returns (quantized_int8, scale_factor).
    """
    max_abs = np.max(np.abs(tensor))
    if max_abs == 0:
        return np.zeros_like(tensor, dtype=np.int8), 1.0

    scale = max_abs / 127.0
    quantized = np.round(tensor / scale)
    quantized = np.clip(quantized, -128, 127).astype(np.int8)
    return quantized, scale


def quantize_layer(w_float, b_float, input_scale=1.0):
    """
    Quantize conv layer weights and biases for INT8 inference.

    The DPU computes: acc = sum(w_int8[i] * x_int8[i]) + bias_int32
    In floating point: acc_float = sum(w_float[i] * x_float[i]) + b_float
    Relationship: w_int8 * x_int8 = (w_float / w_scale) * (x_float / x_scale)
                  So acc_int32 = acc_float / (w_scale * x_scale)
                  bias_int32 = b_float / (w_scale * x_scale)
    """
    w_int8, w_scale = quantize_symmetric_int8(w_float)

    # Bias scale is w_scale * input_scale
    bias_scale = w_scale * input_scale
    b_int32 = np.round(b_float / bias_scale).astype(np.int64)
    b_int32 = np.clip(b_int32, -2**31, 2**31 - 1).astype(np.int32)

    # Output scale: one INT32 accumulator unit = w_scale * input_scale
    output_scale = w_scale * input_scale

    return w_int8, b_int32, w_scale, output_scale


def compute_requant_scale(output_scale, target_output_scale, shift=16):
    """
    Compute the fixed-point requantization multiplier.
    req_scale = (output_scale / target_output_scale) * 2^shift
    """
    ratio = output_scale / target_output_scale
    multiplier = int(round(ratio * (1 << shift)))
    multiplier = max(0, min(multiplier, 65535))  # uint16
    return multiplier


def to_hex8(x):
    return f"{(int(x) & 0xff):02x}"


def write_hex8_file(path, data_flat):
    with open(path, 'w') as f:
        for v in data_flat:
            f.write(to_hex8(v) + '\n')


def write_bias_hex(path, bias_array):
    with open(path, 'w') as f:
        for b in bias_array:
            val = int(b) & 0xffffffff
            f.write(f"{(val >> 0) & 0xff:02x}\n")
            f.write(f"{(val >> 8) & 0xff:02x}\n")
            f.write(f"{(val >> 16) & 0xff:02x}\n")
            f.write(f"{(val >> 24) & 0xff:02x}\n")


def main():
    # Download weights
    if not download_weights():
        return 1

    # Parse darknet weights
    layers = parse_darknet_weights(WEIGHTS_FILE)

    # Fold BN into conv weights
    print("\n[FOLD BN] Folding batch normalization...")
    folded = []
    for layer in layers:
        w_folded, b_folded = fold_bn(layer)
        folded.append({
            'dpu_idx': layer['dpu_idx'],
            'c_in': layer['c_in'], 'c_out': layer['c_out'],
            'kernel': layer['kernel'], 'stride': layer['stride'],
            'w_float': w_folded,
            'b_float': b_folded,
        })
        print(f"  Layer {layer['dpu_idx']:2d}: w_range=[{w_folded.min():.4f}, {w_folded.max():.4f}]  "
              f"b_range=[{b_folded.min():.1f}, {b_folded.max():.1f}]")

    # Quantize
    print("\n[QUANTIZE] Symmetric INT8 quantization...")
    # For the first layer, input scale is based on image normalization
    # YOLOv4-tiny normalizes pixels to [0, 1], so pixel_scale ≈ 1/255
    # But our DPU expects INT8 input, so we normalize to [-128, 127]
    # Input scale = 1/127 (one INT8 unit = 1/127 in float)
    # For simplicity, we use scale=1.0 and just quantize everything independently

    quantized = {}
    output_scales = {}

    # Simple approach: quantize each layer independently with scale=1 for input
    # This won't give accurate detection but will verify hardware correctness
    input_scale = 1.0  # Start with 1.0, meaning INT8 input values are literal

    for layer in folded:
        idx = layer['dpu_idx']
        w_int8, b_int32, w_scale, output_scale = quantize_layer(
            layer['w_float'], layer['b_float'], input_scale
        )
        quantized[idx] = {'w': w_int8, 'b': b_int32}
        output_scales[idx] = output_scale

        print(f"  Layer {idx:2d}: w_scale={w_scale:.6f}  out_scale={output_scale:.6f}  "
              f"w_range=[{w_int8.min():4d},{w_int8.max():4d}]  "
              f"b_range=[{b_int32.min():6d},{b_int32.max():6d}]")

    # Export
    print("\n[EXPORT] Writing hex files...")
    out_dir = PROJECT_ROOT / "image_sim_out" / "dpu_top_real"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Write quantized weights and biases
    for idx, data in quantized.items():
        write_hex8_file(out_dir / f"layer{idx}_weights.hex", data['w'].flatten())
        write_bias_hex(out_dir / f"layer{idx}_bias.hex", data['b'])
        print(f"  layer{idx}_weights.hex ({data['w'].size} bytes)")
        print(f"  layer{idx}_bias.hex ({data['b'].size} x 4 bytes)")

    # Save quantized weights as numpy for the golden model
    np.savez(
        out_dir / "quantized_weights.npz",
        **{f"w{idx}": data['w'] for idx, data in quantized.items()},
        **{f"b{idx}": data['b'] for idx, data in quantized.items()},
    )
    print(f"  quantized_weights.npz saved")

    print("\n[DONE] Real YOLOv4-tiny weights converted to INT8")
    print(f"  Output directory: {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
