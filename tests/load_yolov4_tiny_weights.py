#!/usr/bin/env python3
"""
Load real YOLOv4-tiny darknet weights, fold batch normalization,
quantize to INT8/INT32, and export for the DPU golden model.

Downloads yolov4-tiny.weights if not present, parses the darknet binary
format, folds BN parameters into conv weights, and quantizes.

Covers ALL 21 conv layers in YOLOv4-tiny:
  - 10 backbone conv layers (darknet 0-15)
  - 11 detection head conv layers (darknet 18-29, 32, 35-36)

Two modes:
  Basic:      python tests/load_yolov4_tiny_weights.py
              Uses input_scale=1.0 (biases inaccurate but OK for RTL verification)

  Calibrated: python tests/load_yolov4_tiny_weights.py --calibrate dog.jpg
              Runs float inference to compute per-layer input_scale.
              Biases are correctly quantized for detection quality.

Output: image_sim_out/dpu_top_real/ directory with hex files + quantized_weights.npz
"""

import sys
import struct
import argparse
from pathlib import Path
import numpy as np

SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SCRIPT_DIR))

WEIGHTS_URL = "https://github.com/AlexeyAB/darknet/releases/download/yolov4/yolov4-tiny.weights"
WEIGHTS_FILE = PROJECT_ROOT / "yolov4-tiny.weights"

# DPU conv layer definitions — ALL 21 conv layers in YOLOv4-tiny.
# Each: (dpu_layer, c_in, c_out, kernel, stride, has_bn)
# Listed in darknet config order (weight file reads sequentially).
# dpu_layer = internal layer index (0-35 for 36-layer DPU).
CONV_LAYERS = [
    # --- 1st CSP block (backbone) ---
    (0,   3,  32, 3, 2, True),    # darknet 0
    (1,  32,  64, 3, 2, True),    # darknet 1
    (2,  64,  64, 3, 1, True),    # darknet 2
    # darknet 3 = route split (no weights)
    (4,  32,  32, 3, 1, True),    # darknet 4
    (5,  32,  32, 3, 1, True),    # darknet 5
    # darknet 6 = route concat (no weights)
    (7,  64,  64, 1, 1, True),    # darknet 7
    # darknet 8 = route concat, 9 = maxpool (no weights)
    # --- 2nd CSP block ---
    (10, 128, 128, 3, 1, True),   # darknet 10
    # darknet 11 = route split (no weights)
    (12,  64,  64, 3, 1, True),   # darknet 12
    (13,  64,  64, 3, 1, True),   # darknet 13
    # darknet 14 = route concat (no weights)
    (15, 128, 128, 1, 1, True),   # darknet 15
    # darknet 16 = route concat, 17 = maxpool (no weights)
    # --- 3rd CSP block ---
    (18, 256, 256, 3, 1, True),   # darknet 18
    # darknet 19 = route split (no weights)
    (20, 128, 128, 3, 1, True),   # darknet 20
    (21, 128, 128, 3, 1, True),   # darknet 21
    # darknet 22 = route concat (no weights)
    (23, 256, 256, 1, 1, True),   # darknet 23
    # darknet 24 = route concat, 25 = maxpool (no weights)
    # --- Detection head 1 ---
    (26, 512, 512, 3, 1, True),   # darknet 26
    (27, 512, 256, 1, 1, True),   # darknet 27
    (28, 256, 512, 3, 1, True),   # darknet 28
    (29, 512, 255, 1, 1, False),  # darknet 29 (detection output, NO batch norm)
    # darknet 30 = YOLO decode (skipped), 31 = route (no weights)
    # --- Bridge + Detection head 2 ---
    (31, 256, 128, 1, 1, True),   # darknet 32 → internal 31
    # darknet 33 = upsample, 34 = route concat (no weights)
    (34, 384, 256, 3, 1, True),   # darknet 35 → internal 34
    (35, 256, 255, 1, 1, False),  # darknet 36 → internal 35 (detection output, NO batch norm)
]


# ── Layer topology (all 36 internal layers) ──────────────────────────────
# Mirrors LAYER_DEFS from dpu_top_37layer_golden.py
LAYER_DEFS = [
    ('conv3x3',        3,   32, 2, 3),   # 0
    ('conv3x3',       32,   64, 2, 3),   # 1
    ('conv3x3',       64,   64, 1, 3),   # 2
    ('route_split',   64,   32, 0, 0),   # 3
    ('conv3x3',       32,   32, 1, 3),   # 4
    ('conv3x3',       32,   32, 1, 3),   # 5
    ('route_concat',  32,   64, 0, 0),   # 6
    ('conv1x1',       64,   64, 1, 1),   # 7
    ('route_concat',  64,  128, 0, 0),   # 8
    ('maxpool',      128,  128, 2, 0),   # 9
    ('conv3x3',      128,  128, 1, 3),   # 10
    ('route_split',  128,   64, 0, 0),   # 11
    ('conv3x3',       64,   64, 1, 3),   # 12
    ('conv3x3',       64,   64, 1, 3),   # 13
    ('route_concat',  64,  128, 0, 0),   # 14
    ('conv1x1',      128,  128, 1, 1),   # 15
    ('route_concat', 128,  256, 0, 0),   # 16
    ('maxpool',      256,  256, 2, 0),   # 17
    ('conv3x3',      256,  256, 1, 3),   # 18
    ('route_split',  256,  128, 0, 0),   # 19
    ('conv3x3',      128,  128, 1, 3),   # 20
    ('conv3x3',      128,  128, 1, 3),   # 21
    ('route_concat', 128,  256, 0, 0),   # 22
    ('conv1x1',      256,  256, 1, 1),   # 23
    ('route_concat', 256,  512, 0, 0),   # 24
    ('maxpool',      512,  512, 2, 0),   # 25
    ('conv3x3',      512,  512, 1, 3),   # 26
    ('conv1x1',      512,  256, 1, 1),   # 27
    ('conv3x3',      256,  512, 1, 3),   # 28
    ('conv1x1_linear', 512, 255, 1, 1),  # 29
    ('route_save',   256,  256, 0, 0),   # 30
    ('conv1x1',      256,  128, 1, 1),   # 31
    ('upsample',     128,  128, 0, 0),   # 32
    ('route_concat', 128,  384, 0, 0),   # 33
    ('conv3x3',      384,  256, 1, 3),   # 34
    ('conv1x1_linear', 256, 255, 1, 1),  # 35
]

CONCAT_MAP = {
    6:  lambda o, s: np.concatenate([o[5],  s[4]],  axis=0),
    8:  lambda o, s: np.concatenate([s[2],  o[7]],  axis=0),
    14: lambda o, s: np.concatenate([o[13], s[12]], axis=0),
    16: lambda o, s: np.concatenate([s[10], o[15]], axis=0),
    22: lambda o, s: np.concatenate([o[21], s[20]], axis=0),
    24: lambda o, s: np.concatenate([s[18], o[23]], axis=0),
    33: lambda o, s: np.concatenate([o[32], s[23]], axis=0),
}
SAVE_INDICES = {2, 4, 10, 12, 18, 20, 23, 27}


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
        print(f"[PARSE] Loaded {len(layers)} conv layers, {remaining} bytes remaining")

    return layers


def fold_bn(layer, eps=1e-5):
    """Fold batch normalization into conv weights and biases.

    In darknet format, when batch_normalize=1:
      - 'biases' is actually the BN beta parameter (not conv bias)
      - There is no separate conv bias (it's zero)
      - BN: y = gamma * (x - mean) / sqrt(var + eps) + beta

    Folded into conv:
      w_new = w * gamma / sqrt(var + eps)
      b_new = beta - mean * gamma / sqrt(var + eps)
    """
    w = layer['weights']  # [c_out, c_in, k, k]
    b = layer['biases']   # [c_out] — BN beta when has_bn, conv bias otherwise

    if layer['bn_scales'] is None:
        return w, b

    gamma = layer['bn_scales']  # [c_out]
    mean = layer['bn_means']    # [c_out]
    var = layer['bn_vars']      # [c_out]

    std = np.sqrt(var + eps)
    scale = gamma / std

    w_folded = w * scale.reshape(-1, 1, 1, 1)
    # CORRECT: beta - mean * scale (beta is NOT scaled by gamma/std)
    b_folded = b - mean * scale

    return w_folded, b_folded


def quantize_symmetric_int8(tensor, percentile=99.9):
    """
    Symmetric INT8 quantization with percentile clipping.
    Returns (quantized_int8, scale_factor).
    """
    abs_vals = np.abs(tensor)
    if np.max(abs_vals) == 0:
        return np.zeros_like(tensor, dtype=np.int8), 1.0

    clip_val = np.percentile(abs_vals, percentile)
    if clip_val == 0:
        clip_val = np.max(abs_vals)  # fallback to max

    scale = clip_val / 127.0
    quantized = np.round(tensor / scale)
    quantized = np.clip(quantized, -128, 127).astype(np.int8)
    return quantized, scale


def quantize_layer(w_float, b_float, input_scale=1.0):
    """
    Quantize conv layer weights and biases for INT8 inference.

    The DPU computes: acc = sum(w_int8[i] * x_int8[i]) + bias_int32
    In floating point: acc_float = sum(w_float[i] * x_float[i]) + b_float
    Relationship: w_int8 = w_float / w_scale, x_int8 = x_float / x_scale
                  acc_int32 = acc_float / (w_scale * x_scale)
                  bias_int32 = b_float / (w_scale * x_scale)
    """
    w_int8, w_scale = quantize_symmetric_int8(w_float)

    # Bias scale is w_scale * input_scale
    bias_scale = w_scale * input_scale
    b_int32 = np.round(b_float / bias_scale).astype(np.int64)
    b_int32 = np.clip(b_int32, -2**31, 2**31 - 1).astype(np.int32)

    # Output scale: one INT32 accumulator unit = w_scale * input_scale in float
    output_scale = w_scale * input_scale

    return w_int8, b_int32, w_scale, output_scale


# ── Float operations for calibration ─────────────────────────────────────

def _conv2d_float(input_fm, weights, bias, kernel_size, stride=1):
    """Float conv using im2col + matmul (fast)."""
    C_in, H, W = input_fm.shape
    C_out = weights.shape[0]
    x = input_fm.astype(np.float64)
    if kernel_size == 3:
        x = np.pad(x, ((0, 0), (1, 1), (1, 1)), mode='constant')
    H_out = (H + (2 if kernel_size == 3 else 0) - kernel_size) // stride + 1
    W_out = (W + (2 if kernel_size == 3 else 0) - kernel_size) // stride + 1

    if kernel_size == 3:
        cols = np.empty((C_in * 9, H_out * W_out), dtype=np.float64)
        idx = 0
        for c in range(C_in):
            for ky in range(3):
                for kx in range(3):
                    cols[idx] = x[c, ky:ky + H_out * stride:stride,
                                   kx:kx + W_out * stride:stride].ravel()
                    idx += 1
    else:
        # 1x1: just reshape
        cols = x.reshape(C_in, H * W)
        H_out, W_out = H, W

    w_flat = weights.reshape(C_out, -1).astype(np.float64)
    out = w_flat @ cols + bias.astype(np.float64).reshape(-1, 1)
    return out.reshape(C_out, H_out, W_out)


def _leaky_relu_float(x, alpha=0.1):
    return np.where(x > 0, x, x * alpha)


def _maxpool_2x2_float(x):
    C, H, W = x.shape
    return x.reshape(C, H // 2, 2, W // 2, 2).max(axis=(2, 4))


def _upsample_2x_float(x):
    return np.repeat(np.repeat(x, 2, axis=1), 2, axis=2)


def _run_one_calibration(float_weights, float_biases, x):
    """Run float inference on one preprocessed CHW image, return per-layer input ranges."""
    input_ranges = {}
    outputs = [None] * 36
    saves = {}
    current = x

    for i, (ltype, c_in, c_out, stride, kernel) in enumerate(LAYER_DEFS):
        if ltype in ('conv3x3', 'conv1x1', 'conv1x1_linear'):
            # Record activation range using 99.99th percentile (reduces outlier impact)
            abs_vals = np.abs(current)
            input_ranges[i] = float(np.percentile(abs_vals, 99.99))

            conv_out = _conv2d_float(current, float_weights[i], float_biases[i],
                                     kernel, stride)
            if ltype == 'conv1x1_linear':
                out = conv_out
            else:
                out = _leaky_relu_float(conv_out)

        elif ltype == 'route_split':
            out = current[current.shape[0] // 2:]

        elif ltype == 'route_concat':
            out = CONCAT_MAP[i](outputs, saves)

        elif ltype == 'maxpool':
            out = _maxpool_2x2_float(current)

        elif ltype == 'route_save':
            out = saves[27].copy()

        elif ltype == 'upsample':
            out = _upsample_2x_float(current)

        else:
            raise ValueError(f"Unknown layer type: {ltype}")

        outputs[i] = out
        current = out
        if i in SAVE_INDICES:
            saves[i] = out.copy()

    return input_ranges


def run_calibration(float_weights, float_biases, image_paths, cal_size=416):
    """Run float inference on calibration image(s), return per-conv-layer input_scale.

    Each conv layer's input_scale = percentile(|float_input|, 99.99) / 127.
    Uses 99.99th percentile instead of max to reduce outlier impact.
    Averages scales across multiple images for better coverage.

    Args:
        image_paths: single path string or list of path strings
    """
    from PIL import Image

    if isinstance(image_paths, str):
        image_paths = [image_paths]

    all_ranges = []
    for img_path in image_paths:
        img = Image.open(img_path).convert('RGB')
        img = img.resize((cal_size, cal_size), Image.BILINEAR)
        x = np.array(img, dtype=np.float64) / 255.0  # [0, 1] HWC
        x = x.transpose(2, 0, 1)  # CHW
        ranges = _run_one_calibration(float_weights, float_biases, x)
        all_ranges.append(ranges)
        print(f"    Calibrated on {Path(img_path).name}")

    # Average input ranges across images
    input_scales = {}
    all_keys = all_ranges[0].keys()
    for key in all_keys:
        avg_range = np.mean([r[key] for r in all_ranges])
        x_scale = avg_range / 127.0 if avg_range > 0 else 1.0 / 127.0
        input_scales[key] = x_scale

    return input_scales


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
    parser = argparse.ArgumentParser(
        description='Load YOLOv4-tiny weights, fold BN, quantize to INT8')
    parser.add_argument('--calibrate', type=str, nargs='+', default=None,
                        help='Calibration image path(s) (enables per-layer input_scale)')
    parser.add_argument('--cal-size', type=int, default=416,
                        help='Calibration resolution (default: 416)')
    args = parser.parse_args()

    # Download weights
    if not download_weights():
        return 1

    # Parse darknet weights
    layers = parse_darknet_weights(WEIGHTS_FILE)

    # Fold BN into conv weights
    print("\n[FOLD BN] Folding batch normalization...")
    folded = []
    float_weights = {}
    float_biases = {}
    for layer in layers:
        w_folded, b_folded = fold_bn(layer)
        idx = layer['dpu_idx']
        folded.append({
            'dpu_idx': idx,
            'c_in': layer['c_in'], 'c_out': layer['c_out'],
            'kernel': layer['kernel'], 'stride': layer['stride'],
            'w_float': w_folded,
            'b_float': b_folded,
        })
        float_weights[idx] = w_folded
        float_biases[idx] = b_folded
        print(f"  Layer {idx:2d}: w_range=[{w_folded.min():.4f}, {w_folded.max():.4f}]  "
              f"b_range=[{b_folded.min():.1f}, {b_folded.max():.1f}]")

    # Calibration
    if args.calibrate:
        for cp in args.calibrate:
            if not Path(cp).exists():
                print(f"\n[ERROR] Calibration image not found: {cp}")
                return 1
        cal_names = [Path(cp).name for cp in args.calibrate]
        print(f"\n[CALIBRATE] Float inference on {', '.join(cal_names)} "
              f"at {args.cal_size}x{args.cal_size}...")
        input_scales = run_calibration(float_weights, float_biases,
                                       args.calibrate, args.cal_size)
        print("  Per-layer calibrated input_scale:")
        for idx in sorted(input_scales.keys()):
            print(f"    Layer {idx:2d}: x_scale={input_scales[idx]:.8e}")
    else:
        print("\n[WARN] No calibration image (use --calibrate <image>)")
        print("  Using input_scale=1.0 (biases may be inaccurate for detection)")
        input_scales = {layer['dpu_idx']: 1.0 for layer in folded}

    # Quantize
    print("\n[QUANTIZE] Symmetric INT8 quantization...")
    quantized = {}
    output_scales = {}

    for layer in folded:
        idx = layer['dpu_idx']
        x_scale = input_scales[idx]
        w_int8, b_int32, w_scale, output_scale = quantize_layer(
            layer['w_float'], layer['b_float'], x_scale
        )
        quantized[idx] = {'w': w_int8, 'b': b_int32}
        output_scales[idx] = output_scale

        print(f"  Layer {idx:2d}: x_scale={x_scale:.4e}  w_scale={w_scale:.6f}  "
              f"out_scale={output_scale:.4e}  "
              f"w_range=[{w_int8.min():4d},{w_int8.max():4d}]  "
              f"b_range=[{b_int32.min():8d},{b_int32.max():8d}]")

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

    # Export per-layer output scales for dequantization
    np.savez(
        out_dir / "output_scales.npz",
        **{f"scale{idx}": np.float64(s) for idx, s in output_scales.items()},
    )
    print(f"  output_scales.npz saved")

    # Export per-layer input scales (for reference/debugging)
    np.savez(
        out_dir / "input_scales.npz",
        **{f"scale{idx}": np.float64(input_scales[idx])
           for idx in input_scales},
    )
    print(f"  input_scales.npz saved")

    cal_str = (f"calibrated on {', '.join(args.calibrate)}"
               if args.calibrate else "uncalibrated")
    print(f"\n[DONE] YOLOv4-tiny weights -> INT8 ({cal_str})")
    print(f"  {len(quantized)} conv layers exported to: {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
