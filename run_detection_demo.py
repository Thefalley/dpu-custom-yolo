#!/usr/bin/env python3
"""
End-to-End YOLOv4-tiny Detection Demo

Runs full 36-layer inference with real YOLOv4-tiny weights, then decodes
detection head outputs into bounding boxes with COCO class labels.

Two modes:
  --mode float  (default): Full float32 inference using original darknet weights.
                Produces best detection quality. Demonstrates the YOLO pipeline.
  --mode int8:  INT8 inference matching the DPU hardware. Detection outputs use
                auto-calibrated dequantization. Shows the actual FPGA datapath.

Usage:
  python run_detection_demo.py --input-image image_sim_out/rtl_validation/dog.jpg
  python run_detection_demo.py --input-image image_sim_out/rtl_validation/dog.jpg --mode int8
  python run_detection_demo.py --input-image image_sim_out/rtl_validation/dog.jpg --size 608
"""

import sys
import argparse
import time
import struct
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "tests"))

from phase3_dpu_functional_model import (
    leaky_relu_hardware,
    requantize_fixed_point,
    maxpool_2x2,
    route_split,
    route_concat,
    upsample_2x,
)
from dpu_top_37layer_golden import (
    LAYER_DEFS,
    CONV_LAYERS,
    load_real_weights,
    SHIFT_RTL,
)
from yolo_postprocess import (
    decode_yolo,
    nms,
    draw_detections,
    ANCHORS,
    NUM_CLASSES,
    COCO_CLASSES,
)


# ── Optimized INT8 convolutions (im2col + BLAS matmul) ───────────────────

def conv2d_3x3_int(input_fm, weights, bias, stride=1, padding=1):
    """3x3 conv using im2col + matrix multiply (INT8 in, INT32 out)."""
    C_in, H, W = input_fm.shape
    C_out = weights.shape[0]
    x = (np.pad(input_fm, ((0, 0), (padding, padding), (padding, padding)),
                mode='constant', constant_values=0)
         if padding > 0 else input_fm)
    H_out = (H + 2 * padding - 3) // stride + 1
    W_out = (W + 2 * padding - 3) // stride + 1
    cols = np.empty((C_in * 9, H_out * W_out), dtype=np.int64)
    idx = 0
    for c in range(C_in):
        for ky in range(3):
            for kx in range(3):
                cols[idx] = x[c, ky:ky + H_out * stride:stride,
                               kx:kx + W_out * stride:stride].ravel()
                idx += 1
    w_flat = weights.reshape(C_out, -1).astype(np.int64)
    out = w_flat @ cols + bias.astype(np.int64).reshape(-1, 1)
    return out.reshape(C_out, H_out, W_out).astype(np.int32)


def conv2d_1x1_int(input_fm, weights, bias):
    """1x1 conv via matrix multiply (INT8 in, INT32 out)."""
    C_in, H, W = input_fm.shape
    C_out = weights.shape[0]
    w = weights.reshape(C_out, C_in).astype(np.int64)
    x = input_fm.reshape(C_in, H * W).astype(np.int64)
    out = w @ x + bias.astype(np.int64).reshape(-1, 1)
    return out.reshape(C_out, H, W).astype(np.int32)


# ── Float convolutions ───────────────────────────────────────────────────

def conv2d_3x3_float(input_fm, weights, bias, stride=1, padding=1):
    """3x3 conv in float32."""
    C_in, H, W = input_fm.shape
    C_out = weights.shape[0]
    x = input_fm.astype(np.float64)
    if padding > 0:
        x = np.pad(x, ((0, 0), (padding, padding), (padding, padding)),
                   mode='constant', constant_values=0)
    H_out = (H + 2 * padding - 3) // stride + 1
    W_out = (W + 2 * padding - 3) // stride + 1
    cols = np.empty((C_in * 9, H_out * W_out), dtype=np.float64)
    idx = 0
    for c in range(C_in):
        for ky in range(3):
            for kx in range(3):
                cols[idx] = x[c, ky:ky + H_out * stride:stride,
                               kx:kx + W_out * stride:stride].ravel()
                idx += 1
    w_flat = weights.reshape(C_out, -1).astype(np.float64)
    out = w_flat @ cols + bias.astype(np.float64).reshape(-1, 1)
    return out.reshape(C_out, H_out, W_out)


def conv2d_1x1_float(input_fm, weights, bias):
    """1x1 conv in float32."""
    C_in, H, W = input_fm.shape
    C_out = weights.shape[0]
    w = weights.reshape(C_out, C_in).astype(np.float64)
    x = input_fm.reshape(C_in, H * W).astype(np.float64)
    out = w @ x + bias.astype(np.float64).reshape(-1, 1)
    return out.reshape(C_out, H, W)


def leaky_relu_float(x, alpha=0.1):
    """Float leaky ReLU with exact alpha (darknet default 0.1)."""
    return np.where(x > 0, x, x * alpha)


def maxpool_2x2_float(input_fm):
    """Float 2x2 maxpool with stride 2."""
    C, H, W = input_fm.shape
    return input_fm.reshape(C, H // 2, 2, W // 2, 2).max(axis=(2, 4))


def upsample_2x_float(input_fm):
    """Float 2x nearest-neighbor upsample."""
    return np.repeat(np.repeat(input_fm, 2, axis=1), 2, axis=2)


# ── Load original darknet weights ────────────────────────────────────────

def load_darknet_float_weights():
    """Parse yolov4-tiny.weights, fold BN, return float32 weights/biases."""
    from load_yolov4_tiny_weights import CONV_LAYERS as CL_DEFS

    weights_path = PROJECT_ROOT / "yolov4-tiny.weights"
    if not weights_path.exists():
        print(f"[ERROR] Weights not found: {weights_path}")
        print("  Run: python tests/load_yolov4_tiny_weights.py")
        return None, None

    float_weights = {}
    float_biases = {}

    with open(weights_path, 'rb') as f:
        major, minor, _ = struct.unpack('3i', f.read(12))
        if major * 10 + minor >= 2:
            f.read(8)  # images_seen (uint64)
        else:
            f.read(4)  # images_seen (uint32)

        for dpu_idx, c_in, c_out, kernel, stride, has_bn in CL_DEFS:
            n_w = c_out * c_in * kernel * kernel
            biases = np.frombuffer(f.read(c_out * 4), dtype=np.float32).copy()

            if has_bn:
                bn_scales = np.frombuffer(f.read(c_out * 4), dtype=np.float32).copy()
                bn_means = np.frombuffer(f.read(c_out * 4), dtype=np.float32).copy()
                bn_vars = np.frombuffer(f.read(c_out * 4), dtype=np.float32).copy()
                weights = np.frombuffer(f.read(n_w * 4), dtype=np.float32).copy()
                weights = weights.reshape(c_out, c_in, kernel, kernel)

                # Fold BN: w_new = w * gamma/sqrt(var+eps)
                #          b_new = beta - mean * gamma/sqrt(var+eps)
                # Note: 'biases' here is BN beta (not conv bias) in darknet format
                eps = 1e-5
                std = np.sqrt(bn_vars + eps)
                scale = bn_scales / std
                weights = weights * scale.reshape(-1, 1, 1, 1)
                biases = biases - bn_means * scale
            else:
                weights = np.frombuffer(f.read(n_w * 4), dtype=np.float32).copy()
                weights = weights.reshape(c_out, c_in, kernel, kernel)

            float_weights[dpu_idx] = weights
            float_biases[dpu_idx] = biases

    print(f"  Loaded {len(float_weights)} conv layers from darknet weights (BN folded)")
    return float_weights, float_biases


# ── Image loading ────────────────────────────────────────────────────────

def load_input_image_int8(image_path, h0, w0):
    """Load image for INT8 mode: resize, symmetric mapping [0,1] -> [0,127]."""
    from PIL import Image
    img = Image.open(image_path).convert('RGB')
    img = img.resize((w0, h0), Image.BILINEAR)
    arr = np.array(img, dtype=np.float32)
    # Symmetric: x_float = pixel/255 in [0,1], x_int8 = round(x_float * 127)
    arr = np.round(arr / 255.0 * 127.0)
    return np.clip(arr, -128, 127).astype(np.int8).transpose(2, 0, 1)


def load_input_image_float(image_path, h0, w0):
    """Load image for float mode: resize, normalize to [0, 1]."""
    from PIL import Image
    img = Image.open(image_path).convert('RGB')
    img = img.resize((w0, h0), Image.BILINEAR)
    arr = np.array(img, dtype=np.float32) / 255.0  # [0, 1]
    return arr.transpose(2, 0, 1)  # CHW


# ── Requant scale calibration (INT8 mode) ────────────────────────────────

def _requant_scale(conv_out, linear=False):
    vals = conv_out if linear else leaky_relu_hardware(conv_out)
    abs_vals = np.abs(vals)
    p9999 = float(np.percentile(abs_vals, 99.99))
    if p9999 == 0:
        p9999 = float(np.max(abs_vals))
    if p9999 == 0:
        return 655
    scale = int(127 * (1 << SHIFT_RTL) / p9999)
    return max(1, min(scale, 65535))


# ── Inference engine ─────────────────────────────────────────────────────

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
DETECTION_LAYERS = {29, 35}


def run_inference_float(input_img, weights, biases, verbose=True):
    """Run 36 layers in float32. Returns layer outputs and detection logits."""
    outputs = [None] * 36
    saves = {}
    det_logits = {}
    current = input_img
    total_t = time.time()

    for i, (ltype, c_in, c_out, stride, kernel) in enumerate(LAYER_DEFS):
        t0 = time.time()

        if ltype == 'conv3x3':
            conv_out = conv2d_3x3_float(current, weights[i], biases[i],
                                        stride=stride)
            out = leaky_relu_float(conv_out)

        elif ltype == 'conv1x1':
            conv_out = conv2d_1x1_float(current, weights[i], biases[i])
            out = leaky_relu_float(conv_out)

        elif ltype == 'conv1x1_linear':
            conv_out = conv2d_1x1_float(current, weights[i], biases[i])
            if i in DETECTION_LAYERS:
                det_logits[i] = conv_out.copy()
            out = conv_out  # no activation

        elif ltype == 'route_split':
            out = current[current.shape[0] // 2:]  # second half

        elif ltype == 'route_concat':
            out = CONCAT_MAP[i](outputs, saves)

        elif ltype == 'maxpool':
            out = maxpool_2x2_float(current)

        elif ltype == 'route_save':
            out = saves[27].copy()

        elif ltype == 'upsample':
            out = upsample_2x_float(current)

        else:
            raise ValueError(f"Unknown layer type: {ltype}")

        outputs[i] = out
        current = out
        if i in SAVE_INDICES:
            saves[i] = out.copy()

        dt = time.time() - t0
        if verbose:
            shape_s = str(tuple(out.shape))
            mn, mx = out.min(), out.max()
            print(f"  L{i:2d} {ltype:16s} {c_in:3d}->{c_out:3d}  "
                  f"{shape_s:22s} [{mn:8.2f},{mx:8.2f}]  {dt:.2f}s")

    if verbose:
        print(f"\n  Total inference: {time.time() - total_t:.1f}s")
    return outputs, det_logits


def run_inference_int8(input_img, weights, biases, verbose=True):
    """Run 36 layers in INT8 (DPU simulation). Returns INT32 detection accs."""
    outputs = [None] * 36
    saves = {}
    det_int32 = {}
    current = input_img
    total_t = time.time()

    for i, (ltype, c_in, c_out, stride, kernel) in enumerate(LAYER_DEFS):
        t0 = time.time()

        if ltype == 'conv3x3':
            conv_out = conv2d_3x3_int(current, weights[i], biases[i],
                                      stride=stride)
            scale = _requant_scale(conv_out)
            out = requantize_fixed_point(leaky_relu_hardware(conv_out),
                                         np.int32(scale), SHIFT_RTL)

        elif ltype == 'conv1x1':
            conv_out = conv2d_1x1_int(current, weights[i], biases[i])
            scale = _requant_scale(conv_out)
            out = requantize_fixed_point(leaky_relu_hardware(conv_out),
                                         np.int32(scale), SHIFT_RTL)

        elif ltype == 'conv1x1_linear':
            conv_out = conv2d_1x1_int(current, weights[i], biases[i])
            if i in DETECTION_LAYERS:
                det_int32[i] = conv_out.copy()
            scale = _requant_scale(conv_out, linear=True)
            out = requantize_fixed_point(conv_out, np.int32(scale), SHIFT_RTL)

        elif ltype == 'route_split':
            out = route_split(current, groups=2, group_id=1)

        elif ltype == 'route_concat':
            out = CONCAT_MAP[i](outputs, saves)

        elif ltype == 'maxpool':
            out = maxpool_2x2(current)

        elif ltype == 'route_save':
            out = saves[27].copy()

        elif ltype == 'upsample':
            out = upsample_2x(current)

        else:
            raise ValueError(f"Unknown layer type: {ltype}")

        outputs[i] = out
        current = out
        if i in SAVE_INDICES:
            saves[i] = out.copy()

        dt = time.time() - t0
        if verbose:
            shape_s = str(tuple(out.shape))
            print(f"  L{i:2d} {ltype:16s} {c_in:3d}->{c_out:3d}  "
                  f"{shape_s:22s} [{out.min():4d},{out.max():4d}]  "
                  f"{dt:.2f}s")

    if verbose:
        print(f"\n  Total inference: {time.time() - total_t:.1f}s")
    return outputs, det_int32


# ── Dequantization (INT8 mode) ───────────────────────────────────────────

def load_output_scales():
    """Load per-layer output_scale from calibration (for dequantization)."""
    npz_path = PROJECT_ROOT / "image_sim_out" / "dpu_top_real" / "output_scales.npz"
    if not npz_path.exists():
        return {}
    data = np.load(npz_path)
    scales = {}
    for key in data.files:
        idx = int(key.replace('scale', ''))
        scales[idx] = float(data[key])
    return scales


def dequantize_detection(int32_acc, layer_idx, output_scale=None):
    """Convert INT32 accumulators to float logits for YOLO decode.

    With calibrated output_scale:
      float_logit = int32_acc * output_scale
      (since acc_int32 ≈ acc_float / output_scale)

    Without: fall back to auto-calibration heuristic.
    """
    if output_scale is not None and output_scale > 0:
        logits = int32_acc.astype(np.float64) * output_scale
        print(f"    L{layer_idx} dequant scale={output_scale:.6e}  "
              f"logit range=[{logits.min():.2f}, {logits.max():.2f}]")
        return logits

    # Fallback: auto-calibration
    p99_raw = np.percentile(np.abs(int32_acc), 99)
    if p99_raw == 0:
        print(f"    L{layer_idx} WARNING: all-zero accumulator")
        return int32_acc.astype(np.float64)
    k = 5.0 / p99_raw
    logits = int32_acc.astype(np.float64) * k
    print(f"    L{layer_idx} auto-cal k={k:.8f}  raw_p99={p99_raw:.0f}  "
          f"logit range=[{logits.min():.1f}, {logits.max():.1f}]")
    return logits


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='End-to-End YOLOv4-tiny Detection Demo')
    parser.add_argument('--input-image', type=str, required=True,
                        help='Path to input image (JPEG/PNG)')
    parser.add_argument('--size', type=int, default=416,
                        help='Input resolution, must be multiple of 32 '
                             '(default: 416)')
    parser.add_argument('--mode', choices=['float', 'int8'], default='float',
                        help='Inference mode (default: float)')
    parser.add_argument('--conf-threshold', type=float, default=0.25,
                        help='Confidence threshold (default: 0.25)')
    parser.add_argument('--nms-threshold', type=float, default=0.45,
                        help='NMS IoU threshold (default: 0.45)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output image path (default: detections_<name>.png)')
    parser.add_argument('--quiet', action='store_true',
                        help='Suppress per-layer output')
    args = parser.parse_args()

    h0 = w0 = args.size
    if h0 % 32 != 0:
        print(f"[ERROR] --size must be a multiple of 32 (got {h0})")
        return 1

    verbose = not args.quiet
    image_path = Path(args.input_image)
    if not image_path.exists():
        print(f"[ERROR] Image not found: {image_path}")
        return 1

    use_float = (args.mode == 'float')

    print("=" * 70)
    mode_str = "FLOAT32" if use_float else "INT8 DPU simulation"
    print(f"YOLOv4-tiny Detection Demo  ({mode_str})")
    print("=" * 70)
    print(f"  Input image: {image_path.name}")
    print(f"  Resolution:  {h0}x{w0}")
    print(f"  Mode:        {args.mode}")
    print(f"  Conf thresh: {args.conf_threshold}")
    print(f"  NMS thresh:  {args.nms_threshold}")
    print()

    # ── 1. Load weights ──────────────────────────────────────────────────
    print("[1/5] Loading YOLOv4-tiny weights...")
    if use_float:
        weights, biases = load_darknet_float_weights()
    else:
        weights, biases = load_real_weights()
        if weights is not None:
            print(f"  Loaded {len(weights)} conv layers (INT8 quantized)")
    if weights is None:
        return 1

    # ── 2. Load & preprocess image ───────────────────────────────────────
    print(f"\n[2/5] Preprocessing image -> {h0}x{w0}...")
    if use_float:
        input_img = load_input_image_float(str(image_path), h0, w0)
        print(f"  Shape: {input_img.shape}  range=[{input_img.min():.3f}, "
              f"{input_img.max():.3f}]")
    else:
        input_img = load_input_image_int8(str(image_path), h0, w0)
        print(f"  Shape: {input_img.shape}  range=[{input_img.min()}, "
              f"{input_img.max()}]")

    # ── 3. 36-layer inference ────────────────────────────────────────────
    print(f"\n[3/5] Running 36-layer inference...")
    if use_float:
        outputs, det_data = run_inference_float(
            input_img, weights, biases, verbose=verbose)
    else:
        outputs, det_data = run_inference_int8(
            input_img, weights, biases, verbose=verbose)

    # ── 4. Decode detections ─────────────────────────────────────────────
    print(f"\n[4/5] Decoding YOLO detections...")

    # Load output scales for INT8 dequantization
    out_scales = {}
    if not use_float:
        out_scales = load_output_scales()
        if out_scales:
            print(f"  Loaded {len(out_scales)} output scales for dequantization")

    all_candidates = []

    # Layer 29 -> small feature map (scale 1, large anchors)
    if 29 in det_data:
        raw = det_data[29]
        print(f"  Layer 29: shape={raw.shape}  "
              f"range=[{raw.min():.2f}, {raw.max():.2f}]")
        if use_float:
            logits = raw  # already float logits
        else:
            logits = dequantize_detection(raw, 29, out_scales.get(29))
        boxes = decode_yolo(logits, ANCHORS[1], NUM_CLASSES, (h0, w0),
                            pre_scaled=True)
        print(f"    -> {len(boxes)} candidates")
        if len(boxes) > 0:
            all_candidates.append(boxes)

    # Layer 35 -> large feature map (scale 0, small anchors)
    if 35 in det_data:
        raw = det_data[35]
        print(f"  Layer 35: shape={raw.shape}  "
              f"range=[{raw.min():.2f}, {raw.max():.2f}]")
        if use_float:
            logits = raw
        else:
            logits = dequantize_detection(raw, 35, out_scales.get(35))
        boxes = decode_yolo(logits, ANCHORS[0], NUM_CLASSES, (h0, w0),
                            pre_scaled=True)
        print(f"    -> {len(boxes)} candidates")
        if len(boxes) > 0:
            all_candidates.append(boxes)

    if not all_candidates:
        print("\n  No candidate detections found.")
        return 0

    all_boxes = np.concatenate(all_candidates, axis=0)
    print(f"\n  Total candidates: {len(all_boxes)}")

    # Filter by confidence
    mask = all_boxes[:, 4] > args.conf_threshold
    filtered = all_boxes[mask]
    print(f"  After conf > {args.conf_threshold}: {len(filtered)}")

    if len(filtered) == 0:
        print(f"  No detections above threshold. Top candidates:")
        order = np.argsort(-all_boxes[:, 4])
        for j in order[:5]:
            b = all_boxes[j]
            cname = COCO_CLASSES[int(b[5])] if int(b[5]) < NUM_CLASSES else '?'
            print(f"    {cname}: conf={b[4]:.4f}")
        filtered = all_boxes[order[:20]]

    # NMS
    final_boxes = nms(filtered, args.nms_threshold)

    # ── 5. Report + Visualize ────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"[5/5] DETECTIONS: {len(final_boxes)} object(s)")
    print(f"{'='*70}")

    detections = []
    for box in final_boxes[:30]:
        x1, y1, x2, y2, score, cls_id = box
        cls_name = (COCO_CLASSES[int(cls_id)]
                    if int(cls_id) < NUM_CLASSES else 'unknown')
        detections.append({
            'box': [float(x1), float(y1), float(x2), float(y2)],
            'score': float(score),
            'class_id': int(cls_id),
            'class_name': cls_name,
        })
        print(f"  {cls_name:15s}  conf={score:.4f}  "
              f"box=[{x1:.3f}, {y1:.3f}, {x2:.3f}, {y2:.3f}]")

    if not detections:
        print("  (none)")
        return 0

    # Draw and save
    out_path = args.output or f"detections_{image_path.stem}.png"
    draw_detections(str(image_path), detections, out_path)

    # Display
    try:
        from PIL import Image
        Image.open(out_path).show()
        print(f"\nResult image displayed and saved to: {out_path}")
    except Exception:
        print(f"\nResult saved to: {out_path}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
