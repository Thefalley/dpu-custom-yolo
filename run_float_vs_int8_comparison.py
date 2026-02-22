#!/usr/bin/env python3
"""
Layer-by-layer comparison of Float vs INT8 inference.

Runs both float32 and INT8 (DPU simulation) inference side by side,
comparing accumulator values and outputs at each layer to quantify
quantization error.

Usage:
  python run_float_vs_int8_comparison.py --input-image image_sim_out/rtl_validation/dog.jpg
"""

import sys
import argparse
import time
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "tests"))

from run_detection_demo import (
    LAYER_DEFS, CONCAT_MAP, SAVE_INDICES, DETECTION_LAYERS,
    conv2d_3x3_float, conv2d_1x1_float, leaky_relu_float,
    maxpool_2x2_float, upsample_2x_float,
    conv2d_3x3_int, conv2d_1x1_int,
    load_darknet_float_weights, load_input_image_float,
    load_input_image_int8, _requant_scale,
)
from phase3_dpu_functional_model import (
    leaky_relu_hardware, requantize_fixed_point,
    maxpool_2x2, route_split, route_concat, upsample_2x,
)
from dpu_top_37layer_golden import load_real_weights, SHIFT_RTL


def main():
    parser = argparse.ArgumentParser(
        description='Float vs INT8 layer-by-layer comparison')
    parser.add_argument('--input-image', type=str, required=True)
    parser.add_argument('--size', type=int, default=416)
    args = parser.parse_args()

    h0 = w0 = args.size
    image_path = Path(args.input_image)

    print("=" * 80)
    print("Float vs INT8 Layer-by-Layer Comparison")
    print("=" * 80)
    print(f"  Image: {image_path.name}  Resolution: {h0}x{w0}")
    print()

    # Load weights
    print("[1] Loading weights...")
    float_w, float_b = load_darknet_float_weights()
    int8_w, int8_b = load_real_weights()
    if float_w is None or int8_w is None:
        return 1

    # Load images
    img_float = load_input_image_float(str(image_path), h0, w0)
    img_int8 = load_input_image_int8(str(image_path), h0, w0)

    print(f"  Float input: shape={img_float.shape} range=[{img_float.min():.3f}, {img_float.max():.3f}]")
    print(f"  INT8  input: shape={img_int8.shape} range=[{img_int8.min()}, {img_int8.max()}]")
    print()

    # Run both side by side
    print("[2] Running layer-by-layer comparison...")
    print()
    print(f"{'Layer':>5s}  {'Type':16s}  {'Float range':>24s}  "
          f"{'INT8 range':>14s}  {'Corr':>6s}  {'RMSE':>10s}  {'MaxErr':>10s}")
    print("-" * 100)

    f_outputs = [None] * 36
    f_saves = {}
    f_current = img_float

    i_outputs = [None] * 36
    i_saves = {}
    i_current = img_int8

    for i, (ltype, c_in, c_out, stride, kernel) in enumerate(LAYER_DEFS):
        # ── Float path ──
        if ltype == 'conv3x3':
            f_conv = conv2d_3x3_float(f_current, float_w[i], float_b[i], stride=stride)
            f_out = leaky_relu_float(f_conv)
        elif ltype == 'conv1x1':
            f_conv = conv2d_1x1_float(f_current, float_w[i], float_b[i])
            f_out = leaky_relu_float(f_conv)
        elif ltype == 'conv1x1_linear':
            f_conv = conv2d_1x1_float(f_current, float_w[i], float_b[i])
            f_out = f_conv
        elif ltype == 'route_split':
            f_out = f_current[f_current.shape[0] // 2:]
        elif ltype == 'route_concat':
            f_out = CONCAT_MAP[i](f_outputs, f_saves)
        elif ltype == 'maxpool':
            f_out = maxpool_2x2_float(f_current)
        elif ltype == 'route_save':
            f_out = f_saves[27].copy()
        elif ltype == 'upsample':
            f_out = upsample_2x_float(f_current)

        f_outputs[i] = f_out
        f_current = f_out
        if i in SAVE_INDICES:
            f_saves[i] = f_out.copy()

        # ── INT8 path ──
        if ltype == 'conv3x3':
            i_conv = conv2d_3x3_int(i_current, int8_w[i], int8_b[i], stride=stride)
            scale = _requant_scale(i_conv)
            i_out = requantize_fixed_point(leaky_relu_hardware(i_conv),
                                           np.int32(scale), SHIFT_RTL)
        elif ltype == 'conv1x1':
            i_conv = conv2d_1x1_int(i_current, int8_w[i], int8_b[i])
            scale = _requant_scale(i_conv)
            i_out = requantize_fixed_point(leaky_relu_hardware(i_conv),
                                           np.int32(scale), SHIFT_RTL)
        elif ltype == 'conv1x1_linear':
            i_conv = conv2d_1x1_int(i_current, int8_w[i], int8_b[i])
            scale = _requant_scale(i_conv, linear=True)
            i_out = requantize_fixed_point(i_conv, np.int32(scale), SHIFT_RTL)
        elif ltype == 'route_split':
            i_out = route_split(i_current, groups=2, group_id=1)
        elif ltype == 'route_concat':
            i_out = CONCAT_MAP[i](i_outputs, i_saves)
        elif ltype == 'maxpool':
            i_out = maxpool_2x2(i_current)
        elif ltype == 'route_save':
            i_out = i_saves[27].copy()
        elif ltype == 'upsample':
            i_out = upsample_2x(i_current)

        i_outputs[i] = i_out
        i_current = i_out
        if i in SAVE_INDICES:
            i_saves[i] = i_out.copy()

        # ── Compare ──
        f_flat = f_out.flatten().astype(np.float64)
        i_flat = i_out.flatten().astype(np.float64)

        # Normalize INT8 to float range for comparison
        # Float output has some range; INT8 output is in [-128, 127]
        # Correlation tells us if the patterns match regardless of scale
        if f_flat.std() > 0 and i_flat.std() > 0:
            corr = np.corrcoef(f_flat, i_flat)[0, 1]
        else:
            corr = 0.0

        # Scale-aware RMSE: normalize float to [-127, 127] range
        f_max = np.max(np.abs(f_flat))
        if f_max > 0:
            f_norm = f_flat / f_max * 127.0
            rmse = np.sqrt(np.mean((f_norm - i_flat) ** 2))
            max_err = np.max(np.abs(f_norm - i_flat))
        else:
            rmse = 0.0
            max_err = 0.0

        f_range = f"[{f_out.min():8.2f}, {f_out.max():8.2f}]"
        i_range = f"[{i_out.min():4d}, {i_out.max():4d}]"

        print(f"L{i:2d}    {ltype:16s}  {f_range:>24s}  "
              f"{i_range:>14s}  {corr:6.4f}  {rmse:10.2f}  {max_err:10.2f}")

    # Detection layer analysis
    print()
    print("=" * 80)
    print("Detection Layer Analysis (pre-requant INT32 accumulators)")
    print("=" * 80)

    # Load output scales
    scales_path = PROJECT_ROOT / "image_sim_out" / "dpu_top_real" / "output_scales.npz"
    out_scales = {}
    if scales_path.exists():
        data = np.load(scales_path)
        for key in data.files:
            idx = int(key.replace('scale', ''))
            out_scales[idx] = float(data[key])

    for det_layer in [29, 35]:
        print(f"\n--- Layer {det_layer} ---")

        # Re-run just this layer to get INT32 accumulators
        if det_layer == 29:
            f_input = f_outputs[28] if f_outputs[28] is not None else f_current
            i_input = i_outputs[28] if i_outputs[28] is not None else i_current
        else:
            f_input = f_outputs[34] if f_outputs[34] is not None else f_current
            i_input = i_outputs[34] if i_outputs[34] is not None else i_current

        # Float conv (raw logits)
        f_logits = conv2d_1x1_float(f_input, float_w[det_layer], float_b[det_layer])

        # INT8 conv (INT32 accumulators)
        i_acc = conv2d_1x1_int(i_input, int8_w[det_layer], int8_b[det_layer])

        # Dequantize INT32 to float using output_scale
        osc = out_scales.get(det_layer, None)
        if osc:
            i_logits = i_acc.astype(np.float64) * osc
        else:
            i_logits = i_acc.astype(np.float64)

        print(f"  Float logits:  range=[{f_logits.min():.2f}, {f_logits.max():.2f}]  "
              f"shape={f_logits.shape}")
        print(f"  INT8  logits:  range=[{i_logits.min():.2f}, {i_logits.max():.2f}]  "
              f"shape={i_logits.shape}")

        # Correlation
        f_flat = f_logits.flatten()
        i_flat = i_logits.flatten()
        if f_flat.std() > 0 and i_flat.std() > 0:
            corr = np.corrcoef(f_flat, i_flat)[0, 1]
        else:
            corr = 0.0
        rmse = np.sqrt(np.mean((f_flat - i_flat) ** 2))
        print(f"  Correlation:   {corr:.6f}")
        print(f"  RMSE:          {rmse:.4f}")
        print(f"  Output scale:  {osc}")

        # Check objectness values (channel 4 for each anchor)
        # YOLO format: [tx, ty, tw, th, obj, class0, class1, ...] x 3 anchors
        # 255 channels = 3 anchors x 85 (4 + 1 + 80)
        print(f"\n  Objectness values (sigmoid) per anchor:")
        for a in range(3):
            obj_ch = a * 85 + 4  # objectness channel
            f_obj = 1.0 / (1.0 + np.exp(-f_logits[obj_ch]))
            i_obj = 1.0 / (1.0 + np.exp(-i_logits[obj_ch]))

            f_max_obj = f_obj.max()
            i_max_obj = i_obj.max()
            print(f"    Anchor {a}: float max_obj={f_max_obj:.4f}  "
                  f"int8 max_obj={i_max_obj:.4f}  "
                  f"diff={abs(f_max_obj - i_max_obj):.4f}")

    print("\n[DONE] Comparison complete.")
    return 0


if __name__ == '__main__':
    sys.exit(main())
