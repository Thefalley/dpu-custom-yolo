#!/usr/bin/env python3
"""
Golden model for dpu_top 36-layer YOLOv4-tiny (H0=32, W0=32 for full network).
Runs all 36 internal layers (darknet 0-29, 31-36; skipping YOLO layers 30, 37)
using phase3_dpu_functional_model primitives, exports hex files for RTL TB comparison.

IMPORTANT: Uses hardware-accurate requantize_fixed_point to match RTL exactly.

Internal layer mapping:
  Internal 0-29  = Darknet 0-29
  Internal 30    = Darknet 31  (route from saved L27)
  Internal 31    = Darknet 32  (conv1x1)
  Internal 32    = Darknet 33  (upsample 2x)
  Internal 33    = Darknet 34  (route concat)
  Internal 34    = Darknet 35  (conv3x3)
  Internal 35    = Darknet 36  (conv1x1 linear)

Output directory: image_sim_out/dpu_top_37/
  - input_image.hex          (3*32*32 = 3072 bytes)
  - layerN_weights.hex       (per conv layer)
  - layerN_bias.hex          (per conv layer, 4 bytes LE per channel)
  - layerN_expected.hex      (output fmap for each layer)
  - final_output_expected.hex
  - scales.hex               (36 entries x 2 bytes LE = 72 hex lines)

Usage:
  python tests/dpu_top_37layer_golden.py                  # synthetic random weights
  python tests/dpu_top_37layer_golden.py --real-weights    # real YOLOv4-tiny weights
  python tests/dpu_top_37layer_golden.py --per-layer-scale # calibrate per-layer scale
"""

import sys
import argparse
from pathlib import Path
import numpy as np

SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from phase3_dpu_functional_model import (
    conv2d_3x3,
    conv2d_1x1,
    leaky_relu_hardware,
    requantize_fixed_point,
    maxpool_2x2,
    route_split,
    route_concat,
    upsample_2x,
    INT8_MIN,
    INT8_MAX,
)

# Dimensions — H0=32 needed because layer 25 maxpool would produce 0x0 with H0=16
H0 = 32
W0 = 32
SCALE_RTL = 655  # 0.01 * 65536
SHIFT_RTL = 16
NUM_LAYERS = 36  # internal indices 0-35

# Layer descriptors: (type, c_in, c_out, stride, kernel)
# Internal layer numbering 0-35 (skips darknet YOLO layers 30 and 37)
LAYER_DEFS = [
    # --- 1st CSP block ---
    ('conv3x3',        3,   32, 2, 3),   # 0
    ('conv3x3',       32,   64, 2, 3),   # 1
    ('conv3x3',       64,   64, 1, 3),   # 2
    ('route_split',   64,   32, 0, 0),   # 3  (groups=2, group_id=1)
    ('conv3x3',       32,   32, 1, 3),   # 4
    ('conv3x3',       32,   32, 1, 3),   # 5
    ('route_concat',  32,   64, 0, 0),   # 6  (L5 + save_l4)
    ('conv1x1',       64,   64, 1, 1),   # 7
    ('route_concat',  64,  128, 0, 0),   # 8  (save_l2 + L7)
    ('maxpool',      128,  128, 2, 0),   # 9
    # --- 2nd CSP block ---
    ('conv3x3',      128,  128, 1, 3),   # 10
    ('route_split',  128,   64, 0, 0),   # 11 (groups=2, group_id=1)
    ('conv3x3',       64,   64, 1, 3),   # 12
    ('conv3x3',       64,   64, 1, 3),   # 13
    ('route_concat',  64,  128, 0, 0),   # 14 (L13 + save_l12)
    ('conv1x1',      128,  128, 1, 1),   # 15
    ('route_concat', 128,  256, 0, 0),   # 16 (save_l10 + L15)
    ('maxpool',      256,  256, 2, 0),   # 17
    # --- 3rd CSP block ---
    ('conv3x3',      256,  256, 1, 3),   # 18
    ('route_split',  256,  128, 0, 0),   # 19
    ('conv3x3',      128,  128, 1, 3),   # 20
    ('conv3x3',      128,  128, 1, 3),   # 21
    ('route_concat', 128,  256, 0, 0),   # 22 (L21 + save_l20)
    ('conv1x1',      256,  256, 1, 1),   # 23
    ('route_concat', 256,  512, 0, 0),   # 24 (save_l18 + L23)
    ('maxpool',      512,  512, 2, 0),   # 25
    # --- Detection head 1 ---
    ('conv3x3',      512,  512, 1, 3),   # 26
    ('conv1x1',      512,  256, 1, 1),   # 27
    ('conv3x3',      256,  512, 1, 3),   # 28
    ('conv1x1_linear', 512, 255, 1, 1),  # 29 OUTPUT 1
    # --- Bridge + Detection head 2 ---
    # (skip darknet layer 30 = YOLO)
    ('route_save',   256,  256, 0, 0),   # 30 (internal; route from save_l27)
    ('conv1x1',      256,  128, 1, 1),   # 31
    ('upsample',     128,  128, 0, 0),   # 32
    ('route_concat', 128,  384, 0, 0),   # 33 (L32_ups + save_l23) 128+256=384
    ('conv3x3',      384,  256, 1, 3),   # 34
    ('conv1x1_linear', 256, 255, 1, 1),  # 35 OUTPUT 2
]

# Conv layers (layers that have weights/biases)
CONV_LAYERS = [i for i, (lt, *_) in enumerate(LAYER_DEFS)
               if lt in ('conv3x3', 'conv1x1', 'conv1x1_linear')]


def conv_bn_leaky_hw(input_fm, weights, bias, stride=1, kernel_size=3,
                     scale=SCALE_RTL):
    """Hardware-accurate conv+BN+LeakyReLU+requantize using fixed-point scale."""
    if kernel_size == 3:
        conv_out = conv2d_3x3(input_fm, weights, bias, stride=stride)
    else:
        conv_out = conv2d_1x1(input_fm, weights, bias)

    relu_out = leaky_relu_hardware(conv_out)
    output = requantize_fixed_point(relu_out, np.int32(scale), SHIFT_RTL)
    return output


def conv_linear_hw(input_fm, weights, bias, scale=SCALE_RTL):
    """Conv1x1 with LINEAR activation (no LeakyReLU)."""
    conv_out = conv2d_1x1(input_fm, weights, bias)
    # Skip LeakyReLU — go directly to requantize
    output = requantize_fixed_point(conv_out, np.int32(scale), SHIFT_RTL)
    return output


def calibrate_scale(input_fm, weights, bias, stride=1, kernel_size=3,
                    linear=False):
    """Compute optimal requant scale for a conv layer by examining accumulator range."""
    if kernel_size == 3:
        conv_out = conv2d_3x3(input_fm, weights, bias, stride=stride)
    else:
        conv_out = conv2d_1x1(input_fm, weights, bias)

    if not linear:
        relu_out = leaky_relu_hardware(conv_out)
    else:
        # Linear activation: no LeakyReLU, use raw conv output
        relu_out = conv_out

    max_abs = int(np.max(np.abs(relu_out)))
    if max_abs == 0:
        return SCALE_RTL
    # Scale to map max_abs -> 127 in INT8: out = (relu * scale) >> 16
    # We want: max_abs * scale >> 16 = 127
    # scale = 127 * 2^16 / max_abs
    scale = int(127 * (1 << SHIFT_RTL) / max_abs)
    scale = max(1, min(scale, 65535))
    return scale


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


def load_real_weights():
    """Load real YOLOv4-tiny quantized weights from npz file."""
    npz_path = PROJECT_ROOT / "image_sim_out" / "dpu_top_real" / "quantized_weights.npz"
    if not npz_path.exists():
        print(f"[ERROR] Real weights not found: {npz_path}")
        print("Run: python tests/load_yolov4_tiny_weights.py")
        return None, None

    data = np.load(npz_path)
    weights = {}
    biases = {}
    for idx in CONV_LAYERS:
        key_w = f'w{idx}'
        key_b = f'b{idx}'
        if key_w in data and key_b in data:
            weights[idx] = data[key_w]
            biases[idx] = data[key_b]
        else:
            print(f"[WARN] Real weights missing for layer {idx}, using random")
            ltype, c_in, c_out, stride, kernel = LAYER_DEFS[idx]
            k = kernel if kernel > 0 else 1
            weights[idx] = np.random.randint(-20, 20,
                                             (c_out, c_in, k, k), dtype=np.int8)
            biases[idx] = np.random.randint(-200, 200, c_out, dtype=np.int32)
    return weights, biases


def load_input_image(image_path):
    """Load a real image, resize to H0xW0, convert to INT8 CHW format."""
    from PIL import Image
    img = Image.open(image_path).convert('RGB')
    img = img.resize((W0, H0), Image.BILINEAR)
    arr = np.array(img, dtype=np.float32)  # HWC, 0-255
    # YOLOv4-tiny normalizes to [0,1] then processes; we map to signed INT8 [-128,127]
    # Center around 0: pixel_int8 = round(pixel / 255 * 254 - 127)
    arr = np.round(arr / 255.0 * 254.0 - 127.0)
    arr = np.clip(arr, -128, 127).astype(np.int8)
    # HWC -> CHW
    arr = arr.transpose(2, 0, 1)
    return arr


def export_conv_weights(out_dir, layer_idx, ltype, weights_i, biases_i,
                        c_in, c_out, kernel):
    """Export weights and biases in hex format for a conv layer."""
    if ltype == 'conv3x3':
        # Conv 3x3: transpose [co][ci][ky][kx] -> [co][ky][kx][ci], flatten
        w_reordered = weights_i.transpose(0, 2, 3, 1)  # [co][ky][kx][cin]
        write_hex8_file(out_dir / f"layer{layer_idx}_weights.hex",
                        w_reordered.flatten())
    elif ltype in ('conv1x1', 'conv1x1_linear'):
        # 1x1: [co][ci][1][1] -> [co][ci] (cin-contiguous)
        write_hex8_file(out_dir / f"layer{layer_idx}_weights.hex",
                        weights_i.reshape(c_out, c_in).flatten())
    write_bias_hex(out_dir / f"layer{layer_idx}_bias.hex", biases_i)


def main():
    parser = argparse.ArgumentParser(description='DPU 36-layer golden model (YOLOv4-tiny full)')
    parser.add_argument('--real-weights', action='store_true',
                        help='Use real YOLOv4-tiny quantized weights')
    parser.add_argument('--per-layer-scale', action='store_true',
                        help='Calibrate per-layer requant scale (auto with --real-weights)')
    parser.add_argument('--input-image', type=str, default=None,
                        help='Path to input image (resized to H0xW0, converted to INT8)')
    args = parser.parse_args()
    use_per_layer_scale = args.per_layer_scale or args.real_weights

    np.random.seed(42)

    out_dir = PROJECT_ROOT / "image_sim_out" / "dpu_top_37"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Generate or load input image
    if args.input_image:
        print(f"[INPUT] Loading image: {args.input_image}")
        input_img = load_input_image(args.input_image)
        print(f"  Resized to {W0}x{H0}, range=[{input_img.min()}, {input_img.max()}]")
    else:
        input_img = np.random.randint(-50, 50, (3, H0, W0), dtype=np.int8)
    write_hex8_file(out_dir / "input_image.hex", input_img.flatten())

    # Storage
    layer_outputs = [None] * NUM_LAYERS
    save_l2 = None
    save_l4 = None
    save_l10 = None
    save_l12 = None
    save_l18 = None
    save_l20 = None
    save_l23 = None
    save_l27 = None

    # Load or generate weights
    weights = {}
    biases = {}

    if args.real_weights:
        print("[MODE] Using REAL YOLOv4-tiny weights")
        weights, biases = load_real_weights()
        if weights is None:
            return 1
    else:
        print("[MODE] Using synthetic random weights")
        for i, (ltype, c_in, c_out, stride, kernel) in enumerate(LAYER_DEFS):
            if ltype in ('conv3x3', 'conv1x1', 'conv1x1_linear'):
                k = kernel if kernel > 0 else 1
                w = np.random.randint(-20, 20, (c_out, c_in, k, k), dtype=np.int8)
                b = np.random.randint(-200, 200, c_out, dtype=np.int32)
                weights[i] = w
                biases[i] = b

    layer_scales = [SCALE_RTL] * NUM_LAYERS  # default: uniform scale
    print(f"[CONFIG] H0={H0}, W0={W0}, NUM_LAYERS={NUM_LAYERS}")
    print(f"[CONFIG] default_scale={SCALE_RTL} (0x{SCALE_RTL:04X}), shift={SHIFT_RTL}")
    if use_per_layer_scale:
        print("[CONFIG] Per-layer scale calibration ENABLED")

    # Run all layers
    current_fmap = input_img

    for i, (ltype, c_in, c_out, stride, kernel) in enumerate(LAYER_DEFS):
        if ltype == 'conv3x3':
            if use_per_layer_scale:
                layer_scales[i] = calibrate_scale(current_fmap, weights[i], biases[i],
                                                  stride=stride, kernel_size=3)
            out = conv_bn_leaky_hw(current_fmap, weights[i], biases[i],
                                   stride=stride, kernel_size=3, scale=layer_scales[i])
            export_conv_weights(out_dir, i, ltype, weights[i], biases[i],
                                c_in, c_out, kernel)

        elif ltype == 'conv1x1':
            if use_per_layer_scale:
                layer_scales[i] = calibrate_scale(current_fmap, weights[i], biases[i],
                                                  stride=1, kernel_size=1)
            out = conv_bn_leaky_hw(current_fmap, weights[i], biases[i],
                                   stride=1, kernel_size=1, scale=layer_scales[i])
            export_conv_weights(out_dir, i, ltype, weights[i], biases[i],
                                c_in, c_out, kernel)

        elif ltype == 'conv1x1_linear':
            if use_per_layer_scale:
                layer_scales[i] = calibrate_scale(current_fmap, weights[i], biases[i],
                                                  stride=1, kernel_size=1, linear=True)
            conv_out = conv2d_1x1(current_fmap, weights[i], biases[i])
            # NO LeakyReLU — go directly to requantize
            out = requantize_fixed_point(conv_out, np.int32(layer_scales[i]), SHIFT_RTL)
            export_conv_weights(out_dir, i, ltype, weights[i], biases[i],
                                c_in, c_out, kernel)

        elif ltype == 'route_split':
            out = route_split(current_fmap, groups=2, group_id=1)

        elif ltype == 'route_concat':
            if i == 6:
                out = route_concat([layer_outputs[5], save_l4])
            elif i == 8:
                out = route_concat([save_l2, layer_outputs[7]])
            elif i == 14:
                out = route_concat([layer_outputs[13], save_l12])
            elif i == 16:
                out = route_concat([save_l10, layer_outputs[15]])
            elif i == 22:
                out = route_concat([layer_outputs[21], save_l20])
            elif i == 24:
                out = route_concat([save_l18, layer_outputs[23]])
            elif i == 33:
                out = route_concat([layer_outputs[32], save_l23])
            else:
                raise ValueError(f"Unknown concat layer {i}")

        elif ltype == 'maxpool':
            out = maxpool_2x2(current_fmap)

        elif ltype == 'route_save':
            # Route from saved layer 27 output (darknet layer 31 = route to L27)
            assert save_l27 is not None, "save_l27 must be set before route_save at layer 30"
            out = save_l27.copy()

        elif ltype == 'upsample':
            out = upsample_2x(current_fmap)

        else:
            raise ValueError(f"Unknown layer type {ltype}")

        layer_outputs[i] = out
        current_fmap = out

        # Save points for later route/concat references
        if i == 2:
            save_l2 = out.copy()
        elif i == 4:
            save_l4 = out.copy()
        elif i == 10:
            save_l10 = out.copy()
        elif i == 12:
            save_l12 = out.copy()
        elif i == 18:
            save_l18 = out.copy()
        elif i == 20:
            save_l20 = out.copy()
        elif i == 23:
            save_l23 = out.copy()
        elif i == 27:
            save_l27 = out.copy()

        write_hex8_file(out_dir / f"layer{i}_expected.hex", out.flatten())

        # Count non-zero and saturated values for diagnostics
        nz = np.count_nonzero(out)
        sat = np.sum((out == 127) | (out == -128))
        total = out.size
        scale_str = (f"scale={layer_scales[i]:5d}"
                     if ltype in ('conv3x3', 'conv1x1', 'conv1x1_linear')
                     else "           ")
        print(f"Layer {i:2d} ({ltype:16s}): {c_in:3d} -> {c_out:3d}  "
              f"shape={str(out.shape):20s}  range=[{out.min():4d}, {out.max():3d}]  "
              f"nonzero={nz}/{total} sat={sat} {scale_str}")

    # Save output 1 (layer 29) and output 2 (layer 35) as named files
    write_hex8_file(out_dir / "output1_expected.hex", layer_outputs[29].flatten())
    write_hex8_file(out_dir / "output2_expected.hex", layer_outputs[35].flatten())
    write_hex8_file(out_dir / "final_output_expected.hex", current_fmap.flatten())

    # Export per-layer scales (36 entries x 2 bytes LE = 72 hex lines)
    with open(out_dir / "scales.hex", 'w') as f:
        for li in range(NUM_LAYERS):
            s = layer_scales[li] & 0xffff
            f.write(f"{s & 0xff:02x}\n")
            f.write(f"{(s >> 8) & 0xff:02x}\n")

    print(f"\nOutput 1 shape (layer 29): {layer_outputs[29].shape}")
    print(f"Output 2 shape (layer 35): {layer_outputs[35].shape}")
    print(f"Final output shape: {current_fmap.shape}")
    print(f"Files written to: {out_dir}")
    print("GOLDEN COMPLETE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
