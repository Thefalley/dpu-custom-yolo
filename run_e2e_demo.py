#!/usr/bin/env python3
"""
End-to-End Detection Demo — DPU Custom YOLOv4-tiny

Complete pipeline:
  1. Load a real image (or use test pattern)
  2. Run DPU backbone golden model (18 layers, real weights)
  3. Run YOLO post-processing (detection head + NMS)
  4. Draw bounding boxes on image
  5. Save annotated result as PNG

Usage:
  python run_e2e_demo.py                           # Test pattern, synthetic weights
  python run_e2e_demo.py --real-weights             # Test pattern, real weights
  python run_e2e_demo.py --image dog.jpg            # Real image, synthetic weights
  python run_e2e_demo.py --image dog.jpg --real-weights  # Full real demo
  python run_e2e_demo.py --resolution 416           # Full 416x416 resolution
"""
import sys
import argparse
import time
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "tests"))

from phase3_dpu_functional_model import (
    conv2d_3x3, conv2d_1x1,
    leaky_relu_hardware, requantize_fixed_point,
    maxpool_2x2, route_split, route_concat,
)

# YOLOv4-tiny layer definitions (same as golden model)
LAYER_DEFS = [
    ('conv3x3',      3,   32, 2, 3),
    ('conv3x3',     32,   64, 2, 3),
    ('conv3x3',     64,   64, 1, 3),
    ('route_split', 64,   32, 0, 0),
    ('conv3x3',     32,   32, 1, 3),
    ('conv3x3',     32,   32, 1, 3),
    ('route_concat',32,   64, 0, 0),
    ('conv1x1',     64,   64, 1, 1),
    ('route_concat',64,  128, 0, 0),
    ('maxpool',    128,  128, 2, 2),
    ('conv3x3',    128,  128, 1, 3),
    ('route_split',128,   64, 0, 0),
    ('conv3x3',     64,   64, 1, 3),
    ('conv3x3',     64,   64, 1, 3),
    ('route_concat',64,  128, 0, 0),
    ('conv1x1',    128,  128, 1, 1),
    ('route_concat',128, 256, 0, 0),
    ('maxpool',    256,  256, 2, 2),
]


def load_image(path, h, w):
    """Load image, resize to (h, w), convert to INT8 CHW."""
    try:
        from PIL import Image
        img = Image.open(path).convert('RGB').resize((w, h))
        arr = np.array(img, dtype=np.float32)
        # Normalize to [-128, 127]
        arr = arr - 128.0
        arr = np.clip(arr, -128, 127).astype(np.int8)
        # HWC -> CHW
        arr = arr.transpose(2, 0, 1)
        return arr, img
    except ImportError:
        print("[WARN] PIL not available, using test pattern")
        return None, None


def generate_test_pattern(h, w):
    """Generate a synthetic test pattern."""
    np.random.seed(42)
    return np.random.randint(-128, 128, (3, h, w), dtype=np.int8)


def run_backbone(input_fmap, weights_dict, layer_scales, H0, W0):
    """Run the 18-layer YOLOv4-tiny backbone."""
    layer_outputs = [None] * 18
    current_fmap = input_fmap
    save_l2 = save_l4 = save_l10 = save_l12 = None

    for i, (ltype, c_in, c_out, stride, kernel) in enumerate(LAYER_DEFS):
        if ltype in ('conv3x3', 'conv1x1'):
            w = weights_dict[f'layer{i}_weights']
            b = weights_dict[f'layer{i}_bias']
            scale = layer_scales[i]

            if ltype == 'conv3x3':
                out32 = conv2d_3x3(current_fmap, w, b, stride)
            else:
                out32 = conv2d_1x1(current_fmap, w, b)

            out32 = leaky_relu_hardware(out32)
            out = requantize_fixed_point(out32, scale, 16)

        elif ltype == 'maxpool':
            out = maxpool_2x2(current_fmap)

        elif ltype == 'route_split':
            out = route_split(current_fmap)

        elif ltype == 'route_concat':
            if i == 6:
                out = route_concat([layer_outputs[5], save_l4])
            elif i == 8:
                out = route_concat([save_l2, layer_outputs[7]])
            elif i == 14:
                out = route_concat([layer_outputs[13], save_l12])
            elif i == 16:
                out = route_concat([save_l10, layer_outputs[15]])
            else:
                raise ValueError(f"Unexpected route_concat at layer {i}")
        else:
            raise ValueError(f"Unknown layer type {ltype}")

        layer_outputs[i] = out
        current_fmap = out

        if i == 2:   save_l2  = out.copy()
        elif i == 4: save_l4  = out.copy()
        elif i == 10: save_l10 = out.copy()
        elif i == 12: save_l12 = out.copy()

        shape_str = f"{out.shape}"
        print(f"  Layer {i:2d} ({ltype:14s}): {shape_str:20s} range=[{out.min():4d}, {out.max():3d}]")

    return layer_outputs


def generate_weights_synthetic(layer_scales):
    """Generate synthetic random weights for all conv layers."""
    np.random.seed(12345)
    weights = {}
    for i, (ltype, c_in, c_out, stride, kernel) in enumerate(LAYER_DEFS):
        if ltype in ('conv3x3', 'conv1x1'):
            k = 3 if ltype == 'conv3x3' else 1
            weights[f'layer{i}_weights'] = np.random.randint(-5, 6, (c_out, c_in, k, k), dtype=np.int8)
            weights[f'layer{i}_bias'] = np.random.randint(-100, 100, c_out, dtype=np.int32)
            layer_scales[i] = 655  # default
    return weights


def load_real_weights(layer_scales):
    """Load real YOLOv4-tiny quantized weights."""
    npz_path = PROJECT_ROOT / "image_sim_out" / "dpu_top_real" / "quantized_weights.npz"
    if not npz_path.exists():
        # Try to generate them
        print("  Generating real weights (downloading YOLOv4-tiny)...")
        import subprocess
        r = subprocess.run(
            [sys.executable, str(PROJECT_ROOT / "tests" / "load_yolov4_tiny_weights.py")],
            cwd=str(PROJECT_ROOT), capture_output=True, text=True
        )
        if r.returncode != 0:
            print(f"  [WARN] Cannot load real weights: {r.stderr[-200:]}")
            return None

    if not npz_path.exists():
        return None

    data = np.load(npz_path)
    weights = {}
    for i, (ltype, c_in, c_out, stride, kernel) in enumerate(LAYER_DEFS):
        if ltype in ('conv3x3', 'conv1x1'):
            wkey = f'layer{i}_weights'
            bkey = f'layer{i}_bias'
            skey = f'layer{i}_scale'
            if wkey in data:
                weights[wkey] = data[wkey]
                weights[bkey] = data[bkey]
                if skey in data:
                    layer_scales[i] = int(data[skey])
    return weights


def draw_detections(image_np, detections, input_h, input_w, orig_h, orig_w):
    """Draw bounding boxes on image. Returns annotated image as numpy array."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches

        fig, ax = plt.subplots(1, figsize=(12, 8))
        # Convert image for display
        if image_np is not None:
            ax.imshow(image_np)
        else:
            # Create blank image
            ax.set_xlim(0, orig_w)
            ax.set_ylim(orig_h, 0)

        colors = plt.cm.Set3(np.linspace(0, 1, 20))

        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            # Scale from input coords to original image coords
            sx = orig_w / input_w
            sy = orig_h / input_h
            x1, x2 = x1 * sx, x2 * sx
            y1, y2 = y1 * sy, y2 * sy

            cls_id = det['class_id']
            cls_name = det['class_name']
            conf = det['confidence']
            color = colors[cls_id % len(colors)]

            rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                      linewidth=2, edgecolor=color, facecolor='none')
            ax.add_patch(rect)
            ax.text(x1, y1 - 3, f"{cls_name} {conf:.2f}",
                    color='white', fontsize=8, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor=color, alpha=0.8))

        ax.set_title(f"YOLOv4-tiny Detection — {len(detections)} objects",
                     fontsize=14, fontweight='bold')
        ax.axis('off')

        plt.tight_layout()
        return fig

    except ImportError:
        print("[WARN] matplotlib not available, skipping visualization")
        return None


def main():
    ap = argparse.ArgumentParser(description="End-to-End YOLOv4-tiny Detection Demo")
    ap.add_argument("--image", type=str, default=None, help="Path to input image")
    ap.add_argument("--real-weights", action="store_true", help="Use real YOLOv4-tiny weights")
    ap.add_argument("--resolution", type=int, default=16, help="Input resolution (16 or 416)")
    ap.add_argument("--conf-threshold", type=float, default=0.1, help="Detection confidence threshold")
    ap.add_argument("--nms-threshold", type=float, default=0.45, help="NMS IoU threshold")
    args = ap.parse_args()

    H0 = W0 = args.resolution

    print("=" * 70)
    print("  YOLOv4-tiny End-to-End Detection Demo")
    print(f"  Resolution: {H0}x{W0}")
    print(f"  Weights: {'real' if args.real_weights else 'synthetic'}")
    print(f"  Image: {args.image or 'test pattern'}")
    print("=" * 70)

    # ---- Step 1: Load/generate input ----
    print("\n[1/5] Preparing input...")
    orig_img = None
    if args.image:
        input_fmap, orig_img = load_image(args.image, H0, W0)
        if input_fmap is None:
            print("  Falling back to test pattern")
            input_fmap = generate_test_pattern(H0, W0)
    else:
        input_fmap = generate_test_pattern(H0, W0)
    print(f"  Input shape: {input_fmap.shape}")

    # ---- Step 2: Load weights ----
    print("\n[2/5] Loading weights...")
    layer_scales = [655] * 18  # default
    if args.real_weights:
        weights = load_real_weights(layer_scales)
        if weights is None:
            print("  Real weights not available, using synthetic")
            weights = generate_weights_synthetic(layer_scales)
        else:
            print(f"  Loaded real YOLOv4-tiny weights")
    else:
        weights = generate_weights_synthetic(layer_scales)
        print("  Using synthetic weights")

    # ---- Step 3: Run backbone ----
    print("\n[3/5] Running DPU backbone (18 layers)...")
    t0 = time.time()
    layer_outputs = run_backbone(input_fmap, weights, layer_scales, H0, W0)
    elapsed = time.time() - t0
    print(f"  Backbone complete in {elapsed:.2f}s")

    backbone_l8  = layer_outputs[8]   # 128ch, for skip connection
    backbone_l15 = layer_outputs[15]  # 128ch, scale 1 input
    backbone_l17 = layer_outputs[17]  # 256ch, final output

    print(f"  L8:  {backbone_l8.shape}")
    print(f"  L15: {backbone_l15.shape}")
    print(f"  L17: {backbone_l17.shape}")

    # ---- Step 4: Run YOLO post-processing ----
    print("\n[4/5] Running YOLO post-processing...")
    from yolo_postprocess import DetectionHead, decode_yolo, nms, COCO_CLASSES, ANCHORS

    det_head = DetectionHead()
    det_scale0, det_scale1 = det_head.forward(backbone_l15, backbone_l8, backbone_l17)

    print(f"  Detection scale 0: {det_scale0.shape}")
    print(f"  Detection scale 1: {det_scale1.shape}")

    # Decode predictions
    from yolo_postprocess import NUM_CLASSES
    all_boxes = []
    for scale_idx, (det_tensor, anchors) in enumerate(
            zip([det_scale0, det_scale1], ANCHORS)):
        boxes = decode_yolo(det_tensor, anchors, NUM_CLASSES, (H0, W0))
        print(f"  Scale {scale_idx}: {len(boxes)} raw detections")
        if len(boxes) > 0:
            all_boxes.append(boxes)

    if all_boxes:
        all_boxes = np.concatenate(all_boxes, axis=0)
        # Filter by confidence
        mask = all_boxes[:, 4] > args.conf_threshold
        all_boxes = all_boxes[mask]
        print(f"  After conf filter (>{args.conf_threshold}): {len(all_boxes)} detections")
    else:
        all_boxes = np.zeros((0, 6))

    # NMS
    final_dets = nms(all_boxes, iou_threshold=args.nms_threshold)
    print(f"  After NMS: {len(final_dets)} detections")

    # Format detections
    detections = []
    if len(final_dets) > 0:
        for row in final_dets:
            x1, y1, x2, y2, conf, cls_id = row[0], row[1], row[2], row[3], row[4], int(row[5])
            # Scale bbox from [0,1] to pixel coordinates
            detections.append({
                'bbox': (x1 * W0, y1 * H0, x2 * W0, y2 * H0),
                'confidence': float(conf),
                'class_id': cls_id,
                'class_name': COCO_CLASSES[cls_id] if cls_id < len(COCO_CLASSES) else f"cls{cls_id}",
            })

    # Print detections
    if detections:
        print(f"\n  Detected objects:")
        for d in detections[:20]:
            print(f"    {d['class_name']:15s}  conf={d['confidence']:.3f}  "
                  f"bbox=({d['bbox'][0]:.0f},{d['bbox'][1]:.0f},{d['bbox'][2]:.0f},{d['bbox'][3]:.0f})")
        if len(detections) > 20:
            print(f"    ... and {len(detections) - 20} more")
    else:
        print("  No detections above threshold")

    # ---- Step 5: Visualize ----
    print("\n[5/5] Generating visualization...")
    orig_h, orig_w = H0, W0
    display_img = None
    if orig_img is not None:
        display_img = np.array(orig_img)
        orig_h, orig_w = display_img.shape[:2]

    fig = draw_detections(display_img, detections, H0, W0, orig_h, orig_w)

    out_dir = PROJECT_ROOT / "image_sim_out" / "e2e_demo"
    out_dir.mkdir(parents=True, exist_ok=True)

    if fig is not None:
        out_path = out_dir / "detection_result.png"
        fig.savefig(str(out_path), dpi=150, bbox_inches='tight')
        import matplotlib.pyplot as plt
        plt.close(fig)
        print(f"  Saved: {out_path}")

    # Save detections as text
    det_path = out_dir / "detections.txt"
    with open(det_path, 'w') as f:
        f.write(f"YOLOv4-tiny Detection Results\n")
        f.write(f"Input: {args.image or 'test pattern'}\n")
        f.write(f"Resolution: {H0}x{W0}\n")
        f.write(f"Weights: {'real' if args.real_weights else 'synthetic'}\n")
        f.write(f"Detections: {len(detections)}\n\n")
        for d in detections:
            f.write(f"{d['class_name']},{d['confidence']:.4f},"
                    f"{d['bbox'][0]:.1f},{d['bbox'][1]:.1f},"
                    f"{d['bbox'][2]:.1f},{d['bbox'][3]:.1f}\n")
    print(f"  Saved: {det_path}")

    # Save backbone outputs as npz for further analysis
    npz_path = out_dir / "backbone_outputs.npz"
    np.savez(str(npz_path),
             input=input_fmap,
             layer8=backbone_l8,
             layer15=backbone_l15,
             layer17=backbone_l17)
    print(f"  Saved: {npz_path}")

    # ---- Summary ----
    print(f"\n{'=' * 70}")
    print(f"  END-TO-END DEMO COMPLETE")
    print(f"  Resolution: {H0}x{W0}")
    print(f"  Backbone: 18 layers, {elapsed:.2f}s")
    print(f"  Detections: {len(detections)} objects")
    print(f"  Output: {out_dir}")
    print(f"{'=' * 70}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
