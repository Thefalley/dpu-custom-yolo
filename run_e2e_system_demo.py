#!/usr/bin/env python3
"""
End-to-End System Demo — Full YOLOv4-tiny on DPU (36 layers)

Complete pipeline:
  1. Load a real image (dog.jpg by default)
  2. Download + quantize real YOLOv4-tiny weights (all 21 conv layers)
  3. Run 36-layer DPU golden model (Python)
  4. Optionally run 36-layer RTL simulation (Icarus Verilog)
  5. Decode YOLO detection outputs from both heads (layers 29 & 35)
  6. Apply NMS and draw bounding boxes on original image
  7. Save annotated result

Usage:
  python run_e2e_system_demo.py                           # Python golden only
  python run_e2e_system_demo.py --rtl                     # Include RTL simulation
  python run_e2e_system_demo.py --image horses.jpg        # Different image
  python run_e2e_system_demo.py --synthetic-weights       # Use random weights (fast)
"""
import sys
import os
import time
import argparse
import subprocess
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "tests"))

H0, W0 = 32, 32  # DPU input resolution
NUM_LAYERS = 36

# COCO class names (80 classes)
COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep",
    "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
    "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
    "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv",
    "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
    "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush"
]

# YOLOv4-tiny anchors (2 scales, 3 anchors each)
# Scale 0 = larger feature map (4x4 for 32x32 input), Scale 1 = smaller (2x2)
ANCHORS = [
    [(81, 82), (135, 169), (344, 319)],    # Scale 0 (layer 29 output, 2x2 grid)
    [(10, 14), (23, 27), (37, 58)],        # Scale 1 (layer 35 output, 4x4 grid)
]


def load_image(path, h, w):
    """Load image, resize to (h, w), convert to INT8 CHW."""
    from PIL import Image
    img = Image.open(path).convert('RGB')
    orig_size = img.size  # (W, H)
    img_resized = img.resize((w, h), Image.BILINEAR)
    arr = np.array(img_resized, dtype=np.float32)
    # Map to signed INT8: pixel_int8 = round(pixel/255 * 254 - 127)
    arr = np.round(arr / 255.0 * 254.0 - 127.0)
    arr = np.clip(arr, -128, 127).astype(np.int8)
    arr = arr.transpose(2, 0, 1)  # HWC -> CHW
    return arr, img, orig_size


def sigmoid(x):
    x = np.clip(x.astype(np.float64), -500, 500)
    return 1.0 / (1.0 + np.exp(-x))


def decode_yolo(raw_tensor, anchors, num_classes=80, input_size=(32, 32)):
    """
    Decode a YOLO detection tensor.

    Args:
        raw_tensor: (255, grid_h, grid_w) INT8 from DPU
        anchors: list of (anchor_w, anchor_h) for this scale
        num_classes: number of classes (80 for COCO)
        input_size: (H, W) of the DPU input

    Returns:
        boxes: (N, 6) array of [x1, y1, x2, y2, confidence, class_id]
    """
    num_anchors = len(anchors)
    grid_h, grid_w = raw_tensor.shape[1], raw_tensor.shape[2]
    inp_h, inp_w = input_size

    # Reshape: (3, 85, grid_h, grid_w)
    pred = raw_tensor.reshape(num_anchors, 5 + num_classes, grid_h, grid_w)
    pred = pred.astype(np.float64)

    # The detection output is INT8 after requantization.
    # Scale back to a reasonable range for sigmoid/exp operations.
    # The linear activation preserves sign/magnitude, so INT8 values
    # represent the raw logits scaled by the requant factor.
    # A rough dequant: treat INT8 as ~1/16 of the original logit range.
    pred = pred / 16.0

    grid_y, grid_x = np.meshgrid(np.arange(grid_h), np.arange(grid_w), indexing='ij')

    boxes = []
    for a in range(num_anchors):
        tx = pred[a, 0]  # center x offset
        ty = pred[a, 1]  # center y offset
        tw = pred[a, 2]  # width
        th = pred[a, 3]  # height
        to = pred[a, 4]  # objectness

        bx = (sigmoid(tx) + grid_x) / grid_w
        by = (sigmoid(ty) + grid_y) / grid_h

        aw, ah = anchors[a]
        bw = np.exp(np.clip(tw, -5, 5)) * aw / inp_w
        bh = np.exp(np.clip(th, -5, 5)) * ah / inp_h

        obj = sigmoid(to)

        cls_pred = pred[a, 5:]  # (80, grid_h, grid_w)
        cls_prob = sigmoid(cls_pred)

        conf = obj[np.newaxis] * cls_prob

        for gy in range(grid_h):
            for gx in range(grid_w):
                for c in range(num_classes):
                    score = conf[c, gy, gx]
                    if score > 0.001:
                        cx = bx[gy, gx]
                        cy = by[gy, gx]
                        w = bw[gy, gx]
                        h = bh[gy, gx]
                        x1 = max(0, cx - w / 2)
                        y1 = max(0, cy - h / 2)
                        x2 = min(1, cx + w / 2)
                        y2 = min(1, cy + h / 2)
                        boxes.append([x1, y1, x2, y2, score, c])

    return np.array(boxes) if boxes else np.zeros((0, 6))


def nms(boxes, iou_threshold=0.45):
    """Non-Maximum Suppression."""
    if len(boxes) == 0:
        return boxes

    order = np.argsort(-boxes[:, 4])
    boxes = boxes[order]

    keep = []
    suppressed = np.zeros(len(boxes), dtype=bool)

    for i in range(len(boxes)):
        if suppressed[i]:
            continue
        keep.append(i)
        for j in range(i + 1, len(boxes)):
            if suppressed[j] or boxes[i, 5] != boxes[j, 5]:
                continue
            xx1 = max(boxes[i, 0], boxes[j, 0])
            yy1 = max(boxes[i, 1], boxes[j, 1])
            xx2 = min(boxes[i, 2], boxes[j, 2])
            yy2 = min(boxes[i, 3], boxes[j, 3])
            inter = max(0, xx2 - xx1) * max(0, yy2 - yy1)
            area_i = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
            area_j = (boxes[j, 2] - boxes[j, 0]) * (boxes[j, 3] - boxes[j, 1])
            union = area_i + area_j - inter
            if inter / (union + 1e-10) > iou_threshold:
                suppressed[j] = True

    return boxes[keep]


def draw_detections(orig_img, detections, output_path):
    """Draw bounding boxes on the original image."""
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        print("  [WARN] PIL not available, skipping visualization")
        return False

    img = orig_img.copy()
    draw = ImageDraw.Draw(img)
    img_w, img_h = img.size

    colors = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
        (0, 255, 255), (255, 0, 255), (255, 128, 0), (128, 0, 255),
        (0, 128, 255), (255, 0, 128), (128, 255, 0), (0, 255, 128),
    ]

    for det in detections:
        x1, y1, x2, y2 = det['box']
        # Scale from [0,1] normalized coords to pixel coords
        px1, py1 = int(x1 * img_w), int(y1 * img_h)
        px2, py2 = int(x2 * img_w), int(y2 * img_h)

        color = colors[det['class_id'] % len(colors)]
        for t in range(3):  # thick border
            draw.rectangle([px1 - t, py1 - t, px2 + t, py2 + t], outline=color)

        label = f"{det['class_name']} {det['score']:.2f}"
        # Draw label background
        tw = len(label) * 7
        draw.rectangle([px1, max(0, py1 - 16), px1 + tw, py1], fill=color)
        draw.text((px1 + 2, max(0, py1 - 14)), label, fill=(255, 255, 255))

    img.save(str(output_path))
    return True


def run_golden_model(input_image_path=None, real_weights=True):
    """Run the 36-layer golden model and return detection outputs."""
    cmd = [sys.executable, str(PROJECT_ROOT / "tests" / "dpu_top_37layer_golden.py")]
    if real_weights:
        cmd.append("--real-weights")
    if input_image_path:
        cmd.extend(["--input-image", str(input_image_path)])

    r = subprocess.run(cmd, cwd=str(PROJECT_ROOT),
                       capture_output=True, text=True, timeout=120)
    out = (r.stdout or "") + (r.stderr or "")

    if r.returncode != 0 or "GOLDEN COMPLETE" not in out:
        print("  [FAIL] Golden model failed")
        for line in out.strip().splitlines()[-20:]:
            print(f"    {line}")
        return None

    # Print layer summary
    for line in out.strip().splitlines():
        if line.startswith("Layer") or line.startswith("["):
            print(f"    {line}")

    return True


def run_rtl_simulation():
    """Run 36-layer RTL simulation and return pass/fail."""
    sv_py = PROJECT_ROOT / "verilog-sim-py" / "sv_simulator.py"
    files = [
        PROJECT_ROOT / "rtl" / "dpu" / "primitives" / "mac_int8.sv",
        PROJECT_ROOT / "rtl" / "dpu" / "primitives" / "leaky_relu.sv",
        PROJECT_ROOT / "rtl" / "dpu" / "primitives" / "requantize.sv",
        PROJECT_ROOT / "rtl" / "dpu" / "mac_array_32x32.sv",
        PROJECT_ROOT / "rtl" / "dpu" / "post_process_array.sv",
        PROJECT_ROOT / "rtl" / "dpu" / "conv_engine_array.sv",
        PROJECT_ROOT / "rtl" / "dpu" / "maxpool_unit.sv",
        PROJECT_ROOT / "rtl" / "dpu" / "dpu_top.sv",
        PROJECT_ROOT / "rtl" / "tb" / "dpu_top_36layer_tb.sv",
    ]
    cmd = ([sys.executable, str(sv_py), "--no-wave"] +
           [str(f) for f in files] +
           ["--top", "dpu_top_36layer_tb"])

    env = dict(os.environ)
    if "OSS_CAD_PATH" not in env:
        ocp = PROJECT_ROOT / ".oss_cad_path"
        if ocp.exists():
            env["OSS_CAD_PATH"] = ocp.read_text().strip()

    r = subprocess.run(cmd, cwd=str(PROJECT_ROOT), capture_output=True, text=True,
                       timeout=7200, env=env)
    out = (r.stdout or "") + (r.stderr or "")

    passed = "ALL 36 LAYERS PASS" in out

    # Print key lines
    for line in out.strip().splitlines():
        if any(kw in line for kw in ["PASS", "FAIL", "MISMATCH", "Layer",
                                      "TOTAL", "reload_req", "===", "RESULT"]):
            print(f"    {line}")

    return passed, out


def load_detection_outputs():
    """Load the golden model's detection head outputs (layers 29 and 35)."""
    out_dir = PROJECT_ROOT / "image_sim_out" / "dpu_top_37"

    l29_path = out_dir / "layer29_expected.hex"
    l35_path = out_dir / "layer35_expected.hex"

    if not l29_path.exists() or not l35_path.exists():
        print("  [ERROR] Detection outputs not found")
        return None, None

    def read_hex_int8(path):
        vals = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    v = int(line, 16)
                    if v > 127:
                        v -= 256
                    vals.append(v)
        return np.array(vals, dtype=np.int8)

    # Layer 29: 255ch, spatial = total_elements / 255
    # With H0=32: input goes through 4 stride-2 stages (s2,s2,mp,mp,mp) = /32
    # Layer 29 is after 3 maxpool + 2 stride-2 conv = H0/32, W0/32
    # But minimum 1x1, so for H0=32: 1x1
    l29_flat = read_hex_int8(l29_path)
    n29 = l29_flat.size // 255
    h29 = w29 = int(np.sqrt(n29))
    if h29 * w29 * 255 != l29_flat.size:
        # Non-square or odd size — try common dims
        for h_try in range(1, n29 + 1):
            if n29 % h_try == 0:
                h29, w29 = h_try, n29 // h_try
    l29 = l29_flat.reshape(255, h29, w29)

    # Layer 35: 255ch, spatial from the upsampled branch
    l35_flat = read_hex_int8(l35_path)
    n35 = l35_flat.size // 255
    h35 = w35 = int(np.sqrt(n35))
    if h35 * w35 * 255 != l35_flat.size:
        for h_try in range(1, n35 + 1):
            if n35 % h_try == 0:
                h35, w35 = h_try, n35 // h_try
    l35 = l35_flat.reshape(255, h35, w35)

    return l29, l35


def main():
    ap = argparse.ArgumentParser(description="End-to-End YOLOv4-tiny System Demo (36 layers)")
    ap.add_argument("--image", type=str, default=None,
                    help="Path to input image (default: image_sim_out/rtl_validation/dog.jpg)")
    ap.add_argument("--rtl", action="store_true",
                    help="Also run RTL simulation (slow but verifies hardware)")
    ap.add_argument("--synthetic-weights", action="store_true",
                    help="Use synthetic random weights instead of real YOLOv4-tiny")
    ap.add_argument("--conf-threshold", type=float, default=0.1,
                    help="Detection confidence threshold")
    ap.add_argument("--nms-threshold", type=float, default=0.45,
                    help="NMS IoU threshold")
    args = ap.parse_args()

    # Default image
    if args.image is None:
        default_img = PROJECT_ROOT / "image_sim_out" / "rtl_validation" / "dog.jpg"
        if default_img.exists():
            args.image = str(default_img)

    use_real = not args.synthetic_weights

    print("=" * 72)
    print("  YOLOv4-tiny FULL SYSTEM DEMO — 36-Layer DPU")
    print(f"  Resolution: {H0}x{W0}  |  Weights: {'REAL' if use_real else 'synthetic'}")
    print(f"  Image: {Path(args.image).name if args.image else 'test pattern'}")
    print(f"  RTL simulation: {'YES' if args.rtl else 'no (Python golden only)'}")
    print("=" * 72)

    # ================================================================
    # Step 1: Load input image
    # ================================================================
    print("\n[1/6] Loading input image...")
    orig_img = None
    orig_size = (W0, H0)
    if args.image:
        try:
            input_fmap, orig_img, orig_size = load_image(args.image, H0, W0)
            print(f"  Loaded: {args.image}")
            print(f"  Original: {orig_size[0]}x{orig_size[1]} -> DPU: {W0}x{H0}")
            print(f"  INT8 range: [{input_fmap.min()}, {input_fmap.max()}]")
        except Exception as e:
            print(f"  [WARN] Cannot load image: {e}")
            print(f"  Using test pattern")
            args.image = None

    if args.image is None:
        np.random.seed(42)
        input_fmap = np.random.randint(-50, 50, (3, H0, W0), dtype=np.int8)
        print(f"  Using synthetic test pattern ({H0}x{W0})")

    # ================================================================
    # Step 2: Prepare weights
    # ================================================================
    print("\n[2/6] Preparing weights...")
    if use_real:
        npz_path = PROJECT_ROOT / "image_sim_out" / "dpu_top_real" / "quantized_weights.npz"
        if not npz_path.exists():
            print("  Downloading + quantizing real YOLOv4-tiny weights...")
            r = subprocess.run(
                [sys.executable, str(PROJECT_ROOT / "tests" / "load_yolov4_tiny_weights.py")],
                cwd=str(PROJECT_ROOT), capture_output=True, text=True, timeout=120
            )
            if r.returncode != 0:
                print(f"  [ERROR] Weight conversion failed")
                for line in (r.stderr or "").splitlines()[-5:]:
                    print(f"    {line}")
                return 1
        print(f"  Real weights ready: {npz_path.name}")
        print(f"  21 conv layers quantized to INT8")
    else:
        print("  Using synthetic random weights")

    # ================================================================
    # Step 3: Run 36-layer golden model
    # ================================================================
    print("\n[3/6] Running 36-layer golden model (Python)...")
    t0 = time.time()
    ok = run_golden_model(
        input_image_path=args.image,
        real_weights=use_real
    )
    golden_time = time.time() - t0

    if not ok:
        print(f"  [FAIL] Golden model failed after {golden_time:.1f}s")
        return 1
    print(f"  Golden complete in {golden_time:.1f}s")

    # ================================================================
    # Step 4: RTL simulation (optional)
    # ================================================================
    rtl_passed = None
    rtl_time = 0
    if args.rtl:
        print("\n[4/6] Running 36-layer RTL simulation (Icarus Verilog)...")
        t0 = time.time()
        rtl_passed, rtl_out = run_rtl_simulation()
        rtl_time = time.time() - t0
        print(f"  RTL {'PASS' if rtl_passed else 'FAIL'} in {rtl_time:.1f}s")
    else:
        print("\n[4/6] RTL simulation: SKIPPED (use --rtl to enable)")

    # ================================================================
    # Step 5: YOLO decode
    # ================================================================
    print("\n[5/6] Decoding YOLO detection outputs...")

    det_l29, det_l35 = load_detection_outputs()
    if det_l29 is None:
        print("  [ERROR] Cannot load detection outputs")
        return 1

    print(f"  Detection head 1 (layer 29): {det_l29.shape} — {H0//16}x{W0//16} grid")
    print(f"  Detection head 2 (layer 35): {det_l35.shape} — {H0//8}x{W0//8} grid")

    # Decode both scales
    all_boxes = []
    for scale_idx, (det_tensor, anchors) in enumerate(
            zip([det_l29, det_l35], ANCHORS)):
        boxes = decode_yolo(det_tensor, anchors, num_classes=80,
                            input_size=(H0, W0))
        print(f"  Scale {scale_idx}: {len(boxes)} raw predictions (before threshold)")
        if len(boxes) > 0:
            all_boxes.append(boxes)

    if all_boxes:
        all_boxes = np.concatenate(all_boxes, axis=0)
        # Filter by confidence
        mask = all_boxes[:, 4] > args.conf_threshold
        filtered = all_boxes[mask]
        print(f"  After conf > {args.conf_threshold}: {len(filtered)} detections")
    else:
        filtered = np.zeros((0, 6))
        print(f"  No predictions above threshold")

    # NMS
    final_dets = nms(filtered, iou_threshold=args.nms_threshold)
    print(f"  After NMS (IoU>{args.nms_threshold}): {len(final_dets)} detections")

    # Format detections
    detections = []
    for row in final_dets:
        x1, y1, x2, y2, conf, cls_id = row
        cls_id = int(cls_id)
        detections.append({
            'box': [float(x1), float(y1), float(x2), float(y2)],
            'score': float(conf),
            'class_id': cls_id,
            'class_name': COCO_CLASSES[cls_id] if cls_id < 80 else f"class_{cls_id}",
        })

    # Print detections
    if detections:
        print(f"\n  Detected objects:")
        for i, d in enumerate(detections[:30]):
            print(f"    [{i:2d}] {d['class_name']:15s}  conf={d['score']:.4f}  "
                  f"box=({d['box'][0]:.3f}, {d['box'][1]:.3f}, "
                  f"{d['box'][2]:.3f}, {d['box'][3]:.3f})")
        if len(detections) > 30:
            print(f"    ... and {len(detections) - 30} more")
    else:
        print(f"\n  No objects detected above threshold={args.conf_threshold}")
        print(f"  (This is expected with INT8 quantization at low resolution)")

    # ================================================================
    # Step 6: Visualization
    # ================================================================
    print("\n[6/6] Generating output...")
    out_dir = PROJECT_ROOT / "image_sim_out" / "system_demo"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Draw bounding boxes on original image
    if orig_img and detections:
        out_img_path = out_dir / "detection_result.png"
        if draw_detections(orig_img, detections, out_img_path):
            print(f"  Annotated image: {out_img_path}")
    elif orig_img:
        # Save original image even if no detections
        out_img_path = out_dir / "detection_result.png"
        orig_img.save(str(out_img_path))
        print(f"  Original image (no detections): {out_img_path}")

    # Save detection results as text
    det_txt_path = out_dir / "detections.txt"
    with open(det_txt_path, 'w') as f:
        f.write("YOLOv4-tiny Full System Demo — Detection Results\n")
        f.write(f"Image: {args.image or 'test pattern'}\n")
        f.write(f"DPU Resolution: {H0}x{W0}\n")
        f.write(f"Weights: {'real YOLOv4-tiny' if use_real else 'synthetic'}\n")
        f.write(f"Layers: {NUM_LAYERS}\n")
        f.write(f"Golden time: {golden_time:.1f}s\n")
        if rtl_passed is not None:
            f.write(f"RTL: {'PASS' if rtl_passed else 'FAIL'} ({rtl_time:.1f}s)\n")
        f.write(f"Detections: {len(detections)}\n")
        f.write(f"Conf threshold: {args.conf_threshold}\n")
        f.write(f"NMS threshold: {args.nms_threshold}\n\n")
        for d in detections:
            f.write(f"{d['class_name']},{d['score']:.4f},"
                    f"{d['box'][0]:.4f},{d['box'][1]:.4f},"
                    f"{d['box'][2]:.4f},{d['box'][3]:.4f}\n")
    print(f"  Detection log: {det_txt_path}")

    # Save raw detection tensors for analysis
    np.savez(out_dir / "detection_tensors.npz",
             layer29=det_l29, layer35=det_l35)
    print(f"  Raw tensors: {out_dir / 'detection_tensors.npz'}")

    # ================================================================
    # Summary
    # ================================================================
    print(f"\n{'=' * 72}")
    print(f"  SYSTEM DEMO COMPLETE")
    print(f"  {'=' * 68}")
    print(f"  Image:      {Path(args.image).name if args.image else 'test pattern'}")
    print(f"  DPU layers: {NUM_LAYERS} (full YOLOv4-tiny)")
    print(f"  Weights:    {'real YOLOv4-tiny (21 conv layers)' if use_real else 'synthetic'}")
    print(f"  Golden:     PASS ({golden_time:.1f}s)")
    if rtl_passed is not None:
        print(f"  RTL:        {'PASS' if rtl_passed else 'FAIL'} ({rtl_time:.1f}s)")
    else:
        print(f"  RTL:        skipped")
    print(f"  Detections: {len(detections)} objects")
    if detections:
        # Show top 5
        top5 = sorted(detections, key=lambda d: -d['score'])[:5]
        for d in top5:
            print(f"              - {d['class_name']} ({d['score']:.3f})")
    print(f"  Output:     {out_dir}")
    print(f"{'=' * 72}")

    return 0 if (rtl_passed is None or rtl_passed) else 1


if __name__ == "__main__":
    sys.exit(main())
