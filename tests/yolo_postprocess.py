#!/usr/bin/env python3
"""
YOLO Post-Processing: Backbone output -> Bounding Box Detections

YOLOv4-tiny architecture (full):
  Layers 0-17:  CSPDarknet backbone (implemented in DPU)
  Layers 18+:   FPN neck + detection heads (implemented here in software)

This module:
  1. Takes the DPU backbone output (after layer 17)
  2. Runs the remaining detection head layers in NumPy
  3. Decodes raw predictions -> bounding boxes
  4. Applies Non-Maximum Suppression (NMS)
  5. Returns final detections

Usage:
  python yolo_postprocess.py --backbone-output output.npz [--image input.png]
  python yolo_postprocess.py --run-golden     (runs full golden + postprocess)
"""

import argparse
import numpy as np
from pathlib import Path


# =============================================================================
# COCO class names (80 classes)
# =============================================================================
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

NUM_CLASSES = len(COCO_CLASSES)  # 80

# YOLOv4-tiny anchors (2 scales, 3 anchors each)
ANCHORS = [
    [(10, 14), (23, 27), (37, 58)],       # Scale 0 (large feature map)
    [(81, 82), (135, 169), (344, 319)],    # Scale 1 (small feature map)
]
NUM_ANCHORS = 3


# =============================================================================
# INT8 Convolution (software, for detection head layers)
# =============================================================================
def conv2d_int8(x, weight, bias, stride=1, pad=0):
    """INT8 convolution matching DPU behavior."""
    c_out, c_in, kh, kw = weight.shape
    n, c, h, w = x.shape if x.ndim == 4 else (1, *x.shape)
    x = x.reshape(n, c, h, w)

    if pad > 0:
        x = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)),
                    mode='constant', constant_values=0)
        h += 2 * pad
        w += 2 * pad

    h_out = (h - kh) // stride + 1
    w_out = (w - kw) // stride + 1
    out = np.zeros((n, c_out, h_out, w_out), dtype=np.int32)

    for co in range(c_out):
        for ky in range(kh):
            for kx in range(kw):
                for ci in range(c_in):
                    patch = x[0, ci, ky:ky + h_out * stride:stride,
                              kx:kx + w_out * stride:stride].astype(np.int32)
                    out[0, co] += patch * int(weight[co, ci, ky, kx])
        out[0, co] += bias[co]

    return out.squeeze(0)


def leaky_relu_int8(x):
    """Leaky ReLU: x >= 0 ? x : x >> 3 (matches DPU hardware)."""
    return np.where(x >= 0, x, x >> 3)


def requantize(x, scale=655):
    """Requantize INT32 -> INT8 with scale."""
    y = (x.astype(np.int64) * scale) >> 16
    return np.clip(y, -128, 127).astype(np.int8)


# =============================================================================
# Detection Head (runs after DPU backbone)
# =============================================================================
class DetectionHead:
    """
    YOLOv4-tiny detection head.

    After the backbone (layers 0-17), we need:
      - Layer 18: Conv1x1 128->256 (from layer 15 output, 2x2)
      - Layer 19: Conv1x1 256->255 (detection output scale 1, no BN/act)
      - Layer 20: Upsample 2x (layer 18 output -> 4x4)
      - Layer 21: Route concat (upsample + layer 8 output = 128+256=384)
      - Layer 22: Conv1x1 384->128
      - Layer 23: Conv1x1 128->255 (detection output scale 0, no BN/act)

    For the minimal 16x16 DPU test input:
      - Scale 1 feature map: 1x1 -> 1 grid cell
      - Scale 0 feature map: 2x2 -> 4 grid cells

    Note: Weights for these layers would come from the full YOLOv4-tiny model.
    Here we use random weights for structural validation.
    """

    def __init__(self, num_classes=NUM_CLASSES, num_anchors=NUM_ANCHORS):
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        self.out_ch = num_anchors * (5 + num_classes)  # 3*(5+80) = 255

        # Initialize random weights for structural validation
        # In production, load from the full YOLOv4-tiny darknet weights
        np.random.seed(42)
        self.weights = {}
        self._init_weights()

    def _init_weights(self):
        """Initialize detection head weights (random for validation)."""
        sc = 0.01
        # Layer 18: Conv1x1 128->256
        self.weights['l18_w'] = (np.random.randn(256, 128, 1, 1) * sc).astype(np.int8)
        self.weights['l18_b'] = np.zeros(256, dtype=np.int32)
        # Layer 19: Conv1x1 256->255 (detection)
        self.weights['l19_w'] = (np.random.randn(self.out_ch, 256, 1, 1) * sc).astype(np.int8)
        self.weights['l19_b'] = np.zeros(self.out_ch, dtype=np.int32)
        # Layer 22: Conv1x1 384->128
        self.weights['l22_w'] = (np.random.randn(128, 384, 1, 1) * sc).astype(np.int8)
        self.weights['l22_b'] = np.zeros(128, dtype=np.int32)
        # Layer 23: Conv1x1 128->255 (detection)
        self.weights['l23_w'] = (np.random.randn(self.out_ch, 128, 1, 1) * sc).astype(np.int8)
        self.weights['l23_b'] = np.zeros(self.out_ch, dtype=np.int32)

    def load_weights(self, path):
        """Load detection head weights from npz file."""
        data = np.load(path)
        for key in self.weights:
            if key in data:
                self.weights[key] = data[key]

    def forward(self, backbone_l15, backbone_l8, backbone_l17):
        """
        Run detection head.

        Args:
            backbone_l15: Layer 15 output (128, H, W) INT8
            backbone_l8:  Layer 8 output (128, H*2, W*2) INT8 (for skip)
            backbone_l17: Layer 17 output (256, H/2, W/2) INT8

        Returns:
            det_scale0: Raw detection tensor (255, H, W) for large feature map
            det_scale1: Raw detection tensor (255, H/2, W/2) for small feature map
        """
        # Scale 1: small feature map (from backbone_l15)
        # Layer 18: Conv1x1 128->256
        l18 = conv2d_int8(backbone_l15, self.weights['l18_w'], self.weights['l18_b'])
        l18 = requantize(leaky_relu_int8(l18))

        # Layer 19: Conv1x1 256->255 (detection, no activation)
        det_scale1 = conv2d_int8(l18, self.weights['l19_w'], self.weights['l19_b'])

        # Scale 0: large feature map
        # Layer 20: Upsample 2x (nearest neighbor)
        _, h18, w18 = l18.shape
        l20 = np.repeat(np.repeat(l18, 2, axis=1), 2, axis=2)  # upsample

        # Layer 21: Route concat (upsample + layer 8 skip)
        # Ensure shapes match
        _, h8, w8 = backbone_l8.shape
        _, h20, w20 = l20.shape
        min_h = min(h8, h20)
        min_w = min(w8, w20)
        l21 = np.concatenate([l20[:, :min_h, :min_w],
                              backbone_l8[:, :min_h, :min_w]], axis=0)

        # Layer 22: Conv1x1 384->128
        # Adjust if concat channels don't match expected
        c_concat = l21.shape[0]
        if c_concat != 384:
            # Pad or trim to 384
            if c_concat < 384:
                l21 = np.pad(l21, ((0, 384 - c_concat), (0, 0), (0, 0)))
            else:
                l21 = l21[:384]

        l22 = conv2d_int8(l21, self.weights['l22_w'], self.weights['l22_b'])
        l22 = requantize(leaky_relu_int8(l22))

        # Layer 23: Conv1x1 128->255 (detection, no activation)
        det_scale0 = conv2d_int8(l22, self.weights['l23_w'], self.weights['l23_b'])

        return det_scale0, det_scale1


# =============================================================================
# YOLO Decode: raw predictions -> bounding boxes
# =============================================================================
def sigmoid(x):
    """Numerically stable sigmoid for float arrays."""
    x = np.clip(x.astype(np.float64), -500, 500)
    return 1.0 / (1.0 + np.exp(-x))


def decode_yolo(raw_tensor, anchors, num_classes, input_size):
    """
    Decode a single YOLO detection tensor.

    Args:
        raw_tensor: (num_anchors*(5+num_classes), grid_h, grid_w) INT32
        anchors:    list of (anchor_w, anchor_h) for this scale
        num_classes: number of classes
        input_size: (input_h, input_w) of the original image

    Returns:
        boxes: (N, 6) array of [x1, y1, x2, y2, confidence, class_id]
    """
    num_anchors = len(anchors)
    ch = num_anchors * (5 + num_classes)
    grid_h, grid_w = raw_tensor.shape[1], raw_tensor.shape[2]
    inp_h, inp_w = input_size

    # Reshape: (num_anchors, 5+num_classes, grid_h, grid_w)
    pred = raw_tensor.reshape(num_anchors, 5 + num_classes, grid_h, grid_w)
    pred = pred.astype(np.float64)

    # Scale raw INT32 values to reasonable range for sigmoid
    # (In a real system, the detection head outputs would be properly scaled)
    pred = pred / 256.0

    # Grid offsets
    grid_y, grid_x = np.meshgrid(np.arange(grid_h), np.arange(grid_w), indexing='ij')

    boxes = []
    for a in range(num_anchors):
        tx = pred[a, 0]  # center x offset
        ty = pred[a, 1]  # center y offset
        tw = pred[a, 2]  # width
        th = pred[a, 3]  # height
        to = pred[a, 4]  # objectness

        # Decode center
        bx = (sigmoid(tx) + grid_x) / grid_w
        by = (sigmoid(ty) + grid_y) / grid_h

        # Decode size
        aw, ah = anchors[a]
        bw = np.exp(np.clip(tw, -10, 10)) * aw / inp_w
        bh = np.exp(np.clip(th, -10, 10)) * ah / inp_h

        # Objectness
        obj = sigmoid(to)

        # Class probabilities
        cls_pred = pred[a, 5:]  # (num_classes, grid_h, grid_w)
        cls_prob = sigmoid(cls_pred)

        # Combined confidence
        conf = obj[np.newaxis] * cls_prob  # (num_classes, grid_h, grid_w)

        # Convert to corner format and collect
        for gy in range(grid_h):
            for gx in range(grid_w):
                for c in range(num_classes):
                    score = conf[c, gy, gx]
                    if score > 0.01:  # low threshold for collection
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


# =============================================================================
# Non-Maximum Suppression
# =============================================================================
def nms(boxes, iou_threshold=0.45):
    """
    Apply Non-Maximum Suppression.

    Args:
        boxes: (N, 6) array [x1, y1, x2, y2, score, class_id]
        iou_threshold: IoU threshold for suppression

    Returns:
        Filtered boxes array
    """
    if len(boxes) == 0:
        return boxes

    # Sort by score (descending)
    order = np.argsort(-boxes[:, 4])
    boxes = boxes[order]

    keep = []
    suppressed = np.zeros(len(boxes), dtype=bool)

    for i in range(len(boxes)):
        if suppressed[i]:
            continue
        keep.append(i)

        # Compute IoU with remaining boxes of same class
        for j in range(i + 1, len(boxes)):
            if suppressed[j]:
                continue
            if boxes[i, 5] != boxes[j, 5]:
                continue  # different class

            # Compute IoU
            xx1 = max(boxes[i, 0], boxes[j, 0])
            yy1 = max(boxes[i, 1], boxes[j, 1])
            xx2 = min(boxes[i, 2], boxes[j, 2])
            yy2 = min(boxes[i, 3], boxes[j, 3])

            inter = max(0, xx2 - xx1) * max(0, yy2 - yy1)
            area_i = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
            area_j = (boxes[j, 2] - boxes[j, 0]) * (boxes[j, 3] - boxes[j, 1])
            union = area_i + area_j - inter

            iou = inter / (union + 1e-10)
            if iou > iou_threshold:
                suppressed[j] = True

    return boxes[keep]


# =============================================================================
# Full pipeline: backbone output -> detections
# =============================================================================
def detect(backbone_outputs, input_size=(416, 416),
           conf_threshold=0.25, nms_threshold=0.45):
    """
    Full detection pipeline.

    Args:
        backbone_outputs: dict with keys 'layer8', 'layer15', 'layer17'
                         containing INT8 numpy arrays
        input_size: (H, W) of original input image
        conf_threshold: confidence threshold for filtering
        nms_threshold: IoU threshold for NMS

    Returns:
        detections: list of dicts with 'box', 'score', 'class_id', 'class_name'
    """
    # Run detection head
    head = DetectionHead()
    det0, det1 = head.forward(
        backbone_outputs['layer15'],
        backbone_outputs['layer8'],
        backbone_outputs['layer17']
    )

    # Decode both scales
    all_boxes = []
    boxes0 = decode_yolo(det0, ANCHORS[0], NUM_CLASSES, input_size)
    boxes1 = decode_yolo(det1, ANCHORS[1], NUM_CLASSES, input_size)

    if len(boxes0) > 0:
        all_boxes.append(boxes0)
    if len(boxes1) > 0:
        all_boxes.append(boxes1)

    if not all_boxes:
        return []

    all_boxes = np.concatenate(all_boxes, axis=0)

    # Filter by confidence
    mask = all_boxes[:, 4] > conf_threshold
    all_boxes = all_boxes[mask]

    if len(all_boxes) == 0:
        return []

    # Apply NMS
    final_boxes = nms(all_boxes, nms_threshold)

    # Format results
    detections = []
    for box in final_boxes:
        x1, y1, x2, y2, score, cls_id = box
        detections.append({
            'box': [float(x1), float(y1), float(x2), float(y2)],
            'score': float(score),
            'class_id': int(cls_id),
            'class_name': COCO_CLASSES[int(cls_id)] if int(cls_id) < NUM_CLASSES else 'unknown',
        })

    return detections


# =============================================================================
# Visualization
# =============================================================================
def draw_detections(image_path, detections, output_path="detections.png"):
    """Draw bounding boxes on image and save."""
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        print("PIL not available, skipping visualization")
        return

    img = Image.open(image_path).convert('RGB')
    draw = ImageDraw.Draw(img)
    w, h = img.size

    colors = ['red', 'green', 'blue', 'yellow', 'cyan', 'magenta',
              'orange', 'purple', 'pink', 'lime']

    for det in detections:
        x1, y1, x2, y2 = det['box']
        # Scale from [0,1] to pixel coords
        x1, x2 = int(x1 * w), int(x2 * w)
        y1, y2 = int(y1 * h), int(y2 * h)

        color = colors[det['class_id'] % len(colors)]
        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)

        label = f"{det['class_name']} {det['score']:.2f}"
        draw.text((x1, max(0, y1 - 12)), label, fill=color)

    img.save(output_path)
    print(f"Detections saved to {output_path}")


# =============================================================================
# CLI
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description='YOLO Post-Processing')
    parser.add_argument('--backbone-output', type=str,
                        help='Path to backbone output NPZ (layer8, layer15, layer17)')
    parser.add_argument('--image', type=str,
                        help='Input image for visualization')
    parser.add_argument('--conf-threshold', type=float, default=0.25)
    parser.add_argument('--nms-threshold', type=float, default=0.45)
    parser.add_argument('--run-golden', action='store_true',
                        help='Run golden model first, then post-process')
    parser.add_argument('--input-size', type=int, nargs=2, default=[416, 416],
                        help='Input image size (H W)')
    args = parser.parse_args()

    if args.run_golden:
        # Run the golden model and collect intermediate outputs
        print("Running golden model...")
        import sys
        sys.path.insert(0, str(Path(__file__).parent))
        from dpu_top_18layer_golden import run_golden_model
        outputs = run_golden_model()
        backbone_outputs = {
            'layer8':  outputs.get('layer8',  np.zeros((128, 4, 4), dtype=np.int8)),
            'layer15': outputs.get('layer15', np.zeros((128, 2, 2), dtype=np.int8)),
            'layer17': outputs.get('layer17', np.zeros((256, 1, 1), dtype=np.int8)),
        }
    elif args.backbone_output:
        data = np.load(args.backbone_output)
        backbone_outputs = {
            'layer8':  data['layer8'],
            'layer15': data['layer15'],
            'layer17': data['layer17'],
        }
    else:
        # Generate synthetic backbone output for testing
        print("No backbone output provided, using synthetic data for testing...")
        np.random.seed(123)
        backbone_outputs = {
            'layer8':  np.random.randint(-128, 127, (128, 4, 4), dtype=np.int8),
            'layer15': np.random.randint(-128, 127, (128, 2, 2), dtype=np.int8),
            'layer17': np.random.randint(-128, 127, (256, 1, 1), dtype=np.int8),
        }

    input_size = tuple(args.input_size)
    print(f"Input size: {input_size}")
    print(f"Backbone shapes: L8={backbone_outputs['layer8'].shape}, "
          f"L15={backbone_outputs['layer15'].shape}, "
          f"L17={backbone_outputs['layer17'].shape}")

    # Run detection pipeline
    print("\nRunning detection head + decode + NMS...")
    detections = detect(backbone_outputs, input_size,
                        args.conf_threshold, args.nms_threshold)

    # Print results
    print(f"\nFound {len(detections)} detection(s):")
    for i, det in enumerate(detections):
        print(f"  [{i}] {det['class_name']:15s} conf={det['score']:.3f}  "
              f"box=[{det['box'][0]:.3f}, {det['box'][1]:.3f}, "
              f"{det['box'][2]:.3f}, {det['box'][3]:.3f}]")

    # Visualization
    if args.image and detections:
        draw_detections(args.image, detections)

    print("\nPost-processing complete.")
    return detections


if __name__ == '__main__':
    main()
