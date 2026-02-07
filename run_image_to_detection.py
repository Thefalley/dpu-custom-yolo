#!/usr/bin/env python3
"""
Image -> SW detection -> same image through DPU pipeline (Python, then later RTL).

Stage 1:
  - Load an image (file or synthetic).
  - Run a detector in SW if available (YOLO or OpenCV face).
  - Convert image to INT8 tensor (C, H, W) 416x416 (YOLOv4-tiny input).
  - Run first layer(s) through Phase 3 functional model (placeholder weights).
  - Save layer-0 input tensor for future RTL comparison.

Usage (from project root):
  python run_image_to_detection.py
  python run_image_to_detection.py path/to/image.jpg
  python run_image_to_detection.py --synthetic   # Use synthetic image, no file
"""

import sys
import os
import argparse
from pathlib import Path

import numpy as np

# Project root
SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR
sys.path.insert(0, str(PROJECT_ROOT))

# Phase 3 functional model
from phase3_dpu_functional_model import conv_bn_leaky

# YOLOv4-tiny first layers
INPUT_H, INPUT_W = 416, 416
LAYER0_C_OUT = 32
LAYER0_STRIDE = 2
LAYER1_C_OUT = 64
LAYER1_STRIDE = 2
LAYER1_K = 3


def load_image_as_tensor(path: str = None, synthetic: bool = False):
    """
    Load image and return INT8 tensor (C, H, W) of shape (3, INPUT_H, INPUT_W).
    Values in [-128, 127]. If path is None and not synthetic, returns None.
    """
    if synthetic or path is None:
        np.random.seed(42)
        # Synthetic "image" in CHW, INT8
        img = np.random.randint(-128, 128, (3, INPUT_H, INPUT_W), dtype=np.int8)
        return img

    path = Path(path)
    if not path.exists():
        return None

    # Try OpenCV then PIL
    try:
        import cv2
        bgr = cv2.imread(str(path))
        if bgr is None:
            raise FileNotFoundError(f"cv2 could not read {path}")
        # HWC BGR -> resize to 416x416 -> CHW, then to INT8 [-128, 127]
        bgr = cv2.resize(bgr, (INPUT_W, INPUT_H))
        # 0-255 -> -128..127: x - 128
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        img = (rgb.astype(np.int32) - 128).clip(-128, 127).astype(np.int8)
        # HWC -> CHW
        img = np.transpose(img, (2, 0, 1))
        return img
    except Exception:
        pass

    try:
        from PIL import Image
        pil = Image.open(path)
        if pil.mode != "RGB":
            pil = pil.convert("RGB")
        pil = pil.resize((INPUT_W, INPUT_H), Image.BILINEAR)
        arr = np.array(pil)
        img = (arr.astype(np.int32) - 128).clip(-128, 127).astype(np.int8)
        img = np.transpose(img, (2, 0, 1))
        return img
    except Exception:
        pass

    return None


def run_face_detection(path: str):
    """Run OpenCV Haar face detector; return list of (x, y, w, h) or empty."""
    try:
        import cv2
        cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        img = cv2.imread(path)
        if img is None:
            return []
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = cascade.detectMultiScale(gray, 1.1, 4)
        return [tuple(map(int, (x, y, w, h))) for (x, y, w, h) in faces]
    except Exception:
        return []


def run_yolo_detection(path: str):
    """Run Ultralytics YOLO if available; return list of detections or empty."""
    try:
        from ultralytics import YOLO
        model = YOLO("yolov8n.pt")
        results = model(str(path), verbose=False)
        out = []
        for r in results:
            if r.boxes is None:
                continue
            for box in r.boxes:
                xyxy = box.xyxy[0].cpu().numpy()
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                out.append((list(map(float, xyxy)), cls, conf))
        return out
    except Exception:
        return []


def main():
    ap = argparse.ArgumentParser(description="Image -> SW detection -> DPU pipeline (Stage 1)")
    ap.add_argument("image_path", nargs="?", default=None, help="Path to image file")
    ap.add_argument("--synthetic", action="store_true", help="Use synthetic image (no file)")
    ap.add_argument("--no-detector", action="store_true", help="Skip SW detector (only tensor + layer)")
    ap.add_argument("--layers", type=int, default=1, metavar="N", help="Run N layers (1=layer0, 2=layer0+layer1, ...)")
    args = ap.parse_args()

    print("=" * 60)
    print("Image -> Detection -> DPU pipeline (Stage 1)")
    print("=" * 60)

    # 1) Load image as tensor
    if args.synthetic or not args.image_path:
        print("\n[1] Image: synthetic (3, 416, 416) INT8")
        img_tensor = load_image_as_tensor(synthetic=True)
    else:
        print(f"\n[1] Image: {args.image_path}")
        img_tensor = load_image_as_tensor(path=args.image_path)
        if img_tensor is None:
            print("  [WARN] Could not load file; using synthetic image.")
            img_tensor = load_image_as_tensor(synthetic=True)
    assert img_tensor.shape == (3, INPUT_H, INPUT_W), f"Expected (3,{INPUT_H},{INPUT_W}), got {img_tensor.shape}"
    print(f"  Tensor shape: {img_tensor.shape}, dtype: {img_tensor.dtype}")

    # 2) SW detector (optional)
    detections_face = []
    detections_yolo = []
    if not args.no_detector and args.image_path and Path(args.image_path).exists():
        print("\n[2] SW detector")
        detections_face = run_face_detection(args.image_path)
        if detections_face:
            print(f"  OpenCV faces: {len(detections_face)} -> {detections_face}")
        else:
            print("  OpenCV faces: none (or OpenCV not available)")
        detections_yolo = run_yolo_detection(args.image_path)
        if detections_yolo:
            print(f"  YOLO detections: {len(detections_yolo)}")
            for i, (xyxy, cls, conf) in enumerate(detections_yolo[:5]):
                print(f"    box {i+1}: xyxy={xyxy[:2]}..{xyxy[2:]}, class={cls}, conf={conf:.2f}")
        else:
            print("  YOLO: none (or ultralytics not installed)")
    else:
        print("\n[2] SW detector: skipped (--no-detector or no image path)")

    # 3) Layers (Phase 3) with placeholder weights
    sim_out = PROJECT_ROOT / "sim_out"
    if sim_out.exists() and not sim_out.is_dir():
        sim_out = PROJECT_ROOT / "image_sim_out"
    sim_out.mkdir(parents=True, exist_ok=True)
    np.save(sim_out / "image_input_layer0.npy", img_tensor)

    np.random.seed(123)
    num_layers = max(1, min(args.layers, 2))  # 1 or 2 for now
    print(f"\n[3] Running {num_layers} layer(s) (Phase 3 functional model)")

    w0 = np.random.randint(-30, 30, (LAYER0_C_OUT, 3, 3, 3), dtype=np.int8)
    b0 = np.random.randint(-500, 500, LAYER0_C_OUT, dtype=np.int32)
    layer0_out = conv_bn_leaky(img_tensor, w0, b0, scale=0.01, stride=LAYER0_STRIDE)
    expected_h0 = (INPUT_H + 2 - 3) // LAYER0_STRIDE + 1
    expected_w0 = (INPUT_W + 2 - 3) // LAYER0_STRIDE + 1
    assert layer0_out.shape == (LAYER0_C_OUT, expected_h0, expected_w0)
    print(f"  Layer0: {img_tensor.shape} -> {layer0_out.shape}")
    np.save(sim_out / "layer0_weights.npy", w0)
    np.save(sim_out / "layer0_bias.npy", b0)
    np.save(sim_out / "layer0_output_ref.npy", layer0_out)

    if num_layers >= 2:
        w1 = np.random.randint(-30, 30, (LAYER1_C_OUT, LAYER0_C_OUT, LAYER1_K, LAYER1_K), dtype=np.int8)
        b1 = np.random.randint(-500, 500, LAYER1_C_OUT, dtype=np.int32)
        layer1_out = conv_bn_leaky(layer0_out, w1, b1, scale=0.01, stride=LAYER1_STRIDE)
        expected_h1 = (expected_h0 + 2 - 3) // LAYER1_STRIDE + 1
        expected_w1 = (expected_w0 + 2 - 3) // LAYER1_STRIDE + 1
        assert layer1_out.shape == (LAYER1_C_OUT, expected_h1, expected_w1)
        print(f"  Layer1: {layer0_out.shape} -> {layer1_out.shape}")
        np.save(sim_out / "layer1_weights.npy", w1)
        np.save(sim_out / "layer1_bias.npy", b1)
        np.save(sim_out / "layer1_output_ref.npy", layer1_out)

    # 4) Summary
    print(f"\n[4] Saved for RTL: {sim_out}")
    print(f"     image_input_layer0.npy, layer0_weights/bias/output_ref.npy")
    if num_layers >= 2:
        print(f"     layer1_weights/bias/output_ref.npy")

    # Summary
    print("\n" + "-" * 60)
    if detections_face or detections_yolo:
        print("SW detections (reference):")
        if detections_face:
            print(f"  Faces: {len(detections_face)}")
        if detections_yolo:
            print(f"  YOLO: {len(detections_yolo)}")
    print("Pipeline: image -> INT8 tensor -> Layer0 (Python)", end="")
    if num_layers >= 2:
        print(" -> Layer1 (Python)", end="")
    print(" -> output shape OK.")
    print("Next: run same input through RTL layer when available; compare outputs.")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
