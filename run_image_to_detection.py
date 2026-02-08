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
from phase3_dpu_functional_model import conv_bn_leaky, route_split, route_concat, maxpool_2x2

# YOLOv4-tiny first layers
INPUT_H, INPUT_W = 416, 416
LAYER0_C_OUT = 32
LAYER0_STRIDE = 2
LAYER1_C_OUT = 64
LAYER1_STRIDE = 2
LAYER1_K = 3
LAYER2_C_OUT = 64
LAYER2_STRIDE = 1
LAYER2_K = 3
# Layer 3: route split (groups=2, group_id=1) -> takes second half of channels
LAYER3_GROUPS = 2
LAYER3_GROUP_ID = 1
# Layer 4: Conv 3x3, 32->32, stride 1
LAYER4_C_IN = 32
LAYER4_C_OUT = 32
LAYER4_STRIDE = 1
LAYER4_K = 3
# Layer 5: Conv 3x3, 32->32, stride 1
LAYER5_C_IN = 32
LAYER5_C_OUT = 32
LAYER5_STRIDE = 1
LAYER5_K = 3
# Layer 6: route concat (layer5 + layer4) -> 32+32 = 64 channels
# Layer 7: Conv 1x1, 64->64, stride 1
LAYER7_C_IN = 64
LAYER7_C_OUT = 64
LAYER7_STRIDE = 1
LAYER7_K = 1
# Layer 8: route concat (layer2 + layer7) -> 64+64 = 128 channels
# Layer 9: MaxPool 2x2, stride 2 -> 52x52
# Layer 10: Conv 3x3, 128->128, stride 1 (1152 MACs)
LAYER10_C_IN = 128
LAYER10_C_OUT = 128
LAYER10_STRIDE = 1
LAYER10_K = 3
# Layer 11: route split (groups=2, group_id=1) -> 64 channels
LAYER11_GROUPS = 2
LAYER11_GROUP_ID = 1
# Layer 12: Conv 3x3, 64->64, stride 1 (576 MACs)
LAYER12_C_IN = 64
LAYER12_C_OUT = 64
LAYER12_STRIDE = 1
LAYER12_K = 3
# Layer 13: Conv 3x3, 64->64, stride 1 (576 MACs)
LAYER13_C_IN = 64
LAYER13_C_OUT = 64
LAYER13_STRIDE = 1
LAYER13_K = 3
# Layer 14: route concat (layer13 + layer12) -> 128 channels
# Layer 15: Conv 1x1, 128->128, stride 1 (128 MACs)
LAYER15_C_IN = 128
LAYER15_C_OUT = 128
LAYER15_STRIDE = 1
LAYER15_K = 1
# Layer 16: route concat (layer10 + layer15) -> 256 channels
# Layer 17: MaxPool 2x2, stride 2 -> 26x26


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
    num_layers = max(1, min(args.layers, 18))  # 1-18 layers supported
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

    if num_layers >= 3:
        w2 = np.random.randint(-30, 30, (LAYER2_C_OUT, LAYER1_C_OUT, LAYER2_K, LAYER2_K), dtype=np.int8)
        b2 = np.random.randint(-500, 500, LAYER2_C_OUT, dtype=np.int32)
        layer2_out = conv_bn_leaky(layer1_out, w2, b2, scale=0.01, stride=LAYER2_STRIDE)
        expected_h2 = (expected_h1 + 2 - 3) // LAYER2_STRIDE + 1
        expected_w2 = (expected_w1 + 2 - 3) // LAYER2_STRIDE + 1
        assert layer2_out.shape == (LAYER2_C_OUT, expected_h2, expected_w2)
        print(f"  Layer2: {layer1_out.shape} -> {layer2_out.shape}")
        np.save(sim_out / "layer2_weights.npy", w2)
        np.save(sim_out / "layer2_bias.npy", b2)
        np.save(sim_out / "layer2_output_ref.npy", layer2_out)

    if num_layers >= 4:
        # Layer 3: route split (groups=2, group_id=1) - takes second half of channels
        layer3_out = route_split(layer2_out, groups=LAYER3_GROUPS, group_id=LAYER3_GROUP_ID)
        expected_c3 = LAYER2_C_OUT // LAYER3_GROUPS
        assert layer3_out.shape == (expected_c3, expected_h2, expected_w2)
        print(f"  Layer3 (route split): {layer2_out.shape} -> {layer3_out.shape}")
        np.save(sim_out / "layer3_output_ref.npy", layer3_out)
        # Save route config for reference
        import json
        route_config = {"type": "route_split", "groups": LAYER3_GROUPS, "group_id": LAYER3_GROUP_ID}
        with open(sim_out / "layer3_config.json", "w") as f:
            json.dump(route_config, f, indent=2)

    if num_layers >= 5:
        # Layer 4: Conv 3x3, 32->32, stride 1
        w4 = np.random.randint(-30, 30, (LAYER4_C_OUT, LAYER4_C_IN, LAYER4_K, LAYER4_K), dtype=np.int8)
        b4 = np.random.randint(-500, 500, LAYER4_C_OUT, dtype=np.int32)
        layer4_out = conv_bn_leaky(layer3_out, w4, b4, scale=0.01, stride=LAYER4_STRIDE)
        # Output size same as layer 3 (stride=1)
        assert layer4_out.shape == (LAYER4_C_OUT, expected_h2, expected_w2)
        print(f"  Layer4: {layer3_out.shape} -> {layer4_out.shape}")
        np.save(sim_out / "layer4_weights.npy", w4)
        np.save(sim_out / "layer4_bias.npy", b4)
        np.save(sim_out / "layer4_output_ref.npy", layer4_out)

    if num_layers >= 6:
        # Layer 5: Conv 3x3, 32->32, stride 1
        w5 = np.random.randint(-30, 30, (LAYER5_C_OUT, LAYER5_C_IN, LAYER5_K, LAYER5_K), dtype=np.int8)
        b5 = np.random.randint(-500, 500, LAYER5_C_OUT, dtype=np.int32)
        layer5_out = conv_bn_leaky(layer4_out, w5, b5, scale=0.01, stride=LAYER5_STRIDE)
        assert layer5_out.shape == (LAYER5_C_OUT, expected_h2, expected_w2)
        print(f"  Layer5: {layer4_out.shape} -> {layer5_out.shape}")
        np.save(sim_out / "layer5_weights.npy", w5)
        np.save(sim_out / "layer5_bias.npy", b5)
        np.save(sim_out / "layer5_output_ref.npy", layer5_out)

    if num_layers >= 7:
        # Layer 6: route concat (layer5 + layer4) -> 64 channels
        layer6_out = route_concat([layer5_out, layer4_out])
        assert layer6_out.shape == (64, expected_h2, expected_w2)
        print(f"  Layer6 (route concat): {layer5_out.shape} + {layer4_out.shape} -> {layer6_out.shape}")
        np.save(sim_out / "layer6_output_ref.npy", layer6_out)
        import json
        route_config = {"type": "route_concat", "layers": [-1, -2], "description": "layer5 + layer4"}
        with open(sim_out / "layer6_config.json", "w") as f:
            json.dump(route_config, f, indent=2)

    if num_layers >= 8:
        # Layer 7: Conv 1x1, 64->64, stride 1
        w7 = np.random.randint(-30, 30, (LAYER7_C_OUT, LAYER7_C_IN, LAYER7_K, LAYER7_K), dtype=np.int8)
        b7 = np.random.randint(-500, 500, LAYER7_C_OUT, dtype=np.int32)
        layer7_out = conv_bn_leaky(layer6_out, w7, b7, scale=0.01, stride=LAYER7_STRIDE, kernel_size=LAYER7_K)
        assert layer7_out.shape == (LAYER7_C_OUT, expected_h2, expected_w2)
        print(f"  Layer7: {layer6_out.shape} -> {layer7_out.shape}")
        np.save(sim_out / "layer7_weights.npy", w7)
        np.save(sim_out / "layer7_bias.npy", b7)
        np.save(sim_out / "layer7_output_ref.npy", layer7_out)

    if num_layers >= 9:
        # Layer 8: route concat (layer2 + layer7) -> 128 channels
        # Note: We need layer2_out which was saved earlier
        layer8_out = route_concat([layer2_out, layer7_out])
        assert layer8_out.shape == (128, expected_h2, expected_w2)
        print(f"  Layer8 (route concat): {layer2_out.shape} + {layer7_out.shape} -> {layer8_out.shape}")
        np.save(sim_out / "layer8_output_ref.npy", layer8_out)
        import json
        route_config = {"type": "route_concat", "layers": [-6, -1], "description": "layer2 + layer7"}
        with open(sim_out / "layer8_config.json", "w") as f:
            json.dump(route_config, f, indent=2)

    if num_layers >= 10:
        # Layer 9: MaxPool 2x2, stride 2
        layer9_out = maxpool_2x2(layer8_out, stride=2)
        expected_h9 = expected_h2 // 2
        expected_w9 = expected_w2 // 2
        assert layer9_out.shape == (128, expected_h9, expected_w9)
        print(f"  Layer9 (maxpool 2x2): {layer8_out.shape} -> {layer9_out.shape}")
        np.save(sim_out / "layer9_output_ref.npy", layer9_out)
        import json
        maxpool_config = {"type": "maxpool", "size": 2, "stride": 2}
        with open(sim_out / "layer9_config.json", "w") as f:
            json.dump(maxpool_config, f, indent=2)

    if num_layers >= 11:
        # Layer 10: Conv 3x3, 128->128, stride 1 (1152 MACs)
        w10 = np.random.randint(-30, 30, (LAYER10_C_OUT, LAYER10_C_IN, LAYER10_K, LAYER10_K), dtype=np.int8)
        b10 = np.random.randint(-500, 500, LAYER10_C_OUT, dtype=np.int32)
        layer10_out = conv_bn_leaky(layer9_out, w10, b10, scale=0.01, stride=LAYER10_STRIDE)
        assert layer10_out.shape == (LAYER10_C_OUT, expected_h9, expected_w9)
        print(f"  Layer10: {layer9_out.shape} -> {layer10_out.shape}")
        np.save(sim_out / "layer10_weights.npy", w10)
        np.save(sim_out / "layer10_bias.npy", b10)
        np.save(sim_out / "layer10_output_ref.npy", layer10_out)

    if num_layers >= 12:
        # Layer 11: route split (groups=2, group_id=1) -> 64 channels
        layer11_out = route_split(layer10_out, groups=LAYER11_GROUPS, group_id=LAYER11_GROUP_ID)
        assert layer11_out.shape == (64, expected_h9, expected_w9)
        print(f"  Layer11 (route split): {layer10_out.shape} -> {layer11_out.shape}")
        np.save(sim_out / "layer11_output_ref.npy", layer11_out)
        import json
        route_config = {"type": "route_split", "groups": LAYER11_GROUPS, "group_id": LAYER11_GROUP_ID}
        with open(sim_out / "layer11_config.json", "w") as f:
            json.dump(route_config, f, indent=2)

    if num_layers >= 13:
        # Layer 12: Conv 3x3, 64->64, stride 1 (576 MACs)
        w12 = np.random.randint(-30, 30, (LAYER12_C_OUT, LAYER12_C_IN, LAYER12_K, LAYER12_K), dtype=np.int8)
        b12 = np.random.randint(-500, 500, LAYER12_C_OUT, dtype=np.int32)
        layer12_out = conv_bn_leaky(layer11_out, w12, b12, scale=0.01, stride=LAYER12_STRIDE)
        assert layer12_out.shape == (LAYER12_C_OUT, expected_h9, expected_w9)
        print(f"  Layer12: {layer11_out.shape} -> {layer12_out.shape}")
        np.save(sim_out / "layer12_weights.npy", w12)
        np.save(sim_out / "layer12_bias.npy", b12)
        np.save(sim_out / "layer12_output_ref.npy", layer12_out)

    if num_layers >= 14:
        # Layer 13: Conv 3x3, 64->64, stride 1 (576 MACs)
        w13 = np.random.randint(-30, 30, (LAYER13_C_OUT, LAYER13_C_IN, LAYER13_K, LAYER13_K), dtype=np.int8)
        b13 = np.random.randint(-500, 500, LAYER13_C_OUT, dtype=np.int32)
        layer13_out = conv_bn_leaky(layer12_out, w13, b13, scale=0.01, stride=LAYER13_STRIDE)
        assert layer13_out.shape == (LAYER13_C_OUT, expected_h9, expected_w9)
        print(f"  Layer13: {layer12_out.shape} -> {layer13_out.shape}")
        np.save(sim_out / "layer13_weights.npy", w13)
        np.save(sim_out / "layer13_bias.npy", b13)
        np.save(sim_out / "layer13_output_ref.npy", layer13_out)

    if num_layers >= 15:
        # Layer 14: route concat (layer13 + layer12) -> 128 channels
        layer14_out = route_concat([layer13_out, layer12_out])
        assert layer14_out.shape == (128, expected_h9, expected_w9)
        print(f"  Layer14 (route concat): {layer13_out.shape} + {layer12_out.shape} -> {layer14_out.shape}")
        np.save(sim_out / "layer14_output_ref.npy", layer14_out)
        import json
        route_config = {"type": "route_concat", "layers": [-1, -2], "description": "layer13 + layer12"}
        with open(sim_out / "layer14_config.json", "w") as f:
            json.dump(route_config, f, indent=2)

    if num_layers >= 16:
        # Layer 15: Conv 1x1, 128->128, stride 1 (128 MACs)
        w15 = np.random.randint(-30, 30, (LAYER15_C_OUT, LAYER15_C_IN, LAYER15_K, LAYER15_K), dtype=np.int8)
        b15 = np.random.randint(-500, 500, LAYER15_C_OUT, dtype=np.int32)
        layer15_out = conv_bn_leaky(layer14_out, w15, b15, scale=0.01, stride=LAYER15_STRIDE, kernel_size=LAYER15_K)
        assert layer15_out.shape == (LAYER15_C_OUT, expected_h9, expected_w9)
        print(f"  Layer15: {layer14_out.shape} -> {layer15_out.shape}")
        np.save(sim_out / "layer15_weights.npy", w15)
        np.save(sim_out / "layer15_bias.npy", b15)
        np.save(sim_out / "layer15_output_ref.npy", layer15_out)

    if num_layers >= 17:
        # Layer 16: route concat (layer10 + layer15) -> 256 channels
        layer16_out = route_concat([layer10_out, layer15_out])
        assert layer16_out.shape == (256, expected_h9, expected_w9)
        print(f"  Layer16 (route concat): {layer10_out.shape} + {layer15_out.shape} -> {layer16_out.shape}")
        np.save(sim_out / "layer16_output_ref.npy", layer16_out)
        import json
        route_config = {"type": "route_concat", "layers": [-6, -1], "description": "layer10 + layer15"}
        with open(sim_out / "layer16_config.json", "w") as f:
            json.dump(route_config, f, indent=2)

    if num_layers >= 18:
        # Layer 17: MaxPool 2x2, stride 2 -> 26x26
        layer17_out = maxpool_2x2(layer16_out, stride=2)
        expected_h17 = expected_h9 // 2
        expected_w17 = expected_w9 // 2
        assert layer17_out.shape == (256, expected_h17, expected_w17)
        print(f"  Layer17 (maxpool 2x2): {layer16_out.shape} -> {layer17_out.shape}")
        np.save(sim_out / "layer17_output_ref.npy", layer17_out)
        import json
        maxpool_config = {"type": "maxpool", "size": 2, "stride": 2}
        with open(sim_out / "layer17_config.json", "w") as f:
            json.dump(maxpool_config, f, indent=2)

    # 4) Summary
    print(f"\n[4] Saved for RTL: {sim_out}")
    print(f"     image_input_layer0.npy, layer0_weights/bias/output_ref.npy")
    if num_layers >= 2:
        print(f"     layer1_weights/bias/output_ref.npy")
    if num_layers >= 3:
        print(f"     layer2_weights/bias/output_ref.npy")
    if num_layers >= 4:
        print(f"     layer3_output_ref.npy, layer3_config.json (route split)")
    if num_layers >= 5:
        print(f"     layer4_weights/bias/output_ref.npy")
    if num_layers >= 6:
        print(f"     layer5_weights/bias/output_ref.npy")
    if num_layers >= 7:
        print(f"     layer6_output_ref.npy, layer6_config.json (route concat)")
    if num_layers >= 8:
        print(f"     layer7_weights/bias/output_ref.npy (1x1 conv)")
    if num_layers >= 9:
        print(f"     layer8_output_ref.npy, layer8_config.json (route concat)")
    if num_layers >= 10:
        print(f"     layer9_output_ref.npy, layer9_config.json (maxpool 2x2)")
    if num_layers >= 11:
        print(f"     layer10_weights/bias/output_ref.npy (3x3 conv 128ch)")
    if num_layers >= 12:
        print(f"     layer11_output_ref.npy (route split)")
    if num_layers >= 13:
        print(f"     layer12_weights/bias/output_ref.npy")
    if num_layers >= 14:
        print(f"     layer13_weights/bias/output_ref.npy")
    if num_layers >= 15:
        print(f"     layer14_output_ref.npy (route concat)")
    if num_layers >= 16:
        print(f"     layer15_weights/bias/output_ref.npy (1x1 conv)")
    if num_layers >= 17:
        print(f"     layer16_output_ref.npy (route concat)")
    if num_layers >= 18:
        print(f"     layer17_output_ref.npy (maxpool 2x2)")

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
    if num_layers >= 3:
        print(" -> Layer2 (Python)", end="")
    if num_layers >= 4:
        print(" -> Layer3 (route)", end="")
    if num_layers >= 5:
        print(" -> Layer4 (Python)", end="")
    if num_layers >= 6:
        print(" -> Layer5 (Python)", end="")
    if num_layers >= 7:
        print(" -> Layer6 (route)", end="")
    if num_layers >= 8:
        print(" -> Layer7 (1x1 conv)", end="")
    if num_layers >= 9:
        print(" -> Layer8 (route)", end="")
    if num_layers >= 10:
        print(" -> Layer9 (maxpool)", end="")
    if num_layers >= 11:
        print(" -> Layer10", end="")
    if num_layers >= 12:
        print(" -> Layer11 (route)", end="")
    if num_layers >= 13:
        print(" -> Layer12", end="")
    if num_layers >= 14:
        print(" -> Layer13", end="")
    if num_layers >= 15:
        print(" -> Layer14 (route)", end="")
    if num_layers >= 16:
        print(" -> Layer15 (1x1)", end="")
    if num_layers >= 17:
        print(" -> Layer16 (route)", end="")
    if num_layers >= 18:
        print(" -> Layer17 (maxpool)", end="")
    print(" -> output shape OK.")
    print("Next: run same input through RTL layer when available; compare outputs.")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
